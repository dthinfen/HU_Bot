"""
Subprocess Vectorized Environment for Parallel Training

Uses multiprocessing to run environments in parallel, bypassing Python's GIL.
This allows utilizing multiple CPU cores for environment stepping.
"""

import numpy as np
from multiprocessing import Process, Pipe
from typing import List, Optional, Callable, Tuple
import cloudpickle


def worker(remote, parent_remote, env_fn_wrapper):
    """Worker process that runs a single environment."""
    parent_remote.close()
    env = env_fn_wrapper.fn()
    encoder = env_fn_wrapper.encoder_fn()

    while True:
        try:
            cmd, data = remote.recv()

            if cmd == 'step':
                action = data
                obs_raw, reward, done, info = env.step(action)
                obs = _get_obs(env, encoder)
                mask = env.get_action_mask()
                if not mask.any():
                    mask[1] = True  # Safety fallback
                remote.send((obs, mask, reward, done, info))

            elif cmd == 'reset':
                env.reset()
                obs = _get_obs(env, encoder)
                mask = env.get_action_mask()
                remote.send((obs, mask))

            elif cmd == 'reset_if_done':
                is_done, is_terminal = data
                needs_reset = is_done or is_terminal

                if not needs_reset:
                    mask = env.get_action_mask()
                    if not mask.any():
                        needs_reset = True

                reset_happened = False
                if needs_reset:
                    env.reset()
                    reset_happened = True

                obs = _get_obs(env, encoder)
                mask = env.get_action_mask()
                if not mask.any():
                    mask[1] = True
                remote.send((obs, mask, reset_happened))

            elif cmd == 'get_state':
                # Return state info for checking terminal
                is_terminal = env.state.is_terminal() if env.state else True
                remote.send(is_terminal)

            elif cmd == 'set_opponent':
                # Can't easily pass callable to subprocess, use flag instead
                env.opponent_policy = None  # Will use random in subprocess
                remote.send(None)

            elif cmd == 'close':
                remote.close()
                break

        except EOFError:
            break


def _get_obs(env, encoder):
    """Get encoded observation from environment."""
    return encoder.encode(
        hole_cards=[(c.rank - 2, c.suit) for c in env.state.hero_hole],
        board_cards=[(c.rank - 2, c.suit) for c in env.state.board],
        pot=env.state.pot,
        hero_stack=env.state.hero_stack,
        villain_stack=env.state.villain_stack,
        hero_invested=env.state.hero_invested_this_street,
        villain_invested=env.state.villain_invested_this_street,
        street=env.state.street,
        is_button=(env.state.button == 0),
        action_history=encoder._parse_action_history(env.state.action_history)
    )


class EnvFnWrapper:
    """Wrapper to pickle environment creation function."""
    def __init__(self, fn, encoder_fn):
        self.fn = fn
        self.encoder_fn = encoder_fn


class SubprocVecEnv:
    """
    Vectorized environment using subprocesses.

    Each environment runs in its own process, allowing true parallelism.
    """

    def __init__(
        self,
        num_envs: int,
        starting_stack: float = 100.0,
        num_actions: int = 14,
        num_workers: Optional[int] = None
    ):
        self.num_envs = num_envs
        self.starting_stack = starting_stack
        self.num_actions = num_actions
        self.num_workers = num_workers or min(num_envs, 16)  # Cap at 16 workers

        # Track done states
        self.dones = np.zeros(num_envs, dtype=bool)

        # Create environment factory
        def make_env():
            from alphaholdem.src.env import HeadsUpEnv
            return HeadsUpEnv(starting_stack, num_actions)

        def make_encoder():
            from alphaholdem.src.encoder import AlphaHoldemEncoder
            return AlphaHoldemEncoder()

        # Create workers
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])
        self.processes = []

        env_fn = EnvFnWrapper(make_env, make_encoder)

        for work_remote, remote in zip(self.work_remotes, self.remotes):
            process = Process(target=worker, args=(work_remote, remote, env_fn))
            process.daemon = True
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.closed = False

    def set_opponent(self, policy: Optional[Callable]):
        """
        Note: In subprocess mode, we can't easily pass the neural net policy.
        Opponent will use random actions in subprocesses.
        For self-play, use the single-process VectorizedHeadsUpEnv instead.
        """
        pass  # Opponent stays random in subprocesses

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reset all environments."""
        for remote in self.remotes:
            remote.send(('reset', None))

        results = [remote.recv() for remote in self.remotes]
        obs = np.array([r[0] for r in results])
        masks = np.array([r[1] for r in results])
        self.dones[:] = False

        return obs, masks

    def reset_done_envs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reset only environments that are done."""
        # First check terminal states
        for i, remote in enumerate(self.remotes):
            remote.send(('get_state', None))
        terminals = [remote.recv() for remote in self.remotes]

        # Send reset commands
        for i, remote in enumerate(self.remotes):
            remote.send(('reset_if_done', (self.dones[i], terminals[i])))

        results = [remote.recv() for remote in self.remotes]
        obs = np.array([r[0] for r in results])
        masks = np.array([r[1] for r in results])
        reset_mask = np.array([r[2] for r in results])

        self.dones[reset_mask] = False

        return obs, masks, reset_mask

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Step all environments."""
        # Send step commands
        for i, (remote, action) in enumerate(zip(self.remotes, actions)):
            if not self.dones[i]:
                remote.send(('step', int(action)))
            else:
                remote.send(('reset_if_done', (True, True)))

        # Collect results
        observations = []
        action_masks = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = []

        for i, remote in enumerate(self.remotes):
            result = remote.recv()

            if self.dones[i]:
                # Was reset
                obs, mask, _ = result
                observations.append(obs)
                action_masks.append(mask)
                infos.append({})
                self.dones[i] = False
            else:
                obs, mask, reward, done, info = result
                observations.append(obs)
                action_masks.append(mask)
                rewards[i] = reward
                dones[i] = done
                self.dones[i] = done
                infos.append(info)

        return np.array(observations), np.array(action_masks), rewards, dones, infos

    def close(self):
        """Clean up workers."""
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))

        for process in self.processes:
            process.join()

        self.closed = True

    def __del__(self):
        self.close()
