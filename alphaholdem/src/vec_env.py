"""
Vectorized Environment for Parallel Self-Play Training

Runs multiple poker games in parallel on CPU, batches observations for GPU inference.
This dramatically speeds up training by keeping the GPU fed with work.
"""

import numpy as np
from typing import List, Optional, Callable, Tuple
from multiprocessing import Pool, cpu_count
import torch

from alphaholdem.src.env import HeadsUpEnv
from alphaholdem.src.encoder import AlphaHoldemEncoder


class VectorizedHeadsUpEnv:
    """
    Vectorized wrapper for HeadsUpEnv.

    Runs N environments in parallel, batches observations for efficient GPU inference.
    """

    def __init__(
        self,
        num_envs: int,
        starting_stack: float = 100.0,
        num_actions: int = 14
    ):
        self.num_envs = num_envs
        self.starting_stack = starting_stack
        self.num_actions = num_actions

        # Create environments
        self.envs = [HeadsUpEnv(starting_stack, num_actions) for _ in range(num_envs)]
        self.encoder = AlphaHoldemEncoder()

        # Track which envs are done
        self.dones = np.zeros(num_envs, dtype=bool)

        # Opponent policy (shared across all envs)
        self._opponent_policy: Optional[Callable] = None

        # Pre-allocate arrays to avoid repeated allocation
        self._obs_buffer = np.zeros((num_envs, 50, 4, 13), dtype=np.float32)
        self._mask_buffer = np.zeros((num_envs, num_actions), dtype=np.float32)
        self._reward_buffer = np.zeros(num_envs, dtype=np.float32)
        self._done_buffer = np.zeros(num_envs, dtype=bool)
        self._reset_buffer = np.zeros(num_envs, dtype=bool)

    def set_opponent(self, policy: Optional[Callable]):
        """Set opponent policy for all environments."""
        self._opponent_policy = policy
        for env in self.envs:
            if policy is not None:
                env.set_opponent(policy)

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset all environments.

        Returns:
            observations: (num_envs, 50, 4, 13) tensor
            action_masks: (num_envs, num_actions) boolean array
        """
        for i, env in enumerate(self.envs):
            env.reset()
            if self._opponent_policy is not None:
                env.set_opponent(self._opponent_policy)

            self._obs_buffer[i] = self._get_obs(env)
            self._mask_buffer[i] = env.get_action_mask()
            self.dones[i] = False

        return self._obs_buffer.copy(), self._mask_buffer.copy()

    def reset_done_envs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reset only environments that are done.

        Returns:
            observations: (num_envs, 50, 4, 13)
            action_masks: (num_envs, num_actions)
            reset_mask: (num_envs,) boolean - which envs were reset
        """
        self._reset_buffer[:] = False

        for i, env in enumerate(self.envs):
            # Reset if done, terminal, or if action mask would be empty
            needs_reset = self.dones[i] or env.state.is_terminal()

            if not needs_reset:
                # Also check if action mask is empty (shouldn't happen but safety check)
                mask = env.get_action_mask()
                if not mask.any():
                    needs_reset = True

            if needs_reset:
                env.reset()
                if self._opponent_policy is not None:
                    env.set_opponent(self._opponent_policy)
                self.dones[i] = False
                self._reset_buffer[i] = True

            self._obs_buffer[i] = self._get_obs(env)
            self._mask_buffer[i] = env.get_action_mask()

            # Safety: ensure at least one action is valid
            if not self._mask_buffer[i].any():
                self._mask_buffer[i, 1] = True  # Fallback to call

        return self._obs_buffer.copy(), self._mask_buffer.copy(), self._reset_buffer.copy()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """
        Step all environments with given actions.

        Args:
            actions: (num_envs,) array of actions

        Returns:
            observations: (num_envs, 50, 4, 13)
            action_masks: (num_envs, num_actions)
            rewards: (num_envs,)
            dones: (num_envs,)
            infos: list of info dicts
        """
        self._reward_buffer[:] = 0.0
        self._done_buffer[:] = False
        infos = [None] * self.num_envs  # Pre-allocate list

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            # Skip if already done (will be reset next call)
            if self.dones[i]:
                self._obs_buffer[i] = self._get_obs(env)
                self._mask_buffer[i] = env.get_action_mask()
                infos[i] = {}
                continue

            # Check if terminal from opponent's action
            if env.state.is_terminal():
                self._reward_buffer[i] = env.state.get_payoff(0)
                self._done_buffer[i] = True
                self.dones[i] = True
                self._obs_buffer[i] = self._get_obs(env)
                self._mask_buffer[i] = env.get_action_mask()
                infos[i] = {'terminal_from_opponent': True}
                continue

            # Step environment
            _, reward, done, info = env.step(int(action))

            self._reward_buffer[i] = reward
            self._done_buffer[i] = done
            self.dones[i] = done

            self._obs_buffer[i] = self._get_obs(env)
            self._mask_buffer[i] = env.get_action_mask()
            infos[i] = info

        return self._obs_buffer.copy(), self._mask_buffer.copy(), self._reward_buffer.copy(), self._done_buffer.copy(), infos

    def _get_obs(self, env: HeadsUpEnv) -> np.ndarray:
        """Get encoded observation from environment."""
        return self.encoder.encode(
            hole_cards=[(c.rank - 2, c.suit) for c in env.state.hero_hole],
            board_cards=[(c.rank - 2, c.suit) for c in env.state.board],
            pot=env.state.pot,
            hero_stack=env.state.hero_stack,
            villain_stack=env.state.villain_stack,
            hero_invested=env.state.hero_invested_this_street,
            villain_invested=env.state.villain_invested_this_street,
            street=env.state.street,
            is_button=(env.state.button == 0),
            action_history=self.encoder._parse_action_history(env.state.action_history)
        )

    def get_action_masks(self) -> np.ndarray:
        """Get action masks for all environments."""
        return np.array([env.get_action_mask() for env in self.envs])


class VectorizedRolloutBuffer:
    """
    Rollout buffer optimized for vectorized environments.

    Stores transitions from all environments and provides efficient batching.
    """

    def __init__(
        self,
        buffer_size: int,  # Steps per env
        num_envs: int,
        obs_shape: Tuple,
        num_actions: int,
        device: str = 'cpu'
    ):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.device = device

        # Pre-allocate arrays for efficiency
        total_size = buffer_size * num_envs
        self.observations = np.zeros((buffer_size, num_envs, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_envs), dtype=np.int64)
        self.log_probs = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_envs), dtype=bool)
        self.action_masks = np.zeros((buffer_size, num_envs, num_actions), dtype=bool)

        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,  # (num_envs, *obs_shape)
        actions: np.ndarray,  # (num_envs,)
        log_probs: np.ndarray,  # (num_envs,)
        values: np.ndarray,  # (num_envs,)
        rewards: np.ndarray,  # (num_envs,)
        dones: np.ndarray,  # (num_envs,)
        action_masks: np.ndarray  # (num_envs, num_actions)
    ):
        """Add a batch of transitions from all environments."""
        self.observations[self.pos] = obs
        self.actions[self.pos] = actions
        self.log_probs[self.pos] = log_probs
        self.values[self.pos] = values
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        self.action_masks[self.pos] = action_masks

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,  # (num_envs,)
        gamma: float,
        gae_lambda: float
    ):
        """Compute GAE and returns for all environments."""
        # Flatten for computation
        n_steps = self.buffer_size if self.full else self.pos

        self.advantages = np.zeros((n_steps, self.num_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, self.num_envs), dtype=np.float32)

        # GAE calculation (vectorized across envs)
        gae = np.zeros(self.num_envs, dtype=np.float32)

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_values = last_values
                next_dones = np.zeros(self.num_envs)  # Assume not done
            else:
                next_values = self.values[t + 1]
                next_dones = self.dones[t + 1].astype(np.float32)

            delta = self.rewards[t] + gamma * next_values * (1 - next_dones) - self.values[t]
            gae = delta + gamma * gae_lambda * (1 - next_dones) * gae
            self.advantages[t] = gae
            self.returns[t] = self.advantages[t] + self.values[t]

    def get_batches(self, batch_size: int):
        """Generator for minibatches, flattening across environments."""
        n_steps = self.buffer_size if self.full else self.pos
        total_samples = n_steps * self.num_envs

        # Flatten all arrays
        obs_flat = self.observations[:n_steps].reshape(-1, *self.obs_shape)
        actions_flat = self.actions[:n_steps].reshape(-1)
        log_probs_flat = self.log_probs[:n_steps].reshape(-1)
        values_flat = self.values[:n_steps].reshape(-1)
        advantages_flat = self.advantages.reshape(-1)
        returns_flat = self.returns.reshape(-1)
        masks_flat = self.action_masks[:n_steps].reshape(-1, self.num_actions)

        # Random permutation
        indices = np.random.permutation(total_samples)

        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            batch_indices = indices[start:end]

            yield {
                'observations': torch.tensor(obs_flat[batch_indices], dtype=torch.float32, device=self.device),
                'actions': torch.tensor(actions_flat[batch_indices], dtype=torch.long, device=self.device),
                'old_log_probs': torch.tensor(log_probs_flat[batch_indices], dtype=torch.float32, device=self.device),
                'old_values': torch.tensor(values_flat[batch_indices], dtype=torch.float32, device=self.device),
                'advantages': torch.tensor(advantages_flat[batch_indices], dtype=torch.float32, device=self.device),
                'returns': torch.tensor(returns_flat[batch_indices], dtype=torch.float32, device=self.device),
                'action_masks': torch.tensor(masks_flat[batch_indices], dtype=torch.bool, device=self.device),
            }

    def reset(self):
        """Clear buffer."""
        self.pos = 0
        self.full = False

    def __len__(self):
        return (self.buffer_size if self.full else self.pos) * self.num_envs
