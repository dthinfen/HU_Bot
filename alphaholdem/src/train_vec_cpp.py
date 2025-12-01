"""
AlphaHoldem Training with C++ Environment

Same RL/PPO algorithm as train_vec_v2.py but using FastEnv from C++ for 10-50x speedup.
"""

import torch
import numpy as np
import time
import argparse
import gc
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
from collections import deque

# Add cpp_solver build to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'cpp_solver' / 'build'))

try:
    import ares_solver
    HAS_CPP_ENV = True
    print("C++ FastEnv loaded successfully!")
except ImportError as e:
    HAS_CPP_ENV = False
    print(f"Warning: C++ FastEnv not available ({e}), falling back to Python env")

from alphaholdem.src.network import ActorCritic
from alphaholdem.src.ppo import PPO, PPOConfig


@dataclass
class TrainConfig:
    """Training configuration."""
    starting_stack: float = 100.0
    num_actions: int = 14

    # Vectorized training - can use more envs with C++ speed
    num_envs: int = 256
    steps_per_env: int = 128

    # Training
    total_timesteps: int = 100_000_000

    # Self-play
    k_best: int = 5
    update_opponent_every: int = 5
    eval_for_pool_every: int = 25
    min_updates_for_pool: int = 100
    warmup_self_play_updates: int = 50
    elo_games_per_matchup: int = 200
    initial_elo: float = 1500.0
    elo_k_factor: float = 16.0

    # Network
    hidden_dim: int = 256
    num_residual_blocks: int = 4
    fc_hidden_dim: int = 1024
    fc_num_layers: int = 4
    use_cnn: bool = True

    # Checkpointing
    save_every: int = 50
    eval_every: int = 10
    eval_games: int = 500

    # Memory management
    gc_every: int = 10

    # Paths
    checkpoint_dir: str = "alphaholdem/checkpoints"
    log_dir: str = "alphaholdem/logs"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def steps_per_update(self):
        return self.num_envs * self.steps_per_env


class CppVectorizedEnv:
    """Vectorized environment using C++ FastEnv instances."""

    def __init__(self, num_envs: int, starting_stack: float, num_actions: int):
        self.num_envs = num_envs
        self.starting_stack = starting_stack
        self.num_actions = num_actions

        # Create C++ environments
        self.envs = [ares_solver.FastEnv(starting_stack, num_actions) for _ in range(num_envs)]
        self.dones = np.zeros(num_envs, dtype=bool)

    def set_opponent(self, policy_fn):
        """Set opponent policy for all environments."""
        for env in self.envs:
            if policy_fn is not None:
                env.set_opponent(policy_fn)
            else:
                env.clear_opponent()

    def reset(self):
        """Reset all environments."""
        observations = []
        action_masks = []

        for i, env in enumerate(self.envs):
            obs = env.reset()
            mask = env.get_action_mask()
            observations.append(obs)
            action_masks.append(mask)
            self.dones[i] = False

        return np.array(observations), np.array(action_masks)

    def reset_done_envs(self):
        """Reset only environments that are done."""
        observations = []
        action_masks = []
        reset_mask = np.zeros(self.num_envs, dtype=bool)

        for i, env in enumerate(self.envs):
            if self.dones[i] or env.is_terminal():
                env.reset()
                self.dones[i] = False
                reset_mask[i] = True

            obs = env.get_observation()
            mask = env.get_action_mask()

            # Safety: ensure at least one action is valid
            if not mask.any():
                mask[1] = True

            observations.append(obs)
            action_masks.append(mask)

        return np.array(observations), np.array(action_masks), reset_mask

    def step(self, actions: np.ndarray):
        """Step all environments."""
        observations = []
        action_masks = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if self.dones[i]:
                obs = env.get_observation()
                mask = env.get_action_mask()
                observations.append(obs)
                action_masks.append(mask)
                infos.append({})
                continue

            obs, reward, done, info = env.step(int(action))

            rewards[i] = reward
            dones[i] = done
            self.dones[i] = done

            mask = env.get_action_mask()
            if not mask.any():
                mask[1] = True

            observations.append(obs)
            action_masks.append(mask)
            infos.append(info)

        return np.array(observations), np.array(action_masks), rewards, dones, infos


class VectorizedRolloutBuffer:
    """Rollout buffer for PPO training."""

    def __init__(self, buffer_size: int, num_envs: int, obs_shape: tuple, num_actions: int, device: str):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.device = device

        self.observations = np.zeros((buffer_size, num_envs, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_envs), dtype=np.int64)
        self.log_probs = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_envs), dtype=bool)
        self.action_masks = np.zeros((buffer_size, num_envs, num_actions), dtype=bool)

        self.pos = 0
        self.full = False

    def add(self, obs, actions, log_probs, values, rewards, dones, action_masks):
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

    def compute_returns_and_advantages(self, last_values, gamma, gae_lambda):
        n_steps = self.buffer_size if self.full else self.pos
        self.advantages = np.zeros((n_steps, self.num_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, self.num_envs), dtype=np.float32)

        gae = np.zeros(self.num_envs, dtype=np.float32)

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_values = last_values
                next_dones = np.zeros(self.num_envs)
            else:
                next_values = self.values[t + 1]
                next_dones = self.dones[t + 1].astype(np.float32)

            delta = self.rewards[t] + gamma * next_values * (1 - next_dones) - self.values[t]
            gae = delta + gamma * gae_lambda * (1 - next_dones) * gae
            self.advantages[t] = gae
            self.returns[t] = self.advantages[t] + self.values[t]

    def get_batches(self, batch_size: int):
        n_steps = self.buffer_size if self.full else self.pos
        total_samples = n_steps * self.num_envs

        obs_flat = self.observations[:n_steps].reshape(-1, *self.obs_shape)
        actions_flat = self.actions[:n_steps].reshape(-1)
        log_probs_flat = self.log_probs[:n_steps].reshape(-1)
        values_flat = self.values[:n_steps].reshape(-1)
        advantages_flat = self.advantages.reshape(-1)
        returns_flat = self.returns.reshape(-1)
        masks_flat = self.action_masks[:n_steps].reshape(-1, self.num_actions)

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
        self.pos = 0
        self.full = False


class KBestPool:
    """Pool of K-best agents."""

    def __init__(self, k: int, checkpoint_dir: str, initial_elo: float = 1500.0, k_factor: float = 16.0,
                 fc_hidden_dim: int = 1024, fc_num_layers: int = 4):
        self.k = k
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        self.fc_hidden_dim = fc_hidden_dim
        self.fc_num_layers = fc_num_layers
        self.agents: List[Dict] = []

    def add_agent(self, model: ActorCritic, elo: float, update_num: int, win_rate: float = 0.5):
        path = self.checkpoint_dir / f"agent_{update_num}.pt"
        torch.save(model.state_dict(), path)

        self.agents.append({
            'path': str(path),
            'elo': elo,
            'update_num': update_num,
            'win_rate': win_rate
        })

        self.agents.sort(key=lambda x: x['elo'], reverse=True)
        if len(self.agents) > self.k:
            removed = self.agents.pop()
            if Path(removed['path']).exists():
                Path(removed['path']).unlink()
            print(f"  [POOL] Evicted agent_{removed['update_num']} (ELO={removed['elo']:.0f})")

    def sample_opponent(self, model_class: type, num_actions: int = 14):
        if not self.agents:
            return None, None

        elos = np.array([a['elo'] for a in self.agents])
        weights = np.exp((elos - elos.min()) / 100)
        weights /= weights.sum()

        idx = np.random.choice(len(self.agents), p=weights)
        agent_info = self.agents[idx]

        opponent = model_class(
            input_channels=38, use_cnn=True, num_actions=num_actions,
            fc_hidden_dim=self.fc_hidden_dim, fc_num_layers=self.fc_num_layers
        )
        opponent.load_state_dict(torch.load(agent_info['path'], map_location='cpu'))
        opponent.eval()
        return opponent, agent_info

    def get_agent_by_idx(self, idx: int, model_class: type, num_actions: int = 14):
        if idx >= len(self.agents):
            return None
        agent_info = self.agents[idx]
        model = model_class(
            input_channels=38, use_cnn=True, num_actions=num_actions,
            fc_hidden_dim=self.fc_hidden_dim, fc_num_layers=self.fc_num_layers
        )
        model.load_state_dict(torch.load(agent_info['path'], map_location='cpu'))
        model.eval()
        return model

    def get_pool_stats(self) -> str:
        if not self.agents:
            return "Pool empty"
        elos = [a['elo'] for a in self.agents]
        return f"Pool({len(self.agents)}): ELO [{min(elos):.0f}-{max(elos):.0f}], avg={np.mean(elos):.0f}"


class CppTrainer:
    """Trainer using C++ FastEnv for maximum speed."""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = config.device

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        # Model
        self.model = ActorCritic(
            input_channels=38,
            use_cnn=config.use_cnn,
            hidden_dim=config.hidden_dim,
            num_actions=config.num_actions,
            num_residual_blocks=config.num_residual_blocks,
            fc_hidden_dim=config.fc_hidden_dim,
            fc_num_layers=config.fc_num_layers
        ).to(self.device)

        # PPO
        ppo_config = PPOConfig(
            learning_rate=1e-4,
            clip_ratio=0.2,
            num_epochs=3,
            minibatch_size=256,
            batch_size=config.steps_per_update
        )
        self.ppo = PPO(self.model, ppo_config, self.device)

        # Opponent pool
        self.opponent_pool = KBestPool(
            config.k_best, config.checkpoint_dir,
            initial_elo=config.initial_elo, k_factor=config.elo_k_factor,
            fc_hidden_dim=config.fc_hidden_dim, fc_num_layers=config.fc_num_layers
        )

        # C++ Vectorized environment
        if not HAS_CPP_ENV:
            raise RuntimeError("C++ FastEnv not available. Build cpp_solver first.")

        self.vec_env = CppVectorizedEnv(
            config.num_envs, config.starting_stack, config.num_actions
        )

        # Single env for evaluation
        self.eval_env = ares_solver.FastEnv(config.starting_stack, config.num_actions)

        # Buffer
        self.buffer = VectorizedRolloutBuffer(
            config.steps_per_env, config.num_envs,
            (38, 4, 13), config.num_actions, self.device
        )

        # Stats
        self.total_timesteps = 0
        self.total_hands = 0
        self.update_count = 0
        self.episode_rewards: deque = deque(maxlen=5000)
        self.opponent: Optional[ActorCritic] = None
        self._current_opp_elo = None

        # Warmup tracking
        self.warmup_model: Optional[ActorCritic] = None
        self.in_warmup = True

    def train(self):
        print(f"Starting C++ ACCELERATED training on {self.device}")
        print(f"Parallel environments: {self.config.num_envs}")
        print(f"Target timesteps: {self.config.total_timesteps:,}")
        print(f"Model parameters: {self.model.num_parameters:,}")
        print(f"Warmup phase: {self.config.warmup_self_play_updates} updates")

        total_updates = self.config.total_timesteps // self.config.steps_per_update
        self.ppo.setup_scheduler(total_updates)

        start_time = time.time()
        obs, action_masks = self.vec_env.reset()

        while self.total_timesteps < self.config.total_timesteps:
            obs, action_masks = self._collect_rollout(obs, action_masks)

            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                _, last_values = self.model(obs_tensor)
                last_values = last_values.cpu().numpy()

            self.buffer.compute_returns_and_advantages(
                last_values, self.ppo.config.gamma, self.ppo.config.gae_lambda
            )

            adv = self.buffer.advantages
            self.buffer.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

            stats = self._ppo_update()
            self.update_count += 1

            # Handle training phases
            if self.in_warmup:
                if self.update_count >= self.config.warmup_self_play_updates:
                    print(f"  [WARMUP] Completed warmup phase, switching to self-play")
                    self.warmup_model = ActorCritic(
                        input_channels=38, use_cnn=True, num_actions=self.config.num_actions,
                        fc_hidden_dim=self.config.fc_hidden_dim, fc_num_layers=self.config.fc_num_layers
                    ).to(self.device)
                    self.warmup_model.load_state_dict(self.model.state_dict())
                    self.warmup_model.eval()
                    self.opponent = self.warmup_model
                    self.vec_env.set_opponent(self._make_opponent_policy())
                    self.in_warmup = False
                    self._current_opp_elo = self.config.initial_elo

            # Update opponent
            if not self.in_warmup and self.update_count % self.config.update_opponent_every == 0:
                self._update_opponent()

            # Pool evaluation
            if (not self.in_warmup and
                self.update_count >= self.config.min_updates_for_pool and
                self.update_count % self.config.eval_for_pool_every == 0):
                self._evaluate_for_pool()

            # Save
            if self.update_count % self.config.save_every == 0:
                self._save_checkpoint()

            # Memory management
            if self.update_count % self.config.gc_every == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self._log_progress(stats, start_time)
            self.buffer.reset()

        print(f"\nTraining complete! Total time: {(time.time() - start_time)/3600:.1f}h")
        self._save_checkpoint(final=True)

    def _collect_rollout(self, obs, action_masks):
        self.model.eval()

        for step in range(self.config.steps_per_env):
            obs, action_masks, _ = self.vec_env.reset_done_envs()

            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                mask_tensor = torch.tensor(action_masks, dtype=torch.bool, device=self.device)

                logits, values = self.model(obs_tensor, mask_tensor)
                probs = torch.softmax(logits, dim=-1)

                # Exploration noise
                probs = probs * 0.99 + 0.01 / self.config.num_actions
                probs = probs / probs.sum(dim=-1, keepdim=True)

                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

                actions_np = actions.cpu().numpy()
                log_probs_np = log_probs.cpu().numpy()
                values_np = values.cpu().numpy()

            next_obs, next_masks, rewards, dones, infos = self.vec_env.step(actions_np)

            self.buffer.add(obs, actions_np, log_probs_np, values_np, rewards, dones, action_masks)

            self.total_timesteps += self.config.num_envs
            for done, reward in zip(dones, rewards):
                if done:
                    self.episode_rewards.append(reward)
                    self.total_hands += 1

            obs = next_obs
            action_masks = next_masks

        self.model.train()
        return obs, action_masks

    def _ppo_update(self):
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        for epoch in range(self.ppo.config.num_epochs):
            for batch in self.buffer.get_batches(self.ppo.config.minibatch_size):
                obs = batch['observations']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                advantages = batch['advantages']
                returns = batch['returns']
                action_masks = batch['action_masks']

                new_log_probs, values, entropy = self.model.evaluate_actions(obs, actions, action_masks)

                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.ppo.config.clip_ratio, 1 + self.ppo.config.clip_ratio)
                policy_loss = torch.max(-advantages * ratio, -advantages * clipped_ratio).mean()

                value_loss = 0.5 * ((values - returns) ** 2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

                self.ppo.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.ppo.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                num_updates += 1

        if self.ppo.scheduler:
            self.ppo.scheduler.step()

        return {
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'value_loss': total_value_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1),
        }

    def _make_opponent_policy(self):
        """Create opponent policy function for C++ env."""
        def policy(obs, mask):
            if self.opponent is None:
                valid = np.where(mask)[0]
                return int(np.random.choice(valid)) if len(valid) > 0 else 1

            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)

                logits, _ = self.opponent(obs_tensor, mask_tensor)
                probs = torch.softmax(logits, dim=-1)

                # Stochastic
                probs = probs * 0.95 + 0.05 / self.config.num_actions
                probs = probs / probs.sum()

                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                return action.item()

        return policy

    def _update_opponent(self):
        if len(self.opponent_pool.agents) == 0:
            if self.warmup_model is not None:
                self.opponent = self.warmup_model
                self._current_opp_elo = self.config.initial_elo
        else:
            opponent, agent_info = self.opponent_pool.sample_opponent(ActorCritic, self.config.num_actions)
            if opponent is not None:
                self.opponent = opponent.to(self.device)
                self.opponent.eval()
                self._current_opp_elo = agent_info['elo']
                self.vec_env.set_opponent(self._make_opponent_policy())
                print(f"  [OPPONENT] Switched to agent_{agent_info['update_num']} (ELO={agent_info['elo']:.0f})")

    def _evaluate_for_pool(self):
        if len(self.opponent_pool.agents) == 0:
            self.opponent_pool.add_agent(self.model, self.config.initial_elo, self.update_count)
            print(f"  [POOL] Added first agent_{self.update_count}")
        else:
            new_elo, win_rate, _ = self._evaluate_against_pool()
            self.opponent_pool.add_agent(self.model, new_elo, self.update_count, win_rate)
            print(f"  [POOL] Agent_{self.update_count}: ELO={new_elo:.0f}, win_rate={win_rate:.1%}")
            print(f"  [POOL] {self.opponent_pool.get_pool_stats()}")

    def _evaluate_against_pool(self):
        if len(self.opponent_pool.agents) == 0:
            return self.config.initial_elo, 0.5, []

        new_elo = self.config.initial_elo
        total_wins = 0
        total_games = 0
        results = []
        self.model.eval()

        for idx, agent_info in enumerate(self.opponent_pool.agents):
            opponent = self.opponent_pool.get_agent_by_idx(idx, ActorCritic, self.config.num_actions)
            if opponent is None:
                continue

            opponent = opponent.to(self.device)
            opponent.eval()

            old_opponent = self.opponent
            self.opponent = opponent

            wins, losses, draws = 0, 0, 0
            games = self.config.elo_games_per_matchup

            for _ in range(games):
                self.eval_env.reset()
                self.eval_env.set_opponent(self._make_opponent_policy())

                done = False
                game_reward = 0

                while not done:
                    if self.eval_env.is_terminal():
                        break
                    obs = self.eval_env.get_observation()
                    mask = self.eval_env.get_action_mask()

                    with torch.no_grad():
                        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                        mask_t = torch.tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
                        logits, _ = self.model(obs_t, mask_t)
                        probs = torch.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        action = dist.sample()

                    _, reward, done, _ = self.eval_env.step(action.item())
                    game_reward += reward

                if game_reward > 0.5:
                    wins += 1
                elif game_reward < -0.5:
                    losses += 1
                else:
                    draws += 1

            self.opponent = old_opponent
            total_wins += wins
            total_games += games

            opponent_elo = agent_info['elo']
            score = (wins + 0.5 * draws) / games
            expected = 1 / (1 + 10 ** ((opponent_elo - new_elo) / 400))
            new_elo += self.config.elo_k_factor * games * (score - expected)

            results.append({
                'opponent': agent_info['update_num'],
                'wins': wins, 'draws': draws, 'losses': losses, 'score': score
            })

        self.model.train()
        win_rate = total_wins / total_games if total_games > 0 else 0.5
        return new_elo, win_rate, results

    def _save_checkpoint(self, final: bool = False):
        suffix = "final" if final else f"{self.update_count}"
        path = Path(self.config.checkpoint_dir) / f"checkpoint_{suffix}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.ppo.optimizer.state_dict(),
            'update_count': self.update_count,
            'total_timesteps': self.total_timesteps,
            'total_hands': self.total_hands,
            'config': asdict(self.config)
        }, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint.get('update_count', 0)
        self.total_timesteps = checkpoint.get('total_timesteps', 0)
        self.total_hands = checkpoint.get('total_hands', 0)
        print(f"Loaded: update={self.update_count}, steps={self.total_timesteps:,}")

    def _log_progress(self, stats, start_time):
        elapsed = time.time() - start_time
        steps_per_sec = self.total_timesteps / max(elapsed, 1)
        eta = (self.config.total_timesteps - self.total_timesteps) / max(steps_per_sec, 1)

        mean_reward = np.mean(list(self.episode_rewards)) if self.episode_rewards else 0
        bb_per_100 = mean_reward * 100

        if self.in_warmup:
            phase_str = "WARMUP vs RANDOM"
        elif self.opponent is None:
            phase_str = "vs RANDOM"
        else:
            phase_str = f"vs ELO {self._current_opp_elo:.0f}" if self._current_opp_elo else "vs SELF"

        pool_str = f" | {self.opponent_pool.get_pool_stats()}" if len(self.opponent_pool.agents) > 0 else ""

        print(f"Update {self.update_count} | "
              f"Hands: {self.total_hands:,} | "
              f"bb/100: {bb_per_100:+.1f} | "
              f"{phase_str}{pool_str} | "
              f"Steps/s: {steps_per_sec:.0f} | "
              f"ETA: {eta/3600:.1f}h")


def main():
    parser = argparse.ArgumentParser(description="Train AlphaHoldem with C++ acceleration")
    parser.add_argument('--timesteps', type=int, default=100_000_000)
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--stack', type=float, default=100.0)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--checkpoint-dir', type=str, default='alphaholdem/checkpoints')
    parser.add_argument('--log-dir', type=str, default='alphaholdem/logs')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    config = TrainConfig(
        starting_stack=args.stack,
        total_timesteps=args.timesteps,
        num_envs=args.num_envs,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=device
    )

    trainer = CppTrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
