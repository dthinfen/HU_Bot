"""
AlphaHoldem Vectorized Training v2

Fixes:
- Self-play warmup before first pool addition
- Better ELO tracking with draws
- Memory-efficient rollout collection
- Periodic GC to prevent memory leaks
"""

import torch
import numpy as np
import time
import argparse
import gc
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
from pathlib import Path
import random
from collections import deque

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

from alphaholdem.src.network import ActorCritic
from alphaholdem.src.ppo import PPO, PPOConfig
from alphaholdem.src.vec_env import VectorizedHeadsUpEnv, VectorizedRolloutBuffer
from alphaholdem.src.encoder import AlphaHoldemEncoder

# Try to import C++ vectorized environment
try:
    import cpp_vec_env
    HAS_CPP_ENV = True
except ImportError:
    HAS_CPP_ENV = False
    cpp_vec_env = None


class CppVecEnvWrapper:
    """Wrapper for C++ vectorized environment to match Python API."""

    def __init__(self, num_envs: int, starting_stack: float, num_actions: int):
        self.num_envs = num_envs
        self.num_actions = num_actions
        self._env = cpp_vec_env.VectorizedEnv(num_envs, starting_stack)
        self.dones = np.zeros(num_envs, dtype=bool)

    def set_opponent(self, policy):
        # C++ env only supports random opponents
        # For self-play, the Python env should be used instead
        pass

    def reset(self):
        obs, masks = self._env.reset()
        self.dones[:] = False
        return obs, masks

    def reset_done_envs(self):
        obs, masks, reset_mask = self._env.reset_done_envs()
        self.dones[reset_mask] = False
        return obs, masks, reset_mask

    def step(self, actions):
        obs, masks, rewards, dones = self._env.step(actions.astype(np.int32))
        self.dones = dones
        # Return format matching Python env: (obs, masks, rewards, dones, infos)
        infos = [{}] * self.num_envs
        return obs, masks, rewards, dones, infos


class CppVecEnvV2Wrapper:
    """V2 wrapper with batched opponent inference support for self-play."""

    def __init__(self, num_envs: int, starting_stack: float, num_actions: int):
        self.num_envs = num_envs
        self.num_actions = num_actions
        self._env = cpp_vec_env.VectorizedEnvV2(num_envs, starting_stack)
        self.dones = np.zeros(num_envs, dtype=bool)
        self._opponent_policy = None
        self._use_nn_opponent = False

    def set_opponent(self, policy):
        """Set opponent policy for batched inference."""
        self._opponent_policy = policy
        self._use_nn_opponent = policy is not None

    def reset(self):
        obs, masks = self._env.reset()
        self.dones[:] = False
        return obs, masks

    def reset_done_envs(self):
        # Reset done envs and get initial state
        obs, masks = self._env.reset()  # V2 doesn't have reset_done_envs, use full reset
        # Actually need to handle this differently - step will auto-reset done envs
        obs, masks = self._env.get_obs_and_masks()
        reset_mask = self.dones.copy()
        self.dones[:] = False
        return obs, masks, reset_mask

    def step(self, hero_actions):
        """Step with batched opponent inference."""
        hero_actions = hero_actions.astype(np.int32)

        if not self._use_nn_opponent:
            # Random opponent - let C++ handle it
            obs, masks, rewards, dones = self._env.step(hero_actions, None)
        else:
            # NN opponent - get opponent obs and do batched inference
            opp_obs, opp_masks, env_indices, count = self._env.get_opponent_obs()

            if count > 0:
                # Batch inference for opponent
                opp_actions = np.zeros(self.num_envs, dtype=np.int32)
                opp_actions_batch = self._opponent_policy(
                    opp_obs[:count],
                    opp_masks[:count]
                )
                # Scatter back to full array
                for j in range(count):
                    opp_actions[env_indices[j]] = opp_actions_batch[j]

                obs, masks, rewards, dones = self._env.step(hero_actions, opp_actions)
            else:
                obs, masks, rewards, dones = self._env.step(hero_actions, None)

        self.dones = dones
        infos = [{}] * self.num_envs
        return obs, masks, rewards, dones, infos


@dataclass
class TrainConfig:
    """Training configuration."""
    starting_stack: float = 100.0
    num_actions: int = 14

    # Vectorized training
    num_envs: int = 64  # Reduced from 256 to prevent memory issues
    steps_per_env: int = 128

    # Training
    total_timesteps: int = 100_000_000

    # Self-play - key fixes here
    k_best: int = 5
    update_opponent_every: int = 5  # More frequent opponent updates
    eval_for_pool_every: int = 25  # More frequent pool evaluation
    min_updates_for_pool: int = 100  # Wait for 100 updates before first pool agent
    warmup_self_play_updates: int = 50  # Self-play warmup before pool
    elo_games_per_matchup: int = 200  # More games for better estimates
    initial_elo: float = 1500.0
    elo_k_factor: float = 16.0  # Reduced for stability

    # Network
    hidden_dim: int = 256
    num_residual_blocks: int = 4
    fc_hidden_dim: int = 1024
    fc_num_layers: int = 4
    use_cnn: bool = True

    # PPO hyperparameters
    entropy_coef: float = 0.05  # Entropy bonus (higher = more exploration)

    # Checkpointing
    save_every: int = 50
    eval_every: int = 10
    eval_games: int = 500

    # Memory management
    gc_every: int = 10  # Run garbage collection every N updates

    # C++ environment (10-100x faster)
    use_cpp_env: bool = False

    # Paths
    checkpoint_dir: str = "alphaholdem/checkpoints"
    log_dir: str = "alphaholdem/logs"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def steps_per_update(self):
        return self.num_envs * self.steps_per_env


class KBestPool:
    """Pool of K-best agents with improved ELO handling."""

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

    def sample_opponent(self, model_class: type, num_actions: int = 14) -> tuple:
        if not self.agents:
            return None, None

        # Weighted sampling by ELO (prefer stronger opponents)
        elos = np.array([a['elo'] for a in self.agents])
        weights = np.exp((elos - elos.min()) / 100)  # Softmax-ish
        weights /= weights.sum()

        idx = np.random.choice(len(self.agents), p=weights)
        agent_info = self.agents[idx]

        opponent = model_class(
            input_channels=50,
            use_cnn=True,
            num_actions=num_actions,
            fc_hidden_dim=self.fc_hidden_dim,
            fc_num_layers=self.fc_num_layers
        )
        opponent.load_state_dict(torch.load(agent_info['path'], map_location='cpu'))
        opponent.eval()
        return opponent, agent_info

    def get_agent_by_idx(self, idx: int, model_class: type, num_actions: int = 14) -> Optional[ActorCritic]:
        if idx >= len(self.agents):
            return None
        agent_info = self.agents[idx]
        model = model_class(
            input_channels=50,
            use_cnn=True,
            num_actions=num_actions,
            fc_hidden_dim=self.fc_hidden_dim,
            fc_num_layers=self.fc_num_layers
        )
        model.load_state_dict(torch.load(agent_info['path'], map_location='cpu'))
        model.eval()
        return model

    def get_pool_stats(self) -> str:
        if not self.agents:
            return "Pool empty"
        elos = [a['elo'] for a in self.agents]
        return f"Pool({len(self.agents)}): ELO [{min(elos):.0f}-{max(elos):.0f}], avg={np.mean(elos):.0f}"


class VectorizedTrainerV2:
    """Improved vectorized trainer with fixes."""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = config.device

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        # Model
        self.model = ActorCritic(
            input_channels=50,
            use_cnn=config.use_cnn,
            hidden_dim=config.hidden_dim,
            num_actions=config.num_actions,
            num_residual_blocks=config.num_residual_blocks,
            fc_hidden_dim=config.fc_hidden_dim,
            fc_num_layers=config.fc_num_layers
        ).to(self.device)

        # PPO with tuned hyperparameters
        ppo_config = PPOConfig(
            learning_rate=1e-4,  # Lower LR for stability
            clip_ratio=0.2,
            num_epochs=3,  # Fewer epochs to prevent overfitting
            minibatch_size=256,
            batch_size=config.steps_per_update
        )
        self.ppo = PPO(self.model, ppo_config, self.device)

        # Opponent pool
        self.opponent_pool = KBestPool(
            config.k_best,
            config.checkpoint_dir,
            initial_elo=config.initial_elo,
            k_factor=config.elo_k_factor,
            fc_hidden_dim=config.fc_hidden_dim,
            fc_num_layers=config.fc_num_layers
        )

        # Vectorized environment
        self.use_cpp_env = config.use_cpp_env and HAS_CPP_ENV
        if self.use_cpp_env:
            # Use V2 C++ env with batched opponent inference for full training
            self.vec_env = CppVecEnvV2Wrapper(
                config.num_envs,
                config.starting_stack,
                config.num_actions
            )
            self.py_vec_env = None  # Not needed with V2
        else:
            self.vec_env = VectorizedHeadsUpEnv(
                config.num_envs,
                config.starting_stack,
                config.num_actions
            )
            self.py_vec_env = None

        # Single env for evaluation
        from alphaholdem.src.env import HeadsUpEnv
        self.eval_env = HeadsUpEnv(config.starting_stack, config.num_actions)
        self.encoder = AlphaHoldemEncoder()

        # Vectorized buffer
        self.buffer = VectorizedRolloutBuffer(
            config.steps_per_env,
            config.num_envs,
            (50, 4, 13),
            config.num_actions,
            self.device
        )

        # Stats
        self.total_timesteps = 0
        self.total_hands = 0
        self.update_count = 0
        self.episode_rewards: deque = deque(maxlen=5000)  # Smaller window
        self.opponent: Optional[ActorCritic] = None
        self._current_opp_elo = None

        # Warmup tracking
        self.warmup_model: Optional[ActorCritic] = None
        self.in_warmup = True

    def train(self):
        """Main training loop."""
        print(f"Starting VECTORIZED training v2 on {self.device}")
        print(f"Parallel environments: {self.config.num_envs}")
        print(f"Target timesteps: {self.config.total_timesteps:,}")
        print(f"Model parameters: {self.model.num_parameters:,}")
        print(f"Warmup phase: {self.config.warmup_self_play_updates} updates")
        print(f"Entropy coefficient: {self.config.entropy_coef}")
        print(f"Using C++ env: {self.use_cpp_env} (available: {HAS_CPP_ENV})")

        total_updates = self.config.total_timesteps // self.config.steps_per_update
        self.ppo.setup_scheduler(total_updates)

        start_time = time.time()
        obs, action_masks = self.vec_env.reset()

        while self.total_timesteps < self.config.total_timesteps:
            # Collect rollout
            obs, action_masks = self._collect_vectorized_rollout(obs, action_masks)

            # PPO update
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                _, last_values = self.model(obs_tensor)
                last_values = last_values.cpu().numpy()

            self.buffer.compute_returns_and_advantages(
                last_values,
                self.ppo.config.gamma,
                self.ppo.config.gae_lambda
            )

            # Normalize advantages
            adv = self.buffer.advantages
            self.buffer.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

            # PPO epochs
            stats = self._ppo_update()
            self.update_count += 1

            # Handle training phases
            if self.in_warmup:
                if self.update_count >= self.config.warmup_self_play_updates:
                    # End warmup, save warmup model for self-play
                    print(f"  [WARMUP] Completed warmup phase, switching to self-play")

                    self.warmup_model = ActorCritic(
                        input_channels=50, use_cnn=True, num_actions=self.config.num_actions,
                        fc_hidden_dim=self.config.fc_hidden_dim, fc_num_layers=self.config.fc_num_layers
                    ).to(self.device)
                    self.warmup_model.load_state_dict(self.model.state_dict())
                    self.warmup_model.eval()
                    self.opponent = self.warmup_model

                    # Set batched opponent policy (works with both Python and C++ V2 env)
                    self.vec_env.set_opponent(self._batched_opponent_policy)
                    self.in_warmup = False
                    self._current_opp_elo = self.config.initial_elo

            # Update opponent
            if not self.in_warmup and self.update_count % self.config.update_opponent_every == 0:
                self._update_opponent()

            # Pool evaluation - only after min_updates
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

            # Logging
            self._log_progress(stats, start_time)

            # Reset buffer
            self.buffer.reset()

        print(f"\nTraining complete! Total time: {(time.time() - start_time)/3600:.1f}h")
        self._save_checkpoint(final=True)

    def _collect_vectorized_rollout(self, obs: np.ndarray, action_masks: np.ndarray):
        """Collect experience with proper handling."""
        self.model.eval()

        for step in range(self.config.steps_per_env):
            obs, action_masks, _ = self.vec_env.reset_done_envs()

            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                mask_tensor = torch.tensor(action_masks, dtype=torch.bool, device=self.device)

                logits, values = self.model(obs_tensor, mask_tensor)
                probs = torch.softmax(logits, dim=-1)

                # Add small exploration noise for stability
                probs = probs * 0.99 + 0.01 / self.config.num_actions
                # Re-apply mask to zero out invalid actions after adding noise
                probs = probs * mask_tensor.float()
                # Renormalize with epsilon to avoid numerical issues
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
                probs = probs.clamp(min=1e-8)
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
            for i, (done, reward) in enumerate(zip(dones, rewards)):
                if done:
                    self.episode_rewards.append(reward)
                    self.total_hands += 1

            obs = next_obs
            action_masks = next_masks

        self.model.train()
        return obs, action_masks

    def _ppo_update(self) -> Dict[str, float]:
        """PPO update with gradient accumulation."""
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

                loss = policy_loss + 0.5 * value_loss + self.config.entropy_coef * entropy_loss

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

    def _opponent_policy(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        """Opponent policy with exploration noise (single observation)."""
        if self.opponent is None:
            valid = np.where(action_mask)[0]
            return np.random.choice(valid)

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)

            # Use stochastic opponent (not deterministic) for more variance
            logits, _ = self.opponent(obs_tensor, mask_tensor)
            probs = torch.softmax(logits, dim=-1)

            # Small exploration
            probs = probs * 0.95 + 0.05 / self.config.num_actions
            # Re-apply mask to zero out invalid actions after adding noise
            probs = probs * mask_tensor.float()
            probs = probs / (probs.sum() + 1e-8)
            probs = probs.clamp(min=1e-8)
            probs = probs / probs.sum()

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action.item()

    def _batched_opponent_policy(self, obs_batch: np.ndarray, mask_batch: np.ndarray) -> np.ndarray:
        """Batched opponent policy for efficient GPU inference."""
        batch_size = obs_batch.shape[0]

        if self.opponent is None:
            # Random actions for each observation
            actions = np.zeros(batch_size, dtype=np.int32)
            for i in range(batch_size):
                valid = np.where(mask_batch[i])[0]
                actions[i] = np.random.choice(valid) if len(valid) > 0 else 1
            return actions

        with torch.no_grad():
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
            mask_tensor = torch.tensor(mask_batch, dtype=torch.bool, device=self.device)

            # Batched forward pass - single GPU call for all opponents
            logits, _ = self.opponent(obs_tensor, mask_tensor)
            probs = torch.softmax(logits, dim=-1)

            # Small exploration noise
            probs = probs * 0.95 + 0.05 / self.config.num_actions
            # Re-apply mask
            probs = probs * mask_tensor.float()
            # Renormalize (add eps to avoid div by zero)
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
            # Clamp to valid range for Categorical
            probs = probs.clamp(min=1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True)

            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            return actions.cpu().numpy().astype(np.int32)

    def _update_opponent(self):
        """Update opponent from pool or use self-play."""
        if len(self.opponent_pool.agents) == 0:
            # Self-play against current snapshot
            if self.warmup_model is not None:
                self.opponent = self.warmup_model
                self._current_opp_elo = self.config.initial_elo
        else:
            opponent, agent_info = self.opponent_pool.sample_opponent(ActorCritic, self.config.num_actions)
            if opponent is not None:
                self.opponent = opponent.to(self.device)
                self.opponent.eval()
                self._current_opp_elo = agent_info['elo']
                self.vec_env.set_opponent(self._batched_opponent_policy)
                print(f"  [OPPONENT] Switched to agent_{agent_info['update_num']} (ELO={agent_info['elo']:.0f})")

    def _evaluate_for_pool(self):
        """Evaluate and potentially add to pool."""
        if len(self.opponent_pool.agents) == 0:
            # First agent - add with initial ELO
            self.opponent_pool.add_agent(self.model, self.config.initial_elo, self.update_count)
            print(f"  [POOL] Added first agent_{self.update_count}")
        else:
            new_elo, win_rate, results = self._evaluate_against_pool()
            self.opponent_pool.add_agent(self.model, new_elo, self.update_count, win_rate)
            print(f"  [POOL] Agent_{self.update_count}: ELO={new_elo:.0f}, win_rate={win_rate:.1%}")
            print(f"  [POOL] {self.opponent_pool.get_pool_stats()}")

    def _evaluate_against_pool(self) -> tuple:
        """Evaluate with proper win counting."""
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

            wins = 0
            losses = 0
            draws = 0
            games = self.config.elo_games_per_matchup

            for game_idx in range(games):
                self.eval_env.reset()
                self.eval_env.set_opponent(self._eval_opponent_policy)
                done = False
                game_reward = 0

                while not done:
                    if self.eval_env.state.is_terminal():
                        game_reward = self.eval_env.state.get_payoff(0)
                        break
                    obs = self._get_eval_obs()
                    action_mask = self.eval_env.get_action_mask()
                    with torch.no_grad():
                        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                        mask_t = torch.tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
                        # Use stochastic for evaluation too
                        logits, _ = self.model(obs_t, mask_t)
                        probs = torch.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        action = dist.sample()
                    _, reward, done, _ = self.eval_env.step(action.item())
                    game_reward += reward

                # Better win/loss counting
                if game_reward > 0.5:  # Clear win
                    wins += 1
                elif game_reward < -0.5:  # Clear loss
                    losses += 1
                else:  # Draw (chopped pot, etc.)
                    draws += 1

            self.opponent = old_opponent
            total_wins += wins
            total_games += games

            # ELO update with draws counted as 0.5
            opponent_elo = agent_info['elo']
            score = (wins + 0.5 * draws) / games
            expected = 1 / (1 + 10 ** ((opponent_elo - new_elo) / 400))
            new_elo += self.config.elo_k_factor * games * (score - expected)

            results.append({
                'opponent': agent_info['update_num'],
                'games': games,
                'wins': wins,
                'draws': draws,
                'losses': losses,
                'score': score
            })

        self.model.train()
        win_rate = total_wins / total_games if total_games > 0 else 0.5
        return new_elo, win_rate, results

    def _eval_opponent_policy(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        """Opponent policy for evaluation - stochastic."""
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            logits, _ = self.opponent(obs_tensor, mask_tensor)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action.item()

    def _get_eval_obs(self) -> np.ndarray:
        return self.encoder.encode(
            hole_cards=[(c.rank - 2, c.suit) for c in self.eval_env.state.hero_hole],
            board_cards=[(c.rank - 2, c.suit) for c in self.eval_env.state.board],
            pot=self.eval_env.state.pot,
            hero_stack=self.eval_env.state.hero_stack,
            villain_stack=self.eval_env.state.villain_stack,
            hero_invested=self.eval_env.state.hero_invested_this_street,
            villain_invested=self.eval_env.state.villain_invested_this_street,
            street=self.eval_env.state.street,
            is_button=(self.eval_env.state.button == 0),
            action_history=self.encoder._parse_action_history(self.eval_env.state.action_history)
        )

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

    def _log_progress(self, stats: Dict, start_time: float):
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
    parser = argparse.ArgumentParser(description="Train AlphaHoldem v2")
    parser.add_argument('--timesteps', type=int, default=100_000_000)
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--stack', type=float, default=100.0)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--checkpoint-dir', type=str, default='alphaholdem/checkpoints')
    parser.add_argument('--log-dir', type=str, default='alphaholdem/logs')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--entropy-coef', type=float, default=0.05,
                        help='Entropy coefficient (higher = more exploration, default 0.05)')
    parser.add_argument('--use-cpp-env', action='store_true',
                        help='Use C++ environment for 10-100x faster warmup (auto-switches to Python for self-play)')
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
        device=device,
        entropy_coef=args.entropy_coef,
        use_cpp_env=args.use_cpp_env
    )

    trainer = VectorizedTrainerV2(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
