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


def get_raw_state_dict(model):
    """Get state dict from model, stripping torch.compile's _orig_mod. prefix if present."""
    state_dict = model.state_dict()
    # torch.compile wraps model and adds _orig_mod. prefix to all keys
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    return state_dict


@dataclass
class TrainConfig:
    """Training configuration."""
    starting_stack: float = 100.0  # Default/max stack
    min_stack: float = 10.0  # Minimum stack for variation
    vary_stacks: bool = True  # Whether to vary stack sizes
    num_actions: int = 14

    # Vectorized training - more envs = better GPU batching
    num_envs: int = 512  # Increased from 256 for better GPU utilization
    steps_per_env: int = 128

    # Speed optimizations (don't affect model/accuracy)
    use_mixed_precision: bool = True  # FP16 inference, FP32 training

    # Training
    total_timesteps: int = 100_000_000

    # Self-play
    k_best: int = 10  # Larger pool for more diversity
    update_opponent_every: int = 5
    eval_for_pool_every: int = 25
    min_updates_for_pool: int = 100
    warmup_self_play_updates: int = 50
    elo_games_per_matchup: int = 10000  # More hands = lower variance (poker is high variance)
    eval_num_envs: int = 256  # Parallel envs for fast evaluation
    initial_elo: float = 1500.0
    elo_k_factor: float = 32.0  # Standard chess K-factor (applied per matchup, not per game)

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
    gc_every: int = 50  # Less frequent GC for less overhead

    # Paths
    checkpoint_dir: str = "alphaholdem/checkpoints"
    log_dir: str = "alphaholdem/logs"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def steps_per_update(self):
        return self.num_envs * self.steps_per_env


class CppVectorizedEnv:
    """Vectorized environment using C++ FastEnv instances with BATCHED opponent inference."""

    def __init__(self, num_envs: int, starting_stack: float, num_actions: int,
                 vary_stacks: bool = True, min_stack: float = 10.0):
        self.num_envs = num_envs
        self.starting_stack = starting_stack
        self.num_actions = num_actions
        self.vary_stacks = vary_stacks
        self.min_stack = min_stack

        # Stack sizes to use (common tournament/cash game depths)
        # Distribution weighted towards medium stacks where most play happens
        if vary_stacks:
            self.stack_sizes = [10, 15, 20, 25, 30, 40, 50, 75, 100]
            self.stack_sizes = [s for s in self.stack_sizes if min_stack <= s <= starting_stack]
            if not self.stack_sizes:
                self.stack_sizes = [starting_stack]
        else:
            self.stack_sizes = [starting_stack]

        # Create C++ environments with varying stack sizes
        self.envs = []
        self.env_stacks = []  # Track which stack each env uses
        for i in range(num_envs):
            stack = self.stack_sizes[i % len(self.stack_sizes)]
            self.envs.append(ares_solver.FastEnv(float(stack), num_actions))
            self.env_stacks.append(stack)

        self.dones = np.zeros(num_envs, dtype=bool)

        # Opponent model for batched inference (set by trainer)
        self.opponent_model = None
        self.opponent_device = 'cpu'
        self.use_amp = False  # Mixed precision flag

        print(f"Created {num_envs} envs with stack sizes: {sorted(set(self.env_stacks))}")

    def set_opponent_model(self, model, device, use_amp=False):
        """Set opponent model for BATCHED inference (fast!)."""
        self.opponent_model = model
        self.opponent_device = device
        self.use_amp = use_amp and (device != 'cpu')  # AMP only on CUDA

    def clear_opponent(self):
        """Clear opponent (random actions)."""
        self.opponent_model = None

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

    def _batched_opponent_inference(self, indices):
        """Do BATCHED inference for all opponents that need to act."""
        if not indices:
            return {}

        # Collect observations and masks
        obs_list = []
        mask_list = []
        for i in indices:
            obs_list.append(self.envs[i].get_opponent_observation())
            mask_list.append(self.envs[i].get_opponent_action_mask())

        obs_batch = np.stack(obs_list)
        mask_batch = np.stack(mask_list)

        if self.opponent_model is None:
            # Random actions
            actions = {}
            for idx, i in enumerate(indices):
                valid = np.where(mask_batch[idx])[0]
                actions[i] = int(np.random.choice(valid)) if len(valid) > 0 else 1
            return actions

        # BATCHED neural network inference with FP16 (fast!)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.opponent_device)
            mask_tensor = torch.as_tensor(mask_batch, dtype=torch.bool, device=self.opponent_device)

            # Use mixed precision for faster inference (doesn't affect accuracy)
            with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
                logits, _ = self.opponent_model(obs_tensor, mask_tensor)

            probs = torch.softmax(logits.float(), dim=-1)

            # Stochastic with exploration (vectorized)
            probs = probs * 0.95 + 0.05 / self.num_actions
            probs = probs / probs.sum(dim=-1, keepdim=True)

            # Sample actions for all at once
            action_indices = torch.multinomial(probs, 1).squeeze(-1).cpu().numpy()

        return {i: int(action_indices[idx]) for idx, i in enumerate(indices)}

    def step(self, actions: np.ndarray):
        """Step all environments with BATCHED opponent inference."""
        observations = []
        action_masks = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = []

        # Phase 1: Apply hero actions
        needs_opponent = []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if self.dones[i]:
                observations.append(env.get_observation())
                action_masks.append(env.get_action_mask())
                infos.append({})
                continue

            # Use step_hero_only to not trigger per-env callback
            obs, reward, done, needs_opp = env.step_hero_only(int(action))

            rewards[i] = reward
            dones[i] = done
            self.dones[i] = done
            observations.append(obs)
            infos.append({'pot': env.get_pot()})

            if needs_opp and not done:
                needs_opponent.append(i)

            mask = env.get_action_mask()
            if not mask.any():
                mask[1] = True
            action_masks.append(mask)

        # Phase 2: BATCHED opponent inference (one forward pass for ALL opponents)
        while needs_opponent:
            opponent_actions = self._batched_opponent_inference(needs_opponent)

            # Apply opponent actions and check if more actions needed
            next_needs_opponent = []
            for i in needs_opponent:
                done, reward = self.envs[i].apply_opponent_action(opponent_actions[i])

                if done:
                    rewards[i] = reward
                    dones[i] = True
                    self.dones[i] = True
                elif self.envs[i].needs_opponent_action():
                    # Opponent acts again (e.g., after check-raise)
                    next_needs_opponent.append(i)

                # Update observation and mask
                observations[i] = self.envs[i].get_observation()
                mask = self.envs[i].get_action_mask()
                if not mask.any():
                    mask[1] = True
                action_masks[i] = mask

            needs_opponent = next_needs_opponent

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
                'old_values': torch.tensor(values_flat[batch_indices], dtype=torch.float32, device=self.device),  # For value clipping
                'advantages': torch.tensor(advantages_flat[batch_indices], dtype=torch.float32, device=self.device),
                'returns': torch.tensor(returns_flat[batch_indices], dtype=torch.float32, device=self.device),
                'action_masks': torch.tensor(masks_flat[batch_indices], dtype=torch.bool, device=self.device),
            }

    def reset(self):
        self.pos = 0
        self.full = False


class VectorizedEvaluator:
    """Fast vectorized evaluation for ELO calculation."""

    def __init__(self, num_envs: int, starting_stack: float, num_actions: int, device: str,
                 vary_stacks: bool = True, min_stack: float = 10.0):
        self.num_envs = num_envs
        self.starting_stack = starting_stack
        self.num_actions = num_actions
        self.device = device

        # Stack sizes to use for evaluation (same as training for fair eval)
        if vary_stacks:
            stack_sizes = [10, 15, 20, 25, 30, 40, 50, 75, 100]
            stack_sizes = [s for s in stack_sizes if min_stack <= s <= starting_stack]
            if not stack_sizes:
                stack_sizes = [starting_stack]
        else:
            stack_sizes = [starting_stack]

        # Create C++ environments with varying stack sizes
        self.envs = []
        for i in range(num_envs):
            stack = stack_sizes[i % len(stack_sizes)]
            self.envs.append(ares_solver.FastEnv(float(stack), num_actions))
        self.dones = np.zeros(num_envs, dtype=bool)

    def evaluate_matchup(self, hero_model: ActorCritic, opponent_model: ActorCritic,
                         num_games: int, use_amp: bool = False) -> float:
        """Play num_games between hero and opponent, return total profit for hero."""
        total_profit = 0.0
        games_completed = 0

        hero_model.eval()
        opponent_model.eval()

        # Reset all environments
        for env in self.envs:
            env.reset()
        self.dones[:] = False

        while games_completed < num_games:
            # Reset done environments
            active_indices = []
            for i, env in enumerate(self.envs):
                if self.dones[i] or env.is_terminal():
                    env.reset()
                    self.dones[i] = False
                active_indices.append(i)

            if not active_indices:
                break

            # Collect hero observations
            hero_obs_list = []
            hero_mask_list = []
            needs_hero = []

            for i in active_indices:
                if not self.envs[i].is_terminal() and not self.envs[i].needs_opponent_action():
                    hero_obs_list.append(self.envs[i].get_observation())
                    hero_mask_list.append(self.envs[i].get_action_mask())
                    needs_hero.append(i)

            # Batched hero inference
            if needs_hero:
                hero_actions = self._batched_inference(
                    hero_model, hero_obs_list, hero_mask_list, use_amp
                )

                # Apply hero actions
                for idx, i in enumerate(needs_hero):
                    if games_completed >= num_games:
                        break
                    obs, reward, done, needs_opp = self.envs[i].step_hero_only(hero_actions[idx])
                    if done:
                        total_profit += reward
                        games_completed += 1
                        self.dones[i] = True

            # Handle opponent actions
            needs_opponent = []
            for i in active_indices:
                if not self.dones[i] and not self.envs[i].is_terminal() and self.envs[i].needs_opponent_action():
                    needs_opponent.append(i)

            while needs_opponent and games_completed < num_games:
                opp_obs_list = []
                opp_mask_list = []
                for i in needs_opponent:
                    opp_obs_list.append(self.envs[i].get_opponent_observation())
                    opp_mask_list.append(self.envs[i].get_opponent_action_mask())

                opp_actions = self._batched_inference(
                    opponent_model, opp_obs_list, opp_mask_list, use_amp
                )

                next_needs_opponent = []
                for idx, i in enumerate(needs_opponent):
                    if games_completed >= num_games:
                        break
                    done, reward = self.envs[i].apply_opponent_action(opp_actions[idx])
                    if done:
                        total_profit += reward
                        games_completed += 1
                        self.dones[i] = True
                    elif self.envs[i].needs_opponent_action():
                        next_needs_opponent.append(i)

                needs_opponent = next_needs_opponent

        return total_profit

    def _batched_inference(self, model: ActorCritic, obs_list: list, mask_list: list,
                           use_amp: bool) -> list:
        """Batched neural network inference."""
        if not obs_list:
            return []

        obs_batch = np.stack(obs_list)
        mask_batch = np.stack(mask_list)

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
            mask_tensor = torch.as_tensor(mask_batch, dtype=torch.bool, device=self.device)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp and self.device != 'cpu'):
                logits, _ = model(obs_tensor, mask_tensor)

            probs = torch.softmax(logits.float(), dim=-1)
            # Small exploration noise for evaluation stability
            probs = probs * 0.98 + 0.02 / self.num_actions
            probs = probs / probs.sum(dim=-1, keepdim=True)

            actions = torch.multinomial(probs, 1).squeeze(-1).cpu().numpy()

        return actions.tolist()


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

    def add_agent(self, model: ActorCritic, elo: float, update_num: int, bb_per_100: float = 0.0):
        path = self.checkpoint_dir / f"agent_{update_num}.pt"
        torch.save(get_raw_state_dict(model), path)

        self.agents.append({
            'path': str(path),
            'elo': elo,
            'update_num': update_num,
            'bb_per_100': bb_per_100  # Changed from win_rate to bb/100
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

        # FIXED: Use uniform sampling for diversity (not ELO-weighted)
        # High ELO weighting caused overfitting to one opponent
        idx = np.random.randint(len(self.agents))
        agent_info = self.agents[idx]

        opponent = model_class(
            input_channels=50, use_cnn=True, num_actions=num_actions,
            fc_hidden_dim=self.fc_hidden_dim, fc_num_layers=self.fc_num_layers
        )
        opponent.load_state_dict(torch.load(agent_info['path'], map_location='cpu'))
        opponent.eval()
        return opponent, agent_info

    def get_average_elo(self) -> float:
        """Get average ELO of pool for new agent initialization."""
        if not self.agents:
            return self.initial_elo
        return np.mean([a['elo'] for a in self.agents])

    def get_agent_by_idx(self, idx: int, model_class: type, num_actions: int = 14):
        if idx >= len(self.agents):
            return None
        agent_info = self.agents[idx]
        model = model_class(
            input_channels=50, use_cnn=True, num_actions=num_actions,
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
            input_channels=50,
            use_cnn=config.use_cnn,
            hidden_dim=config.hidden_dim,
            num_actions=config.num_actions,
            num_residual_blocks=config.num_residual_blocks,
            fc_hidden_dim=config.fc_hidden_dim,
            fc_num_layers=config.fc_num_layers
        ).to(self.device)

        # Compile model for faster training (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.device == 'cuda':
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("Model compiled with torch.compile for faster training")
            except Exception as e:
                print(f"torch.compile not available: {e}")

        # PPO - AlphaHoldem paper settings
        ppo_config = PPOConfig(
            learning_rate=3e-4,  # Paper: 0.0003
            clip_ratio=3.0,      # Paper: δ1=3 for Trinal-Clip (NOT 0.2)
            gamma=0.999,         # Paper: 0.999 discount factor
            gae_lambda=0.95,     # Paper: λ=0.95
            num_epochs=4,        # Paper: 4 epochs
            minibatch_size=256,  # Keep original for consistent training dynamics
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
            config.num_envs, config.starting_stack, config.num_actions,
            vary_stacks=config.vary_stacks, min_stack=config.min_stack
        )

        # Vectorized evaluation for fast ELO calculation
        self.evaluator = VectorizedEvaluator(
            config.eval_num_envs, config.starting_stack, config.num_actions, self.device,
            vary_stacks=config.vary_stacks, min_stack=config.min_stack
        )

        # Buffer
        self.buffer = VectorizedRolloutBuffer(
            config.steps_per_env, config.num_envs,
            (50, 4, 13), config.num_actions, self.device
        )

        # Stats
        self.total_timesteps = 0
        self.total_hands = 0
        self.update_count = 0
        self.episode_rewards: deque = deque(maxlen=5000)
        self.opponent: Optional[ActorCritic] = None
        self._current_opp_elo = None
        self._current_opp_id = -1  # Track current opponent for diversity

        # Speed tracking (instantaneous)
        self._last_update_time = None
        self._last_timesteps = 0

        # Mixed precision for faster inference (FP16 inference, FP32 training)
        self.use_amp = config.use_mixed_precision and (self.device != 'cpu')

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
                        input_channels=50, use_cnn=True, num_actions=self.config.num_actions,
                        fc_hidden_dim=self.config.fc_hidden_dim, fc_num_layers=self.config.fc_num_layers
                    ).to(self.device)
                    self.warmup_model.load_state_dict(get_raw_state_dict(self.model))
                    self.warmup_model.eval()
                    self.opponent = self.warmup_model
                    # Use BATCHED opponent inference with FP16 (fast!)
                    self.vec_env.set_opponent_model(self.opponent, self.device, self.use_amp)
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
                # Use as_tensor + non_blocking for faster CPU->GPU transfer
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                mask_tensor = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device)

                # FP16 inference for speed (doesn't affect model accuracy)
                with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
                    logits, values = self.model(obs_tensor, mask_tensor)

                probs = torch.softmax(logits.float(), dim=-1)

                # Exploration noise (vectorized)
                probs = probs * 0.99 + 0.01 / self.config.num_actions
                probs = probs / probs.sum(dim=-1, keepdim=True)

                # Sample actions directly without creating distribution object (faster)
                actions = torch.multinomial(probs, 1).squeeze(-1)
                log_probs = torch.log(probs.gather(1, actions.unsqueeze(-1)).squeeze(-1) + 1e-8)

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
        """
        PPO update with Trinal-Clip from AlphaHoldem paper.

        Trinal-Clip uses three clipping bounds:
        - δ1 = 3.0: Upper bound on policy ratio (prevents too aggressive updates)
        - δ2, δ3: Dynamic bounds for negative advantages (based on pot size)

        For negative advantages (bad actions), we want to decrease probability
        but not too aggressively on high-variance hands.
        """
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        # Trinal-Clip hyperparameters (from paper)
        delta1 = self.ppo.config.clip_ratio  # 3.0 (upper bound for positive advantages)

        for epoch in range(self.ppo.config.num_epochs):
            for batch in self.buffer.get_batches(self.ppo.config.minibatch_size):
                obs = batch['observations']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                old_values = batch['old_values']
                advantages = batch['advantages']
                returns = batch['returns']
                action_masks = batch['action_masks']

                new_log_probs, values, entropy = self.model.evaluate_actions(obs, actions, action_masks)

                ratio = torch.exp(new_log_probs - old_log_probs)

                # Trinal-Clip Policy Loss
                # For positive advantages: clip(ratio, 1-ε, δ1) where δ1=3
                # For negative advantages: DYNAMIC clip based on pot size (return magnitude)

                # Standard PPO term
                surr1 = ratio * advantages

                # Dynamic δ2, δ3 based on return magnitude (proxy for pot size/stakes)
                # High stakes (large |return|) = tighter clipping for stability
                # Low stakes (small |return|) = looser clipping for faster learning
                #
                # IMPORTANT: Must ensure delta2 < delta3 always!
                # pot_scale: 0 = small pot, 1 = large pot (clamped)
                pot_scale = (torch.abs(returns) / self.config.starting_stack).clamp(0, 1)
                # Low pot: [0.5, 1.5] (loose), High pot: [0.7, 1.3] (tight)
                delta2 = 0.5 + 0.2 * pot_scale  # Range: 0.5 to 0.7
                delta3 = 1.5 - 0.2 * pot_scale  # Range: 1.5 to 1.3
                # Verify: at pot_scale=0: [0.5, 1.5], at pot_scale=1: [0.7, 1.3]
                # delta2 < delta3 always (0.7 < 1.3) ✓

                # Trinal-Clip: different clipping for positive vs negative advantages
                pos_adv_mask = (advantages >= 0).float()
                neg_adv_mask = (advantages < 0).float()

                # Positive advantages: use upper clip δ1=3 (allow ratio up to 3x)
                clipped_ratio_pos = torch.clamp(ratio, 1.0 - 0.2, delta1)

                # Negative advantages: use DYNAMIC tighter clip [δ2, δ3] for stability
                # Per-sample clipping based on pot size
                clipped_ratio_neg = torch.max(torch.min(ratio, delta3), delta2)

                clipped_ratio = pos_adv_mask * clipped_ratio_pos + neg_adv_mask * clipped_ratio_neg
                surr2 = clipped_ratio * advantages

                # Take minimum (pessimistic bound) - this is the PPO objective
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping (prevents value function from changing too fast)
                value_clipped = old_values + torch.clamp(values - old_values, -0.2, 0.2)
                value_loss1 = (values - returns) ** 2
                value_loss2 = (value_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

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

    def _update_opponent(self):
        if len(self.opponent_pool.agents) == 0:
            if self.warmup_model is not None:
                self.opponent = self.warmup_model
                self._current_opp_elo = self.config.initial_elo
                self._current_opp_id = -1  # No pool agent
                self.vec_env.set_opponent_model(self.opponent, self.device, self.use_amp)
        else:
            # Sample uniformly from pool (K-best already provides diversity)
            # Note: Removed self-play against current model - it produces ~50/50 outcomes
            # which provides no useful learning signal. Diversity comes from the K-best pool.
            opponent, agent_info = self.opponent_pool.sample_opponent(ActorCritic, self.config.num_actions)
            if opponent is not None:
                new_opp_id = agent_info['update_num']
                # Only print when actually switching to different agent
                if new_opp_id != self._current_opp_id:
                    self.opponent = opponent.to(self.device)
                    self.opponent.eval()
                    self._current_opp_elo = agent_info['elo']
                    self._current_opp_id = new_opp_id
                    self.vec_env.set_opponent_model(self.opponent, self.device, self.use_amp)
                    print(f"  [OPPONENT] Switched to agent_{new_opp_id} (ELO={agent_info['elo']:.0f})")

    def _evaluate_for_pool(self):
        if len(self.opponent_pool.agents) == 0:
            self.opponent_pool.add_agent(self.model, self.config.initial_elo, self.update_count)
            print(f"  [POOL] Added first agent_{self.update_count}")
        else:
            new_elo, bb_per_100, _ = self._evaluate_against_pool()
            self.opponent_pool.add_agent(self.model, new_elo, self.update_count, bb_per_100)
            print(f"  [POOL] Agent_{self.update_count}: ELO={new_elo:.0f}, bb/100={bb_per_100:+.1f}")
            print(f"  [POOL] {self.opponent_pool.get_pool_stats()}")

    def _evaluate_against_pool(self):
        """Vectorized evaluation against all pool agents for ELO calculation."""
        if len(self.opponent_pool.agents) == 0:
            return self.config.initial_elo, 0.0, []

        # Start at pool average ELO
        new_elo = self.opponent_pool.get_average_elo()
        total_profit = 0.0
        total_games = 0
        results = []
        self.model.eval()

        games_per_matchup = self.config.elo_games_per_matchup

        for idx, agent_info in enumerate(self.opponent_pool.agents):
            opponent = self.opponent_pool.get_agent_by_idx(idx, ActorCritic, self.config.num_actions)
            if opponent is None:
                continue

            opponent = opponent.to(self.device)
            opponent.eval()

            # Use vectorized evaluation for speed
            matchup_profit = self.evaluator.evaluate_matchup(
                self.model, opponent, games_per_matchup, use_amp=self.use_amp
            )

            total_profit += matchup_profit
            total_games += games_per_matchup

            # Convert profit to 0-1 score for ELO calculation
            # Use wider range (±25 bb/hand) to avoid clamping during early training
            bb_per_hand = matchup_profit / games_per_matchup
            score = 0.5 + (bb_per_hand / 50.0)  # ±25bb/hand maps to 0-1
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

            opponent_elo = agent_info['elo']
            expected = 1 / (1 + 10 ** ((opponent_elo - new_elo) / 400))
            # K-factor per matchup (not per game) - standard ELO formula
            new_elo += self.config.elo_k_factor * (score - expected)

            results.append({
                'opponent': agent_info['update_num'],
                'opponent_elo': opponent_elo,
                'profit': matchup_profit,
                'bb_per_hand': bb_per_hand,
                'score': score
            })

        self.model.train()
        # Return bb/100 as the metric (more meaningful than win rate)
        bb_per_100 = (total_profit / total_games * 100) if total_games > 0 else 0.0
        return new_elo, bb_per_100, results

    def _save_checkpoint(self, final: bool = False):
        suffix = "final" if final else f"{self.update_count}"
        path = Path(self.config.checkpoint_dir) / f"checkpoint_{suffix}.pt"
        torch.save({
            'model_state_dict': get_raw_state_dict(self.model),
            'optimizer_state_dict': self.ppo.optimizer.state_dict(),
            'update_count': self.update_count,
            'total_timesteps': self.total_timesteps,
            'total_hands': self.total_hands,
            'config': asdict(self.config)
        }, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        # Handle loading into compiled model - strip _orig_mod prefix if present
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        # For compiled models, load into the underlying module
        if hasattr(self.model, '_orig_mod'):
            self.model._orig_mod.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
        self.ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint.get('update_count', 0)
        self.total_timesteps = checkpoint.get('total_timesteps', 0)
        self.total_hands = checkpoint.get('total_hands', 0)
        print(f"Loaded: update={self.update_count}, steps={self.total_timesteps:,}")

    def _log_progress(self, stats, start_time):
        current_time = time.time()
        elapsed = current_time - start_time

        # Instantaneous speed (this update only)
        if self._last_update_time is not None:
            update_elapsed = current_time - self._last_update_time
            update_steps = self.total_timesteps - self._last_timesteps
            instant_speed = update_steps / max(update_elapsed, 0.001)
        else:
            instant_speed = self.total_timesteps / max(elapsed, 1)

        self._last_update_time = current_time
        self._last_timesteps = self.total_timesteps

        # ETA based on instantaneous speed (more accurate)
        remaining = self.config.total_timesteps - self.total_timesteps
        eta = remaining / max(instant_speed, 1)

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
              f"Steps/s: {instant_speed:.0f} | "
              f"ETA: {eta/3600:.1f}h")


def main():
    parser = argparse.ArgumentParser(description="Train AlphaHoldem with C++ acceleration")
    parser.add_argument('--timesteps', type=int, default=100_000_000)
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--stack', type=float, default=100.0, help="Max starting stack (bb)")
    parser.add_argument('--min-stack', type=float, default=10.0, help="Min starting stack (bb)")
    parser.add_argument('--no-vary-stacks', action='store_true', help="Disable stack size variation")
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--checkpoint-dir', type=str, default='alphaholdem/checkpoints')
    parser.add_argument('--log-dir', type=str, default='alphaholdem/logs')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Enable TF32 for faster matrix multiplication on Ampere+ GPUs (A40, A100, RTX 30xx/40xx)
    # ~10-20% speedup with negligible precision loss for training
    if device == 'cuda':
        torch.set_float32_matmul_precision('high')

    config = TrainConfig(
        starting_stack=args.stack,
        min_stack=args.min_stack,
        vary_stacks=not args.no_vary_stacks,
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
