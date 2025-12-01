"""
AlphaHoldem Vectorized Training Script

Self-play training with parallel environments for ~10x speedup.
"""

import torch
import numpy as np
import time
import argparse
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
import json
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


@dataclass
class TrainConfig:
    """Training configuration."""
    # Environment
    starting_stack: float = 100.0
    num_actions: int = 14

    # Vectorized training
    num_envs: int = 16  # Parallel environments
    steps_per_env: int = 128  # Steps per env before update

    # Training
    total_timesteps: int = 100_000_000

    # Self-play
    k_best: int = 5
    update_opponent_every: int = 10
    eval_for_pool_every: int = 50
    min_hands_for_pool: int = 50_000
    elo_games_per_matchup: int = 100
    initial_elo: float = 1500.0
    elo_k_factor: float = 32.0

    # Network (AlphaHoldem paper: 8.6M params)
    hidden_dim: int = 256
    num_residual_blocks: int = 4
    fc_hidden_dim: int = 1024
    fc_num_layers: int = 4
    use_cnn: bool = True

    # Checkpointing
    save_every: int = 50
    eval_every: int = 10
    eval_games: int = 500

    # Logging
    use_tensorboard: bool = True
    log_every: int = 1

    # Paths
    checkpoint_dir: str = "alphaholdem/checkpoints"
    log_dir: str = "alphaholdem/logs"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def steps_per_update(self):
        return self.num_envs * self.steps_per_env


class KBestPool:
    """Pool of K-best historical agents using ELO-based selection."""

    def __init__(self, k: int, checkpoint_dir: str, initial_elo: float = 1500.0, k_factor: float = 32.0,
                 fc_hidden_dim: int = 1024, fc_num_layers: int = 4):
        self.k = k
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        self.fc_hidden_dim = fc_hidden_dim
        self.fc_num_layers = fc_num_layers
        self.agents: List[Dict] = []

    def add_agent(self, model: ActorCritic, elo: float, update_num: int, games_played: int = 0):
        path = self.checkpoint_dir / f"agent_{update_num}.pt"
        torch.save(model.state_dict(), path)

        self.agents.append({
            'path': str(path),
            'elo': elo,
            'update_num': update_num,
            'games_played': games_played
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

        agent_info = random.choice(self.agents)
        opponent = model_class(
            input_channels=50,
            use_cnn=True,
            num_actions=num_actions,
            fc_hidden_dim=self.fc_hidden_dim,
            fc_num_layers=self.fc_num_layers
        )
        opponent.load_state_dict(torch.load(agent_info['path']))
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
        model.load_state_dict(torch.load(agent_info['path']))
        model.eval()
        return model

    def get_pool_stats(self) -> str:
        if not self.agents:
            return "Pool empty"
        elos = [a['elo'] for a in self.agents]
        return f"Pool({len(self.agents)}): ELO [{min(elos):.0f}-{max(elos):.0f}]"


class VectorizedTrainer:
    """Self-play trainer with vectorized environments for 10x speedup."""

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

        # PPO
        ppo_config = PPOConfig(
            learning_rate=3e-4,
            clip_ratio=0.2,
            num_epochs=4,
            minibatch_size=256,  # Larger for GPU efficiency
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
        self.vec_env = VectorizedHeadsUpEnv(
            config.num_envs,
            config.starting_stack,
            config.num_actions
        )

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
        self.episode_rewards: deque = deque(maxlen=10000)
        self.opponent: Optional[ActorCritic] = None
        self._current_opp_elo = None

        # TensorBoard
        self.writer = None
        if config.use_tensorboard and HAS_TENSORBOARD:
            tb_dir = Path(config.log_dir) / f"tensorboard_{datetime.now():%Y%m%d_%H%M%S}"
            self.writer = SummaryWriter(tb_dir)

    def train(self):
        """Main training loop with vectorized collection."""
        print(f"Starting VECTORIZED training on {self.device}")
        print(f"Parallel environments: {self.config.num_envs}")
        print(f"Target timesteps: {self.config.total_timesteps:,}")
        print(f"Model parameters: {self.model.num_parameters:,}")

        total_updates = self.config.total_timesteps // self.config.steps_per_update
        self.ppo.setup_scheduler(total_updates)

        start_time = time.time()

        # Initial reset
        obs, action_masks = self.vec_env.reset()

        while self.total_timesteps < self.config.total_timesteps:
            # Collect rollout from all envs
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

            # Update opponent
            if self.update_count % self.config.update_opponent_every == 0:
                self._update_opponent()

            # Evaluate
            if self.update_count % self.config.eval_every == 0:
                eval_stats = self._evaluate()
                stats.update(eval_stats)

            # Pool evaluation
            if (self.update_count % self.config.eval_for_pool_every == 0 and
                self.total_hands >= self.config.min_hands_for_pool):
                self._evaluate_for_pool()

            # Save
            if self.update_count % self.config.save_every == 0:
                self._save_checkpoint()

            # Logging
            self._log_progress(stats, start_time)

            # Reset buffer
            self.buffer.reset()

        print(f"\nTraining complete! Total time: {(time.time() - start_time)/3600:.1f}h")
        self._save_checkpoint(final=True)

    def _collect_vectorized_rollout(self, obs: np.ndarray, action_masks: np.ndarray):
        """Collect experience from all environments in parallel."""
        self.model.eval()

        for step in range(self.config.steps_per_env):
            # Reset any done environments
            obs, action_masks, reset_mask = self.vec_env.reset_done_envs()

            # Get actions for all envs at once (batched GPU inference)
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                mask_tensor = torch.tensor(action_masks, dtype=torch.bool, device=self.device)

                logits, values = self.model(obs_tensor, mask_tensor)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

                actions_np = actions.cpu().numpy()
                log_probs_np = log_probs.cpu().numpy()
                values_np = values.cpu().numpy()

            # Step all environments
            next_obs, next_masks, rewards, dones, infos = self.vec_env.step(actions_np)

            # Store in buffer
            self.buffer.add(obs, actions_np, log_probs_np, values_np, rewards, dones, action_masks)

            # Update stats
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
        """Perform PPO update on collected data."""
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        num_updates = 0

        for epoch in range(self.ppo.config.num_epochs):
            for batch in self.buffer.get_batches(self.ppo.config.minibatch_size):
                obs = batch['observations']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                old_values = batch['old_values']
                advantages = batch['advantages']
                returns = batch['returns']
                action_masks = batch['action_masks']

                # Forward pass
                new_log_probs, values, entropy = self.model.evaluate_actions(obs, actions, action_masks)

                # Policy loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.ppo.config.clip_ratio, 1 + self.ppo.config.clip_ratio)
                policy_loss = torch.max(-advantages * ratio, -advantages * clipped_ratio).mean()

                # Value loss
                value_loss = 0.5 * ((values - returns) ** 2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

                # Gradient step
                self.ppo.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.ppo.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                with torch.no_grad():
                    total_kl += (old_log_probs - new_log_probs).mean().item()
                num_updates += 1

        if self.ppo.scheduler:
            self.ppo.scheduler.step()

        return {
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'value_loss': total_value_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1),
            'kl': total_kl / max(num_updates, 1),
        }

    def _opponent_policy(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        """Get opponent action (used by environments)."""
        if self.opponent is None:
            valid = np.where(action_mask)[0]
            return np.random.choice(valid)

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            action, _, _ = self.opponent.get_action(obs_tensor, mask_tensor, deterministic=True)
            return action.item()

    def _update_opponent(self):
        """Update opponent from pool."""
        opponent, agent_info = self.opponent_pool.sample_opponent(ActorCritic, self.config.num_actions)
        if opponent is not None:
            self.opponent = opponent.to(self.device)
            self.opponent.eval()
            self._current_opp_elo = agent_info['elo']
            # Update vec_env opponent
            self.vec_env.set_opponent(self._opponent_policy)
            print(f"  [OPPONENT] Switched to agent_{agent_info['update_num']} (ELO={agent_info['elo']:.0f})")
        else:
            self.opponent = None
            self._current_opp_elo = None
            self.vec_env.set_opponent(None)

    def _evaluate(self) -> Dict[str, float]:
        """Evaluate current policy."""
        wins = 0
        rewards = []
        self.model.eval()

        # Use single eval env
        for _ in range(self.config.eval_games):
            self.eval_env.reset()
            if self.opponent is not None:
                self.eval_env.set_opponent(self._opponent_policy)

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
                    action, _, _ = self.model.get_action(obs_t, mask_t, deterministic=True)

                _, reward, done, _ = self.eval_env.step(action.item())
                game_reward += reward

            rewards.append(game_reward)
            if game_reward > 0:
                wins += 1

        self.model.train()

        mean_reward = np.mean(rewards)
        bb_per_100 = mean_reward * 100
        win_rate = wins / len(rewards)

        return {'win_rate': win_rate, 'eval_bb_per_100': bb_per_100}

    def _get_eval_obs(self) -> np.ndarray:
        """Get observation for evaluation env."""
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

    def _evaluate_for_pool(self):
        """Evaluate against pool and potentially add agent."""
        if len(self.opponent_pool.agents) == 0:
            self.opponent_pool.add_agent(self.model, self.config.initial_elo, self.update_count)
            print(f"  [POOL] Added first agent_{self.update_count}")
        else:
            new_elo, results = self._evaluate_against_pool_elo()
            total_games = sum(r['games'] for r in results)
            total_wins = sum(r['wins'] for r in results)
            win_rate = total_wins / total_games if total_games > 0 else 0
            self.opponent_pool.add_agent(self.model, new_elo, self.update_count, total_games)
            print(f"  [POOL] Agent_{self.update_count}: ELO={new_elo:.0f}, win_rate={win_rate:.1%}")
            print(f"  [POOL] {self.opponent_pool.get_pool_stats()}")

    def _evaluate_against_pool_elo(self) -> tuple:
        """Evaluate against all pool agents."""
        if len(self.opponent_pool.agents) == 0:
            return self.config.initial_elo, []

        new_elo = self.config.initial_elo
        results = []
        self.model.eval()

        for idx, agent_info in enumerate(self.opponent_pool.agents):
            opponent = self.opponent_pool.get_agent_by_idx(idx, ActorCritic, self.config.num_actions)
            if opponent is None:
                continue

            opponent = opponent.to(self.device)
            opponent.eval()

            # Temporarily set opponent
            old_opponent = self.opponent
            self.opponent = opponent

            wins = 0
            games = self.config.elo_games_per_matchup

            for _ in range(games):
                self.eval_env.reset()
                self.eval_env.set_opponent(self._opponent_policy)
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
                        action, _, _ = self.model.get_action(obs_t, mask_t, deterministic=True)
                    _, reward, done, _ = self.eval_env.step(action.item())
                    game_reward += reward
                if game_reward > 0:
                    wins += 1

            self.opponent = old_opponent

            # Update ELO
            opponent_elo = agent_info['elo']
            for _ in range(wins):
                expected = 1 / (1 + 10 ** ((opponent_elo - new_elo) / 400))
                new_elo += self.config.elo_k_factor * (1.0 - expected)
            for _ in range(games - wins):
                expected = 1 / (1 + 10 ** ((opponent_elo - new_elo) / 400))
                new_elo += self.config.elo_k_factor * (0.0 - expected)

            results.append({'opponent': agent_info['update_num'], 'games': games, 'wins': wins})

        self.model.train()
        return new_elo, results

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
        """Load training state from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint.get('update_count', 0)
        self.total_timesteps = checkpoint.get('total_timesteps', 0)
        self.total_hands = checkpoint.get('total_hands', 0)
        print(f"Loaded checkpoint: update={self.update_count}, timesteps={self.total_timesteps:,}, hands={self.total_hands:,}")

    def _log_progress(self, stats: Dict, start_time: float):
        elapsed = time.time() - start_time
        steps_per_sec = self.total_timesteps / max(elapsed, 1)
        eta = (self.config.total_timesteps - self.total_timesteps) / max(steps_per_sec, 1)

        mean_reward = np.mean(list(self.episode_rewards)) if self.episode_rewards else 0
        bb_per_100 = mean_reward * 100

        if self.opponent is None:
            opp_str = "vs RANDOM"
            pool_str = ""
        else:
            opp_str = f"vs ELO {self._current_opp_elo:.0f}" if self._current_opp_elo else "vs POOL"
            pool_str = f" | {self.opponent_pool.get_pool_stats()}"

        print(f"Update {self.update_count} | "
              f"Hands: {self.total_hands:,} | "
              f"bb/100: {bb_per_100:+.1f} | "
              f"{opp_str}{pool_str} | "
              f"Steps/s: {steps_per_sec:.0f} | "
              f"ETA: {eta/3600:.1f}h")

        if self.writer:
            self.writer.add_scalar('train/steps_per_sec', steps_per_sec, self.total_timesteps)
            self.writer.add_scalar('episode/bb_per_100', bb_per_100, self.total_timesteps)
            self.writer.add_scalar('episode/total_hands', self.total_hands, self.total_timesteps)


def main():
    parser = argparse.ArgumentParser(description="Train AlphaHoldem with vectorized environments")
    parser.add_argument('--timesteps', type=int, default=100_000_000)
    parser.add_argument('--num-envs', type=int, default=16, help='Parallel environments')
    parser.add_argument('--stack', type=float, default=100.0)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--checkpoint-dir', type=str, default='alphaholdem/checkpoints')
    parser.add_argument('--log-dir', type=str, default='alphaholdem/logs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
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

    trainer = VectorizedTrainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from {args.resume}")

    trainer.train()


if __name__ == "__main__":
    main()
