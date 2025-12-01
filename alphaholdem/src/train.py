"""
AlphaHoldem Training Script

Self-play training with K-Best opponent selection.
"""

import torch
import numpy as np
import time
import argparse
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
import random
from collections import deque

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

from alphaholdem.src.network import ActorCritic, PseudoSiameseNetwork
from alphaholdem.src.ppo import PPO, PPOConfig, RolloutBuffer
from alphaholdem.src.encoder import AlphaHoldemEncoder
from alphaholdem.src.env import HeadsUpEnv


@dataclass
class TrainConfig:
    """Training configuration."""
    # Environment
    starting_stack: float = 100.0
    num_actions: int = 14  # Expanded action space

    # Training
    total_timesteps: int = 10_000_000
    steps_per_update: int = 2048
    num_envs: int = 1  # Parallel environments (future)

    # Self-play (following AlphaHoldem paper - ELO-based selection)
    k_best: int = 5  # Number of best historical agents (AlphaHoldem used 5)
    update_opponent_every: int = 10  # Updates between opponent refresh
    eval_for_pool_every: int = 50  # Evaluate for pool addition every N updates
    min_hands_for_pool: int = 50_000  # Minimum hands before first agent added
    elo_games_per_matchup: int = 100  # Games played per matchup for ELO calculation
    initial_elo: float = 1500.0  # Starting ELO for new agents
    elo_k_factor: float = 32.0  # ELO K-factor (how much ratings change per game)

    # Network (AlphaHoldem paper: 8.6M params total)
    # Paper: 1.8M ConvNets + 6.8M FC layers
    hidden_dim: int = 256  # Backbone output dim
    num_residual_blocks: int = 4  # CNN residual blocks
    fc_hidden_dim: int = 1024  # FC layer width
    fc_num_layers: int = 4  # FC layer depth (gives ~6.8M FC params)
    use_cnn: bool = True

    # Checkpointing
    save_every: int = 50  # Updates between saves
    eval_every: int = 10  # Updates between evaluations
    eval_games: int = 1000  # Hands per evaluation (1000 for reasonable variance)

    # Logging
    use_tensorboard: bool = True
    log_every: int = 1  # Log to tensorboard every N updates

    # Paths
    checkpoint_dir: str = "alphaholdem/checkpoints"
    log_dir: str = "alphaholdem/logs"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class KBestPool:
    """
    Pool of K-best historical agents using ELO-based selection.

    Based on AlphaHoldem paper: agents compete against each other,
    ELO scores are calculated from matchup results, and K-best by ELO are kept.
    """

    def __init__(self, k: int, checkpoint_dir: str, initial_elo: float = 1500.0, k_factor: float = 32.0,
                 fc_hidden_dim: int = 1024, fc_num_layers: int = 4):
        self.k = k
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        self.fc_hidden_dim = fc_hidden_dim
        self.fc_num_layers = fc_num_layers

        # List of {path, elo, update_num, games_played}
        self.agents: List[Dict] = []

    def add_agent(self, model: ActorCritic, elo: float, update_num: int, games_played: int = 0):
        """Add agent to pool with given ELO score."""
        # Save checkpoint
        path = self.checkpoint_dir / f"agent_{update_num}.pt"
        torch.save(model.state_dict(), path)

        self.agents.append({
            'path': str(path),
            'elo': elo,
            'update_num': update_num,
            'games_played': games_played
        })

        # Keep only K-best by ELO score
        self.agents.sort(key=lambda x: x['elo'], reverse=True)
        if len(self.agents) > self.k:
            # Remove lowest ELO agent
            removed = self.agents.pop()
            if Path(removed['path']).exists():
                Path(removed['path']).unlink()
            print(f"  [POOL] Evicted agent_{removed['update_num']} (ELO={removed['elo']:.0f}) - pool full")

    def update_elo(self, agent_idx: int, opponent_idx: int, win: bool):
        """
        Update ELO scores after a game.

        Uses standard ELO formula:
        E_a = 1 / (1 + 10^((R_b - R_a) / 400))
        R'_a = R_a + K * (S_a - E_a)

        where S_a = 1 for win, 0.5 for draw, 0 for loss
        """
        if agent_idx >= len(self.agents) or opponent_idx >= len(self.agents):
            return

        r_a = self.agents[agent_idx]['elo']
        r_b = self.agents[opponent_idx]['elo']

        # Expected score
        e_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
        e_b = 1 - e_a

        # Actual score (win=1, loss=0, draw=0.5)
        s_a = 1.0 if win else 0.0
        s_b = 1.0 - s_a

        # Update ratings
        self.agents[agent_idx]['elo'] = r_a + self.k_factor * (s_a - e_a)
        self.agents[opponent_idx]['elo'] = r_b + self.k_factor * (s_b - e_b)
        self.agents[agent_idx]['games_played'] += 1
        self.agents[opponent_idx]['games_played'] += 1

    def get_agent_by_idx(self, idx: int, model_class: type, num_actions: int = 14) -> Optional[ActorCritic]:
        """Get agent model by index."""
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

    def sample_opponent(self, model_class: type, num_actions: int = 14) -> tuple:
        """Sample a random opponent from the pool. Returns (opponent, agent_info) or (None, None)."""
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

    def get_best_opponent(self, model_class: type, num_actions: int = 14) -> Optional[ActorCritic]:
        """Get the highest ELO opponent from the pool."""
        if not self.agents:
            return None

        best = max(self.agents, key=lambda x: x['elo'])
        opponent = model_class(
            input_channels=50,
            use_cnn=True,
            num_actions=num_actions,
            fc_hidden_dim=self.fc_hidden_dim,
            fc_num_layers=self.fc_num_layers
        )
        opponent.load_state_dict(torch.load(best['path']))
        opponent.eval()
        return opponent

    def get_pool_stats(self) -> str:
        """Get summary of pool agents and their ELO scores."""
        if not self.agents:
            return "Pool empty"
        elos = [a['elo'] for a in self.agents]
        return f"Pool({len(self.agents)}): ELO range [{min(elos):.0f}-{max(elos):.0f}], avg={np.mean(elos):.0f}"


class SelfPlayTrainer:
    """
    Self-play trainer with K-Best opponent selection.

    Based on AlphaHoldem training procedure.
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = config.device

        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        # Initialize model (AlphaHoldem paper: 8.6M params)
        self.model = ActorCritic(
            input_channels=50,
            use_cnn=config.use_cnn,
            hidden_dim=config.hidden_dim,
            num_actions=config.num_actions,
            num_residual_blocks=config.num_residual_blocks,
            fc_hidden_dim=config.fc_hidden_dim,
            fc_num_layers=config.fc_num_layers
        ).to(self.device)

        # PPO trainer
        ppo_config = PPOConfig(
            learning_rate=3e-4,
            clip_ratio=0.2,
            num_epochs=4,
            minibatch_size=64
        )
        self.ppo = PPO(self.model, ppo_config, self.device)

        # K-Best opponent pool with ELO-based selection
        self.opponent_pool = KBestPool(
            config.k_best,
            config.checkpoint_dir,
            initial_elo=config.initial_elo,
            k_factor=config.elo_k_factor,
            fc_hidden_dim=config.fc_hidden_dim,
            fc_num_layers=config.fc_num_layers
        )

        # Current opponent
        self.opponent: Optional[ActorCritic] = None

        # Environment with expanded action space
        self.env = HeadsUpEnv(config.starting_stack, num_actions=config.num_actions)
        self.encoder = AlphaHoldemEncoder()

        # Rollout buffer
        self.buffer = RolloutBuffer(
            config.steps_per_update,
            (50, 4, 13),
            self.device
        )

        # Stats tracking
        self.total_timesteps = 0
        self.total_hands = 0  # Track actual hands played
        self.update_count = 0
        self.episode_rewards: deque = deque(maxlen=10000)  # Rolling window of hands
        self.episode_lengths: deque = deque(maxlen=10000)
        self.eval_win_rates: List[float] = []
        self.eval_rewards: List[float] = []

        # TensorBoard logging
        self.writer = None
        if config.use_tensorboard and HAS_TENSORBOARD:
            tb_dir = Path(config.log_dir) / f"tensorboard_{datetime.now():%Y%m%d_%H%M%S}"
            self.writer = SummaryWriter(tb_dir)
            print(f"TensorBoard logging to: {tb_dir}")

        # JSON logging
        self.log_file = Path(config.log_dir) / f"train_{datetime.now():%Y%m%d_%H%M%S}.json"
        self.logs: List[Dict] = []

    def train(self):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Target timesteps: {self.config.total_timesteps:,}")
        print(f"Model parameters: {self.model.num_parameters:,}")

        # Setup LR scheduler
        total_updates = self.config.total_timesteps // self.config.steps_per_update
        self.ppo.setup_scheduler(total_updates)

        start_time = time.time()

        while self.total_timesteps < self.config.total_timesteps:
            # Collect rollout
            self._collect_rollout()

            # PPO update
            last_obs = self._get_current_obs()
            with torch.no_grad():
                _, last_value = self.model(
                    torch.tensor(last_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                )
                last_value = last_value.item()

            stats = self.ppo.update(self.buffer, last_value)
            self.update_count += 1

            # Update opponent periodically
            if self.update_count % self.config.update_opponent_every == 0:
                self._update_opponent()

            # Evaluate periodically
            if self.update_count % self.config.eval_every == 0:
                eval_stats = self._evaluate()
                stats.update(eval_stats)

            # Evaluate for pool addition using ELO-based selection (AlphaHoldem method)
            # New agent plays against ALL pool agents, ELO is calculated, K-best by ELO kept
            if (self.update_count % self.config.eval_for_pool_every == 0 and
                self.total_hands >= self.config.min_hands_for_pool):

                if len(self.opponent_pool.agents) == 0:
                    # First agent - add with initial ELO
                    self.opponent_pool.add_agent(
                        self.model,
                        self.config.initial_elo,
                        self.update_count,
                        games_played=0
                    )
                    print(f"  [POOL] Added first agent_{self.update_count} (ELO={self.config.initial_elo:.0f})")
                else:
                    # Evaluate against ALL pool agents (round-robin) and calculate ELO
                    new_elo, results = self._evaluate_against_pool_elo()
                    total_games = sum(r['games'] for r in results)
                    total_wins = sum(r['wins'] for r in results)
                    win_rate = total_wins / total_games if total_games > 0 else 0

                    # Add to pool (will be sorted by ELO, lowest evicted if pool full)
                    self.opponent_pool.add_agent(
                        self.model,
                        new_elo,
                        self.update_count,
                        games_played=total_games
                    )
                    print(f"  [POOL] Agent_{self.update_count} added: ELO={new_elo:.0f}, win_rate={win_rate:.1%} vs pool")
                    print(f"  [POOL] {self.opponent_pool.get_pool_stats()}")

            # Save checkpoint
            if self.update_count % self.config.save_every == 0:
                self._save_checkpoint()

            # Logging
            self._log_stats(stats)

            # Print progress
            elapsed = time.time() - start_time
            steps_per_sec = self.total_timesteps / max(elapsed, 1)
            eta = (self.config.total_timesteps - self.total_timesteps) / max(steps_per_sec, 1)

            mean_reward = np.mean(list(self.episode_rewards)) if self.episode_rewards else 0
            # Calculate bb/100 (standard poker metric)
            bb_per_100 = mean_reward * 100 if self.episode_rewards else 0

            # Build opponent/pool info string
            if self.opponent is None:
                opp_str = "vs RANDOM"
                pool_str = ""
            else:
                # Get current opponent ELO if available
                opp_elo = getattr(self, '_current_opp_elo', None)
                opp_str = f"vs ELO {opp_elo:.0f}" if opp_elo else "vs POOL"
                pool_str = f" | {self.opponent_pool.get_pool_stats()}"

            print(f"Update {self.update_count} | "
                  f"Hands: {self.total_hands:,} | "
                  f"bb/100: {bb_per_100:+.1f} | "
                  f"{opp_str}{pool_str} | "
                  f"Steps/s: {steps_per_sec:.0f} | "
                  f"ETA: {eta/3600:.1f}h")

            # Clear buffer
            self.buffer.reset()

        print(f"\nTraining complete! Total time: {(time.time() - start_time)/3600:.1f}h")
        self._save_checkpoint(final=True)

    def _collect_rollout(self):
        """Collect experience for one rollout."""
        obs = self._reset_env()
        episode_reward = 0
        episode_length = 0

        for _ in range(self.config.steps_per_update):
            # Check if we need to reset (game already done from opponent)
            if self.env.state.is_terminal():
                reward = self.env.state.get_payoff(0)
                episode_reward += reward
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.total_hands += 1
                obs = self._reset_env()
                episode_reward = 0
                episode_length = 0
                continue

            # Get action mask
            action_mask = self.env.get_action_mask()

            # Get action from policy
            action, log_prob, value = self.ppo.get_action(obs, action_mask)

            # Step environment
            next_obs, reward, done, info = self.env.step(action)
            episode_reward += reward
            episode_length += 1
            self.total_timesteps += 1

            # Store in buffer
            self.buffer.add(obs, action, log_prob, value, reward, done, action_mask)

            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.total_hands += 1
                obs = self._reset_env()
                episode_reward = 0
                episode_length = 0
            else:
                # Encode new observation
                obs = self._get_current_obs()

    def _reset_env(self) -> np.ndarray:
        """Reset environment and get initial observation."""
        self.env.reset()

        # Set opponent policy if we have one
        if self.opponent is not None:
            self.env.set_opponent(self._opponent_policy)

        return self._get_current_obs()

    def _get_current_obs(self) -> np.ndarray:
        """Get current observation from environment."""
        return self.encoder.encode(
            hole_cards=[(c.rank - 2, c.suit) for c in self.env.state.hero_hole],
            board_cards=[(c.rank - 2, c.suit) for c in self.env.state.board],
            pot=self.env.state.pot,
            hero_stack=self.env.state.hero_stack,
            villain_stack=self.env.state.villain_stack,
            hero_invested=self.env.state.hero_invested_this_street,
            villain_invested=self.env.state.villain_invested_this_street,
            street=self.env.state.street,
            is_button=(self.env.state.button == 0),
            action_history=self.encoder._parse_action_history(self.env.state.action_history)
        )

    def _opponent_policy(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        """Get opponent action."""
        if self.opponent is None:
            # Random policy
            valid = np.where(action_mask)[0]
            return np.random.choice(valid)

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            action, _, _ = self.opponent.get_action(obs_tensor, mask_tensor, deterministic=True)
            return action.item()

    def _update_opponent(self):
        """Update opponent from K-best pool."""
        opponent, agent_info = self.opponent_pool.sample_opponent(ActorCritic, self.config.num_actions)
        if opponent is not None:
            self.opponent = opponent.to(self.device)
            self.opponent.eval()
            self._current_opp_elo = agent_info['elo']
            print(f"  [OPPONENT] Switched to agent_{agent_info['update_num']} (ELO={agent_info['elo']:.0f})")
        else:
            self.opponent = None
            self._current_opp_elo = None
            # Only log this once at the start
            if not hasattr(self, '_logged_random_opponent'):
                print("  [OPPONENT] No agents in pool yet, using RANDOM opponent")
                self._logged_random_opponent = True

    def _evaluate(self) -> Dict[str, float]:
        """
        Evaluate current policy against opponent.

        Returns bb/100 with confidence interval estimate.
        Standard error of bb/100 ≈ 100 * σ / √n where σ ≈ 50-100 bb per hand.
        """
        wins = 0
        rewards = []

        self.model.eval()

        for _ in range(self.config.eval_games):
            obs = self._reset_env()
            done = False
            game_reward = 0

            while not done:
                # Check if terminal from opponent's action
                if self.env.state.is_terminal():
                    reward = self.env.state.get_payoff(0)
                    game_reward += reward
                    break

                action_mask = self.env.get_action_mask()
                action, _, _ = self.ppo.get_action(obs, action_mask, deterministic=True)
                _, reward, done, _ = self.env.step(action)
                game_reward += reward

                if not done:
                    obs = self._get_current_obs()

            rewards.append(game_reward)
            if game_reward > 0:
                wins += 1

        self.model.train()

        # Calculate statistics
        n = len(rewards)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) if n > 1 else 0

        # bb/100 and confidence interval
        bb_per_100 = mean_reward * 100
        # 95% CI: ±1.96 * SE, where SE = σ/√n * 100
        std_error = (std_reward / np.sqrt(n)) * 100 if n > 0 else 0
        ci_95 = 1.96 * std_error

        win_rate = wins / n if n > 0 else 0
        print(f"  [EVAL] Evaluated {n} games: win_rate={win_rate:.1%}, bb/100={bb_per_100:+.1f}")

        return {
            'win_rate': win_rate,
            'eval_reward': mean_reward,
            'eval_bb_per_100': bb_per_100,
            'eval_std': std_reward,
            'eval_ci_95': ci_95,
            'eval_hands': n
        }

    def _evaluate_against_pool_elo(self) -> tuple:
        """
        Evaluate current model against ALL agents in the pool using round-robin.
        Returns (new_elo, results) where results is list of {opponent, games, wins, elo_change}.

        This is the AlphaHoldem ELO-based selection method:
        - New agent plays against each pool agent
        - ELO is calculated from the matchup results
        - Agent is added to pool, K-best by ELO are kept
        """
        if len(self.opponent_pool.agents) == 0:
            return self.config.initial_elo, []

        # Start with initial ELO
        new_elo = self.config.initial_elo
        results = []

        original_opponent = self.opponent
        self.model.eval()

        # Play against each agent in pool
        for idx, agent_info in enumerate(self.opponent_pool.agents):
            opponent = self.opponent_pool.get_agent_by_idx(idx, ActorCritic, self.config.num_actions)
            if opponent is None:
                continue

            opponent = opponent.to(self.device)
            opponent.eval()
            self.opponent = opponent

            wins = 0
            games = self.config.elo_games_per_matchup

            for _ in range(games):
                obs = self._reset_env()
                done = False
                game_reward = 0

                while not done:
                    if self.env.state.is_terminal():
                        reward = self.env.state.get_payoff(0)
                        game_reward += reward
                        break

                    action_mask = self.env.get_action_mask()
                    action, _, _ = self.ppo.get_action(obs, action_mask, deterministic=True)
                    _, reward, done, _ = self.env.step(action)
                    game_reward += reward

                    if not done:
                        obs = self._get_current_obs()

                if game_reward > 0:
                    wins += 1

            # Calculate ELO change from this matchup
            opponent_elo = agent_info['elo']

            # Update ELO for each game result
            for _ in range(wins):
                # Win against this opponent
                expected = 1 / (1 + 10 ** ((opponent_elo - new_elo) / 400))
                new_elo += self.config.elo_k_factor * (1.0 - expected)

            for _ in range(games - wins):
                # Loss against this opponent
                expected = 1 / (1 + 10 ** ((opponent_elo - new_elo) / 400))
                new_elo += self.config.elo_k_factor * (0.0 - expected)

            results.append({
                'opponent': agent_info['update_num'],
                'opponent_elo': opponent_elo,
                'games': games,
                'wins': wins,
                'win_rate': wins / games
            })

        self.model.train()
        self.opponent = original_opponent

        return new_elo, results

    def _save_checkpoint(self, final: bool = False):
        """Save training checkpoint."""
        suffix = "final" if final else f"{self.update_count}"
        path = Path(self.config.checkpoint_dir) / f"checkpoint_{suffix}.pt"

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.ppo.optimizer.state_dict(),
            'update_count': self.update_count,
            'total_timesteps': self.total_timesteps,
            'config': asdict(self.config)
        }, path)

        print(f"Saved checkpoint: {path}")

    def _log_stats(self, stats: Dict):
        """Log training statistics."""
        mean_reward = np.mean(list(self.episode_rewards)[-100:]) if self.episode_rewards else 0
        mean_length = np.mean(list(self.episode_lengths)[-100:]) if self.episode_lengths else 0

        log_entry = {
            'update': self.update_count,
            'timesteps': self.total_timesteps,
            'mean_reward': mean_reward,
            'mean_length': mean_length,
            **stats
        }
        self.logs.append(log_entry)

        # TensorBoard logging
        if self.writer is not None and self.update_count % self.config.log_every == 0:
            # Training metrics
            self.writer.add_scalar('train/policy_loss', stats.get('policy_loss', 0), self.total_timesteps)
            self.writer.add_scalar('train/value_loss', stats.get('value_loss', 0), self.total_timesteps)
            self.writer.add_scalar('train/entropy', stats.get('entropy', 0), self.total_timesteps)
            self.writer.add_scalar('train/kl', stats.get('kl', 0), self.total_timesteps)
            self.writer.add_scalar('train/learning_rate', stats.get('lr', 0), self.total_timesteps)

            # Episode metrics
            bb_per_100 = mean_reward * 100  # Standard poker metric
            self.writer.add_scalar('episode/mean_reward', mean_reward, self.total_timesteps)
            self.writer.add_scalar('episode/bb_per_100', bb_per_100, self.total_timesteps)
            self.writer.add_scalar('episode/mean_length', mean_length, self.total_timesteps)
            self.writer.add_scalar('episode/total_hands', self.total_hands, self.total_timesteps)

            # Evaluation metrics (if available)
            if 'win_rate' in stats:
                self.writer.add_scalar('eval/win_rate', stats['win_rate'], self.total_timesteps)
                self.writer.add_scalar('eval/bb_per_100', stats.get('eval_bb_per_100', 0), self.total_timesteps)
                self.writer.add_scalar('eval/ci_95', stats.get('eval_ci_95', 0), self.total_timesteps)
                self.writer.add_scalar('eval/std', stats.get('eval_std', 0), self.total_timesteps)
                self.eval_win_rates.append(stats['win_rate'])
                self.eval_rewards.append(stats.get('eval_reward', 0))

            # Progress metrics
            self.writer.add_scalar('progress/updates', self.update_count, self.total_timesteps)
            self.writer.add_scalar('progress/episodes', len(self.episode_rewards), self.total_timesteps)

        # Save JSON logs periodically
        if self.update_count % 10 == 0:
            with open(self.log_file, 'w') as f:
                json.dump(self.logs, f, indent=2)

    def close(self):
        """Cleanup resources."""
        if self.writer is not None:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train AlphaHoldem-style poker AI")
    parser.add_argument('--timesteps', type=int, default=10_000_000, help='Total training timesteps')
    parser.add_argument('--stack', type=float, default=100.0, help='Starting stack in BB')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--checkpoint-dir', type=str, default='alphaholdem/checkpoints')
    parser.add_argument('--log-dir', type=str, default='alphaholdem/logs')
    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    config = TrainConfig(
        starting_stack=args.stack,
        total_timesteps=args.timesteps,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=device
    )

    trainer = SelfPlayTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
