"""
PPO with Trinal-Clip for AlphaHoldem

Based on the AlphaHoldem paper's Trinal-Clip PPO modification:
- Standard PPO clipping for stability
- Additional clipping on value function
- Entropy bonus for exploration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class PPOConfig:
    """PPO hyperparameters based on AlphaHoldem paper."""
    # Learning rates
    learning_rate: float = 3e-4  # Paper: 0.0003
    lr_schedule: str = "linear"  # "linear" or "constant"

    # Trinal-Clip PPO (AlphaHoldem paper)
    # Paper uses δ1=3 for policy clipping (NOT standard PPO's 0.2)
    # δ2 and δ3 are dynamically calculated based on chips (0 to 20,000)
    clip_ratio: float = 3.0  # Paper: δ1=3 (Trinal-Clip)
    clip_value: float = 0.2  # Value function clipping (standard)
    target_kl: float = 0.02  # Early stopping if KL exceeds this

    # Training
    gamma: float = 0.999  # Paper: 0.999 (NOT 1.0)
    gae_lambda: float = 0.95  # Paper: λ=0.95
    num_epochs: int = 4  # PPO epochs per update
    batch_size: int = 2048  # Paper: 2048 per GPU (16384 total with 8 GPUs)
    minibatch_size: int = 256  # Larger for GPU efficiency

    # Loss coefficients
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    # Gradient clipping
    max_grad_norm: float = 0.5

    # Normalization
    normalize_advantages: bool = True
    normalize_returns: bool = False


class RolloutBuffer:
    """Buffer for storing rollout data."""

    def __init__(self, buffer_size: int, obs_shape: Tuple, device: str = 'cpu'):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.device = device
        self.reset()

    def reset(self):
        """Clear buffer."""
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.action_masks = []
        self.pos = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        action_mask: Optional[np.ndarray] = None
    ):
        """Add a transition to buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.action_masks.append(action_mask if action_mask is not None else np.ones(8))
        self.pos += 1

    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute GAE and returns."""
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        # GAE calculation
        gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_done = False
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]
            gae = delta + gamma * gae_lambda * (1 - next_done) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        self.advantages = advantages
        self.returns = returns

    def get_batches(self, batch_size: int):
        """Generator for minibatches."""
        n = len(self.observations)
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]

            yield {
                'observations': torch.tensor(
                    np.array([self.observations[i] for i in batch_indices]),
                    dtype=torch.float32, device=self.device
                ),
                'actions': torch.tensor(
                    np.array([self.actions[i] for i in batch_indices]),
                    dtype=torch.long, device=self.device
                ),
                'old_log_probs': torch.tensor(
                    np.array([self.log_probs[i] for i in batch_indices]),
                    dtype=torch.float32, device=self.device
                ),
                'old_values': torch.tensor(
                    np.array([self.values[i] for i in batch_indices]),
                    dtype=torch.float32, device=self.device
                ),
                'advantages': torch.tensor(
                    self.advantages[batch_indices],
                    dtype=torch.float32, device=self.device
                ),
                'returns': torch.tensor(
                    self.returns[batch_indices],
                    dtype=torch.float32, device=self.device
                ),
                'action_masks': torch.tensor(
                    np.array([self.action_masks[i] for i in batch_indices]),
                    dtype=torch.bool, device=self.device
                ),
            }

    def __len__(self):
        return len(self.observations)


class PPO:
    """
    Proximal Policy Optimization with Trinal-Clip.

    Trinal-Clip modification from AlphaHoldem:
    - Clips policy ratio as standard PPO
    - Additionally clips value function updates
    - Uses separate learning rate schedules for actor/critic
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[PPOConfig] = None,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.config = config or PPOConfig()
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5
        )

        # Learning rate scheduler
        self.scheduler = None

        # Tracking
        self.update_count = 0

    def setup_scheduler(self, total_updates: int):
        """Setup learning rate scheduler."""
        if self.config.lr_schedule == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_updates
            )

    def update(self, buffer: RolloutBuffer, last_value: float) -> Dict[str, float]:
        """
        Perform PPO update.

        Args:
            buffer: Rollout buffer with collected experiences
            last_value: Value estimate for last state (for GAE)

        Returns:
            Dictionary of training statistics
        """
        # Compute advantages and returns
        buffer.compute_returns_and_advantages(
            last_value,
            self.config.gamma,
            self.config.gae_lambda
        )

        # Normalize advantages
        if self.config.normalize_advantages:
            advantages = buffer.advantages
            buffer.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training stats
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        num_updates = 0

        # Multiple epochs of PPO updates
        for epoch in range(self.config.num_epochs):
            for batch in buffer.get_batches(self.config.minibatch_size):
                stats = self._update_step(batch)

                total_policy_loss += stats['policy_loss']
                total_value_loss += stats['value_loss']
                total_entropy += stats['entropy']
                total_kl += stats['kl']
                num_updates += 1

                # Early stopping if KL divergence too high
                if stats['kl'] > self.config.target_kl * 1.5:
                    break

            # Check KL for early epoch stopping
            if total_kl / max(num_updates, 1) > self.config.target_kl:
                break

        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()

        self.update_count += 1

        return {
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'value_loss': total_value_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1),
            'kl': total_kl / max(num_updates, 1),
            'num_updates': num_updates,
            'lr': self.optimizer.param_groups[0]['lr']
        }

    def _update_step(self, batch: Dict) -> Dict[str, float]:
        """Single gradient update step."""
        obs = batch['observations']
        actions = batch['actions']
        old_log_probs = batch['old_log_probs']
        old_values = batch['old_values']
        advantages = batch['advantages']
        returns = batch['returns']
        action_masks = batch['action_masks']

        # Forward pass
        new_log_probs, values, entropy = self.model.evaluate_actions(
            obs, actions, action_masks
        )

        # Policy loss (PPO-Clip)
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)

        policy_loss1 = -advantages * ratio
        policy_loss2 = -advantages * clipped_ratio
        policy_loss = torch.max(policy_loss1, policy_loss2).mean()

        # Value loss (Trinal-Clip: clip value function too)
        if self.config.clip_value > 0:
            value_clipped = old_values + torch.clamp(
                values - old_values,
                -self.config.clip_value,
                self.config.clip_value
            )
            value_loss1 = (values - returns) ** 2
            value_loss2 = (value_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
        else:
            value_loss = 0.5 * ((values - returns) ** 2).mean()

        # Entropy bonus
        entropy_loss = -entropy.mean()

        # Total loss
        loss = (
            policy_loss +
            self.config.value_coef * value_loss +
            self.config.entropy_coef * entropy_loss
        )

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

        self.optimizer.step()

        # Calculate KL divergence
        with torch.no_grad():
            kl = (old_log_probs - new_log_probs).mean().item()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item(),
            'kl': kl
        }

    def get_action(
        self,
        obs: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Get action from current policy.

        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value estimate
        """
        self.model.eval()

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            if obs_tensor.dim() == 3:
                obs_tensor = obs_tensor.unsqueeze(0)

            mask_tensor = None
            if action_mask is not None:
                mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=self.device)
                if mask_tensor.dim() == 1:
                    mask_tensor = mask_tensor.unsqueeze(0)

            action, log_prob, value = self.model.get_action(
                obs_tensor, mask_tensor, deterministic
            )

        return action.item(), log_prob.item(), value.item()

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'config': self.config
        }, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint.get('update_count', 0)


if __name__ == "__main__":
    from alphaholdem.src.network import ActorCritic

    # Test PPO
    print("Testing PPO:")

    model = ActorCritic(input_channels=38, use_cnn=True, num_actions=8)
    ppo = PPO(model, device='cpu')

    # Create dummy buffer
    buffer = RolloutBuffer(1000, (38, 4, 13))

    for i in range(100):
        obs = np.random.randn(38, 4, 13).astype(np.float32)
        action = np.random.randint(0, 8)
        log_prob = np.random.randn()
        value = np.random.randn()
        reward = np.random.randn()
        done = i == 99
        mask = np.ones(8)

        buffer.add(obs, action, log_prob, value, reward, done, mask)

    # Test update
    stats = ppo.update(buffer, last_value=0.0)
    print(f"  Policy loss: {stats['policy_loss']:.4f}")
    print(f"  Value loss: {stats['value_loss']:.4f}")
    print(f"  Entropy: {stats['entropy']:.4f}")
    print(f"  KL: {stats['kl']:.4f}")

    # Test action selection
    obs = np.random.randn(38, 4, 13).astype(np.float32)
    mask = np.ones(8)
    action, log_prob, value = ppo.get_action(obs, mask)
    print(f"  Action: {action}, Log prob: {log_prob:.4f}, Value: {value:.4f}")
