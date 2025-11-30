"""
Policy Network for ARES-HU

Predicts action probabilities from Public Belief State (PBS) encoding.
Uses the same architecture as the value network but with softmax output.

The network outputs logits for a fixed set of action buckets:
- 0: Fold
- 1: Check/Call
- 2: Bet/Raise 0.5x pot
- 3: Bet/Raise 1x pot
- 4: Bet/Raise 2x pot
- 5: All-in

Illegal actions are masked before softmax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


# Action bucket indices - aligned with GTO solver standards
# Reference: GTO Wizard uses 3 sizes (small/medium/large) + check + fold + all-in
# See: https://blog.gtowizard.com/pot-geometry/
ACTION_FOLD = 0
ACTION_CHECK_CALL = 1
ACTION_BET_SMALL = 2      # 25-40% pot (probe, block, range bet)
ACTION_BET_MEDIUM = 3     # 40-75% pot (standard value/bluff)
ACTION_BET_LARGE = 4      # 75-125% pot (geometric, polarized)
ACTION_BET_OVERBET = 5    # 125%+ pot (overbets, max pressure)
ACTION_ALL_IN = 6
NUM_ACTION_BUCKETS = 7


class PolicyNetwork(nn.Module):
    """
    Neural network that predicts action probabilities from PBS encoding.

    Architecture:
    - Input: PBS encoding (hand + board + betting history)
    - Hidden: N layers with LayerNorm + ReLU
    - Output: Logits for each action bucket (softmax applied externally with masking)
    """

    def __init__(
        self,
        input_dim: int = 3040,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_actions: int = NUM_ACTION_BUCKETS,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_actions = num_actions

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer (logits, no activation)
        layers.append(nn.Linear(hidden_dim, num_actions))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Initialize with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: PBS encoding -> action logits.

        Args:
            x: PBS encoding [batch, input_dim]

        Returns:
            Action logits [batch, num_actions] (apply softmax externally)
        """
        return self.network(x)

    def get_action_probs(
        self,
        x: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Get action probabilities with optional masking and temperature.

        Args:
            x: PBS encoding [batch, input_dim]
            action_mask: Boolean mask [batch, num_actions], True = legal action
            temperature: Softmax temperature (1.0 = normal, <1 = sharper, >1 = softer)

        Returns:
            Action probabilities [batch, num_actions]
        """
        logits = self.forward(x) / temperature

        if action_mask is not None:
            # Set illegal actions to -inf so softmax gives 0
            logits = logits.masked_fill(~action_mask, float('-inf'))

        return F.softmax(logits, dim=-1)

    def sample_action(
        self,
        x: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.

        Args:
            x: PBS encoding [batch, input_dim]
            action_mask: Boolean mask for legal actions
            temperature: Softmax temperature

        Returns:
            (action_indices, action_probs) tuple
        """
        probs = self.get_action_probs(x, action_mask, temperature)

        # Sample from categorical distribution
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()

        return actions, probs

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PolicyNetworkTrainer:
    """Training loop for policy network using cross-entropy loss."""

    def __init__(
        self,
        model: PolicyNetwork,
        learning_rate: float = 3e-4,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        self.train_losses = []
        self.val_losses = []

    def train_epoch(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        masks_train: Optional[torch.Tensor] = None,
        batch_size: int = 1024
    ) -> float:
        """
        Train for one epoch.

        Args:
            X_train: PBS encodings [N, input_dim]
            y_train: Target action probabilities [N, num_actions]
            masks_train: Legal action masks [N, num_actions] (optional)
            batch_size: Batch size

        Returns:
            Average training loss
        """
        self.model.train()

        n = len(X_train)
        indices = torch.randperm(n)
        total_loss = 0.0
        num_batches = 0

        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train[batch_idx].to(self.device)
            y_batch = y_train[batch_idx].to(self.device)

            self.optimizer.zero_grad()

            # Get logits
            logits = self.model(X_batch)

            # Apply mask if provided
            if masks_train is not None:
                mask_batch = masks_train[batch_idx].to(self.device)
                logits = logits.masked_fill(~mask_batch, float('-inf'))

            # Cross-entropy loss with soft targets (KL divergence)
            # Use numerically stable computation
            probs = F.softmax(logits, dim=-1)

            # Clamp both predictions and targets to avoid log(0)
            probs = probs.clamp(min=1e-7, max=1-1e-7)
            y_batch_safe = y_batch.clamp(min=1e-7)

            # KL divergence: sum(p * log(p/q)) = sum(p * log(p)) - sum(p * log(q))
            # We only need the cross-entropy part: -sum(p * log(q))
            loss = -(y_batch_safe * probs.log()).sum(dim=-1).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def evaluate(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        masks_val: Optional[torch.Tensor] = None,
        batch_size: int = 1024
    ) -> Tuple[float, float]:
        """
        Evaluate on validation set.

        Returns:
            (loss, accuracy) tuple
        """
        self.model.eval()

        with torch.no_grad():
            n = len(X_val)
            total_loss = 0.0
            total_correct = 0
            num_batches = 0

            for i in range(0, n, batch_size):
                X_batch = X_val[i:i+batch_size].to(self.device)
                y_batch = y_val[i:i+batch_size].to(self.device)

                logits = self.model(X_batch)

                if masks_val is not None:
                    mask_batch = masks_val[i:i+batch_size].to(self.device)
                    logits = logits.masked_fill(~mask_batch, float('-inf'))

                probs = F.softmax(logits, dim=-1)
                probs = probs.clamp(min=1e-7, max=1-1e-7)
                y_batch_safe = y_batch.clamp(min=1e-7)
                loss = -(y_batch_safe * probs.log()).sum(dim=-1).mean()

                total_loss += loss.item()

                # Top-1 accuracy (predicted max vs target max)
                pred_actions = logits.argmax(dim=-1)
                target_actions = y_batch.argmax(dim=-1)
                total_correct += (pred_actions == target_actions).sum().item()

                num_batches += 1

            accuracy = total_correct / n

        return total_loss / num_batches, accuracy

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        masks_train: Optional[torch.Tensor] = None,
        masks_val: Optional[torch.Tensor] = None,
        epochs: int = 100,
        batch_size: int = 1024,
        patience: int = 15,
        verbose: bool = True
    ) -> dict:
        """Full training loop with early stopping."""

        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(X_train, y_train, masks_train, batch_size)
            val_loss, val_acc = self.evaluate(X_val, y_val, masks_val, batch_size)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.1%}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)

        final_loss, final_acc = self.evaluate(X_val, y_val, masks_val)

        return {
            'best_epoch': best_epoch + 1,
            'best_val_loss': best_val_loss,
            'final_accuracy': final_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers,
                'num_actions': self.model.num_actions
            }
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'PolicyNetworkTrainer':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint['config']

        model = PolicyNetwork(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_actions=config['num_actions']
        )
        model.load_state_dict(checkpoint['model_state_dict'])

        trainer = cls(model, device=device)
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.train_losses = checkpoint.get('train_losses', [])
        trainer.val_losses = checkpoint.get('val_losses', [])

        return trainer


def map_action_to_bucket(action_type: str, bet_size: float = 0, pot_size: float = 1) -> int:
    """
    Map a concrete action to an action bucket.

    Uses GTO-standard sizing buckets:
    - Small: 25-40% pot (probe, block, range bet)
    - Medium: 40-75% pot (standard value/bluff)
    - Large: 75-125% pot (geometric, polarized)
    - Overbet: 125%+ pot (overbets)

    Args:
        action_type: 'fold', 'check', 'call', 'bet', 'raise', 'allin'
        bet_size: Size of bet/raise (in bb or absolute)
        pot_size: Current pot size for ratio calculation

    Returns:
        Action bucket index
    """
    action_type = action_type.lower()

    if action_type == 'fold':
        return ACTION_FOLD
    elif action_type in ('check', 'call'):
        return ACTION_CHECK_CALL
    elif action_type == 'allin':
        return ACTION_ALL_IN
    elif action_type in ('bet', 'raise'):
        # Map bet size to bucket based on pot ratio (GTO-standard boundaries)
        if pot_size > 0:
            ratio = bet_size / pot_size
            if ratio <= 0.40:
                return ACTION_BET_SMALL      # 25-40% pot
            elif ratio <= 0.75:
                return ACTION_BET_MEDIUM     # 40-75% pot
            elif ratio <= 1.25:
                return ACTION_BET_LARGE      # 75-125% pot
            else:
                return ACTION_BET_OVERBET    # 125%+ pot
        else:
            return ACTION_BET_MEDIUM  # Default
    else:
        return ACTION_CHECK_CALL  # Default fallback


def bucket_to_action_name(bucket: int) -> str:
    """Convert bucket index to human-readable name."""
    names = [
        'fold',
        'check/call',
        'bet small (25-40%)',
        'bet medium (40-75%)',
        'bet large (75-125%)',
        'overbet (125%+)',
        'all-in'
    ]
    return names[bucket] if 0 <= bucket < len(names) else 'unknown'
