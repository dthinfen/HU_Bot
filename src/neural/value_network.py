"""
Value Network for ARES-HU

Predicts counterfactual values from Public Belief State (PBS) encoding.
Based on ReBeL architecture: 6 layers × 1536 hidden units.

For now, we use a smaller network for faster training on CPU.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class ValueNetwork(nn.Module):
    """
    Neural network that predicts counterfactual values from PBS encoding.

    Architecture options:
    - Small (CPU): 4 layers × 256 hidden → ~1M params
    - Medium: 4 layers × 512 hidden → ~4M params
    - Large (ReBeL): 6 layers × 1536 hidden → ~30M params
    """

    def __init__(
        self,
        input_dim: int = 3040,
        hidden_dim: int = 256,
        num_layers: int = 4,
        output_dim: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

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

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: PBS encoding → predicted value."""
        return self.network(x)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ValueNetworkTrainer:
    """Training loop for value network."""

    def __init__(
        self,
        model: ValueNetwork,
        learning_rate: float = 3e-4,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []

    def train_epoch(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        batch_size: int = 1024
    ) -> float:
        """Train for one epoch."""
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
            pred = self.model(X_batch)
            loss = self.criterion(pred, y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def evaluate(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        batch_size: int = 1024
    ) -> Tuple[float, float]:
        """Evaluate on validation set. Returns (loss, RMSE in bb)."""
        self.model.eval()

        with torch.no_grad():
            n = len(X_val)
            total_loss = 0.0
            total_se = 0.0
            num_batches = 0

            for i in range(0, n, batch_size):
                X_batch = X_val[i:i+batch_size].to(self.device)
                y_batch = y_val[i:i+batch_size].to(self.device)

                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)

                total_loss += loss.item()
                total_se += ((pred - y_batch) ** 2).sum().item()
                num_batches += 1

            mse = total_se / n
            rmse = np.sqrt(mse)

        return total_loss / num_batches, rmse

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 1024,
        patience: int = 10,
        verbose: bool = True
    ) -> dict:
        """
        Full training loop with early stopping.

        Returns:
            Training statistics dict
        """
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(X_train, y_train, batch_size)
            val_loss, val_rmse = self.evaluate(X_val, y_val, batch_size)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | RMSE: {val_rmse:.3f} bb")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                # Save best model
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)

        final_loss, final_rmse = self.evaluate(X_val, y_val)

        return {
            'best_epoch': best_epoch + 1,
            'best_val_loss': best_val_loss,
            'final_rmse': final_rmse,
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
                'num_layers': self.model.num_layers
            }
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'ValueNetworkTrainer':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']

        model = ValueNetwork(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])

        trainer = cls(model, device=device)
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.train_losses = checkpoint.get('train_losses', [])
        trainer.val_losses = checkpoint.get('val_losses', [])

        return trainer
