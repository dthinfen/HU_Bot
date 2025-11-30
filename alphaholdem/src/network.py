"""
AlphaHoldem Neural Network Architecture

Actor-Critic network with:
- CNN backbone for processing 3D tensor state
- Separate policy and value heads
- Optional: Transformer attention for betting sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class ResidualBlock(nn.Module):
    """Residual block with two conv layers."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class CNNBackbone(nn.Module):
    """
    CNN backbone for processing 3D tensor state.

    Architecture based on AlphaHoldem:
    - Initial conv to expand channels
    - Residual blocks for deep feature extraction
    - Global average pooling
    """

    def __init__(
        self,
        input_channels: int = 38,
        hidden_channels: int = 128,
        num_residual_blocks: int = 4,
        output_dim: int = 256
    ):
        super().__init__()

        # Initial convolution
        self.input_conv = nn.Conv2d(input_channels, hidden_channels, 3, padding=1)
        self.input_bn = nn.BatchNorm2d(hidden_channels)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_residual_blocks)
        ])

        # Final processing
        self.output_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.output_bn = nn.BatchNorm2d(hidden_channels)

        # Output dimension (after global average pooling)
        self.output_dim = output_dim
        self.fc = nn.Linear(hidden_channels, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, height=4, width=13)
        x = F.relu(self.input_bn(self.input_conv(x)))

        for block in self.residual_blocks:
            x = block(x)

        x = F.relu(self.output_bn(self.output_conv(x)))

        # Global average pooling
        x = x.mean(dim=[2, 3])  # (batch, hidden_channels)

        # Final projection
        x = F.relu(self.fc(x))

        return x


class MLPBackbone(nn.Module):
    """
    Simple MLP backbone for flat vector input.

    Alternative to CNN for faster training/inference.
    """

    def __init__(
        self,
        input_dim: int = 376,  # SimpleEncoder total dim
        hidden_dim: int = 512,
        num_layers: int = 4,
        output_dim: int = 256
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PolicyHead(nn.Module):
    """
    Policy head: outputs action probabilities.

    AlphaHoldem paper: 6.8M params in FC layers total.
    Using deep FC network to match paper architecture.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 1024,  # Paper has large FC layers
        num_layers: int = 3,  # Deeper network
        num_actions: int = 8
    ):
        super().__init__()

        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        self.fc_layers = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, num_actions)

    def forward(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.fc_layers(x)
        logits = self.output(x)

        # Apply action mask if provided (set invalid actions to -inf)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), float('-inf'))

        return logits


class ValueHead(nn.Module):
    """
    Value head: outputs state value estimate.

    AlphaHoldem paper: 6.8M params in FC layers total.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 1024,  # Paper has large FC layers
        num_layers: int = 3  # Deeper network
    ):
        super().__init__()

        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        self.fc_layers = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_layers(x)
        value = self.output(x)
        return value.squeeze(-1)


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO.

    Architecture based on AlphaHoldem paper:
    - Shared CNN backbone (1.8M params in paper)
    - Separate policy and value heads (6.8M params total in paper)

    Paper total: 8.6M params
    """

    def __init__(
        self,
        input_channels: int = 38,
        use_cnn: bool = True,
        hidden_dim: int = 256,  # Backbone output dim
        num_actions: int = 8,
        num_residual_blocks: int = 4,
        fc_hidden_dim: int = 1024,  # FC layer width (paper: large FC layers)
        fc_num_layers: int = 3  # FC layer depth
    ):
        super().__init__()

        self.use_cnn = use_cnn
        self.num_actions = num_actions

        # Backbone
        if use_cnn:
            self.backbone = CNNBackbone(
                input_channels=input_channels,
                hidden_channels=128,
                num_residual_blocks=num_residual_blocks,
                output_dim=hidden_dim
            )
        else:
            from alphaholdem.src.encoder import SimpleEncoder
            self.backbone = MLPBackbone(
                input_dim=SimpleEncoder.TOTAL_DIM,
                hidden_dim=512,
                num_layers=4,
                output_dim=hidden_dim
            )

        # Heads with configurable depth/width (paper has 6.8M params in FC)
        self.policy_head = PolicyHead(hidden_dim, fc_hidden_dim, fc_num_layers, num_actions)
        self.value_head = ValueHead(hidden_dim, fc_hidden_dim, fc_num_layers)

    def forward(
        self,
        x: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: State tensor (batch, channels, 4, 13) for CNN or (batch, dim) for MLP
            action_mask: Boolean mask for valid actions (batch, num_actions)

        Returns:
            logits: Action logits (batch, num_actions)
            value: State value (batch,)
        """
        features = self.backbone(x)
        logits = self.policy_head(features, action_mask)
        value = self.value_head(features)

        return logits, value

    def get_action(
        self,
        x: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Returns:
            action: Sampled action (batch,)
            log_prob: Log probability of action (batch,)
            value: State value (batch,)
        """
        logits, value = self.forward(x, action_mask)

        # Create distribution
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate_actions(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Returns:
            log_prob: Log probability of actions (batch,)
            value: State value (batch,)
            entropy: Policy entropy (batch,)
        """
        logits, value = self.forward(x, action_mask)

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, value, entropy

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PseudoSiameseNetwork(nn.Module):
    """
    Pseudo-Siamese architecture from AlphaHoldem.

    Two identical networks:
    - Current policy (being trained)
    - Historical policy (frozen, from K-best pool)

    Used for diverse self-play training.
    """

    def __init__(
        self,
        input_channels: int = 38,
        hidden_dim: int = 256,
        num_actions: int = 8,
        num_residual_blocks: int = 4
    ):
        super().__init__()

        # Current network (trainable)
        self.current = ActorCritic(
            input_channels=input_channels,
            use_cnn=True,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            num_residual_blocks=num_residual_blocks
        )

        # Historical network (frozen during training)
        self.historical = ActorCritic(
            input_channels=input_channels,
            use_cnn=True,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            num_residual_blocks=num_residual_blocks
        )

        # Freeze historical network
        for param in self.historical.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, use_historical: bool = False, action_mask: Optional[torch.Tensor] = None):
        """Forward pass through either current or historical network."""
        if use_historical:
            with torch.no_grad():
                return self.historical(x, action_mask)
        return self.current(x, action_mask)

    def update_historical(self, checkpoint_path: str):
        """Load historical network from checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.historical.load_state_dict(state_dict)
        for param in self.historical.parameters():
            param.requires_grad = False

    def sync_historical(self):
        """Copy current weights to historical (for initial sync)."""
        self.historical.load_state_dict(self.current.state_dict())
        for param in self.historical.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    # Test networks
    batch_size = 32
    num_actions = 8

    # Test CNN backbone
    print("Testing CNN Actor-Critic:")
    model = ActorCritic(input_channels=38, use_cnn=True, num_actions=num_actions)
    x = torch.randn(batch_size, 38, 4, 13)
    mask = torch.ones(batch_size, num_actions, dtype=torch.bool)
    mask[:, 0] = False  # Mask out fold

    logits, value = model(x, mask)
    print(f"  Input shape: {x.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Parameters: {model.num_parameters:,}")

    # Test action sampling
    action, log_prob, value = model.get_action(x, mask)
    print(f"  Action shape: {action.shape}")
    print(f"  Log prob shape: {log_prob.shape}")

    # Test MLP backbone
    print("\nTesting MLP Actor-Critic:")
    model_mlp = ActorCritic(use_cnn=False, num_actions=num_actions)
    x_flat = torch.randn(batch_size, 376)

    logits, value = model_mlp(x_flat, mask)
    print(f"  Input shape: {x_flat.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Parameters: {model_mlp.num_parameters:,}")

    # Test pseudo-siamese
    print("\nTesting Pseudo-Siamese:")
    siamese = PseudoSiameseNetwork(num_actions=num_actions)
    print(f"  Current params: {siamese.current.num_parameters:,}")
    print(f"  Total params: {sum(p.numel() for p in siamese.parameters()):,}")
