"""
Base Agent interface for ARES-HU.

All poker agents inherit from BaseAgent and implement the act() method.
"""

from abc import ABC, abstractmethod
from typing import Optional
from src.game.holdem_state import HoldemState, Action


class BaseAgent(ABC):
    """
    Abstract base class for poker agents.

    An agent observes a game state and chooses an action to take.
    Different agent types use different decision-making strategies:
    - CFRAgent: Uses pre-computed CFR blueprint
    - NeuralAgent: Uses neural network value predictions
    - RandomAgent: Random baseline for testing
    """

    def __init__(self, name: str = "Agent"):
        """
        Initialize agent.

        Args:
            name: Agent name for logging/display
        """
        self.name = name
        self.hands_played = 0
        self.total_winnings = 0.0

    @abstractmethod
    def act(self, state: HoldemState, player: int) -> Action:
        """
        Choose an action given the current game state.

        Args:
            state: Current game state
            player: Player number (0 or 1)

        Returns:
            Action to take

        Note: This is the core method every agent must implement.
        """
        pass

    def reset_stats(self):
        """Reset agent statistics."""
        self.hands_played = 0
        self.total_winnings = 0.0

    def update_result(self, winnings: float):
        """
        Update agent statistics after a hand completes.

        Args:
            winnings: Net winnings for this hand (positive = won, negative = lost)
        """
        self.hands_played += 1
        self.total_winnings += winnings

    def get_stats(self) -> dict:
        """
        Get agent performance statistics.

        Returns:
            Dictionary with stats (hands played, total winnings, win rate, etc.)
        """
        avg_winnings = self.total_winnings / self.hands_played if self.hands_played > 0 else 0.0

        return {
            'name': self.name,
            'hands_played': self.hands_played,
            'total_winnings': self.total_winnings,
            'avg_winnings_per_hand': avg_winnings,
            'bb_per_100': avg_winnings * 100  # Convert to big blinds per 100 hands
        }

    def __str__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        return (f"{self.name}: {stats['hands_played']} hands, "
                f"{stats['bb_per_100']:.2f} bb/100")
