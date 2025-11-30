"""
Random Agent - baseline player that chooses actions randomly.

Useful for testing and as a sanity check (trained agents should beat random).
"""

import numpy as np
from src.agent.base_agent import BaseAgent
from src.game.holdem_state import HoldemState, Action


class RandomAgent(BaseAgent):
    """
    Agent that chooses actions uniformly at random from legal actions.

    This is the weakest possible strategy and serves as a baseline.
    Any trained agent should easily beat a random player.
    """

    def __init__(self, name: str = "RandomAgent", seed: int = None):
        """
        Initialize random agent.

        Args:
            name: Agent name
            seed: Random seed for reproducibility (None = random)
        """
        super().__init__(name)
        self.rng = np.random.RandomState(seed)

    def act(self, state: HoldemState, player: int) -> Action:
        """
        Choose a random legal action.

        Args:
            state: Current game state
            player: Player number (0 or 1)

        Returns:
            Randomly chosen action from legal actions
        """
        legal_actions = state.get_legal_actions()

        if not legal_actions:
            raise ValueError(f"No legal actions available in state: {state}")

        # Choose uniformly at random
        action_idx = self.rng.randint(len(legal_actions))
        return legal_actions[action_idx]


class CallFoldAgent(BaseAgent):
    """
    Simple baseline agent that either calls or folds (never bets/raises).

    Slightly smarter than random - folds weak hands, calls with strong hands.
    Still very exploitable.
    """

    def __init__(
        self,
        name: str = "CallFoldAgent",
        fold_threshold: float = 0.3,
        seed: int = None
    ):
        """
        Initialize call/fold agent.

        Args:
            name: Agent name
            fold_threshold: Probability of folding when facing a bet (0-1)
            seed: Random seed
        """
        super().__init__(name)
        self.fold_threshold = fold_threshold
        self.rng = np.random.RandomState(seed)

    def act(self, state: HoldemState, player: int) -> Action:
        """
        Choose to call or fold (never bet/raise).

        Args:
            state: Current game state
            player: Player number

        Returns:
            CALL, CHECK, or FOLD action
        """
        legal_actions = state.get_legal_actions()

        # Separate actions by type
        from src.game.holdem_state import ActionType

        can_check = any(a.type == ActionType.CHECK for a in legal_actions)
        can_call = any(a.type == ActionType.CALL for a in legal_actions)
        can_fold = any(a.type == ActionType.FOLD for a in legal_actions)

        # If can check (no bet to call), always check
        if can_check:
            return next(a for a in legal_actions if a.type == ActionType.CHECK)

        # Facing a bet: call or fold based on threshold
        if can_call and can_fold:
            if self.rng.random() < self.fold_threshold:
                return next(a for a in legal_actions if a.type == ActionType.FOLD)
            else:
                return next(a for a in legal_actions if a.type == ActionType.CALL)

        # Fallback: choose first legal action
        return legal_actions[0]
