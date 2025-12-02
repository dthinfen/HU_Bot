"""
Simple Heads-Up Poker Environment

Lightweight environment for RL training.
"""

import numpy as np
import random
from typing import Tuple, Dict, Optional, Callable
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.game.cards import Card, Deck
from src.game.holdem_state import HoldemState, Action, ActionType


class HeadsUpEnv:
    """
    Heads-up no-limit hold'em environment for RL training.

    Simple interface:
    - reset() -> starts new hand
    - step(action) -> (obs, reward, done, info)
    - get_action_mask() -> valid action mask
    """

    # Expanded action space (14 actions total)
    # Based on common poker bet sizes used in solvers/bots
    DEFAULT_RAISE_FRACTIONS = [
        0.33,   # 1/3 pot (common c-bet size)
        0.5,    # 1/2 pot
        0.67,   # 2/3 pot (common value bet)
        0.75,   # 3/4 pot
        1.0,    # Pot size
        1.25,   # 1.25x pot
        1.5,    # 1.5x pot (common 3-bet size)
        2.0,    # 2x pot (polarized bet)
        2.5,    # 2.5x pot (over-bet)
        3.0,    # 3x pot (big over-bet)
    ]

    def __init__(
        self,
        starting_stack: float = 100.0,
        num_actions: int = 14,
        raise_fractions: Optional[list] = None
    ):
        """
        Initialize environment.

        Args:
            starting_stack: Starting stack in big blinds
            num_actions: Number of discrete actions
                0: Fold
                1: Check/Call
                2-11: Raise sizes (configurable pot fractions)
                12: Min-raise
                13: All-in
            raise_fractions: List of pot fractions for raise sizes
        """
        self.starting_stack = starting_stack
        self.num_actions = num_actions
        self.raise_fractions = raise_fractions or self.DEFAULT_RAISE_FRACTIONS[:num_actions - 4]

        self.state: Optional[HoldemState] = None
        self.deck: Optional[Deck] = None
        self.opponent_policy: Optional[Callable] = None
        self.hero_player: int = 0  # We always train player 0

        # Cache encoder to avoid creating new one every opponent action
        self._opponent_encoder = None

    def reset(self) -> None:
        """Reset environment for new hand."""
        # Random button
        button = random.randint(0, 1)

        # Deal cards
        self.deck = Deck()
        hero_cards = self.deck.deal(2)
        villain_cards = self.deck.deal(2)

        # Setup blinds
        sb = 0.5
        bb = 1.0

        if button == 0:  # Hero is button (SB)
            hero_stack = self.starting_stack - sb
            villain_stack = self.starting_stack - bb
            hero_invested = sb
            villain_invested = bb
            current_player = 0  # Button acts first preflop
        else:  # Villain is button (Hero is BB)
            hero_stack = self.starting_stack - bb
            villain_stack = self.starting_stack - sb
            hero_invested = bb
            villain_invested = sb
            current_player = 1  # Button (villain) acts first

        self.state = HoldemState(
            hero_hole=tuple(hero_cards),
            villain_hole=tuple(villain_cards),
            board=(),
            street=0,
            pot=sb + bb,
            hero_stack=hero_stack,
            villain_stack=villain_stack,
            hero_invested_this_street=hero_invested,
            villain_invested_this_street=villain_invested,
            action_history="",
            current_player=current_player,
            button=button
        )

        # Let opponent act first if needed
        self._maybe_opponent_acts()

    def step(self, action: int) -> Tuple[None, float, bool, Dict]:
        """
        Take action in environment.

        Args:
            action: Discrete action index

        Returns:
            obs: None (caller should encode state separately)
            reward: Reward (0 until terminal)
            done: Whether hand is over
            info: Additional info
        """
        assert self.state is not None, "Must call reset() first"
        assert not self.state.is_terminal(), "Episode is done"
        assert self.state.current_player == self.hero_player, "Not hero's turn"

        # Convert to poker action
        poker_action = self._action_to_poker_action(action)

        # Apply action
        self.state = self.state.apply_action(poker_action)

        # Handle street advancement
        self._maybe_advance_street()

        # Check terminal
        done = self.state.is_terminal()
        reward = 0.0

        if done:
            reward = self.state.get_payoff(self.hero_player)
        else:
            # Let opponent act
            self._maybe_opponent_acts()
            done = self.state.is_terminal()
            if done:
                reward = self.state.get_payoff(self.hero_player)

        info = {
            'pot': self.state.pot,
            'hero_stack': self.state.hero_stack,
            'villain_stack': self.state.villain_stack,
            'street': self.state.street
        }

        return None, reward, done, info

    def get_action_mask(self) -> np.ndarray:
        """Get mask of valid actions."""
        mask = np.zeros(self.num_actions, dtype=np.float32)

        if self.state is None or self.state.is_terminal():
            return mask

        to_call = self.state.to_call()
        stack = (self.state.hero_stack if self.state.current_player == 0
                 else self.state.villain_stack)

        if stack <= 0:
            # Player is all-in, can only check
            mask[1] = 1.0  # Check (no action needed)
            return mask

        if to_call > 0:
            mask[0] = 1.0  # Fold
            mask[1] = 1.0  # Call (or all-in if stack < to_call)
            if stack > to_call:
                for i in range(2, self.num_actions):
                    mask[i] = 1.0  # Raises + all-in
        else:
            # Not facing bet
            mask[1] = 1.0  # Check
            if stack > 0:
                for i in range(2, self.num_actions):
                    mask[i] = 1.0  # Bets + all-in

        return mask

    def set_opponent(self, policy: Callable) -> None:
        """
        Set opponent policy function.

        Args:
            policy: Function(obs, action_mask) -> action
        """
        self.opponent_policy = policy

    def _action_to_poker_action(self, action: int) -> Action:
        """Convert discrete action index to poker Action."""
        to_call = self.state.to_call()
        stack = (self.state.hero_stack if self.state.current_player == 0
                 else self.state.villain_stack)

        if action == 0:  # Fold
            if to_call > 0:
                return Action(ActionType.FOLD)
            return Action(ActionType.CHECK)

        elif action == 1:  # Check/Call
            if to_call > 0:
                if stack <= 0:
                    # Already all-in, just check
                    return Action(ActionType.CHECK)
                if stack <= to_call:
                    return Action(ActionType.ALL_IN, stack)
                return Action(ActionType.CALL)
            return Action(ActionType.CHECK)

        elif action == self.num_actions - 1:  # All-in
            if stack <= 0:
                # No chips to go all-in with, check/fold instead
                if to_call > 0:
                    return Action(ActionType.FOLD)
                return Action(ActionType.CHECK)
            return Action(ActionType.ALL_IN, stack)

        elif action == self.num_actions - 2:  # Min-raise
            if stack <= 0:
                # No chips, check/fold
                if to_call > 0:
                    return Action(ActionType.FOLD)
                return Action(ActionType.CHECK)
            min_raise = self.state.min_raise()
            if min_raise >= stack:
                return Action(ActionType.ALL_IN, stack)
            elif to_call > 0:
                return Action(ActionType.RAISE, min_raise)
            else:
                return Action(ActionType.BET, max(min_raise, 1.0))  # Min 1bb bet

        else:  # Raise sizes (2 to num_actions-3)
            if stack <= 0:
                # No chips, check/fold
                if to_call > 0:
                    return Action(ActionType.FOLD)
                return Action(ActionType.CHECK)

            raise_idx = action - 2
            if raise_idx < len(self.raise_fractions):
                frac = self.raise_fractions[raise_idx]
            else:
                frac = 1.0

            pot_after_call = self.state.pot + to_call
            raise_amount = to_call + (pot_after_call * frac)

            # Ensure minimum raise is respected
            min_raise = self.state.min_raise()
            raise_amount = max(raise_amount, min_raise)
            raise_amount = min(raise_amount, stack)

            if raise_amount >= stack:
                return Action(ActionType.ALL_IN, stack)
            elif to_call > 0:
                return Action(ActionType.RAISE, raise_amount)
            else:
                return Action(ActionType.BET, raise_amount)

    def _maybe_advance_street(self) -> None:
        """Advance street if betting round is complete."""
        while (self.state.should_advance_street() and
               not self.state.is_terminal() and
               self.state.street < 3):

            new_street = self.state.street + 1

            # Deal new cards
            if new_street == 1:  # Flop
                new_cards = self.deck.deal(3)
                new_board = tuple(new_cards)
            else:  # Turn or River
                new_card = self.deck.deal(1)[0]
                new_board = self.state.board + (new_card,)

            # OOP acts first postflop
            first_to_act = 1 - self.state.button

            self.state = HoldemState(
                hero_hole=self.state.hero_hole,
                villain_hole=self.state.villain_hole,
                board=new_board,
                street=new_street,
                pot=self.state.pot,
                hero_stack=self.state.hero_stack,
                villain_stack=self.state.villain_stack,
                hero_invested_this_street=0.0,
                villain_invested_this_street=0.0,
                action_history=self.state.action_history + '/',
                current_player=first_to_act,
                button=self.state.button
            )

    def _maybe_opponent_acts(self) -> None:
        """Let opponent act if it's their turn."""
        while (not self.state.is_terminal() and
               self.state.current_player != self.hero_player):

            mask = self._get_opponent_action_mask()

            if self.opponent_policy is not None:
                # Get observation for opponent
                obs = self._get_opponent_obs()
                action = self.opponent_policy(obs, mask)
            else:
                # Random action
                valid = np.where(mask)[0]
                action = np.random.choice(valid)

            # Apply action
            poker_action = self._action_to_poker_action(action)
            self.state = self.state.apply_action(poker_action)
            self._maybe_advance_street()

    def _get_opponent_action_mask(self) -> np.ndarray:
        """Get action mask from opponent's perspective."""
        mask = np.zeros(self.num_actions, dtype=np.float32)

        to_call = self.state.to_call()
        # Opponent is villain when hero_player is 0
        stack = (self.state.villain_stack if self.hero_player == 0
                 else self.state.hero_stack)

        if stack <= 0:
            # Player is all-in, can only check
            mask[1] = 1.0  # Check (no action needed)
            return mask

        if to_call > 0:
            mask[0] = 1.0  # Fold
            mask[1] = 1.0  # Call (or all-in if stack < to_call)
            if stack > to_call:
                for i in range(2, self.num_actions):
                    mask[i] = 1.0
        else:
            mask[1] = 1.0  # Check
            if stack > 0:
                for i in range(2, self.num_actions):
                    mask[i] = 1.0

        return mask

    def _get_opponent_obs(self) -> np.ndarray:
        """Get observation from opponent's perspective."""
        # Use cached encoder to avoid creating new object every call
        if self._opponent_encoder is None:
            from alphaholdem.src.encoder import AlphaHoldemEncoder
            self._opponent_encoder = AlphaHoldemEncoder()
        encoder = self._opponent_encoder

        # Swap hero/villain perspective
        if self.hero_player == 0:
            hole = [(c.rank - 2, c.suit) for c in self.state.villain_hole]
            hero_stack = self.state.villain_stack
            villain_stack = self.state.hero_stack
            hero_invested = self.state.villain_invested_this_street
            villain_invested = self.state.hero_invested_this_street
            is_button = (self.state.button == 1)
        else:
            hole = [(c.rank - 2, c.suit) for c in self.state.hero_hole]
            hero_stack = self.state.hero_stack
            villain_stack = self.state.villain_stack
            hero_invested = self.state.hero_invested_this_street
            villain_invested = self.state.villain_invested_this_street
            is_button = (self.state.button == 0)

        return encoder.encode(
            hole_cards=hole,
            board_cards=[(c.rank - 2, c.suit) for c in self.state.board],
            pot=self.state.pot,
            hero_stack=hero_stack,
            villain_stack=villain_stack,
            hero_invested=hero_invested,
            villain_invested=villain_invested,
            street=self.state.street,
            is_button=is_button,
            action_history=encoder._parse_action_history(self.state.action_history)
        )


if __name__ == "__main__":
    # Quick test
    env = HeadsUpEnv(starting_stack=100.0)
    env.reset()

    print(f"Initial state:")
    print(f"  Hero hole: {env.state.hero_hole}")
    print(f"  Pot: {env.state.pot}")
    print(f"  Hero stack: {env.state.hero_stack}")
    print(f"  Action mask: {env.get_action_mask()}")

    # Play random hand
    done = False
    total_reward = 0
    steps = 0
    while not done:
        mask = env.get_action_mask()
        valid = np.where(mask)[0]
        action = np.random.choice(valid)
        _, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

    print(f"\nHand complete after {steps} steps")
    print(f"  Final reward: {total_reward:.2f}bb")
    print(f"  Final pot: {info['pot']:.2f}bb")
