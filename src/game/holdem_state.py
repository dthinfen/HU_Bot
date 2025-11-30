"""
Heads-Up No-Limit Hold'em game state representation.

Immutable state design similar to KuhnState but for full Hold'em.
Optimized for memory efficiency and CFR traversal speed.

Design principles:
- Immutable (thread-safe, cacheable)
- Minimal memory (<200 bytes per state)
- Fast operations (will be called millions of times in CFR)
- Support continuous bet sizing (abstractions layer handles discretization)
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum
from src.game.cards import Card, HandEvaluator


class ActionType(IntEnum):
    """Poker action types"""
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET = 3
    RAISE = 4
    ALL_IN = 5


@dataclass(frozen=True)
class Action:
    """
    Immutable poker action.

    For bets/raises, amount is in big blinds (absolute, not pot fraction).
    All-in is represented as BET or RAISE with amount = remaining stack.
    """

    type: ActionType
    amount: float = 0.0  # In big blinds

    def __post_init__(self):
        # Validate: only BET/RAISE/ALL_IN can have non-zero amounts
        if self.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            assert self.amount > 0, f"Bet/raise must have positive amount, got {self.amount}"
        else:
            assert self.amount == 0, f"{self.type.name} cannot have amount"

    def __str__(self) -> str:
        """Human-readable representation"""
        if self.type == ActionType.FOLD:
            return "Fold"
        elif self.type == ActionType.CHECK:
            return "Check"
        elif self.type == ActionType.CALL:
            return "Call"
        elif self.type == ActionType.BET:
            return f"Bet {self.amount:.1f}bb"
        elif self.type == ActionType.RAISE:
            return f"Raise {self.amount:.1f}bb"
        elif self.type == ActionType.ALL_IN:
            return f"All-in {self.amount:.1f}bb"
        return "Unknown"

    def to_code(self) -> str:
        """
        Encode action as compact string for history.

        Format:
        'f' = fold
        'x' = check
        'c' = call
        'b{amount}' = bet (e.g., 'b5.0')
        'r{amount}' = raise (e.g., 'r12.5')
        'a{amount}' = all-in (e.g., 'a50.0')
        """
        if self.type == ActionType.FOLD:
            return 'f'
        elif self.type == ActionType.CHECK:
            return 'x'
        elif self.type == ActionType.CALL:
            return 'c'
        elif self.type == ActionType.BET:
            return f'b{self.amount:.2f}'
        elif self.type == ActionType.RAISE:
            return f'r{self.amount:.2f}'
        elif self.type == ActionType.ALL_IN:
            return f'a{self.amount:.2f}'
        return '?'

    @staticmethod
    def from_code(code: str) -> 'Action':
        """Parse action from code string"""
        if code == 'f':
            return Action(ActionType.FOLD)
        elif code == 'x':
            return Action(ActionType.CHECK)
        elif code == 'c':
            return Action(ActionType.CALL)
        elif code[0] == 'b':
            amount = float(code[1:])
            return Action(ActionType.BET, amount)
        elif code[0] == 'r':
            amount = float(code[1:])
            return Action(ActionType.RAISE, amount)
        elif code[0] == 'a':
            amount = float(code[1:])
            return Action(ActionType.ALL_IN, amount)
        else:
            raise ValueError(f"Invalid action code: {code}")


@dataclass(frozen=True)
class HoldemState:
    """
    Immutable state representation for Heads-Up No-Limit Hold'em.

    Design for minimal memory (~150-200 bytes) and fast operations.
    All monetary amounts in big blinds for stack-size independence.
    """

    # Card information
    hero_hole: Tuple[Card, Card]
    villain_hole: Tuple[Card, Card]
    board: Tuple[Card, ...]  # 0, 3, 4, or 5 cards

    # Game state
    street: int  # 0=preflop, 1=flop, 2=turn, 3=river
    pot: float  # Total pot in bb

    # Player stacks (remaining, not invested)
    hero_stack: float
    villain_stack: float

    # Current street investments
    hero_invested_this_street: float
    villain_invested_this_street: float

    # Tracking
    action_history: str  # Encoded actions (e.g., "xb3.0c/xx/b10.0c")
    current_player: int  # 0=hero, 1=villain
    button: int  # 0=hero is button, 1=villain is button

    def __post_init__(self):
        """Validate state consistency"""
        # Basic validations
        assert len(self.hero_hole) == 2, "Hero must have 2 hole cards"
        assert len(self.villain_hole) == 2, "Villain must have 2 hole cards"
        assert 0 <= self.street <= 3, f"Invalid street: {self.street}"

        # Board size validation
        expected_board_size = [0, 3, 4, 5][self.street]
        if self.street > 0:
            assert len(self.board) == expected_board_size, \
                f"Street {self.street} requires {expected_board_size} board cards, got {len(self.board)}"

        # Money validations (with small tolerance for floating-point precision)
        EPSILON = 0.001
        assert self.pot >= -EPSILON, f"Invalid pot: {self.pot}"
        assert self.hero_stack >= -EPSILON, f"Invalid hero stack: {self.hero_stack}"
        assert self.villain_stack >= -EPSILON, f"Invalid villain stack: {self.villain_stack}"

        # Clamp near-zero values to exactly zero
        if -EPSILON <= self.hero_stack < 0:
            object.__setattr__(self, 'hero_stack', 0.0)
        if -EPSILON <= self.villain_stack < 0:
            object.__setattr__(self, 'villain_stack', 0.0)
        assert self.hero_invested_this_street >= 0
        assert self.villain_invested_this_street >= 0

        # Player validations
        assert self.current_player in (0, 1), f"Invalid current_player: {self.current_player}"
        assert self.button in (0, 1), f"Invalid button: {self.button}"

    def _get_street_actions(self) -> str:
        """
        Get action history for current street only.

        Action history uses '/' to denote street advancement:
        - Preflop: actions before first '/'
        - Flop: actions between first and second '/'
        - Turn: actions between second and third '/'
        - River: actions after third '/'

        Note: 'x' is used for CHECK actions, not street markers.
        """
        # Find positions of all '/' characters (street separators)
        sep_positions = [i for i, c in enumerate(self.action_history) if c == '/']

        if len(sep_positions) == 0:
            # No street advancement yet - all preflop
            return self.action_history
        elif self.street == 0:
            # Preflop: before first '/'
            return self.action_history[:sep_positions[0]]
        elif self.street <= len(sep_positions):
            # Get actions between appropriate '/' markers
            start_idx = sep_positions[self.street - 1] + 1
            if self.street < len(sep_positions):
                end_idx = sep_positions[self.street]
                return self.action_history[start_idx:end_idx]
            else:
                return self.action_history[start_idx:]
        else:
            # Beyond available '/' markers - return everything after last '/'
            return self.action_history[sep_positions[-1] + 1:]

    def is_terminal(self) -> bool:
        """Check if this state is terminal (hand over)"""
        # Someone folded (check entire action history, not just current street)
        if 'f' in self.action_history:
            return True

        # River action complete (showdown)
        if self.street == 3:  # River
            street_actions = self._get_street_actions()
            # Check if betting round is complete
            if self._is_street_complete(street_actions):
                return True

        # All-in situation (no more actions possible)
        if self.hero_stack == 0 or self.villain_stack == 0:
            # Check if both players have equal investment or one folded
            if self.hero_invested_this_street == self.villain_invested_this_street:
                return True

        return False

    def _is_street_complete(self, street_actions: str) -> bool:
        """Check if betting round is complete"""
        if not street_actions:
            return False

        # Both players have acted and investments are equal
        # OR someone folded
        # OR someone is all-in and investment is matched

        if 'f' in street_actions:
            return True

        # Count actions
        action_count = len([c for c in street_actions if c in 'xcfbra'])

        # Need at least 2 actions (one from each player) or both passed opportunity
        if action_count < 2:
            return False

        # If investments are equal and both have acted, street is complete
        if self.hero_invested_this_street == self.villain_invested_this_street:
            return True

        return False

    def get_payoff(self, player: int) -> float:
        """
        Get payoff for player at terminal state.

        Returns amount won/lost in big blinds (positive = win, negative = loss).
        """
        assert self.is_terminal(), "Can only get payoff at terminal state"

        # Find who folded (if anyone) - check entire action history
        if 'f' in self.action_history:
            # Determine who folded by counting actions (excluding street markers)
            action_count = 0
            for char in self.action_history:
                if char in 'cfbra':  # actual actions (not 'x' street markers)
                    action_count += 1
                    if char == 'f':
                        folder = (self.button if action_count % 2 == 1 else 1 - self.button)
                        winner = 1 - folder

                        # Winner gets the pot, folder loses their investment
                        if player == winner:
                            # Calculate what player invested total
                            player_invested = self._calculate_total_investment(player)
                            return self.pot - player_invested
                        else:
                            player_invested = self._calculate_total_investment(player)
                            return -player_invested

        # Showdown - evaluate hands
        # Must have at least flop to evaluate (5 cards total)
        if len(self.board) >= 3:
            hero_rank = HandEvaluator.evaluate(self.hero_hole, list(self.board))
            villain_rank = HandEvaluator.evaluate(self.villain_hole, list(self.board))

            result = HandEvaluator.compare(hero_rank, villain_rank)

            player_invested = self._calculate_total_investment(player)

            if result == 0:
                # Tie - split pot
                return (self.pot / 2) - player_invested
            elif (result < 0 and player == 0) or (result > 0 and player == 1):
                # Player wins
                return self.pot - player_invested
            else:
                # Player loses
                return -player_invested
        else:
            # Preflop all-in or error - should not reach here normally
            # For now, split pot
            player_invested = self._calculate_total_investment(player)
            return (self.pot / 2) - player_invested

    def _calculate_total_investment(self, player: int) -> float:
        """Calculate total amount player has invested in pot"""
        # Calculate from starting stack (pot + both stacks should equal starting stacks)
        # Total investment = starting_stack - current_stack
        #
        # Starting stack can be inferred: pot contains all invested money
        # plus whatever is left in stacks. We started with equal stacks.
        # starting_stack = (pot + hero_stack + villain_stack) / 2

        starting_stack = (self.pot + self.hero_stack + self.villain_stack) / 2

        if player == 0:  # Hero
            return starting_stack - self.hero_stack
        else:  # Villain
            return starting_stack - self.villain_stack

    def get_info_set(self, player: int) -> str:
        """
        Get information set string for CFR.

        Information set represents everything the player knows.
        Format: "{hole_cards}|{street}|{stack}|{position}|{action_history}"

        Example: "AsKh|preflop|100.0bb|BTN|xr3.0c"
        """
        # Player's hole cards
        if player == 0:
            hole = f"{self.hero_hole[0]}{self.hero_hole[1]}"
        else:
            hole = f"{self.villain_hole[0]}{self.villain_hole[1]}"

        # Street name
        street_names = ['preflop', 'flop', 'turn', 'river']
        street_name = street_names[self.street]

        # Effective stack
        eff_stack = self.effective_stack()

        # Position
        position = "BTN" if player == self.button else "BB"

        # Action history (current street only for now)
        street_history = self._get_street_actions()

        return f"{hole}|{street_name}|{eff_stack:.1f}bb|{position}|{street_history}"

    def get_legal_actions(self) -> List[Action]:
        """
        Get all legal actions at this state.

        Returns list of Action objects representing legal moves.
        """
        if self.is_terminal():
            return []

        actions = []

        # Determine current street actions
        street_actions = self._get_street_actions()

        # Calculate current bet to call
        to_call = self.to_call()

        # Get current player's stack
        stack = self.hero_stack if self.current_player == 0 else self.villain_stack

        # If player has no chips, they can only check (already all-in)
        if stack <= 0:
            actions.append(Action(ActionType.CHECK))
            return actions

        if to_call == 0:
            # No bet facing - can check or bet
            actions.append(Action(ActionType.CHECK))

            # Can bet any amount up to remaining stack
            # For now, return some standard bet sizes (abstraction will handle this properly)
            # This is just for the game engine to be functional
            if stack > 0:
                # Standard bet sizes: 33%, 50%, 75%, 100% pot
                for fraction in [0.33, 0.5, 0.75, 1.0]:
                    bet_size = min(self.pot * fraction, stack)
                    if bet_size > 0:
                        actions.append(Action(ActionType.BET, bet_size))

                # All-in option
                actions.append(Action(ActionType.ALL_IN, stack))
        else:
            # Facing a bet - can fold, call, or raise
            actions.append(Action(ActionType.FOLD))

            # Can call if have chips
            if stack >= to_call:
                actions.append(Action(ActionType.CALL))
            elif stack > 0:
                # All-in call (calling with remaining chips)
                actions.append(Action(ActionType.ALL_IN, stack))

            # Can raise if have chips beyond call
            if stack > to_call:
                # Minimum raise is previous raise size
                min_raise = self.min_raise()

                # Standard raise sizes
                current_pot_after_call = self.pot + to_call
                for fraction in [0.5, 0.75, 1.0]:
                    raise_total = to_call + (current_pot_after_call * fraction)
                    if raise_total >= min_raise and raise_total <= stack:
                        actions.append(Action(ActionType.RAISE, raise_total))

                # All-in option if not already included
                if stack > min_raise:
                    actions.append(Action(ActionType.ALL_IN, stack))

        # Remove duplicates (can happen with all-in)
        unique_actions = []
        seen = set()
        for action in actions:
            key = (action.type, round(action.amount, 2))
            if key not in seen:
                seen.add(key)
                unique_actions.append(action)

        return unique_actions

    def _is_valid_action(self, action: Action) -> bool:
        """
        Validate action is legal based on poker rules.

        More flexible than exact matching with get_legal_actions(),
        allowing discrete bet sizes to be validated.

        Args:
            action: Action to validate

        Returns:
            True if action is legal, False otherwise
        """
        if self.is_terminal():
            return False

        stack = self.hero_stack if self.current_player == 0 else self.villain_stack
        to_call = self.to_call()

        # If player has no stack, they're already all-in and can only check
        if stack <= 0:
            return action.type == ActionType.CHECK

        if to_call == 0:
            # Not facing a bet
            if action.type == ActionType.CHECK:
                return True
            elif action.type == ActionType.BET:
                return action.amount > 0 and action.amount <= stack
            elif action.type == ActionType.ALL_IN:
                return action.amount > 0 and action.amount <= stack
            else:
                return False  # FOLD, CALL, RAISE not legal
        else:
            # Facing a bet
            if action.type == ActionType.FOLD:
                return True
            elif action.type == ActionType.CALL:
                # Always allow call (if stack < to_call, it's an all-in call)
                return True
            elif action.type == ActionType.RAISE:
                # Match training behavior: allow any raise > to_call
                # (not enforcing strict min_raise requirement)
                return action.amount > to_call and action.amount <= stack
            elif action.type == ActionType.ALL_IN:
                return action.amount > 0 and action.amount <= stack
            else:
                return False  # CHECK, BET not legal when facing bet

    def apply_action(self, action: Action) -> 'HoldemState':
        """
        Apply action and return new state.

        State is immutable, so this creates a new HoldemState object.
        """
        assert not self.is_terminal(), "Cannot apply action to terminal state"

        # Validate action is legal (use flexible validation for discrete action compatibility)
        if not self._is_valid_action(action):
            legal_actions = self.get_legal_actions()
            raise AssertionError(f"Illegal action: {action}. Legal actions: {legal_actions}")

        # Update action history
        new_history = self.action_history + action.to_code()

        # Calculate new state based on action type
        new_pot = self.pot
        new_hero_stack = self.hero_stack
        new_villain_stack = self.villain_stack
        new_hero_invested = self.hero_invested_this_street
        new_villain_invested = self.villain_invested_this_street

        if self.current_player == 0:  # Hero acting
            if action.type == ActionType.FOLD:
                # Hand ends, but we still create valid state
                pass
            elif action.type == ActionType.CHECK:
                # No money moves
                pass
            elif action.type == ActionType.CALL:
                call_amount = self.to_call()
                new_hero_stack -= call_amount
                new_hero_invested += call_amount
                new_pot += call_amount
            elif action.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
                new_hero_stack -= action.amount
                new_hero_invested += action.amount
                new_pot += action.amount
        else:  # Villain acting
            if action.type == ActionType.FOLD:
                pass
            elif action.type == ActionType.CHECK:
                pass
            elif action.type == ActionType.CALL:
                call_amount = self.to_call()
                new_villain_stack -= call_amount
                new_villain_invested += call_amount
                new_pot += call_amount
            elif action.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
                new_villain_stack -= action.amount
                new_villain_invested += action.amount
                new_pot += action.amount

        # Determine next player
        # If street is complete, advance street; otherwise alternate
        new_current_player = 1 - self.current_player

        # Round to avoid floating-point precision issues
        new_pot = round(new_pot, 4)
        new_hero_stack = round(new_hero_stack, 4)
        new_villain_stack = round(new_villain_stack, 4)
        new_hero_invested = round(new_hero_invested, 4)
        new_villain_invested = round(new_villain_invested, 4)

        # Create new state
        new_state = HoldemState(
            hero_hole=self.hero_hole,
            villain_hole=self.villain_hole,
            board=self.board,
            street=self.street,
            pot=new_pot,
            hero_stack=new_hero_stack,
            villain_stack=new_villain_stack,
            hero_invested_this_street=new_hero_invested,
            villain_invested_this_street=new_villain_invested,
            action_history=new_history,
            current_player=new_current_player,
            button=self.button
        )

        # DON'T auto-advance streets - game manager will handle that with card dealing
        return new_state

    def should_advance_street(self) -> bool:
        """
        Check if betting round is complete and ready to advance to next street.

        Game manager should check this after each action and call advance_street()
        with newly dealt cards.
        """
        # Don't advance if terminal
        if self.is_terminal():
            return False

        # Don't advance if already on river
        if self.street >= 3:
            return False

        # Check if betting round complete
        street_actions = self._get_street_actions()
        return self._is_street_complete(street_actions)

    def advance_street(self) -> 'HoldemState':
        """
        Advance to next street, dealing cards and resetting betting.

        Note: This should be called manually when appropriate, or automatically
        by apply_action when betting round completes.
        """
        assert self.street < 3, "Already on river, cannot advance"
        assert not self.is_terminal(), "Cannot advance terminal state"

        # Deal next card(s) - NOTE: This is a placeholder
        # In real implementation, we'll need a deck to deal from
        # For now, board must be provided correctly from game manager

        new_street = self.street + 1

        # Reset street investments
        new_history = self.action_history + '/'

        # Determine who acts first postflop (out of position)
        # Button acts last postflop, so big blind (non-button) acts first
        first_to_act = 1 - self.button

        return HoldemState(
            hero_hole=self.hero_hole,
            villain_hole=self.villain_hole,
            board=self.board,  # NOTE: Board should have been updated by game manager
            street=new_street,
            pot=self.pot,
            hero_stack=self.hero_stack,
            villain_stack=self.villain_stack,
            hero_invested_this_street=0.0,
            villain_invested_this_street=0.0,
            action_history=new_history,
            current_player=first_to_act,
            button=self.button
        )

    # ==================== Helper Methods ====================

    def to_call(self) -> float:
        """Amount current player needs to call"""
        if self.current_player == 0:
            return max(0, self.villain_invested_this_street - self.hero_invested_this_street)
        else:
            return max(0, self.hero_invested_this_street - self.villain_invested_this_street)

    def min_raise(self) -> float:
        """
        Minimum legal raise size.

        Simplified version: just requires doubling the current bet.
        This creates smaller game trees and faster training.

        Returns:
            Minimum total amount for a legal raise
        """
        to_call = self.to_call()

        # Simple rule: must at least double the current bet
        # Or minimum of 2bb preflop, 1bb postflop
        if to_call > 0:
            return to_call * 2
        else:
            return 2.0 if self.street == -1 else 1.0

    def effective_stack(self) -> float:
        """Minimum of both stacks"""
        return min(self.hero_stack, self.villain_stack)

    def spr(self) -> float:
        """Stack-to-pot ratio (effective stack / pot)"""
        if self.pot == 0:
            return float('inf')
        return self.effective_stack() / self.pot

    def pot_odds(self) -> float:
        """Pot odds if facing a bet (amount to call / pot after call)"""
        to_call = self.to_call()
        if to_call == 0:
            return 0.0
        return to_call / (self.pot + to_call)

    def is_all_in(self) -> bool:
        """Check if anyone is all-in"""
        return self.hero_stack == 0 or self.villain_stack == 0

    def __str__(self) -> str:
        """Human-readable representation"""
        street_names = ['Preflop', 'Flop', 'Turn', 'River']

        # Board string
        if len(self.board) > 0:
            board_str = ' '.join(str(c) for c in self.board)
        else:
            board_str = "none"

        return (f"{street_names[self.street]}: "
                f"Pot={self.pot:.1f}bb, "
                f"Board=[{board_str}], "
                f"Hero={self.hero_stack:.1f}bb, "
                f"Villain={self.villain_stack:.1f}bb, "
                f"History: {self.action_history}")
