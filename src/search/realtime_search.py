"""
Real-Time Search Engine for ARES-HU

This is the KEY component that enables "any stack, any bet, any action" play.

Architecture:
1. Given ANY game state (any stack depth, any bet history)
2. Build a depth-limited subgame tree
3. Run CFR search (100-1000 iterations)
4. Use neural network OR heuristic for leaf evaluation
5. Apply QRE for exploitation
6. Return action based on converged strategy

This allows playing at 200bb Slumbot even if trained on 20bb data,
because the search adapts to the specific situation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
import time


class Street(Enum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


class ActionType(Enum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET = 3
    RAISE = 4
    ALL_IN = 5


@dataclass(frozen=True)
class Action:
    type: ActionType
    amount: float = 0  # For bet/raise actions

    def __repr__(self):
        if self.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            return f"{self.type.name}({self.amount:.1f})"
        return self.type.name

    def __hash__(self):
        return hash((self.type, round(self.amount, 2)))

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return self.type == other.type and abs(self.amount - other.amount) < 0.01


@dataclass
class GameState:
    """Represents a poker game state for search."""

    # Core state
    pot: float
    stacks: List[float]  # [p0_stack, p1_stack]
    street: Street
    current_player: int

    # Cards
    hole_cards: List[List[str]]  # [[p0_cards], [p1_cards]]
    board: List[str]

    # Betting state
    bets_this_street: List[float]  # [p0_bet, p1_bet]
    last_raise_size: float = 0
    num_raises_this_street: int = 0

    # Game config
    big_blind: float = 1.0
    max_raises_per_street: int = 4

    # Terminal flags
    is_terminal: bool = False
    terminal_values: Optional[List[float]] = None

    def to_call(self) -> float:
        """Amount current player needs to call."""
        opp = 1 - self.current_player
        return max(0, self.bets_this_street[opp] - self.bets_this_street[self.current_player])

    def effective_stack(self) -> float:
        """Minimum of both stacks (what's actually in play)."""
        return min(self.stacks[0], self.stacks[1])

    def spr(self) -> float:
        """Stack-to-pot ratio."""
        if self.pot <= 0:
            return 100.0
        return self.effective_stack() / self.pot

    def get_legal_actions(self) -> List[Action]:
        """Get all legal actions for current player."""
        if self.is_terminal:
            return []

        actions = []
        to_call = self.to_call()
        my_stack = self.stacks[self.current_player]

        # Can always fold if facing a bet
        if to_call > 0:
            actions.append(Action(ActionType.FOLD))

        # Check (if nothing to call)
        if to_call == 0:
            actions.append(Action(ActionType.CHECK))

        # Call (if facing a bet and have chips)
        if to_call > 0 and my_stack > 0:
            call_amount = min(to_call, my_stack)
            if call_amount >= my_stack:
                actions.append(Action(ActionType.ALL_IN, my_stack))
            else:
                actions.append(Action(ActionType.CALL, call_amount))

        # Bet/Raise (if we have chips and haven't hit max raises)
        if my_stack > to_call and self.num_raises_this_street < self.max_raises_per_street:
            min_raise = max(self.big_blind, self.last_raise_size)
            remaining_after_call = my_stack - to_call

            if remaining_after_call >= min_raise:
                # Generate bet sizes based on pot
                pot_after_call = self.pot + to_call

                # GTO-standard bet sizes
                bet_fractions = [0.33, 0.5, 0.75, 1.0, 1.5]

                for frac in bet_fractions:
                    bet_size = pot_after_call * frac
                    if bet_size >= min_raise and bet_size <= remaining_after_call:
                        total_bet = to_call + bet_size
                        action_type = ActionType.RAISE if to_call > 0 else ActionType.BET
                        actions.append(Action(action_type, total_bet))

                # All-in
                if remaining_after_call > 0:
                    actions.append(Action(ActionType.ALL_IN, my_stack))

        return actions

    def apply_action(self, action: Action) -> 'GameState':
        """Apply action and return new state."""
        new_state = GameState(
            pot=self.pot,
            stacks=self.stacks.copy(),
            street=self.street,
            current_player=1 - self.current_player,  # Switch player
            hole_cards=self.hole_cards,
            board=self.board.copy(),
            bets_this_street=self.bets_this_street.copy(),
            last_raise_size=self.last_raise_size,
            num_raises_this_street=self.num_raises_this_street,
            big_blind=self.big_blind,
            max_raises_per_street=self.max_raises_per_street
        )

        p = self.current_player

        if action.type == ActionType.FOLD:
            # Opponent wins pot
            new_state.is_terminal = True
            new_state.terminal_values = [0.0, 0.0]
            new_state.terminal_values[1 - p] = self.pot + sum(self.bets_this_street)
            new_state.terminal_values[p] = -self.bets_this_street[p]

        elif action.type == ActionType.CHECK:
            # Check if betting round complete
            if self._is_betting_round_complete(new_state):
                new_state = self._advance_street(new_state)

        elif action.type == ActionType.CALL:
            call_amount = self.to_call()
            new_state.stacks[p] -= call_amount
            new_state.bets_this_street[p] += call_amount

            if self._is_betting_round_complete(new_state):
                new_state = self._advance_street(new_state)

        elif action.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            amount = action.amount
            new_state.stacks[p] -= amount
            raise_size = amount - self.bets_this_street[p] - self.to_call()
            new_state.bets_this_street[p] += amount
            new_state.last_raise_size = max(raise_size, self.big_blind)
            new_state.num_raises_this_street += 1

            # Check if opponent is all-in (can't act)
            if new_state.stacks[1 - p] == 0:
                new_state = self._advance_street(new_state)

        return new_state

    def _is_betting_round_complete(self, state: 'GameState') -> bool:
        """Check if betting round is complete."""
        # Both players have acted and bets are equal
        if state.bets_this_street[0] == state.bets_this_street[1]:
            return True
        return False

    def _advance_street(self, state: 'GameState') -> 'GameState':
        """Advance to next street or showdown."""
        # Add bets to pot
        state.pot += sum(state.bets_this_street)
        state.bets_this_street = [0.0, 0.0]
        state.last_raise_size = 0
        state.num_raises_this_street = 0
        state.current_player = 0  # First to act postflop

        if state.street == Street.RIVER or min(state.stacks) == 0:
            # Showdown
            state.is_terminal = True
            # Values will be computed by evaluator
            state.terminal_values = None
        else:
            # Next street
            state.street = Street(state.street.value + 1)

        return state

    def info_key(self, player: int) -> str:
        """Generate information set key for a player."""
        cards = ''.join(sorted(self.hole_cards[player]))
        board = ''.join(sorted(self.board))
        bets = f"{self.bets_this_street[0]:.0f}_{self.bets_this_street[1]:.0f}"
        return f"{cards}|{board}|{self.street.value}|{bets}|{self.pot:.0f}"


@dataclass
class SearchNode:
    """Node in the search tree."""
    state: GameState
    player: int
    actions: List[Action] = field(default_factory=list)

    # CFR data
    regret_sum: np.ndarray = None
    strategy_sum: np.ndarray = None

    def __post_init__(self):
        if not self.state.is_terminal:
            self.actions = self.state.get_legal_actions()
            n_actions = len(self.actions)
            if n_actions > 0:
                self.regret_sum = np.zeros(n_actions)
                self.strategy_sum = np.zeros(n_actions)


class RealtimeSearch:
    """
    Real-time CFR search engine.

    Given any game state, builds a subgame and solves it using CFR.
    Uses neural network or heuristic for leaf evaluation.
    """

    def __init__(
        self,
        leaf_evaluator: Optional[Callable[[GameState, int], float]] = None,
        qre_tau: float = 1.0,
        max_depth: int = 100,  # Max actions before using leaf eval
        verbose: bool = False
    ):
        self.leaf_evaluator = leaf_evaluator or self._default_leaf_eval
        self.qre_tau = qre_tau
        self.max_depth = max_depth
        self.verbose = verbose

        # Search tree
        self.nodes: Dict[str, SearchNode] = {}

        # Stats
        self.iterations = 0
        self.nodes_visited = 0

    def search(
        self,
        state: GameState,
        iterations: int = 1000,
        time_limit: float = None
    ) -> Tuple[Action, Dict[Action, float]]:
        """
        Run CFR search from the given state.

        Args:
            state: Current game state
            iterations: Number of CFR iterations
            time_limit: Optional time limit in seconds

        Returns:
            (best_action, action_probabilities)
        """
        self.nodes = {}
        self.iterations = 0
        self.nodes_visited = 0

        start_time = time.time()

        for i in range(iterations):
            if time_limit and (time.time() - start_time) > time_limit:
                break

            # Run one CFR iteration
            self._cfr(state, depth=0, reach_probs=[1.0, 1.0])
            self.iterations += 1

        elapsed = time.time() - start_time

        if self.verbose:
            print(f"Search: {self.iterations} iters, {self.nodes_visited} nodes, {elapsed:.2f}s")

        # Get strategy for root node
        root_key = state.info_key(state.current_player)
        if root_key not in self.nodes:
            # Fallback: return first legal action
            actions = state.get_legal_actions()
            if actions:
                return actions[0], {a: 1.0/len(actions) for a in actions}
            return None, {}

        node = self.nodes[root_key]
        strategy = self._get_average_strategy(node)

        # Build action->prob dict
        action_probs = {node.actions[i]: strategy[i] for i in range(len(node.actions))}

        # Select action (sample or argmax)
        best_idx = np.argmax(strategy)
        best_action = node.actions[best_idx]

        return best_action, action_probs

    def _cfr(
        self,
        state: GameState,
        depth: int,
        reach_probs: List[float]
    ) -> np.ndarray:
        """
        CFR traversal. Returns expected values for each player.
        """
        self.nodes_visited += 1

        # Terminal node
        if state.is_terminal:
            if state.terminal_values is not None:
                return np.array(state.terminal_values)
            else:
                # Showdown - use evaluator
                ev0 = self._evaluate_showdown(state, 0)
                return np.array([ev0, -ev0])

        # Depth limit - use leaf evaluator
        if depth >= self.max_depth:
            ev0 = self.leaf_evaluator(state, 0)
            return np.array([ev0, -ev0])

        player = state.current_player
        info_key = state.info_key(player)

        # Get or create node
        if info_key not in self.nodes:
            self.nodes[info_key] = SearchNode(state, player)
        node = self.nodes[info_key]

        if len(node.actions) == 0:
            # No legal actions (shouldn't happen)
            return np.array([0.0, 0.0])

        # Get current strategy
        strategy = self._get_strategy(node)

        # Compute action values
        action_values = np.zeros((len(node.actions), 2))

        for a_idx, action in enumerate(node.actions):
            # Update reach probability
            new_reach = reach_probs.copy()
            new_reach[player] *= strategy[a_idx]

            # Apply action and recurse
            new_state = state.apply_action(action)
            action_values[a_idx] = self._cfr(new_state, depth + 1, new_reach)

        # Compute node value
        node_value = np.zeros(2)
        for a_idx in range(len(node.actions)):
            node_value += strategy[a_idx] * action_values[a_idx]

        # Update regrets (for current player)
        opp = 1 - player
        for a_idx in range(len(node.actions)):
            regret = action_values[a_idx, player] - node_value[player]
            node.regret_sum[a_idx] += reach_probs[opp] * regret

        # Update strategy sum
        node.strategy_sum += reach_probs[player] * strategy

        return node_value

    def _get_strategy(self, node: SearchNode) -> np.ndarray:
        """Get current strategy using regret matching (with QRE)."""
        n_actions = len(node.actions)

        if self.qre_tau > 0:
            # QRE: softmax over regrets with numerical stability
            regrets = node.regret_sum / max(1, self.iterations)
            # Clip regrets to prevent overflow
            regrets = np.clip(regrets, -100, 100)
            # Subtract max for numerical stability
            regrets = regrets - regrets.max()
            exp_r = np.exp(regrets / self.qre_tau)
            total = exp_r.sum()
            if total > 0 and np.isfinite(total):
                return exp_r / total
            return np.ones(n_actions) / n_actions
        else:
            # Standard regret matching
            positive = np.maximum(node.regret_sum, 0)
            total = positive.sum()
            if total > 0:
                return positive / total
            return np.ones(n_actions) / n_actions

    def _get_average_strategy(self, node: SearchNode) -> np.ndarray:
        """Get average strategy (what we actually play)."""
        total = node.strategy_sum.sum()
        if total > 0:
            return node.strategy_sum / total
        return np.ones(len(node.actions)) / len(node.actions)

    def _evaluate_showdown(self, state: GameState, player: int) -> float:
        """Evaluate showdown for a player."""
        # Simple heuristic - in real implementation, use hand evaluator
        return self.leaf_evaluator(state, player)

    def _default_leaf_eval(self, state: GameState, player: int) -> float:
        """Default heuristic leaf evaluation."""
        # Return fraction of pot based on stack ratio
        my_stack = state.stacks[player]
        opp_stack = state.stacks[1 - player]
        total = my_stack + opp_stack + state.pot

        if total > 0:
            # Slight advantage to player with more chips
            return (my_stack / total - 0.5) * state.pot
        return 0.0


class NeuralLeafEvaluator:
    """
    Neural network leaf evaluator for real-time search.

    Uses trained value network to estimate EV at leaf nodes.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str):
        """Load trained value network."""
        try:
            import torch
            from src.neural.value_network import ValueNetwork, ValueNetworkTrainer

            trainer = ValueNetworkTrainer.load(path)
            self.model = trainer.model
            self.model.eval()
            print(f"Loaded value network from {path}")
        except Exception as e:
            print(f"Warning: Could not load value network: {e}")
            self.model = None

    def __call__(self, state: GameState, player: int) -> float:
        """Evaluate state for a player."""
        if self.model is None:
            return self._heuristic_eval(state, player)

        # Create PBS encoding
        encoding = self._create_encoding(state)

        # Run through network
        import torch
        with torch.no_grad():
            x = torch.tensor(encoding, dtype=torch.float32).unsqueeze(0)
            values = self.model(x)
            return values[0, player].item()

    def _create_encoding(self, state: GameState) -> np.ndarray:
        """Create PBS encoding for neural network."""
        features = []

        # Normalized features
        total_chips = sum(state.stacks) + state.pot
        features.append(state.pot / max(total_chips, 1))
        features.append(state.stacks[0] / max(total_chips, 1))
        features.append(state.stacks[1] / max(total_chips, 1))
        features.append(state.bets_this_street[0] / max(total_chips, 1))
        features.append(state.bets_this_street[1] / max(total_chips, 1))

        # Street one-hot
        for s in Street:
            features.append(1.0 if s == state.street else 0.0)

        # Current player
        features.append(float(state.current_player))

        # Pad to expected dimension
        while len(features) < 3040:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def _heuristic_eval(self, state: GameState, player: int) -> float:
        """Fallback heuristic evaluation."""
        # Simple: proportional to stack
        my_stack = state.stacks[player]
        total = sum(state.stacks) + state.pot
        if total > 0:
            return (my_stack / total - 0.5) * state.pot
        return 0.0


# =============================================================================
# Helper function to create state from Slumbot format
# =============================================================================

def create_state_from_slumbot(
    hole_cards: List[str],
    board: List[str],
    action_history: str,
    client_pos: int,
    pot: float,
    stacks: List[float]
) -> GameState:
    """
    Create GameState from Slumbot API format.

    Args:
        hole_cards: Hero's hole cards ['Ah', 'Kd']
        board: Board cards ['Ts', '9c', '2h']
        action_history: Slumbot action string 'b200c/kk/b400c'
        client_pos: Our position (0=SB, 1=BB)
        pot: Current pot size
        stacks: [our_stack, opp_stack]
    """
    # Determine street from board
    if len(board) == 0:
        street = Street.PREFLOP
    elif len(board) == 3:
        street = Street.FLOP
    elif len(board) == 4:
        street = Street.TURN
    else:
        street = Street.RIVER

    # Parse who's turn it is from action history
    # This is simplified - full parsing would track all actions
    current_player = client_pos

    state = GameState(
        pot=pot,
        stacks=stacks,
        street=street,
        current_player=current_player,
        hole_cards=[hole_cards, []],  # We don't know opponent's cards
        board=board,
        bets_this_street=[0.0, 0.0],
        big_blind=100  # Slumbot uses 50/100 blinds
    )

    return state


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Create a simple test state
    state = GameState(
        pot=3.0,  # SB + BB + limps
        stacks=[18.5, 18.5],
        street=Street.FLOP,
        current_player=0,
        hole_cards=[['Ah', 'Kd'], ['??', '??']],
        board=['Ts', '9c', '2h'],
        bets_this_street=[0.0, 0.0],
        big_blind=1.0
    )

    print("Test state:")
    print(f"  Pot: {state.pot}")
    print(f"  Stacks: {state.stacks}")
    print(f"  SPR: {state.spr():.1f}")
    print(f"  Street: {state.street}")
    print(f"  Legal actions: {state.get_legal_actions()}")
    print()

    # Run search
    print("Running CFR search (1000 iterations)...")
    search = RealtimeSearch(qre_tau=1.0, verbose=True)
    action, probs = search.search(state, iterations=1000)

    print(f"\nBest action: {action}")
    print("Action probabilities:")
    for a, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"  {a}: {p:.1%}")
