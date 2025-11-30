"""
Agent that uses C++ CFR binary strategy files.

The C++ solver stores strategies in a binary format with uint64 info set keys.
This agent replicates the C++ key hashing to look up strategies.
"""

import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.agent.base_agent import BaseAgent
from src.game.holdem_state import HoldemState, Action, ActionType


class CppCFRAgent(BaseAgent):
    """
    Agent that uses C++ CFR blueprint strategy.

    Loads binary strategy files from C++ solver and replicates
    the info set key hashing for compatible lookups.
    """

    MAGIC = 0x41524553  # "ARES"

    def __init__(
        self,
        blueprint_path: str,
        name: str = "CppCFRAgent",
        seed: Optional[int] = None
    ):
        """
        Initialize C++ CFR agent.

        Args:
            blueprint_path: Path to C++ .bin strategy file
            name: Agent name
            seed: Random seed for action sampling
        """
        super().__init__(name)

        self.blueprint_path = blueprint_path
        self.rng = np.random.RandomState(seed)

        # Load strategy from C++ binary format
        self.strategy, self.config = self._load_cpp_strategy(blueprint_path)

        print(f"Loaded C++ blueprint: {len(self.strategy)} information sets")
        print(f"  Stack: {self.config['stack']}bb, QRE tau: {self.config['qre_tau']}")

    def _load_cpp_strategy(self, path: str) -> Tuple[Dict, Dict]:
        """
        Load strategy from C++ binary format.

        Binary format:
        - magic: uint32 (0x41524553 "ARES")
        - version: uint32
        - eq_type: uint8
        - qre_tau: float32
        - starting_stack: float32
        - num_info_sets: uint64
        - For each info set:
          - key: uint64
          - num_actions: uint32
          - strategy_sum: float32[num_actions]

        Returns:
            (strategy_dict, config_dict)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Strategy file not found: {path}")

        strategy = {}
        config = {}

        with open(path, 'rb') as f:
            # Read header
            magic = struct.unpack('I', f.read(4))[0]
            if magic != self.MAGIC:
                raise ValueError(f"Invalid magic: {hex(magic)}, expected {hex(self.MAGIC)}")

            version = struct.unpack('I', f.read(4))[0]
            eq_type = struct.unpack('B', f.read(1))[0]
            qre_tau = struct.unpack('f', f.read(4))[0]
            stack = struct.unpack('f', f.read(4))[0]

            config = {
                'version': version,
                'equilibrium': 'QRE' if eq_type == 1 else 'Nash',
                'qre_tau': qre_tau,
                'stack': stack
            }

            # Read number of info sets
            num_info_sets = struct.unpack('Q', f.read(8))[0]

            # Read each info set
            for _ in range(num_info_sets):
                key = struct.unpack('Q', f.read(8))[0]
                num_actions = struct.unpack('I', f.read(4))[0]

                # Read strategy_sum
                strategy_sum = np.frombuffer(f.read(num_actions * 4), dtype=np.float32)

                # Convert to average strategy (normalize)
                total = strategy_sum.sum()
                if total > 0:
                    avg_strategy = strategy_sum / total
                else:
                    avg_strategy = np.ones(num_actions) / num_actions

                strategy[key] = avg_strategy

        return strategy, config

    @staticmethod
    def card_to_index_from_str(card_str: str) -> int:
        """
        Convert card string to 0-51 index (C++ compatible).

        Card format: "Ac" = Ace of clubs
        Index = suit * 13 + rank
        Where suit: c=0, d=1, h=2, s=3
              rank: 2=0, 3=1, ..., A=12
        """
        ranks = "23456789TJQKA"
        suits = "cdhs"

        rank_char = card_str[0].upper()
        suit_char = card_str[1].lower()

        rank = ranks.index(rank_char)
        suit = suits.index(suit_char)

        return suit * 13 + rank

    @staticmethod
    def card_to_index(card) -> int:
        """
        Convert Card object to 0-51 index (C++ compatible).

        Python Card: rank 2-14 (2=2, 14=A), suit 0-3 (c,d,h,s)
        C++ index = suit * 13 + (rank - 2)
        """
        # Python card.rank is 2-14 (2=2, 14=Ace)
        # C++ rank is 0-12 (0=2, 12=Ace)
        cpp_rank = card.rank - 2
        cpp_suit = card.suit
        return cpp_suit * 13 + cpp_rank

    @staticmethod
    def hole_cards_canonical_index(card1_idx: int, card2_idx: int) -> int:
        """
        Compute canonical index for hole cards (C++ compatible).

        Ensures card1 > card2 for consistent ordering.
        Uses combinatorial number system: idx1 * (idx1 - 1) / 2 + idx2
        """
        # Ensure canonical ordering (larger index first)
        if card1_idx < card2_idx:
            card1_idx, card2_idx = card2_idx, card1_idx

        return card1_idx * (card1_idx - 1) // 2 + card2_idx

    @staticmethod
    def action_encode(action_type: ActionType) -> int:
        """
        Encode action type to uint8 (C++ compatible).

        ActionType enum values:
        Fold=0, Check=1, Call=2, Bet=3, Raise=4, AllIn=5
        """
        type_map = {
            ActionType.FOLD: 0,
            ActionType.CHECK: 1,
            ActionType.CALL: 2,
            ActionType.BET: 3,
            ActionType.RAISE: 4,
            ActionType.ALL_IN: 5
        }
        return type_map.get(action_type, 0)

    def compute_info_set_key(self, state: HoldemState, player: int) -> int:
        """
        Compute info set key matching C++ implementation.

        Key composition:
        - bits 40+: canonical hole card index
        - bits 16-39: board cards (XOR of shifted indices)
        - bits 0-15: action history (encoded types)
        """
        key = 0

        # Get player's hole cards (hero=0, villain=1)
        if player == 0:
            hand = state.hero_hole
        else:
            hand = state.villain_hole

        card1_idx = self.card_to_index(hand[0])
        card2_idx = self.card_to_index(hand[1])

        # Hole cards contribution
        canonical_idx = self.hole_cards_canonical_index(card1_idx, card2_idx)
        key ^= canonical_idx << 40

        # Board contribution
        board_key = 0
        for i, card in enumerate(state.board):
            if i < 5:
                card_idx = self.card_to_index(card)
                board_key ^= card_idx << (i * 6)
        key ^= board_key << 16

        # Action history contribution - parse from string
        # Python uses string encoding like "r2.0c/xb3.0c"
        # C++ uses action type encoding (Fold=0, Check=1, Call=2, Bet=3, Raise=4, AllIn=5)
        action_key = 0
        action_types = self._parse_action_history(state.action_history)
        for i, action_type in enumerate(action_types[:8]):
            action_key |= action_type << (i * 4)
        key ^= action_key

        return key

    def _parse_action_history(self, history_str: str) -> List[int]:
        """
        Parse Python action history string to list of C++ action type codes.

        Python format: "r2.0c/xb3.0c/b5.0r10.0c"
        - r = raise, c = call, x = check, b = bet, f = fold
        - / separates streets
        - Numbers after r/b are amounts

        C++ encoding: Fold=0, Check=1, Call=2, Bet=3, Raise=4, AllIn=5
        """
        if not history_str:
            return []

        action_types = []
        i = 0
        while i < len(history_str):
            char = history_str[i]

            if char == '/':
                # Street separator, skip
                i += 1
                continue
            elif char == 'f':
                action_types.append(0)  # Fold
                i += 1
            elif char == 'x':
                action_types.append(1)  # Check
                i += 1
            elif char == 'c':
                action_types.append(2)  # Call
                i += 1
            elif char == 'b':
                action_types.append(3)  # Bet
                # Skip the amount
                i += 1
                while i < len(history_str) and (history_str[i].isdigit() or history_str[i] == '.'):
                    i += 1
            elif char == 'r':
                action_types.append(4)  # Raise
                # Skip the amount
                i += 1
                while i < len(history_str) and (history_str[i].isdigit() or history_str[i] == '.'):
                    i += 1
            elif char == 'a':
                action_types.append(5)  # All-in
                # Skip the amount
                i += 1
                while i < len(history_str) and (history_str[i].isdigit() or history_str[i] == '.'):
                    i += 1
            else:
                # Unknown char, skip
                i += 1

        return action_types

    def act(self, state: HoldemState, player: int) -> Action:
        """
        Choose action using C++ CFR strategy.

        Args:
            state: Current game state
            player: Player number (0 or 1)

        Returns:
            Action sampled from strategy
        """
        # Compute info set key
        key = self.compute_info_set_key(state, player)

        # Get legal actions
        legal_actions = state.get_legal_actions()
        num_actions = len(legal_actions)

        if key in self.strategy:
            strategy = self.strategy[key]

            # Handle action count mismatch
            if len(strategy) == num_actions:
                probs = strategy
            elif len(strategy) > num_actions:
                probs = strategy[:num_actions]
            else:
                probs = np.zeros(num_actions)
                probs[:len(strategy)] = strategy

            # Normalize
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = np.ones(num_actions) / num_actions

            # Sample action
            action_idx = self.rng.choice(num_actions, p=probs)
            return legal_actions[action_idx]
        else:
            # Info set not found - use conservative fallback
            return self._conservative_fallback(state, legal_actions)

    def _conservative_fallback(self, state: HoldemState, legal_actions: List[Action]) -> Action:
        """
        Conservative fallback for unknown info sets.
        """
        # Check if we can check (free)
        for action in legal_actions:
            if action.type == ActionType.CHECK:
                return action

        # Facing bet - mix call/fold
        can_call = any(a.type == ActionType.CALL for a in legal_actions)
        can_fold = any(a.type == ActionType.FOLD for a in legal_actions)

        if can_call and can_fold:
            if self.rng.random() < 0.4:
                return next(a for a in legal_actions if a.type == ActionType.CALL)
            else:
                return next(a for a in legal_actions if a.type == ActionType.FOLD)

        if can_fold:
            return next(a for a in legal_actions if a.type == ActionType.FOLD)

        return legal_actions[0]

    def get_strategy_for_key(self, key: int) -> Optional[np.ndarray]:
        """Get strategy for a specific info set key."""
        return self.strategy.get(key)

    @property
    def num_info_sets(self) -> int:
        """Number of info sets in strategy."""
        return len(self.strategy)


def test_cpp_cfr_agent():
    """Test C++ CFR agent loading."""
    import os

    print("Testing C++ CFR Agent...")

    # Check for blueprint
    blueprint_path = "blueprints/cpp_1M/strategy_1M.bin"

    if not os.path.exists(blueprint_path):
        print(f"Blueprint not found at {blueprint_path}")
        print("Run C++ training first")
        return

    # Test loading
    agent = CppCFRAgent(blueprint_path, name="TestCppAgent")
    print(f"\nLoaded {agent.num_info_sets:,} info sets")

    # Test key computation
    print("\nTesting key computation...")

    # Test card_to_index
    assert CppCFRAgent.card_to_index("2c") == 0, "2c should be index 0"
    assert CppCFRAgent.card_to_index("Ac") == 12, "Ac should be index 12"
    assert CppCFRAgent.card_to_index("2d") == 13, "2d should be index 13"
    assert CppCFRAgent.card_to_index("As") == 51, "As should be index 51"
    print("  card_to_index: OK")

    # Test canonical index
    idx_Ac = 12  # Ace of clubs
    idx_Ks = 50  # King of spades
    canonical = CppCFRAgent.hole_cards_canonical_index(idx_Ac, idx_Ks)
    print(f"  AcKs canonical index: {canonical}")

    print("\n Test complete!")


if __name__ == "__main__":
    test_cpp_cfr_agent()
