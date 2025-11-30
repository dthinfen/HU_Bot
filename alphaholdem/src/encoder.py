"""
AlphaHoldem State Encoder

Encodes poker game state as 3D tensor for neural network input.
Based on AlphaHoldem paper's state representation.

Key insight: Represent game state as "image-like" tensor so CNNs can
learn spatial patterns in cards and betting sequences.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class Card:
    """Simple card representation for encoding."""
    rank: int  # 0-12 (2=0, A=12)
    suit: int  # 0-3 (s, h, d, c)

    @staticmethod
    def from_string(s: str) -> 'Card':
        """Parse 'As', 'Kh', etc."""
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5,
                    '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        suit_map = {'s': 0, 'h': 1, 'd': 2, 'c': 3}
        return Card(rank_map[s[0].upper()], suit_map[s[1].lower()])

    def to_index(self) -> int:
        """Convert to 0-51 index."""
        return self.rank * 4 + self.suit


class AlphaHoldemEncoder:
    """
    Encode poker state as 3D tensor.

    Tensor structure (C x H x W):
    - C = channels (different feature types)
    - H = 4 (suits)
    - W = 13 (ranks)

    Channels:
    [0-1]   : Hole cards (2 channels, one per card)
    [2-6]   : Board cards (5 channels, one per card position)
    [7]     : All known cards combined
    [8-11]  : Suit counts (how many of each suit visible)
    [12-15] : Rank counts (pairs, trips, etc.)
    [16-19] : Street indicators (one-hot for preflop/flop/turn/river)
    [20-23] : Position indicators
    [24-27] : Pot/stack ratios
    [28-37] : Betting history (last 10 actions encoded)
    """

    NUM_CHANNELS = 38
    HEIGHT = 4   # Suits
    WIDTH = 13   # Ranks

    def __init__(self):
        self.observation_shape = (self.NUM_CHANNELS, self.HEIGHT, self.WIDTH)

    def encode(
        self,
        hole_cards: List[Tuple[int, int]],  # List of (rank, suit)
        board_cards: List[Tuple[int, int]],
        pot: float,
        hero_stack: float,
        villain_stack: float,
        hero_invested: float,
        villain_invested: float,
        street: int,  # 0-3
        is_button: bool,
        action_history: List[Tuple[int, float]],  # List of (action_type, amount)
        starting_stack: float = 100.0
    ) -> np.ndarray:
        """
        Encode game state as 3D tensor.

        Returns:
            np.ndarray of shape (NUM_CHANNELS, 4, 13)
        """
        tensor = np.zeros((self.NUM_CHANNELS, self.HEIGHT, self.WIDTH), dtype=np.float32)

        # Encode hole cards (channels 0-1)
        for i, (rank, suit) in enumerate(hole_cards[:2]):
            tensor[i, suit, rank] = 1.0

        # Encode board cards (channels 2-6)
        for i, (rank, suit) in enumerate(board_cards[:5]):
            tensor[2 + i, suit, rank] = 1.0

        # Combined known cards (channel 7)
        for rank, suit in hole_cards + board_cards:
            tensor[7, suit, rank] = 1.0

        # Suit counts (channels 8-11)
        all_cards = hole_cards + board_cards
        suit_counts = [0, 0, 0, 0]
        for rank, suit in all_cards:
            suit_counts[suit] += 1
        for s in range(4):
            # Normalize by max possible (7 cards)
            tensor[8 + s, :, :] = suit_counts[s] / 7.0

        # Rank counts (channels 12-15)
        rank_counts = [0] * 13
        for rank, suit in all_cards:
            rank_counts[rank] += 1
        for r in range(13):
            # Encode count as intensity
            count = rank_counts[r]
            if count >= 1:
                tensor[12, :, r] = 1.0  # Has at least one
            if count >= 2:
                tensor[13, :, r] = 1.0  # Has pair
            if count >= 3:
                tensor[14, :, r] = 1.0  # Has trips
            if count >= 4:
                tensor[15, :, r] = 1.0  # Has quads

        # Street indicators (channels 16-19)
        tensor[16 + street, :, :] = 1.0

        # Position (channels 20-23)
        if is_button:
            tensor[20, :, :] = 1.0  # Hero is button
        else:
            tensor[21, :, :] = 1.0  # Hero is big blind

        # Pot/stack info (channels 24-27)
        total_chips = pot + hero_stack + villain_stack
        if total_chips > 0:
            tensor[24, :, :] = pot / total_chips
            tensor[25, :, :] = hero_stack / starting_stack
            tensor[26, :, :] = villain_stack / starting_stack
            tensor[27, :, :] = (hero_invested + villain_invested) / (2 * starting_stack)

        # Betting history (channels 28-37)
        # Encode last 10 actions
        for i, (action_type, amount) in enumerate(action_history[-10:]):
            channel = 28 + i
            # Action type encoding (spread across width)
            # 0=fold, 1=check, 2=call, 3=bet, 4=raise, 5=all-in
            if action_type < self.WIDTH:
                tensor[channel, 0, action_type] = 1.0
            # Amount encoding (normalized, spread across height)
            if starting_stack > 0:
                norm_amount = min(amount / starting_stack, 1.0)
                tensor[channel, 1, :] = norm_amount
            # Hero/villain indicator
            is_hero = (i % 2 == 0) if len(action_history) % 2 == 0 else (i % 2 == 1)
            tensor[channel, 2, :] = 1.0 if is_hero else 0.0

        return tensor

    def encode_from_state(self, state, hero_player: int) -> np.ndarray:
        """
        Encode from HoldemState object.

        Args:
            state: HoldemState instance
            hero_player: Which player we're encoding for (0 or 1)
        """
        # Extract cards
        if hero_player == 0:
            hole_cards = [(c.rank - 2, c.suit) for c in state.hero_hole]
            is_button = (state.button == 0)
        else:
            hole_cards = [(c.rank - 2, c.suit) for c in state.villain_hole]
            is_button = (state.button == 1)

        board_cards = [(c.rank - 2, c.suit) for c in state.board]

        # Parse action history
        action_history = self._parse_action_history(state.action_history)

        return self.encode(
            hole_cards=hole_cards,
            board_cards=board_cards,
            pot=state.pot,
            hero_stack=state.hero_stack if hero_player == 0 else state.villain_stack,
            villain_stack=state.villain_stack if hero_player == 0 else state.hero_stack,
            hero_invested=state.hero_invested_this_street if hero_player == 0 else state.villain_invested_this_street,
            villain_invested=state.villain_invested_this_street if hero_player == 0 else state.hero_invested_this_street,
            street=state.street,
            is_button=is_button,
            action_history=action_history
        )

    def _parse_action_history(self, history: str) -> List[Tuple[int, float]]:
        """Parse action history string to list of (action_type, amount)."""
        actions = []
        i = 0
        while i < len(history):
            c = history[i]
            if c == '/':
                i += 1
                continue
            elif c == 'f':
                actions.append((0, 0.0))  # fold
                i += 1
            elif c == 'x':
                actions.append((1, 0.0))  # check
                i += 1
            elif c == 'c':
                actions.append((2, 0.0))  # call
                i += 1
            elif c == 'b':
                # Parse bet amount
                j = i + 1
                while j < len(history) and (history[j].isdigit() or history[j] == '.'):
                    j += 1
                amount = float(history[i+1:j]) if j > i + 1 else 0.0
                actions.append((3, amount))  # bet
                i = j
            elif c == 'r':
                # Parse raise amount
                j = i + 1
                while j < len(history) and (history[j].isdigit() or history[j] == '.'):
                    j += 1
                amount = float(history[i+1:j]) if j > i + 1 else 0.0
                actions.append((4, amount))  # raise
                i = j
            elif c == 'a':
                # Parse all-in amount
                j = i + 1
                while j < len(history) and (history[j].isdigit() or history[j] == '.'):
                    j += 1
                amount = float(history[i+1:j]) if j > i + 1 else 0.0
                actions.append((5, amount))  # all-in
                i = j
            else:
                i += 1
        return actions


class SimpleEncoder:
    """
    Simpler flat vector encoding (alternative to 3D tensor).

    Good for MLP-based networks or faster training.
    """

    # Feature dimensions
    HOLE_CARDS_DIM = 52 * 2  # One-hot for each card
    BOARD_DIM = 52 * 5
    POT_STACK_DIM = 6  # pot, stacks, invested amounts
    STREET_DIM = 4  # One-hot
    POSITION_DIM = 2
    ACTION_HISTORY_DIM = 50  # Embedded action sequence

    TOTAL_DIM = HOLE_CARDS_DIM + BOARD_DIM + POT_STACK_DIM + STREET_DIM + POSITION_DIM + ACTION_HISTORY_DIM

    def __init__(self):
        self.observation_shape = (self.TOTAL_DIM,)

    def encode(
        self,
        hole_cards: List[Tuple[int, int]],
        board_cards: List[Tuple[int, int]],
        pot: float,
        hero_stack: float,
        villain_stack: float,
        hero_invested: float,
        villain_invested: float,
        street: int,
        is_button: bool,
        action_history: List[Tuple[int, float]],
        starting_stack: float = 100.0
    ) -> np.ndarray:
        """Encode game state as flat vector."""
        vec = np.zeros(self.TOTAL_DIM, dtype=np.float32)
        idx = 0

        # Hole cards (one-hot)
        for i, (rank, suit) in enumerate(hole_cards[:2]):
            card_idx = rank * 4 + suit
            vec[idx + i * 52 + card_idx] = 1.0
        idx += self.HOLE_CARDS_DIM

        # Board cards (one-hot)
        for i, (rank, suit) in enumerate(board_cards[:5]):
            card_idx = rank * 4 + suit
            vec[idx + i * 52 + card_idx] = 1.0
        idx += self.BOARD_DIM

        # Pot/stack info (normalized)
        vec[idx] = pot / (2 * starting_stack)
        vec[idx + 1] = hero_stack / starting_stack
        vec[idx + 2] = villain_stack / starting_stack
        vec[idx + 3] = hero_invested / starting_stack
        vec[idx + 4] = villain_invested / starting_stack
        vec[idx + 5] = (hero_stack + villain_stack) / (2 * starting_stack)  # Effective stack ratio
        idx += self.POT_STACK_DIM

        # Street (one-hot)
        vec[idx + street] = 1.0
        idx += self.STREET_DIM

        # Position
        vec[idx] = 1.0 if is_button else 0.0
        vec[idx + 1] = 0.0 if is_button else 1.0
        idx += self.POSITION_DIM

        # Action history (simplified embedding)
        for i, (action_type, amount) in enumerate(action_history[-10:]):
            base = idx + i * 5
            if base + 4 < len(vec):
                vec[base + min(action_type, 4)] = 1.0  # Action type
                # Amount (normalized)
                if starting_stack > 0:
                    vec[base + 4] = min(amount / starting_stack, 1.0)

        return vec


if __name__ == "__main__":
    # Test encoding
    encoder = AlphaHoldemEncoder()

    # Simulate a game state
    hole = [(12, 0), (12, 1)]  # AA
    board = [(10, 0), (9, 1), (2, 2)]  # QJs2
    actions = [(3, 3.0), (4, 9.0), (2, 0.0)]  # bet 3, raise 9, call

    tensor = encoder.encode(
        hole_cards=hole,
        board_cards=board,
        pot=21.0,
        hero_stack=91.0,
        villain_stack=88.0,
        hero_invested=9.0,
        villain_invested=9.0,
        street=1,
        is_button=True,
        action_history=actions
    )

    print(f"Tensor shape: {tensor.shape}")
    print(f"Non-zero elements: {np.count_nonzero(tensor)}")
    print(f"Observation shape: {encoder.observation_shape}")

    # Test simple encoder
    simple = SimpleEncoder()
    vec = simple.encode(
        hole_cards=hole,
        board_cards=board,
        pot=21.0,
        hero_stack=91.0,
        villain_stack=88.0,
        hero_invested=9.0,
        villain_invested=9.0,
        street=1,
        is_button=True,
        action_history=actions
    )
    print(f"\nSimple encoder shape: {vec.shape}")
    print(f"Non-zero elements: {np.count_nonzero(vec)}")
