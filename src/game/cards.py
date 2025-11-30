"""
Card representation and hand evaluation for Texas Hold'em.

Uses the Treys library for fast hand evaluation (~235k evals/sec).
Provides clean wrapper API for ARES-HU.

Performance: Treys uses lookup tables with bit arithmetic
Memory: ~5MB for lookup tables (loaded once)
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import random
from treys import Card as TreysCard, Evaluator, Deck as TreysDeck


class Card:
    """
    Immutable card representation.

    Wrapper around Treys.Card for integration with ARES-HU.
    Uses Treys integer representation internally for performance.
    """

    # Rank mappings
    RANK_CHARS = '23456789TJQKA'
    RANK_TO_INT = {c: i + 2 for i, c in enumerate(RANK_CHARS)}
    INT_TO_RANK = {i: c for c, i in RANK_TO_INT.items()}

    # Suit mappings
    SUIT_CHARS = 'shdc'  # spades, hearts, diamonds, clubs
    SUIT_TO_INT = {c: i for i, c in enumerate(SUIT_CHARS)}
    INT_TO_SUIT = {i: c for c, i in SUIT_TO_INT.items()}

    __slots__ = ['_treys_card', '_hash']

    def __init__(self, rank: int, suit: int):
        """
        Create a card.

        Args:
            rank: 2-14 (2=deuce, 14=ace)
            suit: 0-3 (0=spades, 1=hearts, 2=diamonds, 3=clubs)
        """
        assert 2 <= rank <= 14, f"Invalid rank: {rank}"
        assert 0 <= suit <= 3, f"Invalid suit: {suit}"

        # Convert to Treys format
        rank_char = self.INT_TO_RANK[rank]
        suit_char = self.INT_TO_SUIT[suit]
        self._treys_card = TreysCard.new(rank_char + suit_char)
        self._hash = hash((rank, suit))

    @property
    def rank(self) -> int:
        """Get rank as integer (2-14)"""
        # Parse from string representation since Treys internal format is complex
        card_str = TreysCard.int_to_str(self._treys_card)
        rank_char = card_str[0]
        return self.RANK_TO_INT[rank_char]

    @property
    def suit(self) -> int:
        """Get suit as integer (0-3)"""
        # Parse from string representation
        card_str = TreysCard.int_to_str(self._treys_card)
        suit_char = card_str[1].lower()
        return self.SUIT_TO_INT[suit_char]

    @property
    def rank_char(self) -> str:
        """Get rank as character ('2'-'9', 'T', 'J', 'Q', 'K', 'A')"""
        return self.INT_TO_RANK[self.rank]

    @property
    def suit_char(self) -> str:
        """Get suit as character ('s', 'h', 'd', 'c')"""
        return self.INT_TO_SUIT[self.suit]

    @property
    def treys_card(self) -> int:
        """Get underlying Treys card integer for hand evaluation"""
        return self._treys_card

    @staticmethod
    def from_string(s: str) -> 'Card':
        """
        Parse card from string.

        Args:
            s: Card string like 'As', 'Kh', '2d', 'Tc'

        Returns:
            Card instance

        Examples:
            >>> Card.from_string('As')  # Ace of spades
            >>> Card.from_string('Kh')  # King of hearts
            >>> Card.from_string('2d')  # Deuce of diamonds
        """
        assert len(s) == 2, f"Invalid card string: {s}"
        rank_char = s[0].upper()
        suit_char = s[1].lower()

        assert rank_char in Card.RANK_CHARS, f"Invalid rank: {rank_char}"
        assert suit_char in Card.SUIT_CHARS, f"Invalid suit: {suit_char}"

        rank = Card.RANK_TO_INT[rank_char]
        suit = Card.SUIT_TO_INT[suit_char]

        return Card(rank, suit)

    @staticmethod
    def from_treys(treys_card: int) -> 'Card':
        """Create Card from Treys integer representation"""
        card_str = TreysCard.int_to_str(treys_card)
        return Card.from_string(card_str)

    def __str__(self) -> str:
        """String representation like 'As', 'Kh', '2d'"""
        return f"{self.rank_char}{self.suit_char}"

    def __repr__(self) -> str:
        return f"Card('{self}')"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Card):
            return False
        return self._treys_card == other._treys_card

    def __hash__(self) -> int:
        return self._hash

    def __lt__(self, other) -> bool:
        """Compare by rank first, then suit"""
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.suit < other.suit


class Deck:
    """
    52-card deck for dealing cards.

    Supports excluding specific cards (for known hole cards/board).
    """

    def __init__(self, exclude: Optional[List[Card]] = None, seed: Optional[int] = None):
        """
        Create deck.

        Args:
            exclude: Cards to exclude from deck (e.g., known hole cards)
            seed: Random seed for deterministic dealing (for testing)
        """
        self.exclude = exclude or []
        self.seed = seed
        self._reset()

    def _reset(self):
        """Reset deck to full 52 cards minus exclusions"""
        # Create all 52 cards
        all_cards = []
        for rank in range(2, 15):
            for suit in range(4):
                card = Card(rank, suit)
                if card not in self.exclude:
                    all_cards.append(card)

        self.cards = all_cards.copy()

        # Shuffle
        if self.seed is not None:
            random.Random(self.seed).shuffle(self.cards)
        else:
            random.shuffle(self.cards)

    def deal(self, n: int = 1) -> List[Card]:
        """
        Deal n cards from deck.

        Args:
            n: Number of cards to deal

        Returns:
            List of dealt cards

        Raises:
            ValueError: If not enough cards in deck
        """
        if len(self.cards) < n:
            raise ValueError(f"Not enough cards in deck: requested {n}, have {len(self.cards)}")

        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt

    def reset(self):
        """Reset deck to full state"""
        self._reset()

    def __len__(self) -> int:
        return len(self.cards)


class HandEvaluator:
    """
    Fast hand evaluation using Treys library.

    Performance: ~235k evaluations per second (5 cards)

    Hand strength represented as integer (lower = stronger):
    - 1 = Royal Flush (best possible)
    - 7462 = 7-high (worst possible)

    This is Treys' native format. Lower is better.
    """

    _evaluator = None  # Singleton evaluator

    @classmethod
    def _get_evaluator(cls) -> Evaluator:
        """Lazy-load singleton evaluator (loads lookup tables once)"""
        if cls._evaluator is None:
            cls._evaluator = Evaluator()
        return cls._evaluator

    @staticmethod
    def evaluate(hole_cards: Tuple[Card, Card], board: List[Card]) -> int:
        """
        Evaluate 5-7 card hand.

        Args:
            hole_cards: Player's two hole cards
            board: Community cards (0-5 cards)

        Returns:
            Hand rank (1-7462, lower is better)
            1 = Royal Flush, 7462 = 7-high

        Examples:
            >>> hero_hole = (Card.from_string('As'), Card.from_string('Ah'))
            >>> board = [Card.from_string(c) for c in ['Ac', 'Ad', '2s']]
            >>> rank = HandEvaluator.evaluate(hero_hole, board)
            >>> HandEvaluator.rank_to_string(rank)
            'Four of a Kind'
        """
        assert len(hole_cards) == 2, "Must have exactly 2 hole cards"
        assert 0 <= len(board) <= 5, "Board must have 0-5 cards"

        # Convert to Treys format
        treys_hole = [c.treys_card for c in hole_cards]
        treys_board = [c.treys_card for c in board]

        # Evaluate
        evaluator = HandEvaluator._get_evaluator()

        # Treys requires exactly 5 cards total for evaluation
        # For <5 cards (e.g., preflop), we can't evaluate yet
        if len(treys_hole) + len(treys_board) < 5:
            raise ValueError(f"Need at least 5 cards to evaluate, got {len(treys_hole) + len(treys_board)}")

        return evaluator.evaluate(treys_board, treys_hole)

    @staticmethod
    def rank_to_string(rank: int) -> str:
        """
        Convert hand rank to readable string.

        Args:
            rank: Hand rank (1-7462)

        Returns:
            Hand class like "Royal Flush", "Straight", "Two Pair"
        """
        evaluator = HandEvaluator._get_evaluator()
        return evaluator.class_to_string(evaluator.get_rank_class(rank))

    @staticmethod
    def compare(rank1: int, rank2: int) -> int:
        """
        Compare two hand ranks.

        Args:
            rank1: First hand rank
            rank2: Second hand rank

        Returns:
            -1 if rank1 wins, 1 if rank2 wins, 0 if tie
        """
        if rank1 < rank2:
            return -1
        elif rank1 > rank2:
            return 1
        else:
            return 0

    @staticmethod
    def hand_summary(rank: int, hole_cards: Tuple[Card, Card], board: List[Card]) -> str:
        """
        Get human-readable hand summary.

        Returns:
            String like "As Ah on Ac Ad 2s 3h 4d: Four of a Kind (rank: 11)"
        """
        hole_str = ' '.join(str(c) for c in hole_cards)
        board_str = ' '.join(str(c) for c in board)
        rank_str = HandEvaluator.rank_to_string(rank)
        return f"{hole_str} on {board_str}: {rank_str} (rank: {rank})"


# Convenience function for quick hand evaluation from strings
def eval_hand(hole: str, board: str = "") -> int:
    """
    Quick hand evaluation from strings.

    Args:
        hole: Hole cards like "AsAh"
        board: Board cards like "AcAd2s"

    Returns:
        Hand rank (1-7462, lower is better)

    Examples:
        >>> eval_hand("AsAh", "AcAd2s")  # Quad aces
        >>> eval_hand("7h2d", "KsQsJsTh9h")  # King high straight
    """
    assert len(hole) == 4, "Hole must be exactly 2 cards (4 chars)"

    hole_cards = (
        Card.from_string(hole[0:2]),
        Card.from_string(hole[2:4])
    )

    board_cards = []
    for i in range(0, len(board), 2):
        board_cards.append(Card.from_string(board[i:i+2]))

    return HandEvaluator.evaluate(hole_cards, board_cards)
