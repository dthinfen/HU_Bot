"""
Comprehensive unit tests for cards module.

Tests card representation, deck functionality, and hand evaluation.
"""

import pytest
from src.game.cards import Card, Deck, HandEvaluator, eval_hand


class TestCard:
    """Test Card class"""

    def test_card_creation(self):
        """Test creating cards with rank and suit"""
        card = Card(14, 0)  # Ace of spades
        assert card.rank == 14
        assert card.suit == 0
        assert card.rank_char == 'A'
        assert card.suit_char == 's'

    def test_card_from_string(self):
        """Test parsing cards from strings"""
        # All ranks
        assert Card.from_string('2s').rank == 2
        assert Card.from_string('Ts').rank == 10
        assert Card.from_string('Js').rank == 11
        assert Card.from_string('Qs').rank == 12
        assert Card.from_string('Ks').rank == 13
        assert Card.from_string('As').rank == 14

        # All suits
        assert Card.from_string('As').suit == 0  # spades
        assert Card.from_string('Ah').suit == 1  # hearts
        assert Card.from_string('Ad').suit == 2  # diamonds
        assert Card.from_string('Ac').suit == 3  # clubs

    def test_card_string_representation(self):
        """Test card string output"""
        assert str(Card.from_string('As')) == 'As'
        assert str(Card.from_string('Kh')) == 'Kh'
        assert str(Card.from_string('2d')) == '2d'
        assert str(Card.from_string('Tc')) == 'Tc'

    def test_card_equality(self):
        """Test card equality comparison"""
        as1 = Card.from_string('As')
        as2 = Card.from_string('As')
        kh = Card.from_string('Kh')

        assert as1 == as2
        assert as1 != kh
        assert as1 != "As"  # Not equal to string

    def test_card_hashing(self):
        """Test cards can be used in sets/dicts"""
        cards = {Card.from_string('As'), Card.from_string('Kh'), Card.from_string('As')}
        assert len(cards) == 2  # Duplicate As removed

    def test_card_comparison(self):
        """Test card ordering"""
        deuce = Card.from_string('2s')
        ace = Card.from_string('As')
        king = Card.from_string('Kh')

        assert deuce < ace
        assert deuce < king
        assert king < ace

    def test_invalid_card_rank(self):
        """Test invalid rank raises assertion"""
        with pytest.raises(AssertionError):
            Card(1, 0)  # Rank too low
        with pytest.raises(AssertionError):
            Card(15, 0)  # Rank too high

    def test_invalid_card_suit(self):
        """Test invalid suit raises assertion"""
        with pytest.raises(AssertionError):
            Card(14, -1)  # Suit too low
        with pytest.raises(AssertionError):
            Card(14, 4)  # Suit too high

    def test_invalid_string_format(self):
        """Test invalid string format raises assertion"""
        with pytest.raises(AssertionError):
            Card.from_string('A')  # Too short
        with pytest.raises(AssertionError):
            Card.from_string('Ass')  # Too long
        with pytest.raises(AssertionError):
            Card.from_string('Ax')  # Invalid suit
        with pytest.raises(AssertionError):
            Card.from_string('Xs')  # Invalid rank


class TestDeck:
    """Test Deck class"""

    def test_deck_creation(self):
        """Test creating a full deck"""
        deck = Deck()
        assert len(deck) == 52

    def test_deck_dealing(self):
        """Test dealing cards from deck"""
        deck = Deck(seed=42)
        cards = deck.deal(5)

        assert len(cards) == 5
        assert len(deck) == 47
        assert all(isinstance(c, Card) for c in cards)

    def test_deck_exclusions(self):
        """Test excluding cards from deck"""
        exclude = [Card.from_string('As'), Card.from_string('Kh')]
        deck = Deck(exclude=exclude)

        assert len(deck) == 50  # 52 - 2 excluded

        # Deal all cards and ensure excluded cards not present
        all_cards = deck.deal(50)
        assert Card.from_string('As') not in all_cards
        assert Card.from_string('Kh') not in all_cards

    def test_deck_deterministic_with_seed(self):
        """Test deck is deterministic with same seed"""
        deck1 = Deck(seed=42)
        deck2 = Deck(seed=42)

        cards1 = deck1.deal(5)
        cards2 = deck2.deal(5)

        assert cards1 == cards2

    def test_deck_different_with_different_seed(self):
        """Test deck is different with different seed"""
        deck1 = Deck(seed=42)
        deck2 = Deck(seed=99)

        cards1 = deck1.deal(5)
        cards2 = deck2.deal(5)

        assert cards1 != cards2

    def test_deck_reset(self):
        """Test deck reset functionality"""
        deck = Deck(seed=42)
        cards1 = deck.deal(5)

        deck.reset()

        assert len(deck) == 52
        cards2 = deck.deal(5)
        assert cards1 == cards2  # Same seed, same cards

    def test_deck_overdeal(self):
        """Test dealing more cards than available raises error"""
        deck = Deck()
        deck.deal(52)  # Deal all cards

        with pytest.raises(ValueError):
            deck.deal(1)  # Try to deal from empty deck


class TestHandEvaluator:
    """Test HandEvaluator class"""

    def test_royal_flush(self):
        """Test royal flush evaluation"""
        hero = (Card.from_string('As'), Card.from_string('Ks'))
        board = [Card.from_string(c) for c in ['Qs', 'Js', 'Ts']]

        rank = HandEvaluator.evaluate(hero, board)
        assert rank == 1  # Best possible hand
        assert HandEvaluator.rank_to_string(rank) == "Royal Flush"

    def test_straight_flush(self):
        """Test straight flush evaluation"""
        hero = (Card.from_string('9s'), Card.from_string('8s'))
        board = [Card.from_string(c) for c in ['7s', '6s', '5s']]

        rank = HandEvaluator.evaluate(hero, board)
        assert HandEvaluator.rank_to_string(rank) == "Straight Flush"
        assert rank < 10  # Very strong hand

    def test_four_of_a_kind(self):
        """Test four of a kind evaluation"""
        hero = (Card.from_string('As'), Card.from_string('Ah'))
        board = [Card.from_string(c) for c in ['Ac', 'Ad', '2s']]

        rank = HandEvaluator.evaluate(hero, board)
        assert HandEvaluator.rank_to_string(rank) == "Four of a Kind"

    def test_full_house(self):
        """Test full house evaluation"""
        hero = (Card.from_string('As'), Card.from_string('Ah'))
        board = [Card.from_string(c) for c in ['Ac', 'Ks', 'Kh']]

        rank = HandEvaluator.evaluate(hero, board)
        assert HandEvaluator.rank_to_string(rank) == "Full House"

    def test_flush(self):
        """Test flush evaluation"""
        hero = (Card.from_string('As'), Card.from_string('Ks'))
        board = [Card.from_string(c) for c in ['Qs', '7s', '2s']]

        rank = HandEvaluator.evaluate(hero, board)
        assert HandEvaluator.rank_to_string(rank) == "Flush"

    def test_straight(self):
        """Test straight evaluation"""
        hero = (Card.from_string('As'), Card.from_string('Kh'))
        board = [Card.from_string(c) for c in ['Qd', 'Jc', 'Ts']]

        rank = HandEvaluator.evaluate(hero, board)
        assert HandEvaluator.rank_to_string(rank) == "Straight"

    def test_three_of_a_kind(self):
        """Test three of a kind evaluation"""
        hero = (Card.from_string('As'), Card.from_string('Ah'))
        board = [Card.from_string(c) for c in ['Ac', 'Ks', '2d']]

        rank = HandEvaluator.evaluate(hero, board)
        assert HandEvaluator.rank_to_string(rank) == "Three of a Kind"

    def test_two_pair(self):
        """Test two pair evaluation"""
        hero = (Card.from_string('As'), Card.from_string('Ah'))
        board = [Card.from_string(c) for c in ['Ks', 'Kh', '2d']]

        rank = HandEvaluator.evaluate(hero, board)
        assert HandEvaluator.rank_to_string(rank) == "Two Pair"

    def test_one_pair(self):
        """Test one pair evaluation"""
        hero = (Card.from_string('As'), Card.from_string('Ah'))
        board = [Card.from_string(c) for c in ['Ks', 'Qd', '2d']]

        rank = HandEvaluator.evaluate(hero, board)
        assert HandEvaluator.rank_to_string(rank) == "Pair"

    def test_high_card(self):
        """Test high card evaluation"""
        hero = (Card.from_string('As'), Card.from_string('Kh'))
        board = [Card.from_string(c) for c in ['Qd', 'Jc', '9s']]

        rank = HandEvaluator.evaluate(hero, board)
        assert HandEvaluator.rank_to_string(rank) == "High Card"
        assert rank > 6000  # Weak hand (7462 is worst possible)

    def test_hand_comparison(self):
        """Test comparing hand strengths"""
        # Royal flush vs quad aces
        royal = eval_hand("AsKs", "QsJsTs")
        quads = eval_hand("AhAs", "AcAd2s")

        assert HandEvaluator.compare(royal, quads) == -1  # Royal wins

        # Ace high flush vs king high flush
        ace_flush = eval_hand("AsKs", "QsJsTs")
        king_flush = eval_hand("KhQh", "JhTh9h")

        result = HandEvaluator.compare(ace_flush, king_flush)
        assert result == -1  # Ace high wins

    def test_identical_hands(self):
        """Test identical hands tie"""
        rank1 = eval_hand("AsKh", "QdJcTh")
        rank2 = eval_hand("AsKh", "QdJcTh")

        assert HandEvaluator.compare(rank1, rank2) == 0

    def test_six_card_hand(self):
        """Test evaluating 6-card hand (turn)"""
        hero = (Card.from_string('As'), Card.from_string('Ah'))
        board = [Card.from_string(c) for c in ['Ac', 'Ad', '2s', 'Kh']]

        rank = HandEvaluator.evaluate(hero, board)
        assert HandEvaluator.rank_to_string(rank) == "Four of a Kind"

    def test_seven_card_hand(self):
        """Test evaluating 7-card hand (river)"""
        hero = (Card.from_string('As'), Card.from_string('Ah'))
        board = [Card.from_string(c) for c in ['Ac', 'Ad', '2s', 'Kh', 'Qd']]

        rank = HandEvaluator.evaluate(hero, board)
        assert HandEvaluator.rank_to_string(rank) == "Four of a Kind"

    def test_too_few_cards(self):
        """Test evaluating with too few cards raises error"""
        hero = (Card.from_string('As'), Card.from_string('Ah'))
        board = []  # Preflop

        with pytest.raises(ValueError):
            HandEvaluator.evaluate(hero, board)

    def test_eval_hand_convenience_function(self):
        """Test convenience eval_hand function"""
        rank = eval_hand("AsKs", "QsJsTs")
        assert rank == 1  # Royal flush

        rank = eval_hand("7h2d", "KsQdJc")
        assert HandEvaluator.rank_to_string(rank) == "High Card"

    def test_hand_summary(self):
        """Test hand summary generation"""
        hero = (Card.from_string('As'), Card.from_string('Ah'))
        board = [Card.from_string(c) for c in ['Ac', 'Ad', '2s']]
        rank = HandEvaluator.evaluate(hero, board)

        summary = HandEvaluator.hand_summary(rank, hero, board)
        assert "As Ah" in summary
        assert "Ac Ad 2s" in summary
        assert "Four of a Kind" in summary

    def test_wheel_straight(self):
        """Test A-2-3-4-5 straight (wheel)"""
        hero = (Card.from_string('Ah'), Card.from_string('2h'))
        board = [Card.from_string(c) for c in ['3s', '4d', '5c']]

        rank = HandEvaluator.evaluate(hero, board)
        assert HandEvaluator.rank_to_string(rank) == "Straight"

    def test_better_hand_wins(self):
        """Test stronger hand always has lower rank"""
        hands = [
            ("AsKs", "QsJsTs", "Royal Flush"),
            ("9s8s", "7s6s5s", "Straight Flush"),
            ("AsAh", "AcAd2s", "Four of a Kind"),
            ("AsAh", "AcKsKh", "Full House"),
            ("AsKs", "Qs7s2s", "Flush"),
            ("AsKh", "QdJcTs", "Straight"),
            ("AsAh", "AcKsQd", "Three of a Kind"),
            ("AsAh", "KsKhQd", "Two Pair"),
            ("AsAh", "KsQdJc", "Pair"),
            ("AsKh", "QdJc9s", "High Card"),
        ]

        ranks = []
        for hole, board, expected in hands:
            rank = eval_hand(hole, board)
            assert HandEvaluator.rank_to_string(rank) == expected
            ranks.append(rank)

        # Each rank should be higher (weaker) than the previous
        for i in range(len(ranks) - 1):
            assert ranks[i] < ranks[i + 1], \
                f"{hands[i][2]} should beat {hands[i+1][2]}"


class TestCardPerformance:
    """Performance tests for cards module"""

    def test_hand_evaluation_performance(self):
        """Test hand evaluation is fast enough"""
        import time

        hero = (Card.from_string('As'), Card.from_string('Ah'))
        board = [Card.from_string(c) for c in ['Ac', 'Ad', '2s']]

        # Warm up
        for _ in range(100):
            HandEvaluator.evaluate(hero, board)

        # Time 10000 evaluations
        start = time.perf_counter()
        for _ in range(10000):
            HandEvaluator.evaluate(hero, board)
        elapsed = time.perf_counter() - start

        evals_per_sec = 10000 / elapsed
        print(f"\nHand evaluations/sec: {evals_per_sec:,.0f}")

        # Should be >200k evals/sec (Treys benchmark)
        # Being conservative here since performance varies
        assert evals_per_sec > 100000, \
            f"Too slow: {evals_per_sec:,.0f} evals/sec (target: >100k)"

    def test_card_creation_performance(self):
        """Test card creation is fast"""
        import time

        start = time.perf_counter()
        for _ in range(100000):
            Card.from_string('As')
        elapsed = time.perf_counter() - start

        creations_per_sec = 100000 / elapsed
        print(f"\nCard creations/sec: {creations_per_sec:,.0f}")

        # Should be very fast (>500k/sec)
        assert creations_per_sec > 100000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
