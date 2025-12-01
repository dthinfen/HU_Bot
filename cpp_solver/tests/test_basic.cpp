#include "ares/core/cards.hpp"
#include "ares/core/hand_evaluator.hpp"
#include "ares/game/holdem_state.hpp"
#include "ares/belief/public_belief_state.hpp"
#include "ares/cfr/dcfr_solver.hpp"
#include <iostream>
#include <cassert>

using namespace ares;

void test_cards() {
    std::cout << "Testing cards... ";

    Card c1 = make_card(Rank::Ace, Suit::Spades);
    assert(get_rank(c1) == Rank::Ace);
    assert(get_suit(c1) == Suit::Spades);

    Card c2 = make_card(Rank::King, Suit::Hearts);
    assert(get_rank(c2) == Rank::King);
    assert(get_suit(c2) == Suit::Hearts);

    // Test string conversion
    assert(card_to_string(c1) == "As");
    assert(card_to_string(c2) == "Kh");

    std::cout << "PASSED\n";
}

void test_hand_evaluator() {
    std::cout << "Testing hand evaluator... ";

    // Initialize lookup tables first
    HandEvaluator::initialize();

    // Royal flush (spades)
    std::array<Card, 7> royal = {
        make_card(Rank::Ace, Suit::Spades),
        make_card(Rank::King, Suit::Spades),
        make_card(Rank::Queen, Suit::Spades),
        make_card(Rank::Jack, Suit::Spades),
        make_card(Rank::Ten, Suit::Spades),
        make_card(Rank::Two, Suit::Hearts),
        make_card(Rank::Three, Suit::Clubs)
    };

    // High card
    std::array<Card, 7> high_card = {
        make_card(Rank::Ace, Suit::Spades),
        make_card(Rank::King, Suit::Hearts),
        make_card(Rank::Queen, Suit::Diamonds),
        make_card(Rank::Jack, Suit::Clubs),
        make_card(Rank::Nine, Suit::Spades),
        make_card(Rank::Two, Suit::Hearts),
        make_card(Rank::Three, Suit::Clubs)
    };

    auto royal_rank = HandEvaluator::evaluate(royal);
    auto high_rank = HandEvaluator::evaluate(high_card);

    // Note: Lower rank = better hand (1 = royal flush, 7462 = worst)
    // Royal flush should be better (lower rank) than high card
    assert(royal_rank < high_rank);  // Lower is better!
    assert(royal_rank <= HandEvaluator::STRAIGHT_FLUSH + 10);  // Should be straight flush category

    std::cout << "PASSED (royal=" << royal_rank << ", high_card=" << high_rank << ")\n";
}

void test_game_state() {
    std::cout << "Testing game state... ";

    // Create state with dealt hole cards
    HoleCards hand0(make_card(Rank::Ace, Suit::Spades), make_card(Rank::King, Suit::Spades));
    HoleCards hand1(make_card(Rank::Queen, Suit::Hearts), make_card(Rank::Jack, Suit::Hearts));

    auto state = HoldemState::create_with_hands(hand0, hand1);
    assert(!state.is_terminal());
    assert(!state.is_chance_node());  // Should not need dealing now
    assert(state.pot() > 0);  // Blinds posted

    auto actions = state.get_legal_actions();
    assert(!actions.empty());

    std::cout << "PASSED (pot=" << state.pot() << ", legal actions: " << actions.size() << ")\n";
}

void test_pbs() {
    std::cout << "Testing PBS... ";

    auto pbs = PublicBeliefState::create_initial(100.0f);
    assert(!pbs.is_terminal());
    assert(pbs.pot() > 0);

    // Test encoding
    auto encoding = pbs.encode();
    assert(encoding.size() > 0);
    std::cout << "PASSED (encoding dim: " << encoding.size() << ")\n";
}

void test_hand_distribution() {
    std::cout << "Testing hand distribution... ";

    HandDistribution dist;
    dist.set_uniform();

    // Should sum to ~1
    float sum = 0.0f;
    for (int i = 0; i < NUM_HOLE_COMBOS; ++i) {
        sum += dist.get(i);
    }
    assert(std::abs(sum - 1.0f) < 0.001f);

    // Test entropy
    float entropy = dist.entropy();
    assert(entropy > 0);  // Uniform distribution has positive entropy

    std::cout << "PASSED (entropy: " << entropy << ")\n";
}

void test_dcfr_config() {
    std::cout << "Testing DCFR config... ";

    auto nash_cfg = DCFRConfig::nash_preset();
    assert(nash_cfg.equilibrium == EquilibriumType::NASH);

    auto qre_cfg = DCFRConfig::qre_preset(1.5f);
    assert(qre_cfg.equilibrium == EquilibriumType::QRE);
    assert(qre_cfg.qre_tau == 1.5f);

    std::cout << "PASSED\n";
}

int main() {
    std::cout << "\n=== ARES-HU C++ Solver Tests ===\n\n";

    test_cards();
    test_hand_evaluator();
    test_game_state();
    test_hand_distribution();
    test_pbs();
    test_dcfr_config();

    std::cout << "\n=== All tests passed! ===\n\n";
    return 0;
}
