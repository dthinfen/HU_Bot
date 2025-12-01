#include "ares/core/hand_evaluator.hpp"
#include "ares/core/cards.hpp"
#include <iostream>
#include <cassert>
#include <chrono>

using namespace ares;

void test_card_basics() {
    std::cout << "Testing card basics..." << std::endl;

    // Test card creation
    Card as = make_card(Rank::Ace, Suit::Spades);
    Card kh = make_card(Rank::King, Suit::Hearts);

    assert(get_rank(as) == Rank::Ace);
    assert(get_suit(as) == Suit::Spades);
    assert(card_to_string(as) == "As");
    assert(card_to_string(kh) == "Kh");

    // Test string parsing
    assert(string_to_card("As") == as);
    assert(string_to_card("Kh") == kh);
    assert(string_to_card("2c") == make_card(Rank::Two, Suit::Clubs));

    std::cout << "  Card basics: PASSED" << std::endl;
}

void test_hand_evaluation() {
    std::cout << "Testing hand evaluation..." << std::endl;

    HandEvaluator::initialize();

    // Royal flush: As Ks Qs Js Ts + 2c 3c
    std::array<Card, 7> royal_flush = {
        string_to_card("As"),
        string_to_card("Ks"),
        string_to_card("Qs"),
        string_to_card("Js"),
        string_to_card("Ts"),
        string_to_card("2c"),
        string_to_card("3c")
    };
    auto rf_rank = HandEvaluator::evaluate(royal_flush);
    std::cout << "  Royal flush rank: " << rf_rank
              << " (" << HandEvaluator::rank_category(rf_rank) << ")" << std::endl;
    assert(rf_rank == 1);  // Best possible hand

    // Four of a kind: As Ah Ad Ac + Ks Kh Kd
    std::array<Card, 7> quads = {
        string_to_card("As"),
        string_to_card("Ah"),
        string_to_card("Ad"),
        string_to_card("Ac"),
        string_to_card("Ks"),
        string_to_card("Kh"),
        string_to_card("2d")
    };
    auto quads_rank = HandEvaluator::evaluate(quads);
    std::cout << "  Quad aces rank: " << quads_rank
              << " (" << HandEvaluator::rank_category(quads_rank) << ")" << std::endl;
    assert(quads_rank >= HandEvaluator::FOUR_OF_A_KIND);
    assert(quads_rank < HandEvaluator::FULL_HOUSE);

    // Full house: As Ah Ad Ks Kh + 2c 3c
    std::array<Card, 7> full_house = {
        string_to_card("As"),
        string_to_card("Ah"),
        string_to_card("Ad"),
        string_to_card("Ks"),
        string_to_card("Kh"),
        string_to_card("2c"),
        string_to_card("3c")
    };
    auto fh_rank = HandEvaluator::evaluate(full_house);
    std::cout << "  Full house rank: " << fh_rank
              << " (" << HandEvaluator::rank_category(fh_rank) << ")" << std::endl;
    assert(fh_rank >= HandEvaluator::FULL_HOUSE);
    assert(fh_rank < HandEvaluator::FLUSH);

    // High card: As Kh Qd Jc 9s 7h 5d (no made hand)
    std::array<Card, 7> high_card = {
        string_to_card("As"),
        string_to_card("Kh"),
        string_to_card("Qd"),
        string_to_card("Jc"),
        string_to_card("9s"),
        string_to_card("7h"),
        string_to_card("5d")
    };
    auto hc_rank = HandEvaluator::evaluate(high_card);
    std::cout << "  High card rank: " << hc_rank
              << " (" << HandEvaluator::rank_category(hc_rank) << ")" << std::endl;
    assert(hc_rank >= HandEvaluator::HIGH_CARD);

    // Verify ordering: royal flush beats everything
    assert(rf_rank < quads_rank);
    assert(quads_rank < fh_rank);
    assert(fh_rank < hc_rank);

    std::cout << "  Hand evaluation: PASSED" << std::endl;
}

void test_hand_comparison() {
    std::cout << "Testing hand comparison..." << std::endl;

    HandEvaluator::initialize();

    // AA vs KK on blank board
    HoleCards aa(string_to_card("As"), string_to_card("Ah"));
    HoleCards kk(string_to_card("Ks"), string_to_card("Kh"));
    std::array<Card, 5> board = {
        string_to_card("2c"),
        string_to_card("5d"),
        string_to_card("8s"),
        string_to_card("Tc"),
        string_to_card("3h")
    };

    auto aa_rank = HandEvaluator::evaluate(aa, board);
    auto kk_rank = HandEvaluator::evaluate(kk, board);

    std::cout << "  AA rank: " << aa_rank << std::endl;
    std::cout << "  KK rank: " << kk_rank << std::endl;
    assert(aa_rank < kk_rank);  // AA wins

    std::cout << "  Hand comparison: PASSED" << std::endl;
}

void benchmark_evaluation() {
    std::cout << "Benchmarking evaluation speed..." << std::endl;

    HandEvaluator::initialize();

    const int NUM_HANDS = 1000000;
    Deck deck;

    auto start = std::chrono::high_resolution_clock::now();

    uint64_t checksum = 0;
    for (int i = 0; i < NUM_HANDS; ++i) {
        deck.reset();
        std::array<Card, 7> cards;
        for (int j = 0; j < 7; ++j) {
            cards[j] = index_to_card((i * 7 + j) % 52);
        }
        auto rank = HandEvaluator::evaluate(cards);
        checksum += rank;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    double hands_per_sec = (double)NUM_HANDS / (duration.count() / 1000.0);
    std::cout << "  Evaluated " << NUM_HANDS << " hands in "
              << duration.count() << "ms" << std::endl;
    std::cout << "  Speed: " << (hands_per_sec / 1e6) << "M hands/sec" << std::endl;
    std::cout << "  Checksum: " << checksum << " (for verification)" << std::endl;
}

int main() {
    std::cout << "=== Hand Evaluator Tests ===" << std::endl;

    test_card_basics();
    test_hand_evaluation();
    test_hand_comparison();
    benchmark_evaluation();

    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
}
