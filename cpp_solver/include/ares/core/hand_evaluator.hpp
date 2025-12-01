#pragma once

#include "ares/core/cards.hpp"
#include <array>
#include <cstdint>

namespace ares {

/**
 * Fast 7-card hand evaluator using lookup tables.
 *
 * Based on the Two-Plus-Two evaluator algorithm with optimizations.
 * Hand ranks are returned as 16-bit values where lower = better.
 *
 * Hand categories (high nibble of rank):
 *   1 = High Card
 *   2 = Pair
 *   3 = Two Pair
 *   4 = Three of a Kind
 *   5 = Straight
 *   6 = Flush
 *   7 = Full House
 *   8 = Four of a Kind
 *   9 = Straight Flush
 *
 * The low 12 bits encode the specific hand within each category.
 * Total unique ranks: 7462 (from worst to best)
 */
class HandEvaluator {
public:
    // Hand rank type (lower is better, 1 = royal flush, 7462 = worst high card)
    using Rank = uint16_t;

    // Hand category constants
    static constexpr Rank STRAIGHT_FLUSH = 1;
    static constexpr Rank FOUR_OF_A_KIND = 11;
    static constexpr Rank FULL_HOUSE = 167;
    static constexpr Rank FLUSH = 323;
    static constexpr Rank STRAIGHT = 1600;
    static constexpr Rank THREE_OF_A_KIND = 1610;
    static constexpr Rank TWO_PAIR = 2468;
    static constexpr Rank ONE_PAIR = 3326;
    static constexpr Rank HIGH_CARD = 6186;
    static constexpr Rank WORST_RANK = 7462;

    /**
     * Initialize lookup tables. Must be called once before evaluation.
     */
    static void initialize();

    /**
     * Check if evaluator is initialized.
     */
    static bool is_initialized() { return initialized_; }

    /**
     * Evaluate a 7-card hand.
     *
     * @param cards Array of 7 cards
     * @return Hand rank (lower is better)
     */
    static Rank evaluate(const std::array<Card, 7>& cards);

    /**
     * Evaluate a 7-card hand from hole cards + board.
     *
     * @param hole Two hole cards
     * @param board Five board cards
     * @return Hand rank (lower is better)
     */
    static Rank evaluate(const HoleCards& hole, const std::array<Card, 5>& board);

    /**
     * Evaluate from card bitmask (must have exactly 7 bits set).
     *
     * @param hand 64-bit hand mask with exactly 7 cards
     * @return Hand rank (lower is better)
     */
    static Rank evaluate(HandMask hand);

    /**
     * Get category name for a rank.
     */
    static const char* rank_category(Rank rank);

    /**
     * Compare two hands. Returns:
     *   <0 if hand1 wins
     *   >0 if hand2 wins
     *   0 if tie
     */
    static int compare(Rank rank1, Rank rank2) {
        return static_cast<int>(rank1) - static_cast<int>(rank2);
    }

private:
    // Lookup tables
    static std::array<Rank, 8192> flush_table_;      // For flush hands
    static std::array<Rank, 8192> unique5_table_;    // For straights/high cards
    static std::array<uint32_t, 52> card_to_prime_;  // Card -> prime number
    static std::array<Rank, 4888> products_table_;   // Prime products -> rank

    static bool initialized_;

    // Prime numbers for each rank (2-A)
    static constexpr std::array<uint32_t, 13> PRIMES = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41
    };

    // Helper functions
    static uint32_t prime_product(const std::array<Card, 7>& cards);
    static uint16_t card_bit(Card c);
    static Rank eval_5_cards(uint16_t q);
    static Rank eval_flush(const std::array<Card, 7>& cards, int flush_suit);
    static int find_flush(const std::array<Card, 7>& cards);
};

/**
 * Batch hand evaluator for SIMD optimization.
 * Evaluates multiple hands in parallel.
 */
class BatchEvaluator {
public:
    /**
     * Evaluate multiple hands.
     *
     * @param hands Array of hand masks (7 cards each)
     * @param ranks Output array for ranks
     * @param count Number of hands to evaluate
     */
    static void evaluate_batch(
        const HandMask* hands,
        HandEvaluator::Rank* ranks,
        size_t count
    );

    /**
     * Calculate equity of hole cards vs range on a board.
     *
     * @param hole Hero's hole cards
     * @param board Current board (0-5 cards)
     * @param board_count Number of board cards
     * @param villain_range Bitmask of villain's possible holdings
     * @return Equity (0.0 to 1.0)
     */
    static float calculate_equity(
        const HoleCards& hole,
        const Card* board,
        int board_count,
        HandMask villain_range = 0
    );
};

}  // namespace ares
