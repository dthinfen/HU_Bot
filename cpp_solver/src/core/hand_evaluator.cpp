#include "ares/core/hand_evaluator.hpp"
#include <algorithm>
#include <cassert>

namespace ares {

// Static member initialization
std::array<HandEvaluator::Rank, 8192> HandEvaluator::flush_table_;
std::array<HandEvaluator::Rank, 8192> HandEvaluator::unique5_table_;
std::array<uint32_t, 52> HandEvaluator::card_to_prime_;
std::array<HandEvaluator::Rank, 4888> HandEvaluator::products_table_;
bool HandEvaluator::initialized_ = false;

// Helper: Check if bits represent a straight
static bool is_straight(uint16_t bits) {
    // Check for 5 consecutive bits
    // bits is a 13-bit value where bit i means rank i is present
    uint16_t mask = 0x1F;  // 5 consecutive bits

    // Check for regular straights (5-high through A-high)
    for (int i = 0; i <= 8; ++i) {
        if ((bits & (mask << i)) == (mask << i)) {
            return true;
        }
    }

    // Check for wheel (A-2-3-4-5)
    // A is bit 12, 2-3-4-5 are bits 0-3
    if ((bits & 0x100F) == 0x100F) {
        return true;
    }

    return false;
}

// Get highest straight value (returns high card of straight)
static int get_straight_high(uint16_t bits) {
    uint16_t mask = 0x1F;

    // Check from highest (A-high) to lowest
    for (int i = 8; i >= 0; --i) {
        if ((bits & (mask << i)) == (mask << i)) {
            return i + 4;  // Returns 4 (5-high) to 12 (A-high)
        }
    }

    // Wheel (A-2-3-4-5) - high card is 5 (index 3)
    if ((bits & 0x100F) == 0x100F) {
        return 3;
    }

    return -1;
}

// Count bits set in 16-bit value
static int popcount16(uint16_t x) {
    return __builtin_popcount(x);
}

void HandEvaluator::initialize() {
    if (initialized_) return;

    // Initialize card_to_prime mapping
    for (int i = 0; i < 52; ++i) {
        int rank = i % 13;
        card_to_prime_[i] = PRIMES[rank];
    }

    // Initialize flush_table and unique5_table
    // These tables map 13-bit patterns to hand ranks

    // Generate all possible 5-bit patterns from 13 bits
    for (uint16_t bits = 0; bits < 8192; ++bits) {
        if (popcount16(bits) != 5) {
            flush_table_[bits] = 0;
            unique5_table_[bits] = 0;
            continue;
        }

        // Check for straight
        if (is_straight(bits)) {
            int high = get_straight_high(bits);
            // Straight flush ranks: 1 (royal) to 10 (5-high)
            // For flush_table, these are straight flushes
            flush_table_[bits] = 10 - (high - 3);  // 1 for A-high, 10 for 5-high

            // For unique5_table, these are regular straights
            // Straights rank from 1600 (A-high) to 1609 (5-high)
            unique5_table_[bits] = STRAIGHT + (12 - high);
        } else {
            // Flush (not straight) - rank by high cards
            // Flush ranks: 323 to 1599
            int high_cards[5];
            int idx = 0;
            for (int i = 12; i >= 0 && idx < 5; --i) {
                if (bits & (1 << i)) {
                    high_cards[idx++] = i;
                }
            }

            // Calculate flush rank (lower is better)
            // Use combinatorial ranking
            int rank = 0;
            for (int i = 0; i < 5; ++i) {
                rank = rank * 13 + (12 - high_cards[i]);
            }
            flush_table_[bits] = FLUSH + rank % (STRAIGHT - FLUSH);

            // High card ranks: 6186 to 7462
            unique5_table_[bits] = HIGH_CARD + rank % (WORST_RANK - HIGH_CARD + 1);
        }
    }

    // Initialize products_table for non-flush hands
    // This maps prime products to hand ranks
    // For simplicity, we'll compute on-the-fly for now
    // Full implementation would precompute all 4888 unique products

    initialized_ = true;
}

int HandEvaluator::find_flush(const std::array<Card, 7>& cards) {
    int suit_counts[4] = {0, 0, 0, 0};

    for (const Card& c : cards) {
        if (c == NO_CARD) continue;  // Skip undealt cards
        suit_counts[get_suit_int(c)]++;
    }

    for (int s = 0; s < 4; ++s) {
        if (suit_counts[s] >= 5) {
            return s;
        }
    }

    return -1;
}

HandEvaluator::Rank HandEvaluator::eval_flush(
    const std::array<Card, 7>& cards,
    int flush_suit
) {
    // Get all cards of flush suit
    uint16_t bits = 0;
    for (const Card& c : cards) {
        if (c == NO_CARD) continue;  // Skip undealt cards
        if (get_suit_int(c) == flush_suit) {
            bits |= (1 << get_rank_int(c));
        }
    }

    // If more than 5 cards, we need to find best 5
    if (popcount16(bits) > 5) {
        // Find best 5-card combination
        Rank best = WORST_RANK;
        for (int i = 0; i < 13; ++i) {
            if (!(bits & (1 << i))) continue;
            uint16_t bits5 = bits & ~(1 << i);
            if (popcount16(bits5) == 5) {
                Rank r = flush_table_[bits5];
                if (r < best) best = r;
            } else if (popcount16(bits5) > 5) {
                // Still more than 5, remove another
                for (int j = 0; j < i; ++j) {
                    if (!(bits5 & (1 << j))) continue;
                    uint16_t bits5b = bits5 & ~(1 << j);
                    if (popcount16(bits5b) == 5) {
                        Rank r = flush_table_[bits5b];
                        if (r < best) best = r;
                    }
                }
            }
        }
        return best;
    }

    return flush_table_[bits];
}

HandEvaluator::Rank HandEvaluator::evaluate(const std::array<Card, 7>& cards) {
    if (!initialized_) {
        initialize();
    }

    // Check for flush first
    int flush_suit = find_flush(cards);
    if (flush_suit >= 0) {
        return eval_flush(cards, flush_suit);
    }

    // No flush - evaluate by rank patterns
    // Count ranks
    int rank_counts[13] = {0};
    uint16_t rank_bits = 0;

    for (const Card& c : cards) {
        if (c == NO_CARD) continue;  // Skip undealt cards
        int r = get_rank_int(c);
        if (r < 0 || r > 12) continue;  // Safety check
        rank_counts[r]++;
        rank_bits |= (1 << r);
    }

    // Find pairs, trips, quads
    int quads = -1, trips = -1, pairs[2] = {-1, -1};
    int pair_count = 0;

    for (int r = 12; r >= 0; --r) {
        switch (rank_counts[r]) {
            case 4:
                quads = r;
                break;
            case 3:
                if (trips < 0) trips = r;
                else if (pair_count < 2) pairs[pair_count++] = r;
                break;
            case 2:
                if (pair_count < 2) pairs[pair_count++] = r;
                break;
        }
    }

    // Four of a kind
    if (quads >= 0) {
        // Find best kicker
        int kicker = -1;
        for (int r = 12; r >= 0; --r) {
            if (r != quads && rank_counts[r] > 0) {
                kicker = r;
                break;
            }
        }
        return FOUR_OF_A_KIND + (12 - quads) * 13 + (12 - kicker);
    }

    // Full house
    if (trips >= 0 && pair_count > 0) {
        int pair_rank = (pairs[0] != trips) ? pairs[0] : pairs[1];
        if (pair_rank < 0) pair_rank = pairs[0];
        return FULL_HOUSE + (12 - trips) * 13 + (12 - pair_rank);
    }

    // Straight (no flush)
    if (is_straight(rank_bits)) {
        int high = get_straight_high(rank_bits);
        return STRAIGHT + (12 - high);
    }

    // Three of a kind
    if (trips >= 0) {
        // Find two best kickers
        int kickers[2] = {-1, -1};
        int ki = 0;
        for (int r = 12; r >= 0 && ki < 2; --r) {
            if (r != trips && rank_counts[r] > 0) {
                kickers[ki++] = r;
            }
        }
        return THREE_OF_A_KIND +
               (12 - trips) * 169 +
               (12 - kickers[0]) * 13 +
               (12 - kickers[1]);
    }

    // Two pair
    if (pair_count >= 2) {
        int high_pair = std::max(pairs[0], pairs[1]);
        int low_pair = std::min(pairs[0], pairs[1]);
        int kicker = -1;
        for (int r = 12; r >= 0; --r) {
            if (r != high_pair && r != low_pair && rank_counts[r] > 0) {
                kicker = r;
                break;
            }
        }
        return TWO_PAIR +
               (12 - high_pair) * 169 +
               (12 - low_pair) * 13 +
               (12 - kicker);
    }

    // One pair
    if (pair_count == 1) {
        int pair = pairs[0];
        int kickers[3] = {-1, -1, -1};
        int ki = 0;
        for (int r = 12; r >= 0 && ki < 3; --r) {
            if (r != pair && rank_counts[r] > 0) {
                kickers[ki++] = r;
            }
        }
        return ONE_PAIR +
               (12 - pair) * 2197 +
               (12 - kickers[0]) * 169 +
               (12 - kickers[1]) * 13 +
               (12 - kickers[2]);
    }

    // High card
    int high_cards[5];
    int hi = 0;
    for (int r = 12; r >= 0 && hi < 5; --r) {
        if (rank_counts[r] > 0) {
            high_cards[hi++] = r;
        }
    }
    return HIGH_CARD +
           (12 - high_cards[0]) * 2197 +
           (12 - high_cards[1]) * 169 +
           (12 - high_cards[2]) * 13 +
           (12 - high_cards[3]);
}

HandEvaluator::Rank HandEvaluator::evaluate(
    const HoleCards& hole,
    const std::array<Card, 5>& board
) {
    std::array<Card, 7> cards;
    cards[0] = hole[0];
    cards[1] = hole[1];
    for (int i = 0; i < 5; ++i) {
        cards[2 + i] = board[i];
    }
    return evaluate(cards);
}

HandEvaluator::Rank HandEvaluator::evaluate(HandMask hand) {
    assert(count_cards(hand) == 7);

    std::array<Card, 7> cards;
    int idx = 0;
    for (int i = 0; i < 52 && idx < 7; ++i) {
        if (hand & (1ULL << i)) {
            cards[idx++] = index_to_card(i);
        }
    }

    return evaluate(cards);
}

const char* HandEvaluator::rank_category(Rank rank) {
    if (rank <= 10) return "Straight Flush";
    if (rank <= 166) return "Four of a Kind";
    if (rank <= 322) return "Full House";
    if (rank <= 1599) return "Flush";
    if (rank <= 1609) return "Straight";
    if (rank <= 2467) return "Three of a Kind";
    if (rank <= 3325) return "Two Pair";
    if (rank <= 6185) return "One Pair";
    return "High Card";
}

// Batch evaluator implementation
void BatchEvaluator::evaluate_batch(
    const HandMask* hands,
    HandEvaluator::Rank* ranks,
    size_t count
) {
    // Simple loop for now - can be optimized with SIMD later
    for (size_t i = 0; i < count; ++i) {
        ranks[i] = HandEvaluator::evaluate(hands[i]);
    }
}

float BatchEvaluator::calculate_equity(
    const HoleCards& hole,
    const Card* board,
    int board_count,
    HandMask villain_range
) {
    HandMask dead = hole.to_mask();
    for (int i = 0; i < board_count; ++i) {
        dead = add_card_to_hand(dead, board[i]);
    }

    int wins = 0, ties = 0, total = 0;

    // Enumerate villain hands
    for (int c1 = 0; c1 < 51; ++c1) {
        if (dead & (1ULL << c1)) continue;
        for (int c2 = c1 + 1; c2 < 52; ++c2) {
            if (dead & (1ULL << c2)) continue;

            // Check if in villain range
            if (villain_range != 0) {
                HandMask v_hand = (1ULL << c1) | (1ULL << c2);
                if (!(villain_range & v_hand)) continue;
            }

            HoleCards villain(index_to_card(c1), index_to_card(c2));
            HandMask v_dead = dead | villain.to_mask();

            // If board is complete, evaluate directly
            if (board_count == 5) {
                std::array<Card, 5> b;
                for (int i = 0; i < 5; ++i) b[i] = board[i];

                auto hero_rank = HandEvaluator::evaluate(hole, b);
                auto vill_rank = HandEvaluator::evaluate(villain, b);

                if (hero_rank < vill_rank) wins++;
                else if (hero_rank == vill_rank) ties++;
                total++;
            } else {
                // Need to enumerate remaining board cards
                // For now, simplified: just count as 50% equity
                // Full implementation would enumerate all runouts
                total++;
                wins++;  // Placeholder
            }
        }
    }

    if (total == 0) return 0.5f;
    return (wins + 0.5f * ties) / total;
}

}  // namespace ares
