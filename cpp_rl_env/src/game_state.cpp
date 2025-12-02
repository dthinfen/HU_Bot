#include "game_state.hpp"
#include <algorithm>
#include <sstream>

namespace rl_env {

float GameState::get_payoff(int player) const {
    if (!is_terminal()) return 0.0f;

    // Calculate starting stack from current state
    float starting_stack = (pot + hero_stack + villain_stack) / 2.0f;

    if (folded) {
        // Someone folded
        int winner = 1 - folder;
        float player_invested = starting_stack -
            (player == 0 ? hero_stack : villain_stack);

        if (player == winner) {
            return pot - player_invested;
        } else {
            return -player_invested;
        }
    }

    // Showdown - evaluate hands
    if (board_size >= 3) {
        uint16_t hero_rank = HandEvaluator::evaluate(
            hero_hole[0], hero_hole[1],
            board.data(), board_size
        );
        uint16_t villain_rank = HandEvaluator::evaluate(
            villain_hole[0], villain_hole[1],
            board.data(), board_size
        );

        float player_invested = starting_stack -
            (player == 0 ? hero_stack : villain_stack);

        if (hero_rank < villain_rank) {
            // Hero wins (lower rank is better)
            return player == 0 ? (pot - player_invested) : -player_invested;
        } else if (villain_rank < hero_rank) {
            // Villain wins
            return player == 1 ? (pot - player_invested) : -player_invested;
        } else {
            // Tie - split pot
            return (pot / 2.0f) - player_invested;
        }
    }

    // Preflop all-in - split for now (shouldn't happen often)
    float player_invested = starting_stack -
        (player == 0 ? hero_stack : villain_stack);
    return (pot / 2.0f) - player_invested;
}

bool GameState::should_advance_street() const {
    if (is_terminal()) return false;
    if (street >= 3) return false;

    // Get current street actions
    std::string street_actions;
    size_t last_sep = action_history.rfind('/');
    if (last_sep == std::string::npos) {
        street_actions = action_history;
    } else {
        street_actions = action_history.substr(last_sep + 1);
    }

    // Count actions (excluding digits after amounts)
    int action_count = 0;
    for (char c : street_actions) {
        if (c == 'f' || c == 'x' || c == 'c' || c == 'b' || c == 'r' || c == 'a') {
            action_count++;
        }
    }

    // Need at least 2 actions
    if (action_count < 2) return false;

    // Investments must be equal
    if (std::abs(hero_invested - villain_invested) > 0.001f) return false;

    return true;
}

GameState GameState::apply_action(const Action& action) const {
    GameState new_state = *this;

    // Update action history
    new_state.action_history += action.to_code();
    if (action.amount > 0) {
        std::ostringstream oss;
        oss << action.amount;
        new_state.action_history += oss.str();
    }

    // Apply action effects
    if (current_player == 0) {  // Hero acting
        switch (action.type) {
            case ActionType::FOLD:
                new_state.terminal = true;
                new_state.folded = true;
                new_state.folder = 0;
                break;

            case ActionType::CHECK:
                // No money moves
                break;

            case ActionType::CALL: {
                float call_amount = to_call();
                new_state.hero_stack -= call_amount;
                new_state.hero_invested += call_amount;
                new_state.pot += call_amount;
                break;
            }

            case ActionType::BET:
            case ActionType::RAISE:
            case ActionType::ALL_IN:
                new_state.hero_stack -= action.amount;
                new_state.hero_invested += action.amount;
                new_state.pot += action.amount;
                break;
        }
    } else {  // Villain acting
        switch (action.type) {
            case ActionType::FOLD:
                new_state.terminal = true;
                new_state.folded = true;
                new_state.folder = 1;
                break;

            case ActionType::CHECK:
                break;

            case ActionType::CALL: {
                float call_amount = to_call();
                new_state.villain_stack -= call_amount;
                new_state.villain_invested += call_amount;
                new_state.pot += call_amount;
                break;
            }

            case ActionType::BET:
            case ActionType::RAISE:
            case ActionType::ALL_IN:
                new_state.villain_stack -= action.amount;
                new_state.villain_invested += action.amount;
                new_state.pot += action.amount;
                break;
        }
    }

    // Switch player
    new_state.current_player = 1 - current_player;

    // Check for all-in showdown
    if (!new_state.folded &&
        (new_state.hero_stack <= 0 || new_state.villain_stack <= 0) &&
        std::abs(new_state.hero_invested - new_state.villain_invested) < 0.001f) {
        new_state.terminal = true;
    }

    // Check river showdown
    if (!new_state.folded && new_state.street == 3) {
        if (new_state.should_advance_street()) {
            new_state.terminal = true;
        }
    }

    return new_state;
}

// Hand evaluator implementation (simplified)
bool HandEvaluator::initialized_ = false;
std::array<uint16_t, 7462> HandEvaluator::flush_lookup_;
std::array<uint16_t, 7462> HandEvaluator::unique5_lookup_;

void HandEvaluator::initialize() {
    if (initialized_) return;
    // For now, use simple evaluation
    // TODO: Add proper lookup tables
    initialized_ = true;
}

uint16_t HandEvaluator::evaluate(Card h1, Card h2, const Card* board, int board_size) {
    // Simple evaluation: combine cards and compute hand rank
    // This is a simplified version - production should use proper 7-card eval

    std::array<Card, 7> cards;
    cards[0] = h1;
    cards[1] = h2;
    for (int i = 0; i < board_size && i < 5; i++) {
        cards[2 + i] = board[i];
    }
    int total = 2 + board_size;

    // Count ranks and suits
    std::array<int, 13> rank_count{};
    std::array<int, 4> suit_count{};
    for (int i = 0; i < total; i++) {
        rank_count[card_rank(cards[i])]++;
        suit_count[card_suit(cards[i])]++;
    }

    // Check for flush
    int flush_suit = -1;
    for (int s = 0; s < 4; s++) {
        if (suit_count[s] >= 5) {
            flush_suit = s;
            break;
        }
    }

    // Check for straight
    auto has_straight = [&]() -> int {
        // Returns high card of straight, or -1
        for (int high = 12; high >= 4; high--) {
            bool found = true;
            for (int r = high; r > high - 5; r--) {
                if (rank_count[r] == 0) {
                    found = false;
                    break;
                }
            }
            if (found) return high;
        }
        // Check A-2-3-4-5 (wheel)
        if (rank_count[12] > 0 && rank_count[0] > 0 && rank_count[1] > 0 &&
            rank_count[2] > 0 && rank_count[3] > 0) {
            return 3;  // 5-high straight
        }
        return -1;
    };

    // Count pairs, trips, quads
    int quads = 0, trips = 0, pairs = 0;
    int quad_rank = -1, trip_rank = -1;
    std::array<int, 2> pair_ranks = {-1, -1};

    for (int r = 12; r >= 0; r--) {
        if (rank_count[r] == 4) { quads++; quad_rank = r; }
        else if (rank_count[r] == 3) { trips++; if (trip_rank < 0) trip_rank = r; }
        else if (rank_count[r] == 2) {
            if (pairs < 2) pair_ranks[pairs] = r;
            pairs++;
        }
    }

    // Compute hand rank (lower is better)
    // Hand rankings: SF=1, Quads=2, FH=3, Flush=4, Straight=5, Trips=6, 2P=7, Pair=8, High=9

    int straight_high = has_straight();

    // Straight flush check (simplified)
    if (flush_suit >= 0 && straight_high >= 0) {
        // Check if straight is in flush suit
        std::array<int, 13> flush_ranks{};
        for (int i = 0; i < total; i++) {
            if (card_suit(cards[i]) == flush_suit) {
                flush_ranks[card_rank(cards[i])]++;
            }
        }
        for (int high = 12; high >= 4; high--) {
            bool sf = true;
            for (int r = high; r > high - 5; r--) {
                if (flush_ranks[r] == 0) { sf = false; break; }
            }
            if (sf) return 1 * 1000 + (12 - high);  // Straight flush
        }
    }

    if (quads > 0) {
        return 2 * 1000 + (12 - quad_rank);  // Quads
    }

    if (trips > 0 && pairs > 0) {
        return 3 * 1000 + (12 - trip_rank) * 13 + (12 - pair_ranks[0]);  // Full house
    }

    if (flush_suit >= 0) {
        // Get top 5 cards of flush suit
        int flush_score = 0;
        int count = 0;
        for (int r = 12; r >= 0 && count < 5; r--) {
            for (int i = 0; i < total && count < 5; i++) {
                if (card_rank(cards[i]) == r && card_suit(cards[i]) == flush_suit) {
                    flush_score = flush_score * 13 + (12 - r);
                    count++;
                }
            }
        }
        return 4 * 1000 + flush_score % 1000;  // Flush
    }

    if (straight_high >= 0) {
        return 5 * 1000 + (12 - straight_high);  // Straight
    }

    if (trips > 0) {
        return 6 * 1000 + (12 - trip_rank);  // Trips
    }

    if (pairs >= 2) {
        return 7 * 1000 + (12 - pair_ranks[0]) * 13 + (12 - pair_ranks[1]);  // Two pair
    }

    if (pairs == 1) {
        return 8 * 1000 + (12 - pair_ranks[0]);  // Pair
    }

    // High card
    int high_score = 0;
    int count = 0;
    for (int r = 12; r >= 0 && count < 5; r--) {
        if (rank_count[r] > 0) {
            high_score = high_score * 13 + (12 - r);
            count++;
        }
    }
    return 9 * 1000 + high_score % 1000;  // High card
}

}  // namespace rl_env
