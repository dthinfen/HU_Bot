#pragma once

#include "ares/core/cards.hpp"
#include "ares/game/action.hpp"
#include <array>
#include <vector>
#include <optional>

namespace ares {

// Number of players (heads-up)
constexpr int NUM_PLAYERS = 2;

// Position constants
constexpr int BUTTON = 0;  // Small blind preflop, acts last postflop
constexpr int BIG_BLIND = 1;

// Game configuration (defined before HoldemState to allow default arguments)
struct HoldemConfig {
    float starting_stack = 20.0f;  // In big blinds
    float small_blind = 0.5f;
    float big_blind = 1.0f;

    static HoldemConfig default_config() { return HoldemConfig{}; }
};

// Game state representation
class HoldemState {
public:
    using Config = HoldemConfig;

    // Create initial state
    static HoldemState create_initial(const Config& config = Config::default_config());

    // Create state with specific hands (for testing/analysis)
    static HoldemState create_with_hands(
        const HoleCards& hand0,
        const HoleCards& hand1,
        const Config& config = Config::default_config()
    );

    // Create state from arbitrary game position (for real-time search)
    static HoldemState create_from_position(
        const HoleCards& hero_hand,
        int hero_position,           // 0 = P0 (BB/OOP), 1 = P1 (BTN/IP)
        const std::vector<Card>& board,
        float pot,
        float hero_stack,
        float villain_stack,
        float hero_bet,
        float villain_bet,
        int current_player,          // Who acts next
        const Config& config = Config::default_config()
    );

    // State queries
    bool is_terminal() const { return is_terminal_; }
    bool is_chance_node() const { return needs_deal_; }
    int current_player() const { return current_player_; }
    Street street() const { return street_; }

    float pot() const { return pot_; }
    float to_call() const;
    float effective_stack() const;

    // Get utility at terminal node (for player)
    float utility(int player) const;

    // Get legal actions
    std::vector<Action> get_legal_actions() const;

    // Apply action and return new state
    HoldemState apply_action(const Action& action) const;

    // Deal cards (for chance nodes)
    HoldemState deal_hole_cards(const HoleCards& hand0,
                                const HoleCards& hand1) const;
    HoldemState deal_board(const std::vector<Card>& cards) const;

    // Random dealing with RNG
    template<typename RNG>
    HoldemState deal_random(RNG& rng) const {
        if (!needs_deal_) return *this;

        Deck deck;
        deck.remove(used_cards());

        if (street_ == Street::Preflop && hands_[0][0] == NO_CARD) {
            // Deal hole cards
            deck.shuffle(rng);
            HoleCards h0(deck.deal(), deck.deal());
            HoleCards h1(deck.deal(), deck.deal());
            return deal_hole_cards(h0, h1);
        } else {
            // Deal board cards for current street
            // street_ is already the new street, so we calculate cards needed
            // based on current board_count_ vs. what current street requires
            int target_cards = street_board_cards(street_);
            int num_new = target_cards - board_count_;
            if (num_new <= 0) {
                // Shouldn't happen but guard against it
                HoldemState new_state = *this;
                new_state.needs_deal_ = false;
                return new_state;
            }
            std::vector<Card> new_cards;
            for (int i = 0; i < num_new; ++i) {
                new_cards.push_back(deck.deal_random(rng));
            }
            return deal_board(new_cards);
        }
    }

    // Accessors
    const HoleCards& hand(int player) const { return hands_[player]; }
    const std::array<Card, 5>& board() const { return board_; }
    int board_count() const { return board_count_; }
    float stack(int player) const { return stacks_[player]; }
    float bet(int player) const { return bets_[player]; }

    // Action history
    const std::vector<Action>& action_history() const { return history_; }

    // Information set key generation
    uint64_t info_set_key(int player) const;

    // String representation
    std::string to_string() const;

private:
    HoldemState() = default;

    void advance_street();
    Street next_street() const;
    HandMask used_cards() const;

    // Betting state
    std::array<float, NUM_PLAYERS> stacks_;
    std::array<float, NUM_PLAYERS> bets_;
    float pot_;

    // Cards
    std::array<HoleCards, NUM_PLAYERS> hands_;
    std::array<Card, 5> board_;
    int board_count_;

    // Game flow
    Street street_;
    int current_player_;
    int last_aggressor_;
    int actions_this_round_;
    bool is_terminal_;
    bool needs_deal_;
    bool showdown_;

    // History
    std::vector<Action> history_;

    // Configuration
    Config config_;
};

// Hash function for info set keys
struct InfoSetKeyHash {
    size_t operator()(uint64_t key) const {
        // MurmurHash-inspired mixing
        key ^= key >> 33;
        key *= 0xff51afd7ed558ccdULL;
        key ^= key >> 33;
        key *= 0xc4ceb9fe1a85ec53ULL;
        key ^= key >> 33;
        return key;
    }
};

}  // namespace ares
