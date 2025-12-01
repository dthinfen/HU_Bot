#pragma once

#include "ares/core/cards.hpp"
#include "ares/game/action.hpp"
#include "ares/game/holdem_state.hpp"
#include <array>
#include <vector>
#include <cmath>

namespace ares {

// Number of possible hole card combinations: C(52,2) = 1326
constexpr int NUM_HOLE_COMBOS = 1326;

// Maximum betting history length to encode
constexpr int MAX_HISTORY_LENGTH = 20;

/**
 * Hand Distribution - probability distribution over all possible hole cards.
 *
 * Each player's belief about what hands they/opponent might have.
 * Uses a 1326-dimensional probability vector.
 */
class HandDistribution {
public:
    HandDistribution();

    // Initialize to uniform distribution over all hands
    void set_uniform();

    // Initialize to uniform over hands not blocked by dead cards
    void set_uniform_unblocked(HandMask dead_cards);

    // Get/set probability for a specific hand
    float get(int combo_index) const { return probs_[combo_index]; }
    void set(int combo_index, float prob) { probs_[combo_index] = prob; }

    // Get probability for specific hole cards
    float get(const HoleCards& hand) const;
    void set(const HoleCards& hand, float prob);

    // Normalize probabilities to sum to 1
    void normalize();

    // Zero out hands blocked by cards
    void block(HandMask dead_cards);

    // Update beliefs based on action (Bayesian update)
    // P(hand | action) ∝ P(action | hand) * P(hand)
    void update_with_action(
        const std::vector<float>& action_probs_per_hand,
        int action_taken
    );

    // Get entropy of distribution
    float entropy() const;

    // Get number of non-zero hands
    int support_size() const;

    // Raw access for neural network encoding
    const std::array<float, NUM_HOLE_COMBOS>& probs() const { return probs_; }
    std::array<float, NUM_HOLE_COMBOS>& probs() { return probs_; }

    // Convert hole cards to combo index (0-1325)
    static int hand_to_index(const HoleCards& hand);
    static HoleCards index_to_hand(int index);

private:
    std::array<float, NUM_HOLE_COMBOS> probs_;
};

/**
 * Public Belief State (PBS) - the key innovation from ReBeL.
 *
 * Transforms imperfect information game into perfect information
 * by representing beliefs about hidden cards as continuous state.
 */
class PublicBeliefState {
public:
    PublicBeliefState() = default;

    // Create from game state
    static PublicBeliefState from_state(const HoldemState& state);

    // Create initial PBS (preflop, no information)
    static PublicBeliefState create_initial(float starting_stack);

    // Update PBS with new action
    PublicBeliefState apply_action(
        const Action& action,
        const std::vector<std::vector<float>>& action_probs  // [player][combo] -> prob of this action
    ) const;

    // Update PBS with new board cards
    PublicBeliefState apply_board(const std::vector<Card>& new_cards) const;

    // Encode PBS as feature vector for neural network
    std::vector<float> encode() const;

    // Get encoding dimension
    static int encoding_dim();

    // Accessors
    Street street() const { return street_; }
    float pot() const { return pot_; }
    float stack(int player) const { return stacks_[player]; }
    const std::array<Card, 5>& board() const { return board_; }
    int board_count() const { return board_count_; }
    int current_player() const { return current_player_; }
    const std::vector<Action>& history() const { return history_; }

    // Belief distributions
    const HandDistribution& belief(int player) const { return beliefs_[player]; }
    HandDistribution& belief(int player) { return beliefs_[player]; }

    // Check if terminal
    bool is_terminal() const { return is_terminal_; }

    // Get legal actions (same as HoldemState)
    std::vector<Action> get_legal_actions() const;

private:
    // Public game state
    Street street_ = Street::Preflop;
    float pot_ = 0.0f;
    std::array<float, 2> stacks_ = {0.0f, 0.0f};
    std::array<float, 2> bets_ = {0.0f, 0.0f};
    std::array<Card, 5> board_;
    int board_count_ = 0;
    int current_player_ = 0;
    bool is_terminal_ = false;

    // Action history
    std::vector<Action> history_;

    // Belief distributions for each player
    std::array<HandDistribution, 2> beliefs_;

    // Configuration
    float starting_stack_ = 100.0f;
    float small_blind_ = 0.5f;
    float big_blind_ = 1.0f;
};

/**
 * PBS Encoder - converts PBS to neural network input features.
 *
 * Features include:
 * - Board card one-hot encoding (52 * 5 = 260)
 * - Pot size (normalized) (1)
 * - Stack sizes (normalized) (2)
 * - Betting history encoding (variable)
 * - Belief distributions (1326 * 2 = 2652)
 * - Board texture features (flush draws, straights, etc.)
 */
class PBSEncoder {
public:
    // Encode full PBS
    static std::vector<float> encode(const PublicBeliefState& pbs);

    // Get total encoding dimension
    static int encoding_dim();

    // Individual encoding components
    static std::vector<float> encode_board(const std::array<Card, 5>& board, int count);
    static std::vector<float> encode_pot_and_stacks(float pot, float stack0, float stack1, float starting);
    static std::vector<float> encode_history(const std::vector<Action>& history);
    static std::vector<float> encode_beliefs(const HandDistribution& belief0, const HandDistribution& belief1);
    static std::vector<float> encode_board_texture(const std::array<Card, 5>& board, int count);
    static std::vector<float> encode_position(int current_player, Street street);

private:
    // Feature dimensions
    static constexpr int BOARD_DIM = 52 * 5;           // One-hot per card slot
    static constexpr int POT_STACK_DIM = 4;            // pot, stack0, stack1, spr
    static constexpr int HISTORY_DIM = MAX_HISTORY_LENGTH * 5;  // Action encoding
    static constexpr int BELIEF_DIM = NUM_HOLE_COMBOS * 2;      // Both players
    static constexpr int TEXTURE_DIM = 20;             // Board texture features
    static constexpr int POSITION_DIM = 4;             // Position + street
};

/**
 * Range utilities for equity calculations and belief updates.
 */
namespace range_utils {

// Calculate equity of range vs range on given board
float range_vs_range_equity(
    const HandDistribution& range1,
    const HandDistribution& range2,
    const std::array<Card, 5>& board,
    int board_count
);

// Get all hands that are not blocked by given cards
std::vector<int> unblocked_combos(HandMask dead_cards);

// Convert canonical hand index to (card1, card2) indices
std::pair<int, int> combo_to_cards(int combo_index);

// Convert (card1, card2) to combo index
int cards_to_combo(int card1, int card2);

}  // namespace range_utils

}  // namespace ares
