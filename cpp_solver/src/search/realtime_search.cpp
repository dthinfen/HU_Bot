/**
 * Real-Time CFR Search for ARES-HU
 *
 * Enables "any stack, any bet" play by running depth-limited CFR
 * from arbitrary game states. This is the key component for neural
 * network + search integration (ReBeL-style).
 *
 * Key features:
 * - Fast CFR iterations from arbitrary states
 * - QRE equilibrium for human exploitation
 * - Configurable search depth and iterations
 */

#include "ares/search/realtime_search.hpp"
#include "ares/cfr/regret_minimizer.hpp"
#include "ares/core/hand_evaluator.hpp"
#include "ares/belief/public_belief_state.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>

namespace ares {

// =============================================================================
// RealtimeSearch Implementation
// =============================================================================

RealtimeSearch::RealtimeSearch(const SearchConfig& config)
    : config_(config), rng_(config.seed)
{
    HandEvaluator::initialize();

    // Force a reasonable max depth if not set
    if (config_.max_depth <= 0) {
        config_.max_depth = 10;  // Limit tree depth for stability
    }
}

SearchResult RealtimeSearch::search(const HoldemState& root_state) {
    if (root_state.is_terminal()) {
        return SearchResult{};
    }

    // Clear previous search data
    nodes_.clear();
    iterations_done_ = 0;

    auto start_time = std::chrono::high_resolution_clock::now();
    auto deadline = start_time + std::chrono::milliseconds(
        static_cast<int>(config_.time_limit * 1000));

    // Run CFR iterations
    for (int iter = 1; iter <= config_.iterations; ++iter) {
        // Check time limit
        if (config_.time_limit > 0) {
            auto now = std::chrono::high_resolution_clock::now();
            if (now >= deadline) {
                break;
            }
        }

        // Run one iteration for each traverser
        for (int traverser = 0; traverser < 2; ++traverser) {
            cfr_iterate(root_state, traverser, 1.0f, 1.0f, 0);
        }

        iterations_done_ = iter;
    }

    // Build result
    SearchResult result;
    result.iterations = iterations_done_;
    result.nodes_visited = nodes_.size();

    auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
    result.time_seconds = std::chrono::duration<float>(elapsed).count();

    // Get strategy at root
    if (!root_state.is_terminal() && !root_state.is_chance_node()) {
        uint64_t root_key = root_state.info_set_key(root_state.current_player());
        auto it = nodes_.find(root_key);

        if (it != nodes_.end()) {
            result.actions = root_state.get_legal_actions();
            result.strategy = get_average_strategy(it->second);

            // Find best action
            if (!result.strategy.empty()) {
                int best_idx = 0;
                float best_prob = result.strategy[0];
                for (size_t i = 1; i < result.strategy.size(); ++i) {
                    if (result.strategy[i] > best_prob) {
                        best_prob = result.strategy[i];
                        best_idx = i;
                    }
                }
                if (best_idx < static_cast<int>(result.actions.size())) {
                    result.best_action = result.actions[best_idx];
                }
            }
        }
    }

    return result;
}

float RealtimeSearch::cfr_iterate(
    const HoldemState& state,
    int traverser,
    float p0,
    float p1,
    int depth
) {
    // Terminal node
    if (state.is_terminal()) {
        return state.utility(traverser);
    }

    // Chance node - sample
    if (state.is_chance_node()) {
        HoldemState dealt = state.deal_random(rng_);
        return cfr_iterate(dealt, traverser, p0, p1, depth);
    }

    // Depth limit - use simple evaluation
    if (config_.max_depth > 0 && depth >= config_.max_depth) {
        return evaluate_leaf(state, traverser);
    }

    // Prevent infinite recursion
    if (depth > 100) {
        return 0.0f;
    }

    int player = state.current_player();
    std::vector<Action> actions = state.get_legal_actions();
    int num_actions = static_cast<int>(actions.size());

    if (num_actions == 0) {
        return 0.0f;
    }

    // Get or create node - IMPORTANT: we must re-lookup after recursive calls
    // because unordered_map can rehash and invalidate references
    uint64_t info_key = state.info_set_key(player);
    get_or_create_node(info_key, num_actions);  // Ensure node exists

    // Get current strategy (copy the regrets before recursion)
    std::vector<float> strategy;
    {
        SearchNode& node = nodes_.at(info_key);
        if (node.regret_sum.size() != static_cast<size_t>(num_actions)) {
            node = SearchNode(num_actions);
        }
        strategy = get_current_strategy(node);
    }

    // Validate strategy size
    if (strategy.size() != static_cast<size_t>(num_actions)) {
        strategy.assign(num_actions, 1.0f / num_actions);
    }

    if (player == traverser) {
        // Traverse all actions
        std::vector<float> action_values(num_actions, 0.0f);
        float node_value = 0.0f;

        for (int a = 0; a < num_actions; ++a) {
            HoldemState next = state.apply_action(actions[a]);
            float new_p0 = (player == 0) ? p0 * strategy[a] : p0;
            float new_p1 = (player == 1) ? p1 * strategy[a] : p1;

            action_values[a] = cfr_iterate(next, traverser, new_p0, new_p1, depth + 1);
            node_value += strategy[a] * action_values[a];
        }

        // Re-lookup node after recursion (map may have rehashed)
        SearchNode& node = nodes_.at(info_key);

        // Safety check: ensure node size matches (can differ if info_key collision)
        if (node.regret_sum.size() != static_cast<size_t>(num_actions)) {
            // Info set key collision - resize node
            node = SearchNode(num_actions);
        }

        // Update regrets
        float reach = (player == 0) ? p1 : p0;  // Opponent reach probability
        for (int a = 0; a < num_actions; ++a) {
            float regret = action_values[a] - node_value;
            node.regret_sum[a] += reach * regret;

            // CFR+ style: clamp negative regrets (optional, controlled by config)
            if (config_.cfr_plus && node.regret_sum[a] < 0) {
                node.regret_sum[a] = 0;
            }
        }

        // Update strategy sum
        float own_reach = (player == 0) ? p0 : p1;
        for (int a = 0; a < num_actions; ++a) {
            node.strategy_sum[a] += own_reach * strategy[a];
        }

        node.visits++;
        return node_value;
    } else {
        // Sample opponent action
        int sampled = sample_action(strategy);
        HoldemState next = state.apply_action(actions[sampled]);

        float new_p0 = (player == 0) ? p0 * strategy[sampled] : p0;
        float new_p1 = (player == 1) ? p1 * strategy[sampled] : p1;

        return cfr_iterate(next, traverser, new_p0, new_p1, depth + 1);
    }
}

float RealtimeSearch::evaluate_leaf(const HoldemState& state, int player) {
    // Use neural network evaluation if available
    if (config_.neural_eval) {
        // Create PBS from state and encode it
        PublicBeliefState pbs = PublicBeliefState::from_state(state);
        std::vector<float> features = PBSEncoder::encode(pbs);

        // Get neural network prediction (value for player 0)
        float value = config_.neural_eval(features);

        // Return value from perspective of requested player
        return (player == 0) ? value : -value;
    }

    // Fallback: Simple heuristic evaluation based on hand strength and pot equity
    // This gives CFR something to work with instead of returning 0

    if (state.is_terminal()) {
        return state.utility(player);
    }

    // Get basic game info
    float pot = state.pot();
    (void)pot;  // Used below

    // Simple hand strength estimate (0-1)
    float hand_strength = 0.5f;  // Default to medium

    // Get hole cards for player
    const HoleCards& hand = state.hand(player);
    if (hand[0] != NO_CARD && hand[1] != NO_CARD) {
        int rank1 = get_rank_int(hand[0]);
        int rank2 = get_rank_int(hand[1]);
        bool suited = get_suit_int(hand[0]) == get_suit_int(hand[1]);
        bool paired = rank1 == rank2;

        int high = std::max(rank1, rank2);
        int low = std::min(rank1, rank2);
        int gap = high - low;

        if (paired) {
            // Pairs: 22=0.5, AA=0.85
            hand_strength = 0.5f + (rank1 / 12.0f) * 0.35f;
        } else {
            // Unpaired: based on high card, low card, suited, connected
            hand_strength = 0.25f + (high / 12.0f) * 0.25f + (low / 12.0f) * 0.1f;
            if (suited) hand_strength += 0.08f;
            if (gap <= 2) hand_strength += 0.05f;
            if (high >= 10 && low >= 10) hand_strength += 0.1f;  // Broadway
        }

        // Clamp to reasonable range
        hand_strength = std::max(0.2f, std::min(0.9f, hand_strength));
    }

    // If postflop, try to estimate made hand strength
    if (state.street() != Street::Preflop && state.board()[0] != NO_CARD) {
        // Use hand evaluator if we have board cards
        auto board = state.board();
        auto rank = HandEvaluator::evaluate(hand, board);

        // Convert hand rank to strength (lower rank = better hand)
        // Ranks typically range from ~300 (best) to ~7500 (worst)
        float normalized_rank = static_cast<float>(rank) / 7500.0f;
        hand_strength = 1.0f - normalized_rank;  // Invert so higher = better
        hand_strength = std::max(0.1f, std::min(0.95f, hand_strength));
    }

    // Calculate expected value based on hand strength and pot
    // If we're ahead, we expect to win a fraction of the pot
    // If we're behind, we expect to lose our investment

    float equity = hand_strength;
    float pot_share = equity * pot;
    float expected_loss = (1.0f - equity) * (pot * 0.5f);  // Rough estimate

    // Return value relative to current investment
    // Positive = ahead, negative = behind
    float ev = pot_share - expected_loss;

    // Scale by pot size for reasonable magnitude
    return ev * 0.5f;
}

SearchNode& RealtimeSearch::get_or_create_node(uint64_t key, int num_actions) {
    auto it = nodes_.find(key);
    if (it == nodes_.end()) {
        auto result = nodes_.emplace(key, SearchNode(num_actions));
        return result.first->second;
    }
    return it->second;
}

std::vector<float> RealtimeSearch::get_current_strategy(const SearchNode& node) const {
    if (config_.qre_tau > 0) {
        return regret_match(node.regret_sum, RegretMatchingType::SOFTMAX_QRE, config_.qre_tau);
    } else {
        return regret_match(node.regret_sum, RegretMatchingType::STANDARD, 0.0f);
    }
}

std::vector<float> RealtimeSearch::get_average_strategy(const SearchNode& node) const {
    return compute_average_strategy(node.strategy_sum);
}

int RealtimeSearch::sample_action(const std::vector<float>& strategy) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);
    float cumsum = 0.0f;

    for (size_t a = 0; a < strategy.size(); ++a) {
        cumsum += strategy[a];
        if (r <= cumsum) {
            return static_cast<int>(a);
        }
    }
    return static_cast<int>(strategy.size()) - 1;
}

// =============================================================================
// SearchNode Implementation
// =============================================================================

SearchNode::SearchNode(int num_actions)
    : regret_sum(num_actions, 0.0f)
    , strategy_sum(num_actions, 0.0f)
    , visits(0)
{}

}  // namespace ares
