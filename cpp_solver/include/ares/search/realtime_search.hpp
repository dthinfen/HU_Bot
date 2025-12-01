#pragma once

/**
 * Real-Time CFR Search for ARES-HU
 *
 * Enables playing at any stack depth by running CFR search
 * from arbitrary game states at decision time.
 */

#include "ares/game/holdem_state.hpp"
#include <unordered_map>
#include <vector>
#include <random>
#include <functional>
#include <memory>

namespace ares {

// Forward declarations
class PublicBeliefState;

/**
 * Callback type for neural network leaf evaluation.
 * Takes a PBS feature vector and returns the predicted value.
 */
using NeuralEvalCallback = std::function<float(const std::vector<float>&)>;

/**
 * Configuration for real-time search.
 */
struct SearchConfig {
    int iterations = 500;       // Number of CFR iterations
    float time_limit = 5.0f;    // Time limit in seconds (0 = no limit)
    int max_depth = 0;          // Max search depth (0 = unlimited)
    float qre_tau = 1.0f;       // QRE temperature (0 = Nash)
    bool cfr_plus = true;       // Use CFR+ (clamp negative regrets)
    unsigned int seed = 42;     // Random seed

    // Optional neural evaluator callback (if null, uses heuristic)
    NeuralEvalCallback neural_eval = nullptr;
};

/**
 * Node data for search tree.
 */
struct SearchNode {
    std::vector<float> regret_sum;
    std::vector<float> strategy_sum;
    int visits;

    SearchNode() = default;
    explicit SearchNode(int num_actions);
};

/**
 * Result of a search.
 */
struct SearchResult {
    std::vector<Action> actions;       // Legal actions at root
    std::vector<float> strategy;       // Probabilities for each action
    Action best_action;                // Highest probability action
    int iterations = 0;                // Iterations completed
    size_t nodes_visited = 0;          // Unique nodes visited
    float time_seconds = 0.0f;         // Time taken
};

/**
 * Real-time CFR search engine.
 *
 * Usage:
 *   RealtimeSearch search(config);
 *   SearchResult result = search.search(game_state);
 *   // result.strategy contains the computed mixed strategy
 *   // result.best_action is the most likely action
 */
class RealtimeSearch {
public:
    explicit RealtimeSearch(const SearchConfig& config = SearchConfig{});

    /**
     * Run CFR search from the given game state.
     *
     * @param root_state The current game state to search from
     * @return SearchResult with computed strategy and best action
     */
    SearchResult search(const HoldemState& root_state);

    /**
     * Get number of iterations completed in last search.
     */
    int iterations_done() const { return iterations_done_; }

private:
    SearchConfig config_;
    std::mt19937 rng_;

    // Search tree (cleared each search)
    std::unordered_map<uint64_t, SearchNode> nodes_;
    int iterations_done_ = 0;

    /**
     * Run one CFR iteration from the given state.
     */
    float cfr_iterate(
        const HoldemState& state,
        int traverser,
        float p0,
        float p1,
        int depth
    );

    /**
     * Evaluate a leaf node (when depth limit reached).
     * Default: random rollout. Override for neural net evaluation.
     */
    float evaluate_leaf(const HoldemState& state, int player);

    /**
     * Get or create a search node.
     */
    SearchNode& get_or_create_node(uint64_t key, int num_actions);

    /**
     * Get current strategy from regrets (for iteration).
     */
    std::vector<float> get_current_strategy(const SearchNode& node) const;

    /**
     * Get average strategy (for final result).
     */
    std::vector<float> get_average_strategy(const SearchNode& node) const;

    /**
     * Sample an action according to strategy.
     */
    int sample_action(const std::vector<float>& strategy);
};

}  // namespace ares
