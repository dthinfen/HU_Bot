#pragma once

#include "ares/game/holdem_state.hpp"
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ares {

// Forward declarations
class CardAbstraction;
class BettingAbstraction;

// Information set data (regrets and strategy)
struct InfoSetData {
    std::vector<float> regret_sum;     // Cumulative regrets
    std::vector<float> strategy_sum;   // Cumulative strategy (for averaging)
    std::atomic<uint32_t> visits{0};   // Number of times visited

    InfoSetData() = default;
    InfoSetData(int num_actions)
        : regret_sum(num_actions, 0.0f)
        , strategy_sum(num_actions, 0.0f) {}

    // Non-copyable due to atomic
    InfoSetData(const InfoSetData& other)
        : regret_sum(other.regret_sum)
        , strategy_sum(other.strategy_sum)
        , visits(other.visits.load()) {}

    InfoSetData& operator=(const InfoSetData& other) {
        regret_sum = other.regret_sum;
        strategy_sum = other.strategy_sum;
        visits = other.visits.load();
        return *this;
    }
};

// Strategy map type
using StrategyMap = std::unordered_map<uint64_t, InfoSetData, InfoSetKeyHash>;

// Equilibrium type to compute
enum class EquilibriumType {
    NASH,       // Standard Nash equilibrium (unexploitable)
    QRE         // Quantal Response Equilibrium (exploits human mistakes)
};

// DCFR+ Solver Configuration
struct DCFRConfig {
    // Iteration settings
    int iterations = 1000000;
    int num_threads = 8;

    // Equilibrium target
    EquilibriumType equilibrium = EquilibriumType::QRE;  // Default to QRE for human opponents
    float qre_tau = 1.0f;  // QRE temperature (lower = closer to Nash, higher = more exploration)
                           // Recommended: 0.5-2.0 for playing humans

    // DCFR+ parameters
    float alpha = 1.5f;   // Positive regret discount (t^alpha / (t^alpha + 1))
    float beta = 0.0f;    // Negative regret discount (t^beta / (t^beta + 1))
    float gamma = 1.0f;   // Strategy contribution (t / (t + 1))^gamma

    // Training settings
    int defer_averaging = 1000;      // Iterations before strategy averaging
    int checkpoint_freq = 10000;     // How often to save checkpoints
    std::string checkpoint_dir;      // Directory for checkpoints

    // Game settings
    float starting_stack = 20.0f;    // Stack size in BB

    // Abstraction (nullptr for no abstraction)
    std::shared_ptr<CardAbstraction> card_abstraction;
    std::shared_ptr<BettingAbstraction> betting_abstraction;

    // Logging
    bool verbose = true;
    std::function<void(int, float)> progress_callback;

    // Presets
    static DCFRConfig nash_preset() {
        DCFRConfig cfg;
        cfg.equilibrium = EquilibriumType::NASH;
        return cfg;
    }

    static DCFRConfig qre_preset(float tau = 1.0f) {
        DCFRConfig cfg;
        cfg.equilibrium = EquilibriumType::QRE;
        cfg.qre_tau = tau;
        return cfg;
    }

    static DCFRConfig competitive_preset() {
        // For beating humans - QRE with optimal settings
        DCFRConfig cfg;
        cfg.equilibrium = EquilibriumType::QRE;
        cfg.qre_tau = 1.0f;
        cfg.iterations = 10000000;
        cfg.alpha = 1.5f;
        cfg.beta = 0.5f;
        cfg.gamma = 2.0f;
        cfg.defer_averaging = 5000;
        return cfg;
    }
};

// DCFR+ Solver
class DCFRPlusSolver {
public:
    explicit DCFRPlusSolver(const DCFRConfig& config);

    // Move constructor (needed because mutex is non-copyable)
    DCFRPlusSolver(DCFRPlusSolver&& other) noexcept
        : config_(std::move(other.config_))
        , strategy_(std::move(other.strategy_))
        , thread_rngs_(std::move(other.thread_rngs_))
        , current_iteration_(other.current_iteration_.load())
        , nodes_visited_(other.nodes_visited_.load())
        , info_sets_updated_(other.info_sets_updated_.load())
    {}

    // Move assignment
    DCFRPlusSolver& operator=(DCFRPlusSolver&& other) noexcept {
        if (this != &other) {
            config_ = std::move(other.config_);
            strategy_ = std::move(other.strategy_);
            thread_rngs_ = std::move(other.thread_rngs_);
            current_iteration_ = other.current_iteration_.load();
            nodes_visited_ = other.nodes_visited_.load();
            info_sets_updated_ = other.info_sets_updated_.load();
        }
        return *this;
    }

    // Non-copyable
    DCFRPlusSolver(const DCFRPlusSolver&) = delete;
    DCFRPlusSolver& operator=(const DCFRPlusSolver&) = delete;

    // Main training interface
    void train();

    // Get final average strategy
    const StrategyMap& get_strategy() const { return strategy_; }

    // Get strategy for specific info set
    std::vector<float> get_avg_strategy(uint64_t info_set_key) const;

    // Serialization
    void save(const std::string& path) const;
    static DCFRPlusSolver load(const std::string& path);

    // Save/load checkpoint (includes regrets)
    void save_checkpoint(const std::string& path) const;
    void load_checkpoint(const std::string& path);

    // Statistics
    size_t num_info_sets() const { return strategy_.size(); }
    int current_iteration() const { return current_iteration_; }
    double exploitability() const;  // Compute exploitability (expensive)

    // Training data export for neural network
    // Exports (PBS_encoding, value) pairs as binary numpy-compatible format
    void export_training_data(
        const std::string& path,
        int num_samples = 100000
    ) const;

private:
    // Single CFR iteration
    void iterate(int iteration);

    // Recursive CFR traversal
    float cfr_traverse(
        const HoldemState& state,
        int traverser,
        float p0,  // Player 0 reach probability
        float p1,  // Player 1 reach probability
        int thread_id
    );

    // Regret matching to get current strategy
    std::vector<float> regret_match(const InfoSetData& data) const;

    // Update regrets and strategy
    void update_regrets(
        InfoSetData& data,
        const std::vector<float>& regrets,
        int iteration
    );

    void update_strategy(
        InfoSetData& data,
        const std::vector<float>& strategy,
        float reach_prob,
        int iteration
    );

    // Discount factors
    float positive_discount(int t) const;
    float negative_discount(int t) const;
    float strategy_discount(int t) const;

    // Get or create info set data
    InfoSetData& get_info_set(uint64_t key, int num_actions);

    // Abstraction helpers
    uint64_t abstract_info_set(const HoldemState& state, int player) const;
    std::vector<Action> abstract_actions(const HoldemState& state) const;

    // Configuration
    DCFRConfig config_;

    // Strategy storage
    StrategyMap strategy_;
    mutable std::mutex strategy_mutex_;  // For thread-safe access

    // Per-thread random number generators
    std::vector<std::mt19937> thread_rngs_;

    // Training state
    std::atomic<int> current_iteration_{0};

    // Performance counters
    std::atomic<uint64_t> nodes_visited_{0};
    std::atomic<uint64_t> info_sets_updated_{0};
};

// Utility functions

// Compute Nash equilibrium distance (sum of immediate regrets)
float compute_immediate_regret(const StrategyMap& strategy);

// Sample action from strategy
template<typename RNG>
int sample_action(const std::vector<float>& probs, RNG& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cumsum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumsum += probs[i];
        if (r < cumsum) return static_cast<int>(i);
    }
    return static_cast<int>(probs.size() - 1);
}

}  // namespace ares
