#pragma once

#ifdef ARES_USE_LIBTORCH
#include <torch/torch.h>
#endif

#include "ares/belief/public_belief_state.hpp"
#include <vector>
#include <memory>
#include <string>

namespace ares {

// Forward declarations
class PublicBeliefState;

/**
 * Neural network configuration for ReBeL-style architecture.
 *
 * Based on Meta's ReBeL paper:
 * - 6 hidden layers × 1536 units each
 * - LayerNorm + ReLU activation
 * - ~15M parameters per network
 */
struct NetworkConfig {
    int input_dim = 2956;       // PBS encoding dimension
    int hidden_dim = 1536;      // Hidden layer size (ReBeL: 1536)
    int num_layers = 6;         // Number of hidden layers (ReBeL: 6)
    float dropout = 0.0f;       // Dropout rate (0 for inference)
    bool use_layer_norm = true; // Use LayerNorm (recommended)
    bool use_residual = false;  // Use residual connections

    // QRE-specific
    float qre_tau = 1.0f;       // Temperature for policy output

    static NetworkConfig rebel_config() {
        NetworkConfig cfg;
        cfg.hidden_dim = 1536;
        cfg.num_layers = 6;
        cfg.use_layer_norm = true;
        return cfg;
    }

    static NetworkConfig small_config() {
        // For development/testing
        NetworkConfig cfg;
        cfg.hidden_dim = 256;
        cfg.num_layers = 3;
        return cfg;
    }
};

#ifdef ARES_USE_LIBTORCH

/**
 * Value Network - predicts expected payoffs for both players.
 *
 * Input: PBS encoding (2956 dims)
 * Output: [EV_player0, EV_player1] (2 dims)
 *
 * Used for:
 * - Leaf evaluation in depth-limited search
 * - Training targets come from CFR results
 */
class ValueNetworkImpl : public torch::nn::Module {
public:
    explicit ValueNetworkImpl(const NetworkConfig& config);

    // Forward pass
    torch::Tensor forward(torch::Tensor x);

    // Evaluate PBS
    std::array<float, 2> evaluate(const PublicBeliefState& pbs);

    // Batch evaluation
    std::vector<std::array<float, 2>> evaluate_batch(
        const std::vector<PublicBeliefState>& pbs_batch
    );

private:
    NetworkConfig config_;
    torch::nn::Sequential layers_;
    torch::nn::Linear output_layer_{nullptr};
};
TORCH_MODULE(ValueNetwork);

/**
 * Policy Network - outputs action probabilities.
 *
 * Input: PBS encoding (2956 dims)
 * Output: Probability distribution over legal actions
 *
 * Uses QRE-style temperature scaling:
 *   P(a) = exp(logit(a) / τ) / Σ exp(logit / τ)
 */
class PolicyNetworkImpl : public torch::nn::Module {
public:
    explicit PolicyNetworkImpl(const NetworkConfig& config);

    // Forward pass - returns logits for all possible actions
    torch::Tensor forward(torch::Tensor x);

    // Get strategy for specific legal actions
    std::vector<float> get_strategy(
        const PublicBeliefState& pbs,
        const std::vector<Action>& legal_actions
    );

    // Batch strategy computation
    std::vector<std::vector<float>> get_strategy_batch(
        const std::vector<PublicBeliefState>& pbs_batch,
        const std::vector<std::vector<Action>>& legal_actions_batch
    );

    // Set QRE temperature
    void set_tau(float tau) { config_.qre_tau = tau; }

private:
    NetworkConfig config_;
    torch::nn::Sequential layers_;
    torch::nn::Linear output_layer_{nullptr};

    // Map action to output index
    static int action_to_index(const Action& action);
    static constexpr int MAX_ACTIONS = 10;  // Max discrete actions
};
TORCH_MODULE(PolicyNetwork);

/**
 * Combined model for value and policy (optional).
 *
 * Shares early layers for efficiency.
 */
class CombinedNetworkImpl : public torch::nn::Module {
public:
    explicit CombinedNetworkImpl(const NetworkConfig& config);

    // Forward pass - returns (values, policy_logits)
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

private:
    NetworkConfig config_;
    torch::nn::Sequential shared_layers_;
    torch::nn::Sequential value_head_;
    torch::nn::Sequential policy_head_;
};
TORCH_MODULE(CombinedNetwork);

#endif  // ARES_USE_LIBTORCH

/**
 * Abstract interface for value estimation.
 *
 * Allows switching between:
 * - Neural network evaluation
 * - Rollout-based estimation
 * - Tabular lookup
 */
class ValueEstimator {
public:
    virtual ~ValueEstimator() = default;

    // Estimate expected values for both players
    virtual std::array<float, 2> estimate(const PublicBeliefState& pbs) = 0;

    // Batch estimation (for efficiency)
    virtual std::vector<std::array<float, 2>> estimate_batch(
        const std::vector<PublicBeliefState>& pbs_batch
    ) {
        std::vector<std::array<float, 2>> results;
        results.reserve(pbs_batch.size());
        for (const auto& pbs : pbs_batch) {
            results.push_back(estimate(pbs));
        }
        return results;
    }
};

/**
 * Abstract interface for policy estimation.
 */
class PolicyEstimator {
public:
    virtual ~PolicyEstimator() = default;

    // Get action probabilities
    virtual std::vector<float> get_strategy(
        const PublicBeliefState& pbs,
        const std::vector<Action>& legal_actions
    ) = 0;
};

/**
 * Uniform policy - baseline for comparison.
 */
class UniformPolicy : public PolicyEstimator {
public:
    std::vector<float> get_strategy(
        const PublicBeliefState& pbs,
        const std::vector<Action>& legal_actions
    ) override {
        int n = legal_actions.size();
        return std::vector<float>(n, 1.0f / n);
    }
};

/**
 * Rollout-based value estimator (no neural network).
 *
 * Estimates values by random rollouts to terminal states.
 */
class RolloutValueEstimator : public ValueEstimator {
public:
    RolloutValueEstimator(int num_rollouts = 100);

    std::array<float, 2> estimate(const PublicBeliefState& pbs) override;

private:
    int num_rollouts_;
};

#ifdef ARES_USE_LIBTORCH

/**
 * Neural value estimator wrapper.
 */
class NeuralValueEstimator : public ValueEstimator {
public:
    explicit NeuralValueEstimator(std::shared_ptr<ValueNetwork> network);

    std::array<float, 2> estimate(const PublicBeliefState& pbs) override;

    std::vector<std::array<float, 2>> estimate_batch(
        const std::vector<PublicBeliefState>& pbs_batch
    ) override;

private:
    std::shared_ptr<ValueNetwork> network_;
};

/**
 * Neural policy estimator wrapper.
 */
class NeuralPolicyEstimator : public PolicyEstimator {
public:
    explicit NeuralPolicyEstimator(std::shared_ptr<PolicyNetwork> network);

    std::vector<float> get_strategy(
        const PublicBeliefState& pbs,
        const std::vector<Action>& legal_actions
    ) override;

    void set_tau(float tau);

private:
    std::shared_ptr<PolicyNetwork> network_;
};

#endif  // ARES_USE_LIBTORCH

/**
 * Model serialization utilities.
 */
namespace model_utils {

#ifdef ARES_USE_LIBTORCH

// Save model to file
void save_model(const torch::nn::Module& model, const std::string& path);

// Load model from file
template<typename T>
std::shared_ptr<T> load_model(const std::string& path, const NetworkConfig& config);

// Export to TorchScript for deployment
void export_torchscript(
    const torch::nn::Module& model,
    const std::string& path,
    const std::vector<int64_t>& example_input_shape
);

#endif  // ARES_USE_LIBTORCH

}  // namespace model_utils

}  // namespace ares
