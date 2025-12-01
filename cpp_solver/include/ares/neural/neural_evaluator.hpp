#pragma once

/**
 * Neural Network Evaluator for ARES-HU
 *
 * Loads a TorchScript model and evaluates game states.
 * Used for leaf node evaluation in real-time search.
 */

#include <torch/script.h>
#include <string>
#include <vector>
#include <memory>

namespace ares {

/**
 * Neural network evaluator using TorchScript.
 *
 * Usage:
 *   NeuralEvaluator eval("models/value_net.torchscript");
 *   float value = eval.evaluate(pbs_encoding);
 */
class NeuralEvaluator {
public:
    /**
     * Load a TorchScript model from disk.
     *
     * @param model_path Path to the .torchscript file
     * @throws std::runtime_error if model cannot be loaded
     */
    explicit NeuralEvaluator(const std::string& model_path);

    /**
     * Check if model is loaded and ready.
     */
    bool is_loaded() const { return model_loaded_; }

    /**
     * Evaluate a single game state.
     *
     * @param pbs_encoding The PBS feature vector (input_dim floats)
     * @return Predicted value
     */
    float evaluate(const std::vector<float>& pbs_encoding);

    /**
     * Evaluate multiple game states in a batch (more efficient).
     *
     * @param batch Vector of PBS encodings
     * @return Vector of predicted values
     */
    std::vector<float> evaluate_batch(const std::vector<std::vector<float>>& batch);

    /**
     * Get the expected input dimension.
     */
    int input_dim() const { return input_dim_; }

private:
    torch::jit::script::Module model_;
    bool model_loaded_ = false;
    int input_dim_ = 3042;  // Default, will be detected from model
};

}  // namespace ares
