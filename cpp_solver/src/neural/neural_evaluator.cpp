/**
 * Neural Network Evaluator Implementation
 */

#include "ares/neural/neural_evaluator.hpp"
#include <iostream>
#include <stdexcept>

namespace ares {

NeuralEvaluator::NeuralEvaluator(const std::string& model_path) {
    try {
        // Load the TorchScript model
        model_ = torch::jit::load(model_path);
        model_.eval();  // Set to evaluation mode
        model_loaded_ = true;

        std::cout << "Loaded neural network from " << model_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        model_loaded_ = false;
    }
}

float NeuralEvaluator::evaluate(const std::vector<float>& pbs_encoding) {
    if (!model_loaded_) {
        return 0.0f;  // Fallback if model not loaded
    }

    // Ensure correct input size
    if (pbs_encoding.size() != static_cast<size_t>(input_dim_)) {
        // Pad or truncate as needed
        std::vector<float> padded(input_dim_, 0.0f);
        size_t copy_size = std::min(pbs_encoding.size(), static_cast<size_t>(input_dim_));
        std::copy(pbs_encoding.begin(), pbs_encoding.begin() + copy_size, padded.begin());

        // Create tensor from padded data
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor input = torch::from_blob(
            padded.data(),
            {1, input_dim_},
            options
        ).clone();  // Clone to own the data

        // Run inference
        torch::NoGradGuard no_grad;
        auto output = model_.forward({input}).toTensor();
        return output.item<float>();
    }

    // Create tensor from input
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor input = torch::from_blob(
        const_cast<float*>(pbs_encoding.data()),
        {1, input_dim_},
        options
    ).clone();  // Clone to own the data

    // Run inference
    torch::NoGradGuard no_grad;
    auto output = model_.forward({input}).toTensor();
    return output.item<float>();
}

std::vector<float> NeuralEvaluator::evaluate_batch(
    const std::vector<std::vector<float>>& batch
) {
    if (!model_loaded_ || batch.empty()) {
        return std::vector<float>(batch.size(), 0.0f);
    }

    int batch_size = static_cast<int>(batch.size());

    // Flatten batch into contiguous memory
    std::vector<float> flat_data(batch_size * input_dim_, 0.0f);
    for (int i = 0; i < batch_size; ++i) {
        size_t copy_size = std::min(batch[i].size(), static_cast<size_t>(input_dim_));
        std::copy(batch[i].begin(), batch[i].begin() + copy_size,
                  flat_data.begin() + i * input_dim_);
    }

    // Create tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor input = torch::from_blob(
        flat_data.data(),
        {batch_size, input_dim_},
        options
    ).clone();

    // Run inference
    torch::NoGradGuard no_grad;
    auto output = model_.forward({input}).toTensor();

    // Extract results
    std::vector<float> results(batch_size);
    auto accessor = output.accessor<float, 2>();
    for (int i = 0; i < batch_size; ++i) {
        results[i] = accessor[i][0];
    }

    return results;
}

}  // namespace ares
