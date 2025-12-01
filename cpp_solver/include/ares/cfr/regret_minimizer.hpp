#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace ares {

/**
 * Regret matching strategies for CFR.
 *
 * Supports:
 * - Standard regret matching (Nash equilibrium)
 * - Regularized/QRE regret matching (Quantal Response Equilibrium)
 */
enum class RegretMatchingType {
    STANDARD,       // Nash equilibrium target
    SOFTMAX_QRE,    // Quantal Response Equilibrium
    HEDGE           // Multiplicative weights
};

/**
 * Convert regrets to strategy using specified method.
 *
 * @param regrets Cumulative regrets for each action
 * @param type Regret matching algorithm
 * @param tau Temperature parameter (for QRE modes)
 * @return Probability distribution over actions
 */
inline std::vector<float> regret_match(
    const std::vector<float>& regrets,
    RegretMatchingType type = RegretMatchingType::STANDARD,
    float tau = 1.0f
) {
    const size_t n = regrets.size();
    std::vector<float> strategy(n);

    if (n == 0) return strategy;
    if (n == 1) {
        strategy[0] = 1.0f;
        return strategy;
    }

    switch (type) {
        case RegretMatchingType::STANDARD: {
            // Standard regret matching: Nash equilibrium
            // σ(a) = max(R(a), 0) / Σ max(R, 0)
            float sum_positive = 0.0f;
            for (size_t i = 0; i < n; ++i) {
                float pos = std::max(regrets[i], 0.0f);
                strategy[i] = pos;
                sum_positive += pos;
            }

            if (sum_positive > 0) {
                for (size_t i = 0; i < n; ++i) {
                    strategy[i] /= sum_positive;
                }
            } else {
                // Uniform if all regrets non-positive
                float uniform = 1.0f / static_cast<float>(n);
                std::fill(strategy.begin(), strategy.end(), uniform);
            }
            break;
        }

        case RegretMatchingType::SOFTMAX_QRE: {
            // QRE: σ(a) = exp(R(a) / τ) / Σ exp(R / τ)
            // This is the logit quantal response equilibrium
            //
            // τ controls "rationality":
            // - τ → 0: Approaches Nash (hard max)
            // - τ → ∞: Approaches uniform (pure exploration)
            // - τ = 1: Balanced QRE
            //
            // For numerical stability, subtract max before exp

            float max_regret = *std::max_element(regrets.begin(), regrets.end());
            float sum_exp = 0.0f;

            for (size_t i = 0; i < n; ++i) {
                float scaled = (regrets[i] - max_regret) / tau;
                strategy[i] = std::exp(scaled);
                sum_exp += strategy[i];
            }

            for (size_t i = 0; i < n; ++i) {
                strategy[i] /= sum_exp;
            }
            break;
        }

        case RegretMatchingType::HEDGE: {
            // Hedge/multiplicative weights update
            // Similar to softmax but uses cumulative weights
            float max_regret = *std::max_element(regrets.begin(), regrets.end());
            float sum_exp = 0.0f;

            for (size_t i = 0; i < n; ++i) {
                float scaled = tau * (regrets[i] - max_regret);
                strategy[i] = std::exp(scaled);
                sum_exp += strategy[i];
            }

            for (size_t i = 0; i < n; ++i) {
                strategy[i] /= sum_exp;
            }
            break;
        }
    }

    return strategy;
}

/**
 * CFR+ style regret update with discounting.
 *
 * @param current Current cumulative regret
 * @param instant Instantaneous regret for this iteration
 * @param iteration Current iteration number
 * @param alpha DCFR+ alpha parameter (positive regret discount)
 * @param beta DCFR+ beta parameter (negative regret discount)
 * @return Updated cumulative regret
 */
inline float update_regret_dcfr_plus(
    float current,
    float instant,
    int iteration,
    float alpha = 1.5f,
    float beta = 0.0f
) {
    // DCFR+ discounting:
    // Positive regrets: multiply by t^α / (t^α + 1)
    // Negative regrets: multiply by t^β / (t^β + 1)

    float t = static_cast<float>(iteration);

    if (current > 0) {
        float discount = std::pow(t, alpha) / (std::pow(t, alpha) + 1.0f);
        current *= discount;
    } else if (current < 0 && beta > 0) {
        float discount = std::pow(t, beta) / (std::pow(t, beta) + 1.0f);
        current *= discount;
    } else if (current < 0) {
        // β = 0: Zero out negative regrets (CFR+ behavior)
        current = 0.0f;
    }

    return current + instant;
}

/**
 * Linear CFR regret update (used by Pluribus for warm-up).
 *
 * Weights iterations linearly: iteration t contributes weight t.
 *
 * @param current Current cumulative regret
 * @param instant Instantaneous regret
 * @param iteration Current iteration
 * @return Updated regret
 */
inline float update_regret_linear(
    float current,
    float instant,
    int iteration
) {
    // Linear weighting: regret_sum = Σ t * r_t
    // Equivalent to: new = old * (t-1)/t + instant
    float t = static_cast<float>(iteration);
    float discount = (t - 1.0f) / t;
    return current * discount + instant;
}

/**
 * Strategy contribution with DCFR-style discounting.
 *
 * @param strategy Current strategy
 * @param reach_prob Reach probability for this player
 * @param iteration Current iteration
 * @param gamma DCFR gamma parameter
 * @param defer_until Don't contribute until this iteration
 * @return Contribution to average strategy
 */
inline std::vector<float> strategy_contribution(
    const std::vector<float>& strategy,
    float reach_prob,
    int iteration,
    float gamma = 1.0f,
    int defer_until = 0
) {
    std::vector<float> contribution(strategy.size(), 0.0f);

    if (iteration < defer_until) {
        return contribution;  // Don't contribute yet
    }

    // Weight by reach probability and iteration discount
    float t = static_cast<float>(iteration);
    float weight = reach_prob * std::pow(t / (t + 1.0f), gamma);

    for (size_t i = 0; i < strategy.size(); ++i) {
        contribution[i] = weight * strategy[i];
    }

    return contribution;
}

/**
 * Compute average strategy from cumulative strategy sum.
 *
 * @param strategy_sum Cumulative weighted strategy
 * @return Normalized average strategy
 */
inline std::vector<float> compute_average_strategy(
    const std::vector<float>& strategy_sum
) {
    std::vector<float> avg(strategy_sum.size());
    float total = std::accumulate(strategy_sum.begin(), strategy_sum.end(), 0.0f);

    if (total > 0) {
        for (size_t i = 0; i < strategy_sum.size(); ++i) {
            avg[i] = strategy_sum[i] / total;
        }
    } else {
        float uniform = 1.0f / static_cast<float>(strategy_sum.size());
        std::fill(avg.begin(), avg.end(), uniform);
    }

    return avg;
}

/**
 * Configuration for the regret minimizer.
 */
struct RegretMinimizerConfig {
    RegretMatchingType type = RegretMatchingType::STANDARD;
    float tau = 1.0f;           // QRE temperature (only for SOFTMAX_QRE)
    float alpha = 1.5f;         // DCFR+ positive regret discount
    float beta = 0.0f;          // DCFR+ negative regret discount
    float gamma = 1.0f;         // Strategy contribution discount
    int defer_averaging = 1000; // Iterations before strategy averaging

    // Presets
    static RegretMinimizerConfig nash() {
        return RegretMinimizerConfig{RegretMatchingType::STANDARD};
    }

    static RegretMinimizerConfig qre(float temperature = 1.0f) {
        RegretMinimizerConfig cfg;
        cfg.type = RegretMatchingType::SOFTMAX_QRE;
        cfg.tau = temperature;
        return cfg;
    }

    static RegretMinimizerConfig cfr_plus() {
        RegretMinimizerConfig cfg;
        cfg.type = RegretMatchingType::STANDARD;
        cfg.alpha = 1.5f;
        cfg.beta = 0.0f;  // Zero negative regrets
        return cfg;
    }

    static RegretMinimizerConfig dcfr() {
        RegretMinimizerConfig cfg;
        cfg.type = RegretMatchingType::STANDARD;
        cfg.alpha = 1.5f;
        cfg.beta = 0.5f;
        cfg.gamma = 2.0f;
        return cfg;
    }
};

}  // namespace ares
