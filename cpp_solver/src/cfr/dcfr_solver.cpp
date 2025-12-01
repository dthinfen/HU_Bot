#include "ares/cfr/dcfr_solver.hpp"
#include "ares/cfr/regret_minimizer.hpp"
#include "ares/core/hand_evaluator.hpp"
#include "ares/belief/public_belief_state.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>

namespace ares {

// =============================================================================
// DCFRPlusSolver Implementation
// =============================================================================

DCFRPlusSolver::DCFRPlusSolver(const DCFRConfig& config)
    : config_(config)
{
    // Initialize per-thread RNGs
    std::random_device rd;
    thread_rngs_.reserve(config_.num_threads);
    for (int i = 0; i < config_.num_threads; ++i) {
        thread_rngs_.emplace_back(rd() + i);
    }

    // Initialize hand evaluator
    HandEvaluator::initialize();

    if (config_.verbose) {
        std::cout << "DCFR+ Solver initialized\n"
                  << "  Equilibrium: "
                  << (config_.equilibrium == EquilibriumType::QRE ? "QRE" : "Nash")
                  << "\n";
        if (config_.equilibrium == EquilibriumType::QRE) {
            std::cout << "  QRE tau: " << config_.qre_tau << "\n";
        }
        std::cout << "  Threads: " << config_.num_threads << "\n"
                  << "  Stack: " << config_.starting_stack << "bb\n"
                  << "  Iterations: " << config_.iterations << "\n";
    }
}

void DCFRPlusSolver::train() {
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int iter = 1; iter <= config_.iterations; ++iter) {
        current_iteration_ = iter;

        // Run one iteration of CFR
        iterate(iter);

        // Progress logging
        if (config_.verbose && (iter % 1000 == 0 || iter == 1)) {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                now - start_time).count();

            double iters_per_sec = iter / std::max(1.0, static_cast<double>(elapsed));
            int remaining = config_.iterations - iter;
            int eta_sec = static_cast<int>(remaining / iters_per_sec);

            std::cout << "\rIteration " << iter << "/" << config_.iterations
                      << " | Info sets: " << strategy_.size()
                      << " | Speed: " << static_cast<int>(iters_per_sec) << " iter/s"
                      << " | ETA: " << eta_sec / 60 << "m " << eta_sec % 60 << "s"
                      << std::flush;
        }

        // Checkpointing
        if (!config_.checkpoint_dir.empty() &&
            iter % config_.checkpoint_freq == 0) {
            std::string path = config_.checkpoint_dir + "/checkpoint_" +
                              std::to_string(iter) + ".bin";
            save_checkpoint(path);
        }

        // Callback
        if (config_.progress_callback) {
            // Could compute exploitability here but it's expensive
            config_.progress_callback(iter, 0.0f);
        }
    }

    if (config_.verbose) {
        std::cout << "\nTraining complete!\n"
                  << "  Total info sets: " << strategy_.size() << "\n"
                  << "  Nodes visited: " << nodes_visited_ << "\n";
    }
}

void DCFRPlusSolver::iterate(int iteration) {
    // Create initial game state
    HoldemState::Config game_config;
    game_config.starting_stack = config_.starting_stack;
    HoldemState root = HoldemState::create_initial(game_config);

    // Traverse for both players (external sampling)
    // In external sampling, we traverse once for each player as the "traverser"
    // and sample opponent's actions while computing regrets for traverser

    for (int traverser = 0; traverser < 2; ++traverser) {
        // Deal random cards
        HoldemState state = root.deal_random(thread_rngs_[0]);

        // Run CFR traversal
        cfr_traverse(state, traverser, 1.0f, 1.0f, 0);
    }
}

float DCFRPlusSolver::cfr_traverse(
    const HoldemState& state,
    int traverser,
    float p0,
    float p1,
    int thread_id
) {
    nodes_visited_++;

    // Terminal node - return utility
    if (state.is_terminal()) {
        return state.utility(traverser);
    }

    // Chance node - deal cards and continue
    if (state.is_chance_node()) {
        HoldemState dealt = state.deal_random(thread_rngs_[thread_id]);
        return cfr_traverse(dealt, traverser, p0, p1, thread_id);
    }

    int player = state.current_player();
    std::vector<Action> actions = state.get_legal_actions();
    int num_actions = static_cast<int>(actions.size());

    if (num_actions == 0) {
        // No legal actions - shouldn't happen but handle gracefully
        return 0.0f;
    }

    // Get or create info set
    uint64_t info_key = abstract_info_set(state, player);
    InfoSetData& info_set = get_info_set(info_key, num_actions);

    // Get current strategy from regrets
    std::vector<float> strategy = regret_match(info_set);

    float reach_prob = (player == 0) ? p0 : p1;

    // External Sampling MCCFR:
    // - Traversing player: explore ALL actions
    // - Opponent: SAMPLE one action according to strategy
    if (player == traverser) {
        // Explore all actions for traversing player
        std::vector<float> action_values(num_actions, 0.0f);
        float node_value = 0.0f;

        for (int a = 0; a < num_actions; ++a) {
            HoldemState next_state = state.apply_action(actions[a]);

            float new_p0 = (player == 0) ? p0 * strategy[a] : p0;
            float new_p1 = (player == 1) ? p1 * strategy[a] : p1;

            action_values[a] = cfr_traverse(next_state, traverser, new_p0, new_p1, thread_id);
            node_value += strategy[a] * action_values[a];
        }

        // Compute instantaneous regrets
        std::vector<float> instant_regrets(num_actions);
        for (int a = 0; a < num_actions; ++a) {
            instant_regrets[a] = action_values[a] - node_value;
        }

        // Update cumulative regrets with DCFR+ discounting
        update_regrets(info_set, instant_regrets, current_iteration_);

        // Update strategy sum (for averaging)
        update_strategy(info_set, strategy, reach_prob, current_iteration_);

        info_set.visits++;
        info_sets_updated_++;

        return node_value;
    } else {
        // Sample one action for opponent (external sampling)
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(thread_rngs_[thread_id]);

        int sampled_action = 0;
        float cumulative = 0.0f;
        for (int a = 0; a < num_actions; ++a) {
            cumulative += strategy[a];
            if (r <= cumulative) {
                sampled_action = a;
                break;
            }
        }

        HoldemState next_state = state.apply_action(actions[sampled_action]);

        float new_p0 = (player == 0) ? p0 * strategy[sampled_action] : p0;
        float new_p1 = (player == 1) ? p1 * strategy[sampled_action] : p1;

        return cfr_traverse(next_state, traverser, new_p0, new_p1, thread_id);
    }
}

std::vector<float> DCFRPlusSolver::regret_match(const InfoSetData& data) const {
    RegretMatchingType rm_type = (config_.equilibrium == EquilibriumType::QRE)
                                  ? RegretMatchingType::SOFTMAX_QRE
                                  : RegretMatchingType::STANDARD;

    return ares::regret_match(data.regret_sum, rm_type, config_.qre_tau);
}

void DCFRPlusSolver::update_regrets(
    InfoSetData& data,
    const std::vector<float>& instant_regrets,
    int iteration
) {
    for (size_t a = 0; a < data.regret_sum.size(); ++a) {
        // Apply DCFR+ discounting
        float current = data.regret_sum[a];

        // Discount existing regret
        if (current > 0) {
            current *= positive_discount(iteration);
        } else if (current < 0) {
            if (config_.beta > 0) {
                current *= negative_discount(iteration);
            } else {
                // CFR+ behavior: zero out negative regrets
                current = 0.0f;
            }
        }

        // Add instantaneous regret
        data.regret_sum[a] = current + instant_regrets[a];
    }
}

void DCFRPlusSolver::update_strategy(
    InfoSetData& data,
    const std::vector<float>& strategy,
    float reach_prob,
    int iteration
) {
    // Don't contribute to average strategy during warm-up
    if (iteration < config_.defer_averaging) {
        return;
    }

    // Strategy contribution with discounting
    float weight = reach_prob * strategy_discount(iteration);

    for (size_t a = 0; a < data.strategy_sum.size(); ++a) {
        data.strategy_sum[a] += weight * strategy[a];
    }
}

float DCFRPlusSolver::positive_discount(int t) const {
    // t^α / (t^α + 1)
    float t_alpha = std::pow(static_cast<float>(t), config_.alpha);
    return t_alpha / (t_alpha + 1.0f);
}

float DCFRPlusSolver::negative_discount(int t) const {
    // t^β / (t^β + 1)
    float t_beta = std::pow(static_cast<float>(t), config_.beta);
    return t_beta / (t_beta + 1.0f);
}

float DCFRPlusSolver::strategy_discount(int t) const {
    // (t / (t + 1))^γ
    float ratio = static_cast<float>(t) / (static_cast<float>(t) + 1.0f);
    return std::pow(ratio, config_.gamma);
}

InfoSetData& DCFRPlusSolver::get_info_set(uint64_t key, int num_actions) {
    std::lock_guard<std::mutex> lock(strategy_mutex_);

    auto it = strategy_.find(key);
    if (it == strategy_.end()) {
        auto result = strategy_.emplace(key, InfoSetData(num_actions));
        return result.first->second;
    }
    return it->second;
}

uint64_t DCFRPlusSolver::abstract_info_set(const HoldemState& state, int player) const {
    // For now, no abstraction - use raw info set key
    // TODO: Add card abstraction support
    return state.info_set_key(player);
}

std::vector<Action> DCFRPlusSolver::abstract_actions(const HoldemState& state) const {
    // For now, no action abstraction - use all legal actions
    // TODO: Add betting abstraction support
    return state.get_legal_actions();
}

std::vector<float> DCFRPlusSolver::get_avg_strategy(uint64_t info_set_key) const {
    auto it = strategy_.find(info_set_key);
    if (it == strategy_.end()) {
        return {};
    }

    return compute_average_strategy(it->second.strategy_sum);
}

double DCFRPlusSolver::exploitability() const {
    // TODO: Implement best-response computation
    // This is expensive (requires full tree traversal for each player)
    return -1.0;
}

// =============================================================================
// Serialization
// =============================================================================

void DCFRPlusSolver::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    // Write magic number and version
    uint32_t magic = 0x41524553;  // "ARES"
    uint32_t version = 1;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Write config
    uint8_t eq_type = static_cast<uint8_t>(config_.equilibrium);
    out.write(reinterpret_cast<const char*>(&eq_type), sizeof(eq_type));
    out.write(reinterpret_cast<const char*>(&config_.qre_tau), sizeof(config_.qre_tau));
    out.write(reinterpret_cast<const char*>(&config_.starting_stack), sizeof(config_.starting_stack));

    // Write strategy map size
    uint64_t size = strategy_.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));

    // Write each info set
    for (const auto& [key, data] : strategy_) {
        out.write(reinterpret_cast<const char*>(&key), sizeof(key));

        uint32_t num_actions = static_cast<uint32_t>(data.strategy_sum.size());
        out.write(reinterpret_cast<const char*>(&num_actions), sizeof(num_actions));

        // Write only strategy_sum (average strategy), not regrets
        out.write(reinterpret_cast<const char*>(data.strategy_sum.data()),
                  num_actions * sizeof(float));
    }

    if (config_.verbose) {
        std::cout << "Saved strategy to " << path << " (" << size << " info sets)\n";
    }
}

DCFRPlusSolver DCFRPlusSolver::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }

    // Read and verify magic number
    uint32_t magic, version;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != 0x41524553) {
        throw std::runtime_error("Invalid file format");
    }

    // Read config
    DCFRConfig config;
    uint8_t eq_type;
    in.read(reinterpret_cast<char*>(&eq_type), sizeof(eq_type));
    config.equilibrium = static_cast<EquilibriumType>(eq_type);
    in.read(reinterpret_cast<char*>(&config.qre_tau), sizeof(config.qre_tau));
    in.read(reinterpret_cast<char*>(&config.starting_stack), sizeof(config.starting_stack));

    DCFRPlusSolver solver(config);

    // Read strategy map
    uint64_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));

    for (uint64_t i = 0; i < size; ++i) {
        uint64_t key;
        in.read(reinterpret_cast<char*>(&key), sizeof(key));

        uint32_t num_actions;
        in.read(reinterpret_cast<char*>(&num_actions), sizeof(num_actions));

        InfoSetData data(num_actions);
        in.read(reinterpret_cast<char*>(data.strategy_sum.data()),
                num_actions * sizeof(float));

        solver.strategy_[key] = std::move(data);
    }

    return solver;
}

void DCFRPlusSolver::save_checkpoint(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open checkpoint file: " + path);
    }

    // Write magic and version
    uint32_t magic = 0x43484B50;  // "CHKP"
    uint32_t version = 1;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Write iteration
    int iter = current_iteration_.load();
    out.write(reinterpret_cast<const char*>(&iter), sizeof(iter));

    // Write full config
    uint8_t eq_type = static_cast<uint8_t>(config_.equilibrium);
    out.write(reinterpret_cast<const char*>(&eq_type), sizeof(eq_type));
    out.write(reinterpret_cast<const char*>(&config_.qre_tau), sizeof(config_.qre_tau));
    out.write(reinterpret_cast<const char*>(&config_.alpha), sizeof(config_.alpha));
    out.write(reinterpret_cast<const char*>(&config_.beta), sizeof(config_.beta));
    out.write(reinterpret_cast<const char*>(&config_.gamma), sizeof(config_.gamma));
    out.write(reinterpret_cast<const char*>(&config_.starting_stack), sizeof(config_.starting_stack));

    // Write strategy map size
    uint64_t size = strategy_.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));

    // Write each info set with regrets
    for (const auto& [key, data] : strategy_) {
        out.write(reinterpret_cast<const char*>(&key), sizeof(key));

        uint32_t num_actions = static_cast<uint32_t>(data.regret_sum.size());
        out.write(reinterpret_cast<const char*>(&num_actions), sizeof(num_actions));

        // Write both regret_sum and strategy_sum
        out.write(reinterpret_cast<const char*>(data.regret_sum.data()),
                  num_actions * sizeof(float));
        out.write(reinterpret_cast<const char*>(data.strategy_sum.data()),
                  num_actions * sizeof(float));

        uint32_t visits = data.visits.load();
        out.write(reinterpret_cast<const char*>(&visits), sizeof(visits));
    }

    if (config_.verbose) {
        std::cout << "Saved checkpoint to " << path << " (iter " << iter << ")\n";
    }
}

void DCFRPlusSolver::load_checkpoint(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open checkpoint file: " + path);
    }

    // Read and verify magic
    uint32_t magic, version;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != 0x43484B50) {
        throw std::runtime_error("Invalid checkpoint format");
    }

    // Read iteration
    int iter;
    in.read(reinterpret_cast<char*>(&iter), sizeof(iter));
    current_iteration_ = iter;

    // Read config (skip for now, use current config)
    uint8_t eq_type;
    float tau, alpha, beta, gamma, stack;
    in.read(reinterpret_cast<char*>(&eq_type), sizeof(eq_type));
    in.read(reinterpret_cast<char*>(&tau), sizeof(tau));
    in.read(reinterpret_cast<char*>(&alpha), sizeof(alpha));
    in.read(reinterpret_cast<char*>(&beta), sizeof(beta));
    in.read(reinterpret_cast<char*>(&gamma), sizeof(gamma));
    in.read(reinterpret_cast<char*>(&stack), sizeof(stack));

    // Read strategy map
    uint64_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));

    strategy_.clear();
    for (uint64_t i = 0; i < size; ++i) {
        uint64_t key;
        in.read(reinterpret_cast<char*>(&key), sizeof(key));

        uint32_t num_actions;
        in.read(reinterpret_cast<char*>(&num_actions), sizeof(num_actions));

        InfoSetData data(num_actions);
        in.read(reinterpret_cast<char*>(data.regret_sum.data()),
                num_actions * sizeof(float));
        in.read(reinterpret_cast<char*>(data.strategy_sum.data()),
                num_actions * sizeof(float));

        uint32_t visits;
        in.read(reinterpret_cast<char*>(&visits), sizeof(visits));
        data.visits = visits;

        strategy_[key] = std::move(data);
    }

    if (config_.verbose) {
        std::cout << "Loaded checkpoint from " << path
                  << " (iter " << iter << ", " << size << " info sets)\n";
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

float compute_immediate_regret(const StrategyMap& strategy) {
    float total_regret = 0.0f;

    for (const auto& [key, data] : strategy) {
        // Sum positive regrets
        for (float r : data.regret_sum) {
            if (r > 0) {
                total_regret += r;
            }
        }
    }

    return total_regret;
}

// =============================================================================
// Training Data Export
// =============================================================================

void DCFRPlusSolver::export_training_data(
    const std::string& path,
    int num_samples
) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    // Header: magic, version, num_samples, pbs_dim
    uint32_t magic = 0x54524E44;  // "TRND"
    uint32_t version = 1;
    uint32_t samples = static_cast<uint32_t>(num_samples);
    uint32_t pbs_dim = 3040;  // PBS encoding dimension

    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&samples), sizeof(samples));
    out.write(reinterpret_cast<const char*>(&pbs_dim), sizeof(pbs_dim));

    // Generate training samples
    std::mt19937 rng(std::random_device{}());
    std::vector<float> pbs_buffer(pbs_dim);

    int samples_written = 0;
    int attempts = 0;
    int max_attempts = num_samples * 100;

    HoldemState::Config game_config;
    game_config.starting_stack = config_.starting_stack;

    while (samples_written < num_samples && attempts < max_attempts) {
        attempts++;

        // Create random game state
        HoldemState state = HoldemState::create_initial(game_config);
        state = state.deal_random(rng);

        // Play random actions to reach different game states
        std::uniform_int_distribution<int> depth_dist(0, 12);
        int target_depth = depth_dist(rng);

        for (int d = 0; d < target_depth && !state.is_terminal(); ++d) {
            if (state.is_chance_node()) {
                state = state.deal_random(rng);
                continue;
            }

            auto actions = state.get_legal_actions();
            if (actions.empty()) break;

            // Choose action according to learned strategy (if available)
            uint64_t info_key = state.info_set_key(state.current_player());
            auto it = strategy_.find(info_key);

            int action_idx = 0;
            if (it != strategy_.end() && !it->second.strategy_sum.empty()) {
                // Sample according to strategy
                std::vector<float> strat = compute_average_strategy(it->second.strategy_sum);
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                float r = dist(rng);
                float cumsum = 0.0f;
                for (size_t a = 0; a < strat.size() && a < actions.size(); ++a) {
                    cumsum += strat[a];
                    if (r <= cumsum) {
                        action_idx = a;
                        break;
                    }
                }
            } else {
                // Random action
                std::uniform_int_distribution<int> action_dist(0, static_cast<int>(actions.size()) - 1);
                action_idx = action_dist(rng);
            }

            action_idx = std::min(action_idx, static_cast<int>(actions.size()) - 1);
            state = state.apply_action(actions[action_idx]);
        }

        // Skip terminal states
        if (state.is_terminal() || state.is_chance_node()) {
            continue;
        }

        // Create simple state encoding
        // (Future: use proper PBS encoding)
        std::vector<float> encoding;
        encoding.reserve(100);

        // Encode pot, stacks, bets
        encoding.push_back(state.pot() / config_.starting_stack);
        encoding.push_back(state.stack(0) / config_.starting_stack);
        encoding.push_back(state.stack(1) / config_.starting_stack);
        encoding.push_back(state.bet(0) / config_.starting_stack);
        encoding.push_back(state.bet(1) / config_.starting_stack);

        // Street one-hot
        int street_int = static_cast<int>(state.street());
        for (int s = 0; s < 4; ++s) {
            encoding.push_back(s == street_int ? 1.0f : 0.0f);
        }

        // Current player
        encoding.push_back(static_cast<float>(state.current_player()));

        // Board cards (6 values per card: rank one-hot simplified + suit one-hot)
        for (int i = 0; i < 5; ++i) {
            Card c = state.board()[i];
            if (c != NO_CARD) {
                encoding.push_back(get_rank_int(c) / 12.0f);  // Normalized rank
                encoding.push_back(get_suit_int(c) / 3.0f);   // Normalized suit
            } else {
                encoding.push_back(0.0f);
                encoding.push_back(0.0f);
            }
        }

        // Compute value for current player using CFR traversal
        // For simplicity, estimate value as average over several rollouts
        float value_p0 = 0.0f;
        float value_p1 = 0.0f;
        int num_rollouts = 10;

        for (int r = 0; r < num_rollouts; ++r) {
            HoldemState rollout_state = state;
            while (!rollout_state.is_terminal()) {
                if (rollout_state.is_chance_node()) {
                    rollout_state = rollout_state.deal_random(rng);
                    continue;
                }

                auto actions = rollout_state.get_legal_actions();
                if (actions.empty()) break;

                // Use strategy if available
                uint64_t info_key = rollout_state.info_set_key(rollout_state.current_player());
                auto it = strategy_.find(info_key);

                int action_idx = 0;
                if (it != strategy_.end() && !it->second.strategy_sum.empty()) {
                    std::vector<float> strat = compute_average_strategy(it->second.strategy_sum);
                    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                    float r_val = dist(rng);
                    float cumsum = 0.0f;
                    for (size_t a = 0; a < strat.size() && a < actions.size(); ++a) {
                        cumsum += strat[a];
                        if (r_val <= cumsum) {
                            action_idx = a;
                            break;
                        }
                    }
                } else {
                    std::uniform_int_distribution<int> action_dist(0, static_cast<int>(actions.size()) - 1);
                    action_idx = action_dist(rng);
                }

                action_idx = std::min(action_idx, static_cast<int>(actions.size()) - 1);
                rollout_state = rollout_state.apply_action(actions[action_idx]);
            }

            if (rollout_state.is_terminal()) {
                value_p0 += rollout_state.utility(0);
                value_p1 += rollout_state.utility(1);
            }
        }

        value_p0 /= num_rollouts;
        value_p1 /= num_rollouts;

        // Write sample: PBS encoding + values
        // Pad/truncate encoding to pbs_dim
        pbs_buffer.assign(pbs_dim, 0.0f);
        for (size_t i = 0; i < std::min(encoding.size(), static_cast<size_t>(pbs_dim)); ++i) {
            pbs_buffer[i] = encoding[i];
        }

        out.write(reinterpret_cast<const char*>(pbs_buffer.data()),
                  pbs_dim * sizeof(float));
        out.write(reinterpret_cast<const char*>(&value_p0), sizeof(float));
        out.write(reinterpret_cast<const char*>(&value_p1), sizeof(float));

        samples_written++;

        if (config_.verbose && samples_written % 10000 == 0) {
            std::cout << "Exported " << samples_written << "/" << num_samples
                      << " training samples\r" << std::flush;
        }
    }

    if (config_.verbose) {
        std::cout << "\nExported " << samples_written << " training samples to "
                  << path << "\n";
    }
}

}  // namespace ares
