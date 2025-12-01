/**
 * ARES-HU Strategy Evaluator
 *
 * Fast evaluation of CFR strategies against various opponents.
 * Can run 100k+ hands in seconds for statistical significance.
 *
 * Usage:
 *   ./eval_cfr --strategy strategy.bin --opponent random --hands 100000
 */

#include "ares/cfr/dcfr_solver.hpp"
#include "ares/game/holdem_state.hpp"
#include "ares/core/hand_evaluator.hpp"
#include <iostream>
#include <string>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>

using namespace ares;

// Opponent types
enum class OpponentType {
    Random,      // Uniform random over discrete actions
    CallFold,    // Simple call/fold with fold threshold
    AlwaysCall,  // Never folds, always calls
    TightAgg     // Tight-aggressive baseline
};

class Evaluator {
public:
    Evaluator(const std::string& strategy_path, unsigned int seed = 42)
        : rng_(seed), seed_(seed) {
        // Load strategy
        std::cout << "Loading strategy from " << strategy_path << "...\n";
        solver_ = std::make_unique<DCFRPlusSolver>(DCFRPlusSolver::load(strategy_path));
        std::cout << "Loaded " << solver_->num_info_sets() << " information sets\n";
    }

    struct EvalResults {
        int hands_played = 0;
        int agent_wins = 0;
        int opponent_wins = 0;
        int ties = 0;
        double total_winnings_bb = 0.0;
        double elapsed_seconds = 0.0;
    };

    EvalResults evaluate(OpponentType opponent_type, int num_hands, float stack_size) {
        EvalResults results;
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

        auto start = std::chrono::high_resolution_clock::now();

        for (int hand = 0; hand < num_hands; ++hand) {
            // Alternate button position
            int button = hand % 2;

            // Play hand and get result
            float result = play_hand(opponent_type, stack_size, button);

            results.total_winnings_bb += result;

            if (result > 0.001f) {
                results.agent_wins++;
            } else if (result < -0.001f) {
                results.opponent_wins++;
            } else {
                results.ties++;
            }
            results.hands_played++;

            // Progress
            if ((hand + 1) % 10000 == 0) {
                std::cout << "  Played " << (hand + 1) << "/" << num_hands
                          << " hands (+" << std::fixed << std::setprecision(1)
                          << (results.total_winnings_bb * 100.0 / (hand + 1)) << " bb/100)\n";
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        results.elapsed_seconds = std::chrono::duration<double>(end - start).count();

        return results;
    }

private:
    std::unique_ptr<DCFRPlusSolver> solver_;
    std::mt19937 rng_;
    unsigned int seed_;

    float play_hand(OpponentType opponent_type, float stack_size, int /*button*/) {
        // Create config with stack size
        HoldemConfig config;
        config.starting_stack = stack_size;

        // Create initial state and deal cards
        HoldemState state = HoldemState::create_initial(config);

        // Deal hole cards
        if (state.is_chance_node()) {
            state = state.deal_random(rng_);
        }

        // Agent is player 0, opponent is player 1
        while (!state.is_terminal()) {
            // Check if we need to deal board cards
            if (state.is_chance_node()) {
                state = state.deal_random(rng_);
                continue;
            }

            int player = state.current_player();
            std::vector<Action> actions = state.get_legal_actions();

            if (actions.empty()) {
                // This shouldn't happen if is_terminal is false
                break;
            }

            Action chosen;
            if (player == 0) {
                // Agent acts using strategy
                chosen = choose_agent_action(state, actions);
            } else {
                // Opponent acts
                chosen = choose_opponent_action(state, actions, opponent_type);
            }

            state = state.apply_action(chosen);
        }

        // Return agent's winnings (player 0)
        return state.utility(0);
    }

    Action choose_agent_action(const HoldemState& state, const std::vector<Action>& actions) {
        uint64_t info_key = state.info_set_key(0);
        std::vector<float> strategy = solver_->get_avg_strategy(info_key);

        if (strategy.empty() || strategy.size() != actions.size()) {
            // Fallback: use uniform random
            std::uniform_int_distribution<int> dist(0, actions.size() - 1);
            return actions[dist(rng_)];
        }

        // Sample from strategy
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        float r = uniform(rng_);
        float cumsum = 0.0f;

        for (size_t i = 0; i < actions.size(); ++i) {
            cumsum += strategy[i];
            if (r < cumsum) {
                return actions[i];
            }
        }

        return actions.back();
    }

    Action choose_opponent_action(const HoldemState& state, const std::vector<Action>& actions,
                                  OpponentType opponent_type) {
        switch (opponent_type) {
            case OpponentType::Random:
                return choose_random_action(actions);

            case OpponentType::CallFold:
                return choose_callfold_action(actions, 0.4f);  // 40% fold

            case OpponentType::AlwaysCall:
                return choose_alwayscall_action(actions);

            case OpponentType::TightAgg:
                return choose_tight_action(state, actions);

            default:
                return choose_random_action(actions);
        }
    }

    Action choose_random_action(const std::vector<Action>& actions) {
        std::uniform_int_distribution<int> dist(0, actions.size() - 1);
        return actions[dist(rng_)];
    }

    Action choose_callfold_action(const std::vector<Action>& actions, float fold_prob) {
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

        // Find call and fold actions
        Action* call_action = nullptr;
        Action* fold_action = nullptr;
        Action* check_action = nullptr;

        for (auto& a : actions) {
            if (a.type == ActionType::Call) call_action = const_cast<Action*>(&a);
            if (a.type == ActionType::Fold) fold_action = const_cast<Action*>(&a);
            if (a.type == ActionType::Check) check_action = const_cast<Action*>(&a);
        }

        // If can check, always check
        if (check_action) return *check_action;

        // If facing bet, call/fold based on probability
        if (call_action && fold_action) {
            if (uniform(rng_) < fold_prob) {
                return *fold_action;
            }
            return *call_action;
        }

        // Fallback to first action
        return actions[0];
    }

    Action choose_alwayscall_action(const std::vector<Action>& actions) {
        // Check if possible
        for (const auto& a : actions) {
            if (a.type == ActionType::Check) return a;
        }
        // Otherwise call
        for (const auto& a : actions) {
            if (a.type == ActionType::Call) return a;
        }
        // Fallback
        return actions[0];
    }

    Action choose_tight_action(const HoldemState& state, const std::vector<Action>& actions) {
        // Simple tight-aggressive: fold most hands, bet/raise with good hands
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

        // Check if possible
        for (const auto& a : actions) {
            if (a.type == ActionType::Check) return a;
        }

        // 60% fold facing aggression
        Action* fold_action = nullptr;
        Action* call_action = nullptr;
        for (const auto& a : actions) {
            if (a.type == ActionType::Fold) fold_action = const_cast<Action*>(&a);
            if (a.type == ActionType::Call) call_action = const_cast<Action*>(&a);
        }

        if (fold_action && call_action) {
            if (uniform(rng_) < 0.6f) return *fold_action;
            return *call_action;
        }

        return actions[0];
    }
};

void print_usage(const char* program) {
    std::cout << "ARES-HU Strategy Evaluator\n\n"
              << "Usage: " << program << " [options]\n\n"
              << "Options:\n"
              << "  --strategy FILE     Strategy file to evaluate (required)\n"
              << "  --opponent TYPE     Opponent type: random, callfold, alwayscall, tight (default: random)\n"
              << "  --hands N           Number of hands to play (default: 100000)\n"
              << "  --stack N           Stack size in BB (default: 20)\n"
              << "  --seed N            Random seed (default: 42)\n"
              << "  --help              Show this help\n";
}

int main(int argc, char* argv[]) {
    // Default configuration
    std::string strategy_file;
    OpponentType opponent_type = OpponentType::Random;
    int num_hands = 100000;
    float stack_size = 20.0f;
    unsigned int seed = 42;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--strategy" && i + 1 < argc) {
            strategy_file = argv[++i];
        } else if (arg == "--opponent" && i + 1 < argc) {
            std::string opp = argv[++i];
            if (opp == "random") opponent_type = OpponentType::Random;
            else if (opp == "callfold") opponent_type = OpponentType::CallFold;
            else if (opp == "alwayscall") opponent_type = OpponentType::AlwaysCall;
            else if (opp == "tight") opponent_type = OpponentType::TightAgg;
            else {
                std::cerr << "Unknown opponent type: " << opp << "\n";
                return 1;
            }
        } else if (arg == "--hands" && i + 1 < argc) {
            num_hands = std::atoi(argv[++i]);
        } else if (arg == "--stack" && i + 1 < argc) {
            stack_size = std::atof(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::atoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (strategy_file.empty()) {
        std::cerr << "Error: --strategy is required\n";
        print_usage(argv[0]);
        return 1;
    }

    // Initialize hand evaluator
    std::cout << "Initializing hand evaluator...\n";
    HandEvaluator::initialize();

    // Create evaluator and run
    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════════╗\n"
              << "║            ARES-HU Strategy Evaluator                    ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Configuration:\n"
              << "  Strategy:     " << strategy_file << "\n"
              << "  Opponent:     ";

    switch (opponent_type) {
        case OpponentType::Random: std::cout << "Random\n"; break;
        case OpponentType::CallFold: std::cout << "Call/Fold (40% fold)\n"; break;
        case OpponentType::AlwaysCall: std::cout << "Always Call\n"; break;
        case OpponentType::TightAgg: std::cout << "Tight Aggressive\n"; break;
    }

    std::cout << "  Hands:        " << num_hands << "\n"
              << "  Stack:        " << stack_size << "bb\n"
              << "  Seed:         " << seed << "\n\n";

    try {
        Evaluator eval(strategy_file, seed);

        std::cout << "\nStarting evaluation...\n\n";

        auto results = eval.evaluate(opponent_type, num_hands, stack_size);

        // Print results
        double win_rate = results.total_winnings_bb * 100.0 / results.hands_played;
        double hands_per_sec = results.hands_played / results.elapsed_seconds;

        // Calculate standard error
        double variance = stack_size * 2;  // Rough estimate
        double std_error = std::sqrt(variance / results.hands_played) * 100;

        std::cout << "\n"
                  << "╔══════════════════════════════════════════════════════════╗\n"
                  << "║                  EVALUATION RESULTS                      ║\n"
                  << "╚══════════════════════════════════════════════════════════╝\n\n";

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Results:\n"
                  << "  Hands played:  " << results.hands_played << "\n"
                  << "  Agent wins:    " << results.agent_wins << " (" << (100.0 * results.agent_wins / results.hands_played) << "%)\n"
                  << "  Opponent wins: " << results.opponent_wins << " (" << (100.0 * results.opponent_wins / results.hands_played) << "%)\n"
                  << "  Ties:          " << results.ties << "\n\n"
                  << "  Total BB won:  " << (results.total_winnings_bb > 0 ? "+" : "") << results.total_winnings_bb << " bb\n"
                  << "  Win rate:      " << (win_rate > 0 ? "+" : "") << win_rate << " bb/100\n"
                  << "  Std error:     ~" << std_error << " bb/100\n\n"
                  << "Performance:\n"
                  << "  Time:          " << results.elapsed_seconds << "s\n"
                  << "  Speed:         " << (int)hands_per_sec << " hands/sec\n\n";

        // Rating
        std::cout << "Rating: ";
        if (win_rate < -50) {
            std::cout << "❌ LOSING - needs improvement\n";
        } else if (win_rate < 50) {
            std::cout << "➖ BREAKEVEN\n";
        } else if (win_rate < 150) {
            std::cout << "✅ WINNING\n";
        } else if (win_rate < 300) {
            std::cout << "✅ STRONG\n";
        } else {
            std::cout << "✅ EXPERT\n";
        }

        std::cout << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
