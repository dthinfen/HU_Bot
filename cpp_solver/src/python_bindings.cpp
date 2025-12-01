/**
 * Python bindings for ARES-HU C++ Solver
 *
 * Uses pybind11 to expose C++ solver to Python.
 * C++ owns all 48M+ infosets, Python only sends high-level commands.
 *
 * Build:
 *   pip install pybind11
 *   c++ -O3 -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) \
 *       python_bindings.cpp -o ares_solver$(python3-config --extension-suffix)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "ares/cfr/dcfr_solver.hpp"
#include "ares/game/holdem_state.hpp"
#include "ares/core/hand_evaluator.hpp"
#include "ares/search/realtime_search.hpp"

#ifdef ARES_USE_LIBTORCH
#include "ares/neural/neural_evaluator.hpp"
#endif

#include <memory>
#include <random>

namespace py = pybind11;
using namespace ares;

/**
 * Wrapper class for Python - provides high-level interface to C++ solver.
 */
class AresSolver {
public:
    AresSolver() : rng_(42) {
        // Initialize hand evaluator on first use
        static bool initialized = false;
        if (!initialized) {
            HandEvaluator::initialize();
            initialized = true;
        }
    }

#ifdef ARES_USE_LIBTORCH
    /**
     * Load neural network model for leaf evaluation.
     *
     * Args:
     *     model_path: Path to TorchScript model file
     *
     * Returns:
     *     True if model loaded successfully
     */
    bool load_neural_model(const std::string& model_path) {
        try {
            neural_eval_ = std::make_unique<NeuralEvaluator>(model_path);
            return neural_eval_->is_loaded();
        } catch (const std::exception& e) {
            std::cerr << "Failed to load neural model: " << e.what() << std::endl;
            return false;
        }
    }

    /**
     * Check if neural model is loaded.
     */
    bool has_neural_model() const {
        return neural_eval_ && neural_eval_->is_loaded();
    }
#endif

    /**
     * Train a new CFR solver.
     *
     * Args:
     *     iterations: Number of CFR iterations
     *     stack_size: Starting stack in big blinds
     *     qre_tau: QRE temperature (0 = Nash, 1.0 = recommended)
     *     threads: Number of threads
     *     verbose: Print progress
     *
     * Returns:
     *     Number of information sets created
     */
    size_t train(
        int iterations,
        float stack_size = 20.0f,
        float qre_tau = 1.0f,
        int threads = 8,
        bool verbose = true
    ) {
        DCFRConfig config;
        config.iterations = iterations;
        config.starting_stack = stack_size;
        config.qre_tau = qre_tau;
        config.equilibrium = (qre_tau > 0) ? EquilibriumType::QRE : EquilibriumType::NASH;
        config.num_threads = threads;
        config.verbose = verbose;

        solver_ = std::make_unique<DCFRPlusSolver>(config);
        solver_->train();

        return solver_->num_info_sets();
    }

    /**
     * Load a trained solver from disk.
     *
     * Args:
     *     path: Path to binary strategy file
     *
     * Returns:
     *     Number of information sets loaded
     */
    size_t load(const std::string& path) {
        solver_ = std::make_unique<DCFRPlusSolver>(DCFRPlusSolver::load(path));
        return solver_->num_info_sets();
    }

    /**
     * Save solver to disk.
     *
     * Args:
     *     path: Output path for binary strategy file
     */
    void save(const std::string& path) {
        if (!solver_) {
            throw std::runtime_error("No solver loaded");
        }
        solver_->save(path);
    }

    /**
     * Get number of information sets in the solver.
     */
    size_t num_info_sets() const {
        return solver_ ? solver_->num_info_sets() : 0;
    }

    /**
     * Get strategy for a specific game state.
     *
     * Args:
     *     hero_cards: List of 2 card strings (e.g., ["Ah", "Ks"])
     *     board_cards: List of 0-5 card strings
     *     action_history: List of action strings (e.g., ["raise", "call"])
     *     hero_stack: Hero's remaining stack in bb
     *     villain_stack: Villain's remaining stack in bb
     *     pot: Current pot in bb
     *     to_call: Amount to call in bb
     *
     * Returns:
     *     Dict mapping action names to probabilities
     */
    py::dict get_strategy(
        const std::vector<std::string>& hero_cards,
        const std::vector<std::string>& board_cards,
        float hero_stack,
        float villain_stack,
        float pot
    ) {
        if (!solver_) {
            throw std::runtime_error("No solver loaded");
        }

        // Build game state
        HoldemConfig config;
        config.starting_stack = hero_stack + villain_stack;  // Approximate

        HoldemState state = HoldemState::create_initial(config);

        // Deal hole cards
        HoleCards hero_hole(string_to_card(hero_cards[0]), string_to_card(hero_cards[1]));

        // Villain needs valid cards (doesn't affect hero's info set key, but needed for valid state)
        // Use cards that don't conflict with hero's cards
        Card v1 = index_to_card(0);  // 2c
        Card v2 = index_to_card(1);  // 3c

        // Avoid conflicts with hero's cards
        int hero_idx0 = card_to_index(hero_hole[0]);
        int hero_idx1 = card_to_index(hero_hole[1]);
        int v_idx = 0;

        while (v_idx == hero_idx0 || v_idx == hero_idx1) v_idx++;
        v1 = index_to_card(v_idx);
        v_idx++;
        while (v_idx == hero_idx0 || v_idx == hero_idx1) v_idx++;
        v2 = index_to_card(v_idx);

        HoleCards villain_hole(v1, v2);
        state = state.deal_hole_cards(hero_hole, villain_hole);

        // Deal board
        if (!board_cards.empty()) {
            std::vector<Card> board;
            for (const auto& s : board_cards) {
                board.push_back(string_to_card(s));
            }
            state = state.deal_board(board);
        }

        // Get info set key and strategy
        uint64_t info_key = state.info_set_key(0);  // Hero is player 0
        std::vector<float> strategy = solver_->get_avg_strategy(info_key);

        // Get legal actions
        std::vector<Action> actions = state.get_legal_actions();

        // If strategy not found or wrong size, return uniform with debug info
        if (strategy.empty() || strategy.size() != actions.size()) {
            py::dict result;
            float uniform = 1.0f / static_cast<float>(actions.size());
            for (const auto& action : actions) {
                result[action_to_string(action).c_str()] = uniform;
            }
            result["_debug_key"] = info_key;
            result["_debug_found"] = !strategy.empty();
            result["_debug_canonical"] = hero_hole.canonical_index();
            return result;
        }

        // Build result dict
        py::dict result;
        for (size_t i = 0; i < actions.size() && i < strategy.size(); ++i) {
            result[action_to_string(actions[i]).c_str()] = strategy[i];
        }

        return result;
    }

    /**
     * Evaluate strategy against an opponent.
     *
     * Args:
     *     opponent: "random", "callfold", "alwayscall"
     *     num_hands: Number of hands to play
     *     stack_size: Starting stack in bb
     *     seed: Random seed
     *
     * Returns:
     *     Dict with evaluation results
     */
    py::dict evaluate(
        const std::string& opponent,
        int num_hands = 100000,
        float stack_size = 20.0f,
        unsigned int seed = 42
    ) {
        if (!solver_) {
            throw std::runtime_error("No solver loaded");
        }

        rng_.seed(seed);

        double total_winnings = 0.0;
        int agent_wins = 0;
        int opponent_wins = 0;
        int ties = 0;

        for (int hand = 0; hand < num_hands; ++hand) {
            float result = play_hand(opponent, stack_size);
            total_winnings += result;

            if (result > 0.001f) agent_wins++;
            else if (result < -0.001f) opponent_wins++;
            else ties++;
        }

        double win_rate = total_winnings * 100.0 / num_hands;

        py::dict results;
        results["hands_played"] = num_hands;
        results["total_bb_won"] = total_winnings;
        results["win_rate_bb_per_100"] = win_rate;
        results["agent_wins"] = agent_wins;
        results["opponent_wins"] = opponent_wins;
        results["ties"] = ties;

        return results;
    }

    /**
     * Export training data for neural network training.
     *
     * Args:
     *     path: Output path for training data
     *     num_samples: Number of samples to export
     */
    void export_training_data(const std::string& path, int num_samples = 100000) {
        if (!solver_) {
            throw std::runtime_error("No solver loaded");
        }
        solver_->export_training_data(path, num_samples);
    }

    /**
     * Run real-time CFR search from a specific game state.
     *
     * This is the key method for playing at arbitrary stack depths.
     * Instead of looking up a pre-computed strategy, it runs CFR
     * iterations from the current state.
     *
     * Args:
     *     hero_cards: Hero's hole cards (e.g., ["Ah", "Ks"])
     *     board_cards: Board cards (0-5 cards)
     *     pot: Current pot in bb
     *     hero_stack: Hero's remaining stack in bb
     *     villain_stack: Villain's remaining stack in bb
     *     hero_bet: Hero's current bet this street
     *     villain_bet: Villain's current bet this street
     *     street: 0=preflop, 1=flop, 2=turn, 3=river
     *     hero_position: 0=OOP (BB), 1=IP (BTN)
     *     iterations: Number of CFR iterations (default 500)
     *     time_limit: Time limit in seconds (default 5.0)
     *     qre_tau: QRE temperature (default 1.0)
     *
     * Returns:
     *     Dict with 'actions' and 'strategy' (probabilities)
     */
    py::dict search(
        const std::vector<std::string>& hero_cards,
        const std::vector<std::string>& board_cards,
        float pot,
        float hero_stack,
        float villain_stack,
        float hero_bet,
        float villain_bet,
        int street,
        int hero_position,
        int iterations = 500,
        float time_limit = 5.0f,
        float qre_tau = 1.0f
    ) {
        // Suppress unused parameter warning
        (void)street;

        // Parse hero hole cards
        HoleCards hero_hole(string_to_card(hero_cards[0]), string_to_card(hero_cards[1]));

        // Parse board cards
        std::vector<Card> board;
        for (const auto& s : board_cards) {
            board.push_back(string_to_card(s));
        }

        // Map hero_position to internal player index:
        // hero_position=0 (BB/OOP) -> internal player 1 (BIG_BLIND)
        // hero_position=1 (BTN/IP) -> internal player 0 (BUTTON)
        //
        // For search, hero is always the current player since we're deciding hero's action
        int current_player = (hero_position == 0) ? 1 : 0;

        // Calculate starting stack for config (total chips in play)
        float total_chips = hero_stack + villain_stack + hero_bet + villain_bet + pot;
        float starting_stack = std::max(total_chips / 2.0f, 20.0f);

        HoldemConfig config;
        config.starting_stack = starting_stack;
        config.big_blind = 1.0f;
        config.small_blind = 0.5f;

        // Create state from current position
        HoldemState state = HoldemState::create_from_position(
            hero_hole,
            hero_position,
            board,
            pot,
            hero_stack,
            villain_stack,
            hero_bet,
            villain_bet,
            current_player,
            config
        );

        // Configure search
        SearchConfig search_config;
        search_config.iterations = iterations;
        search_config.time_limit = time_limit;
        search_config.qre_tau = qre_tau;
        search_config.cfr_plus = true;

#ifdef ARES_USE_LIBTORCH
        // Use neural evaluator if loaded
        if (neural_eval_ && neural_eval_->is_loaded()) {
            search_config.neural_eval = [this](const std::vector<float>& features) {
                return neural_eval_->evaluate(features);
            };
        }
#endif

        // Run search
        RealtimeSearch search_engine(search_config);
        SearchResult result = search_engine.search(state);

        // Build result dict
        py::dict output;

        py::list action_list;
        py::list prob_list;

        for (size_t i = 0; i < result.actions.size(); ++i) {
            action_list.append(action_to_string(result.actions[i]));
            if (i < result.strategy.size()) {
                prob_list.append(result.strategy[i]);
            } else {
                prob_list.append(1.0f / result.actions.size());
            }
        }

        output["actions"] = action_list;
        output["strategy"] = prob_list;
        output["best_action"] = action_to_string(result.best_action);
        output["iterations"] = result.iterations;
        output["nodes"] = result.nodes_visited;
        output["time"] = result.time_seconds;

        return output;
    }

private:
    std::unique_ptr<DCFRPlusSolver> solver_;
    std::mt19937 rng_;

#ifdef ARES_USE_LIBTORCH
    std::unique_ptr<NeuralEvaluator> neural_eval_;
#endif

    Card string_to_card(const std::string& s) {
        static const std::string ranks = "23456789TJQKA";
        static const std::string suits = "cdhs";

        if (s.size() < 2) {
            return make_card(0, 0);  // Default to 2c
        }

        size_t rank_pos = ranks.find(toupper(s[0]));
        size_t suit_pos = suits.find(tolower(s[1]));

        int rank = (rank_pos != std::string::npos) ? static_cast<int>(rank_pos) : 0;
        int suit = (suit_pos != std::string::npos) ? static_cast<int>(suit_pos) : 0;

        return make_card(rank, suit);
    }

    std::string action_to_string(const Action& action) {
        switch (action.type) {
            case ActionType::Fold: return "fold";
            case ActionType::Check: return "check";
            case ActionType::Call: return "call";
            case ActionType::Bet: return "bet_" + std::to_string(action.amount);
            case ActionType::Raise: return "raise_" + std::to_string(action.amount);
            case ActionType::AllIn: return "allin_" + std::to_string(action.amount);
            default: return "unknown";
        }
    }

    float play_hand(const std::string& opponent_type, float stack_size) {
        HoldemConfig config;
        config.starting_stack = stack_size;

        HoldemState state = HoldemState::create_initial(config);

        // Deal hole cards
        if (state.is_chance_node()) {
            state = state.deal_random(rng_);
        }

        while (!state.is_terminal()) {
            if (state.is_chance_node()) {
                state = state.deal_random(rng_);
                continue;
            }

            int player = state.current_player();
            std::vector<Action> actions = state.get_legal_actions();

            if (actions.empty()) break;

            Action chosen;
            if (player == 0) {
                // Agent uses CFR strategy
                uint64_t info_key = state.info_set_key(0);
                std::vector<float> strategy = solver_->get_avg_strategy(info_key);

                if (strategy.empty() || strategy.size() != actions.size()) {
                    // Fallback to random
                    std::uniform_int_distribution<int> dist(0, actions.size() - 1);
                    chosen = actions[dist(rng_)];
                } else {
                    // Sample from strategy
                    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
                    float r = uniform(rng_);
                    float cumsum = 0.0f;

                    for (size_t i = 0; i < actions.size(); ++i) {
                        cumsum += strategy[i];
                        if (r < cumsum) {
                            chosen = actions[i];
                            break;
                        }
                    }
                    if (cumsum == 0) chosen = actions.back();
                }
            } else {
                // Opponent
                chosen = choose_opponent_action(actions, opponent_type);
            }

            state = state.apply_action(chosen);
        }

        return state.utility(0);
    }

    Action choose_opponent_action(const std::vector<Action>& actions, const std::string& type) {
        if (type == "random") {
            std::uniform_int_distribution<int> dist(0, actions.size() - 1);
            return actions[dist(rng_)];
        }
        else if (type == "callfold") {
            // Check if possible
            for (const auto& a : actions) {
                if (a.type == ActionType::Check) return a;
            }
            // Call/fold with 40% fold
            std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
            Action* call_action = nullptr;
            Action* fold_action = nullptr;
            for (auto& a : actions) {
                if (a.type == ActionType::Call) call_action = const_cast<Action*>(&a);
                if (a.type == ActionType::Fold) fold_action = const_cast<Action*>(&a);
            }
            if (call_action && fold_action) {
                return (uniform(rng_) < 0.4f) ? *fold_action : *call_action;
            }
            return actions[0];
        }
        else if (type == "alwayscall") {
            for (const auto& a : actions) {
                if (a.type == ActionType::Check) return a;
            }
            for (const auto& a : actions) {
                if (a.type == ActionType::Call) return a;
            }
            return actions[0];
        }

        // Default: random
        std::uniform_int_distribution<int> dist(0, actions.size() - 1);
        return actions[dist(rng_)];
    }
};


PYBIND11_MODULE(ares_solver, m) {
    m.doc() = "ARES-HU Poker Solver - Fast CFR implementation";

    py::class_<AresSolver>(m, "Solver")
        .def(py::init<>())
        .def("train", &AresSolver::train,
             py::arg("iterations"),
             py::arg("stack_size") = 20.0f,
             py::arg("qre_tau") = 1.0f,
             py::arg("threads") = 8,
             py::arg("verbose") = true,
             "Train a new CFR solver")
        .def("load", &AresSolver::load,
             py::arg("path"),
             "Load a trained solver from disk")
        .def("save", &AresSolver::save,
             py::arg("path"),
             "Save solver to disk")
        .def("num_info_sets", &AresSolver::num_info_sets,
             "Get number of information sets")
        .def("get_strategy", &AresSolver::get_strategy,
             py::arg("hero_cards"),
             py::arg("board_cards"),
             py::arg("hero_stack"),
             py::arg("villain_stack"),
             py::arg("pot"),
             "Get strategy for a game state")
        .def("evaluate", &AresSolver::evaluate,
             py::arg("opponent") = "random",
             py::arg("num_hands") = 100000,
             py::arg("stack_size") = 20.0f,
             py::arg("seed") = 42,
             "Evaluate strategy against an opponent")
        .def("export_training_data", &AresSolver::export_training_data,
             py::arg("path"),
             py::arg("num_samples") = 100000,
             "Export training data for neural network")
        .def("search", &AresSolver::search,
             py::arg("hero_cards"),
             py::arg("board_cards"),
             py::arg("pot"),
             py::arg("hero_stack"),
             py::arg("villain_stack"),
             py::arg("hero_bet"),
             py::arg("villain_bet"),
             py::arg("street"),
             py::arg("hero_position"),
             py::arg("iterations") = 500,
             py::arg("time_limit") = 5.0f,
             py::arg("qre_tau") = 1.0f,
             "Run real-time CFR search from a game state")
#ifdef ARES_USE_LIBTORCH
        .def("load_neural_model", &AresSolver::load_neural_model,
             py::arg("model_path"),
             "Load neural network model for leaf evaluation")
        .def("has_neural_model", &AresSolver::has_neural_model,
             "Check if neural model is loaded")
#endif
        ;
}
