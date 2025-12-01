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


/**
 * Fast RL Environment for training - exposes gym-like interface.
 * This is what you need for PPO/RL training with C++ speed.
 */
class FastPokerEnv {
public:
    FastPokerEnv(float stack_size = 100.0f, int num_actions = 14)
        : stack_size_(stack_size), num_actions_(num_actions), rng_(std::random_device{}()),
          state_(HoldemState::create_initial(HoldemConfig{stack_size, 0.5f, 1.0f})) {
        static bool initialized = false;
        if (!initialized) {
            HandEvaluator::initialize();
            initialized = true;
        }
        reset();
    }

    /**
     * Reset environment to new hand.
     * Returns: observation as numpy array (38, 4, 13)
     */
    py::array_t<float> reset() {
        HoldemConfig config;
        config.starting_stack = stack_size_;
        config.big_blind = 1.0f;
        config.small_blind = 0.5f;

        state_ = HoldemState::create_initial(config);

        // Deal random hole cards
        if (state_.is_chance_node()) {
            state_ = state_.deal_random(rng_);
        }

        // CRITICAL FIX: Alternate hero position between BTN (0) and BB (1)
        // This ensures model learns both in-position and out-of-position play
        hero_player_ = hand_count_ % 2;
        hand_count_++;
        cumulative_reward_ = 0.0f;

        return get_observation();
    }

    /**
     * Step environment with action.
     * Args:
     *     action: Action index (0=fold, 1=call/check, 2+=raise sizes)
     * Returns:
     *     Tuple of (observation, reward, done, info_dict)
     */
    py::tuple step(int action_idx) {
        if (state_.is_terminal()) {
            float reward = state_.utility(hero_player_);
            return py::make_tuple(get_observation(), reward, true, py::dict());
        }

        // Handle chance nodes (deal board cards)
        while (state_.is_chance_node() && !state_.is_terminal()) {
            state_ = state_.deal_random(rng_);
        }

        if (state_.is_terminal()) {
            float reward = state_.utility(hero_player_);
            return py::make_tuple(get_observation(), reward, true, py::dict());
        }

        // Get legal actions and map index to action
        std::vector<Action> legal = state_.get_legal_actions();
        if (legal.empty()) {
            return py::make_tuple(get_observation(), 0.0f, true, py::dict());
        }

        // Clamp action index to valid range
        int clamped_idx = std::min(action_idx, static_cast<int>(legal.size()) - 1);
        clamped_idx = std::max(clamped_idx, 0);

        Action chosen = legal[clamped_idx];

        // Apply hero's action
        state_ = state_.apply_action(chosen);

        // Handle chance nodes and opponent
        while (!state_.is_terminal()) {
            if (state_.is_chance_node()) {
                state_ = state_.deal_random(rng_);
                continue;
            }

            if (state_.current_player() != hero_player_) {
                // Opponent acts (simple policy - can be replaced)
                std::vector<Action> opp_actions = state_.get_legal_actions();
                if (opp_actions.empty()) break;

                // Use callback if set, otherwise random
                Action opp_action;
                if (opponent_callback_) {
                    auto obs = get_observation_for_player(1 - hero_player_);
                    auto mask = get_action_mask_for_player(1 - hero_player_);
                    int opp_idx = opponent_callback_(obs, mask);
                    opp_idx = std::clamp(opp_idx, 0, static_cast<int>(opp_actions.size()) - 1);
                    opp_action = opp_actions[opp_idx];
                } else {
                    std::uniform_int_distribution<int> dist(0, opp_actions.size() - 1);
                    opp_action = opp_actions[dist(rng_)];
                }
                state_ = state_.apply_action(opp_action);
            } else {
                break;  // Hero's turn
            }
        }

        bool done = state_.is_terminal();
        float reward = done ? state_.utility(hero_player_) : 0.0f;

        py::dict info;
        info["pot"] = state_.pot();

        return py::make_tuple(get_observation(), reward, done, info);
    }

    /**
     * Get action mask (which actions are legal).
     * Returns: numpy array of bools (num_actions,)
     */
    py::array_t<bool> get_action_mask() {
        // Use uint8_t instead of bool for numpy compatibility
        std::vector<uint8_t> mask(num_actions_, 0);

        if (state_.is_terminal() || state_.is_chance_node()) {
            mask[1] = 1;  // Default to call/check
            return py::array_t<bool>(mask.size(), reinterpret_cast<bool*>(mask.data()));
        }

        std::vector<Action> legal = state_.get_legal_actions();

        for (const auto& action : legal) {
            int idx = action_to_index(action);
            if (idx >= 0 && idx < num_actions_) {
                mask[idx] = 1;
            }
        }

        // Ensure at least one action is valid
        bool any_valid = false;
        for (auto b : mask) any_valid |= (b != 0);
        if (!any_valid) mask[1] = 1;

        return py::array_t<bool>(mask.size(), reinterpret_cast<bool*>(mask.data()));
    }

    /**
     * Get current observation.
     * Returns: numpy array (38, 4, 13) matching AlphaHoldem encoding
     */
    py::array_t<float> get_observation() {
        return get_observation_for_player(hero_player_);
    }

    /**
     * Set opponent policy callback.
     * Args:
     *     callback: Function taking (obs, mask) -> action_idx
     */
    void set_opponent(py::function callback) {
        opponent_callback_ = [callback](py::array_t<float> obs, py::array_t<bool> mask) -> int {
            py::object result = callback(obs, mask);
            return result.cast<int>();
        };
    }

    /**
     * Clear opponent callback (use random opponent).
     */
    void clear_opponent() {
        opponent_callback_ = nullptr;
    }

    bool is_terminal() const { return state_.is_terminal(); }
    float get_pot() const { return state_.pot(); }
    int current_player() const { return state_.current_player(); }

    /**
     * Check if opponent needs to act (for batched inference).
     * Returns true if it's opponent's turn (not hero's, not terminal, not chance).
     */
    bool needs_opponent_action() const {
        if (state_.is_terminal() || state_.is_chance_node()) return false;
        return state_.current_player() != hero_player_;
    }

    /**
     * Get opponent observation for batched inference.
     */
    py::array_t<float> get_opponent_observation() {
        return get_observation_for_player(1 - hero_player_);
    }

    /**
     * Get opponent action mask for batched inference.
     */
    py::array_t<bool> get_opponent_action_mask() {
        return get_action_mask_for_player(1 - hero_player_);
    }

    /**
     * Apply opponent action (for batched inference - no callback).
     * After this, handles chance nodes until it's hero's turn or terminal.
     * Returns: (done, reward)
     */
    py::tuple apply_opponent_action(int action_idx) {
        if (state_.is_terminal()) {
            return py::make_tuple(true, state_.utility(hero_player_));
        }

        std::vector<Action> legal = state_.get_legal_actions();
        if (legal.empty()) {
            return py::make_tuple(true, 0.0f);
        }

        int clamped_idx = std::clamp(action_idx, 0, static_cast<int>(legal.size()) - 1);
        state_ = state_.apply_action(legal[clamped_idx]);

        // Handle chance nodes until hero's turn or terminal
        while (!state_.is_terminal() && state_.is_chance_node()) {
            state_ = state_.deal_random(rng_);
        }

        bool done = state_.is_terminal();
        float reward = done ? state_.utility(hero_player_) : 0.0f;
        return py::make_tuple(done, reward);
    }

    /**
     * Step without auto-opponent (for batched inference).
     * Only applies hero action, stops when opponent needs to act.
     * Returns: (obs, reward, done, needs_opponent)
     */
    py::tuple step_hero_only(int action_idx) {
        if (state_.is_terminal()) {
            float reward = state_.utility(hero_player_);
            return py::make_tuple(get_observation(), reward, true, false);
        }

        // Handle chance nodes first
        while (state_.is_chance_node() && !state_.is_terminal()) {
            state_ = state_.deal_random(rng_);
        }

        if (state_.is_terminal()) {
            float reward = state_.utility(hero_player_);
            return py::make_tuple(get_observation(), reward, true, false);
        }

        // Apply hero action
        std::vector<Action> legal = state_.get_legal_actions();
        if (legal.empty()) {
            return py::make_tuple(get_observation(), 0.0f, true, false);
        }

        int clamped_idx = std::clamp(action_idx, 0, static_cast<int>(legal.size()) - 1);
        state_ = state_.apply_action(legal[clamped_idx]);

        // Handle chance nodes
        while (!state_.is_terminal() && state_.is_chance_node()) {
            state_ = state_.deal_random(rng_);
        }

        bool done = state_.is_terminal();
        float reward = done ? state_.utility(hero_player_) : 0.0f;
        bool needs_opp = !done && (state_.current_player() != hero_player_);

        return py::make_tuple(get_observation(), reward, done, needs_opp);
    }

private:
    HoldemState state_;
    float stack_size_;
    int num_actions_;
    int hero_player_;
    float cumulative_reward_;
    int hand_count_ = 0;  // For alternating positions
    std::mt19937 rng_;
    std::function<int(py::array_t<float>, py::array_t<bool>)> opponent_callback_;

    py::array_t<float> get_observation_for_player(int player) {
        // Create 50x4x13 observation tensor matching Python AlphaHoldemEncoder exactly
        // Layout: C x H x W where C=50 channels, H=4 suits, W=13 ranks
        //
        // Channel layout:
        // [0-1]:   Hole cards (2)
        // [2-6]:   Board cards (5)
        // [7]:     All known cards combined (1)
        // [8-11]:  Suit counts (4)
        // [12-15]: Rank counts (4)
        // [16-19]: Street indicators (4)
        // [20-21]: Position indicators (2)
        // [22]:    To-call amount normalized (1)
        // [23]:    Facing all-in indicator (1)
        // [24-27]: Pot/stack ratios (4)
        // [28-49]: Betting history - 22 actions (22)
        // Total: 50 channels

        constexpr int NUM_CHANNELS = 50;
        constexpr int H = 4;   // suits
        constexpr int W = 13;  // ranks
        constexpr int PLANE_SIZE = H * W;  // 52

        std::vector<float> obs(NUM_CHANNELS * PLANE_SIZE, 0.0f);

        // Helper to set a value at (channel, suit, rank)
        auto set_obs = [&](int channel, int suit, int rank, float value) {
            obs[channel * PLANE_SIZE + suit * W + rank] = value;
        };

        // Helper to fill entire plane with a value
        auto fill_plane = [&](int channel, float value) {
            for (int i = 0; i < PLANE_SIZE; ++i) {
                obs[channel * PLANE_SIZE + i] = value;
            }
        };

        // Collect all visible cards for suit/rank counting
        std::vector<std::pair<int, int>> all_cards;  // (rank, suit)

        // [0-1] Hole cards (2 channels, one per card)
        const auto& hand = state_.hand(player);
        for (int i = 0; i < 2; ++i) {
            Card c = hand[i];
            if (c != NO_CARD) {
                int rank = get_rank_int(c);
                int suit = get_suit_int(c);
                set_obs(i, suit, rank, 1.0f);
                all_cards.push_back({rank, suit});
            }
        }

        // [2-6] Board cards (5 channels, one per card position)
        const auto& board = state_.board();
        for (int i = 0; i < state_.board_count(); ++i) {
            Card c = board[i];
            if (c != NO_CARD) {
                int rank = get_rank_int(c);
                int suit = get_suit_int(c);
                set_obs(2 + i, suit, rank, 1.0f);
                all_cards.push_back({rank, suit});
            }
        }

        // [7] All known cards combined
        for (const auto& [rank, suit] : all_cards) {
            set_obs(7, suit, rank, 1.0f);
        }

        // [8-11] Suit counts (how many of each suit visible, normalized by 7)
        int suit_counts[4] = {0, 0, 0, 0};
        for (const auto& [rank, suit] : all_cards) {
            suit_counts[suit]++;
        }
        for (int s = 0; s < 4; ++s) {
            fill_plane(8 + s, suit_counts[s] / 7.0f);
        }

        // [12-15] Rank counts (pairs, trips, quads detection)
        int rank_counts[13] = {0};
        for (const auto& [rank, suit] : all_cards) {
            rank_counts[rank]++;
        }
        for (int r = 0; r < 13; ++r) {
            int count = rank_counts[r];
            if (count >= 1) {
                for (int s = 0; s < 4; ++s) set_obs(12, s, r, 1.0f);  // Has at least one
            }
            if (count >= 2) {
                for (int s = 0; s < 4; ++s) set_obs(13, s, r, 1.0f);  // Has pair
            }
            if (count >= 3) {
                for (int s = 0; s < 4; ++s) set_obs(14, s, r, 1.0f);  // Has trips
            }
            if (count >= 4) {
                for (int s = 0; s < 4; ++s) set_obs(15, s, r, 1.0f);  // Has quads
            }
        }

        // [16-19] Street indicators (one-hot)
        int street_idx = static_cast<int>(state_.street());
        fill_plane(16 + street_idx, 1.0f);

        // [20-21] Position indicators
        // In HU: player 0 = button (IP postflop), player 1 = BB (OOP postflop)
        bool is_button = (player == 0);
        if (is_button) {
            fill_plane(20, 1.0f);  // Hero is button
        } else {
            fill_plane(21, 1.0f);  // Hero is big blind
        }

        // [22] To-call amount (normalized by pot for pot odds)
        float to_call_amt = state_.to_call();
        float pot = std::max(state_.pot(), 1.0f);
        fill_plane(22, std::min(to_call_amt / pot, 2.0f) / 2.0f);  // Cap at 2x pot, normalize to 0-1

        // [23] Facing all-in indicator
        // Check if opponent is all-in (their stack is 0 and we have a decision)
        int opp = 1 - player;
        bool facing_allin = (state_.stack(opp) < 0.01f && to_call_amt > 0);
        fill_plane(23, facing_allin ? 1.0f : 0.0f);

        // [24-27] Pot/stack ratios
        float total_chips = state_.pot() + state_.stack(0) + state_.stack(1);
        if (total_chips > 0) {
            fill_plane(24, state_.pot() / total_chips);
        }
        fill_plane(25, state_.stack(player) / stack_size_);
        fill_plane(26, state_.stack(1 - player) / stack_size_);
        // Plane 27: total invested normalized
        float hero_invested = stack_size_ - state_.stack(player);
        float villain_invested = stack_size_ - state_.stack(1 - player);
        fill_plane(27, (hero_invested + villain_invested) / (2 * stack_size_));

        // [28-49] Betting history (last 22 actions - enough for any HU hand)
        // Encodes action type, amount, and who acted
        constexpr int MAX_HISTORY = 22;
        const auto& history = state_.action_history();
        int history_size = static_cast<int>(history.size());
        int start_idx = std::max(0, history_size - MAX_HISTORY);

        for (int i = start_idx; i < history_size; ++i) {
            int channel = 28 + (i - start_idx);
            const Action& action = history[i];

            // Action type encoding (row 0, spread across columns 0-5)
            // 0=fold, 1=check, 2=call, 3=bet, 4=raise, 5=all-in
            int action_type = 0;
            switch (action.type) {
                case ActionType::Fold:  action_type = 0; break;
                case ActionType::Check: action_type = 1; break;
                case ActionType::Call:  action_type = 2; break;
                case ActionType::Bet:   action_type = 3; break;
                case ActionType::Raise: action_type = 4; break;
                case ActionType::AllIn: action_type = 5; break;
                default: action_type = 1; break;
            }
            if (action_type < W) {
                set_obs(channel, 0, action_type, 1.0f);
            }

            // Amount encoding (row 1, normalized by starting stack)
            float norm_amount = std::min(action.amount / stack_size_, 1.0f);
            for (int r = 0; r < W; ++r) {
                set_obs(channel, 1, r, norm_amount);
            }

            // Hero/villain indicator (row 2)
            // Use the player field from the action (properly tracks who acted)
            bool is_hero_action = (action.player == player);
            for (int r = 0; r < W; ++r) {
                set_obs(channel, 2, r, is_hero_action ? 1.0f : 0.0f);
            }
        }

        return py::array_t<float>({NUM_CHANNELS, H, W}, obs.data());
    }

    py::array_t<bool> get_action_mask_for_player(int player) {
        // Simplified - same logic as get_action_mask but for any player
        (void)player;  // Currently same for both players
        // Use uint8_t instead of bool for numpy compatibility
        std::vector<uint8_t> mask(num_actions_, 0);

        if (state_.is_terminal() || state_.is_chance_node()) {
            mask[1] = 1;
            return py::array_t<bool>(mask.size(), reinterpret_cast<bool*>(mask.data()));
        }

        std::vector<Action> legal = state_.get_legal_actions();
        for (const auto& action : legal) {
            int idx = action_to_index(action);
            if (idx >= 0 && idx < num_actions_) {
                mask[idx] = 1;
            }
        }

        bool any_valid = false;
        for (auto b : mask) any_valid |= (b != 0);
        if (!any_valid) mask[1] = 1;

        return py::array_t<bool>(mask.size(), reinterpret_cast<bool*>(mask.data()));
    }

    int action_to_index(const Action& action) {
        switch (action.type) {
            case ActionType::Fold: return 0;
            case ActionType::Check: return 1;
            case ActionType::Call: return 1;
            case ActionType::Bet:
            case ActionType::Raise:
            case ActionType::AllIn: {
                // Map bet sizes to indices 2-13
                float pot = std::max(state_.pot(), 1.0f);
                float ratio = action.amount / pot;
                if (ratio < 0.4f) return 2;
                if (ratio < 0.6f) return 3;
                if (ratio < 0.8f) return 4;
                if (ratio < 1.0f) return 5;
                if (ratio < 1.5f) return 6;
                if (ratio < 2.0f) return 7;
                if (ratio < 3.0f) return 8;
                if (action.type == ActionType::AllIn) return 13;
                return std::min(9 + static_cast<int>(ratio), 12);
            }
            default: return 1;
        }
    }
};


PYBIND11_MODULE(ares_solver, m) {
    m.doc() = "ARES-HU Poker Solver - Fast CFR implementation + RL Environment";

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

    // Fast RL Environment for training
    py::class_<FastPokerEnv>(m, "FastEnv")
        .def(py::init<float, int>(),
             py::arg("stack_size") = 100.0f,
             py::arg("num_actions") = 14,
             "Create fast poker environment for RL training")
        .def("reset", &FastPokerEnv::reset,
             "Reset environment, returns observation")
        .def("step", &FastPokerEnv::step,
             py::arg("action"),
             "Step with action, returns (obs, reward, done, info)")
        .def("get_action_mask", &FastPokerEnv::get_action_mask,
             "Get valid action mask")
        .def("get_observation", &FastPokerEnv::get_observation,
             "Get current observation")
        .def("set_opponent", &FastPokerEnv::set_opponent,
             py::arg("callback"),
             "Set opponent policy function(obs, mask) -> action")
        .def("clear_opponent", &FastPokerEnv::clear_opponent,
             "Clear opponent (use random)")
        .def("is_terminal", &FastPokerEnv::is_terminal,
             "Check if game is over")
        .def("get_pot", &FastPokerEnv::get_pot,
             "Get current pot size")
        .def("current_player", &FastPokerEnv::current_player,
             "Get current player index")
        // Batched inference methods
        .def("needs_opponent_action", &FastPokerEnv::needs_opponent_action,
             "Check if opponent needs to act (for batched inference)")
        .def("get_opponent_observation", &FastPokerEnv::get_opponent_observation,
             "Get opponent observation for batched inference")
        .def("get_opponent_action_mask", &FastPokerEnv::get_opponent_action_mask,
             "Get opponent action mask for batched inference")
        .def("apply_opponent_action", &FastPokerEnv::apply_opponent_action,
             py::arg("action"),
             "Apply opponent action without callback, returns (done, reward)")
        .def("step_hero_only", &FastPokerEnv::step_hero_only,
             py::arg("action"),
             "Step hero only, returns (obs, reward, done, needs_opponent)")
        ;
}
