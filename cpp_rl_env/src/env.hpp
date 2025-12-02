#pragma once

#include "game_state.hpp"
#include <array>
#include <random>
#include <functional>

namespace rl_env {

// Default raise fractions matching Python env.py
constexpr std::array<float, 10> DEFAULT_RAISE_FRACTIONS = {
    0.33f, 0.5f, 0.67f, 0.75f, 1.0f,
    1.25f, 1.5f, 2.0f, 2.5f, 3.0f
};

// Heads-up environment matching Python HeadsUpEnv
class HeadsUpEnv {
public:
    static constexpr int NUM_ACTIONS = 14;

    HeadsUpEnv(float starting_stack = 100.0f, std::mt19937* rng = nullptr)
        : starting_stack_(starting_stack)
        , rng_(rng)
        , owns_rng_(rng == nullptr)
    {
        if (owns_rng_) {
            rng_ = new std::mt19937(std::random_device{}());
        }
        raise_fractions_ = DEFAULT_RAISE_FRACTIONS;
    }

    ~HeadsUpEnv() {
        if (owns_rng_) delete rng_;
    }

    // Reset for new hand
    void reset() {
        // Random button
        std::uniform_int_distribution<int> dist(0, 1);
        int button = dist(*rng_);

        // Shuffle and deal
        deck_.shuffle(*rng_);
        auto hero_cards = deck_.deal_hole();
        auto villain_cards = deck_.deal_hole();

        // Setup state
        state_ = GameState();
        state_.hero_hole = hero_cards;
        state_.villain_hole = villain_cards;
        state_.street = 0;
        state_.button = button;

        // Post blinds
        float sb = 0.5f;
        float bb = 1.0f;

        if (button == 0) {  // Hero is button (SB)
            state_.hero_stack = starting_stack_ - sb;
            state_.villain_stack = starting_stack_ - bb;
            state_.hero_invested = sb;
            state_.villain_invested = bb;
            state_.current_player = 0;  // Button acts first preflop
        } else {  // Villain is button (Hero is BB)
            state_.hero_stack = starting_stack_ - bb;
            state_.villain_stack = starting_stack_ - sb;
            state_.hero_invested = bb;
            state_.villain_invested = sb;
            state_.current_player = 1;  // Button (villain) acts first
        }

        state_.pot = sb + bb;
        state_.action_history = "";
        state_.terminal = false;
        state_.folded = false;
        state_.folder = -1;

        // Let opponent act if needed
        maybe_opponent_acts();
    }

    // Step with discrete action index
    // Returns: reward, done
    std::pair<float, bool> step(int action_idx) {
        Action poker_action = action_to_poker_action(action_idx);
        state_ = state_.apply_action(poker_action);

        // Handle street advancement
        maybe_advance_street();

        // Check terminal
        bool done = state_.is_terminal();
        float reward = 0.0f;

        if (done) {
            reward = state_.get_payoff(0);  // Hero's payoff
        } else {
            // Let opponent act
            maybe_opponent_acts();
            done = state_.is_terminal();
            if (done) {
                reward = state_.get_payoff(0);
            }
        }

        return {reward, done};
    }

    // Get action mask (which actions are legal)
    std::array<float, NUM_ACTIONS> get_action_mask() const {
        std::array<float, NUM_ACTIONS> mask{};
        mask.fill(0.0f);

        if (state_.is_terminal()) {
            return mask;
        }

        float to_call = state_.to_call();
        float stack = state_.current_stack();

        if (stack <= 0) {
            mask[1] = 1.0f;  // Check only
            return mask;
        }

        if (to_call > 0) {
            mask[0] = 1.0f;  // Fold
            mask[1] = 1.0f;  // Call
            if (stack > to_call) {
                for (int i = 2; i < NUM_ACTIONS; i++) {
                    mask[i] = 1.0f;  // Raises + all-in
                }
            }
        } else {
            mask[1] = 1.0f;  // Check
            if (stack > 0) {
                for (int i = 2; i < NUM_ACTIONS; i++) {
                    mask[i] = 1.0f;  // Bets + all-in
                }
            }
        }

        return mask;
    }

    // Set opponent policy (nullptr = random)
    void set_opponent(std::function<int(const GameState&, const std::array<float, NUM_ACTIONS>&)> policy) {
        opponent_policy_ = policy;
    }

    const GameState& state() const { return state_; }
    float starting_stack() const { return starting_stack_; }

private:
    Action action_to_poker_action(int action_idx) const {
        float to_call = state_.to_call();
        float stack = state_.current_stack();

        if (action_idx == 0) {  // Fold
            if (to_call > 0) {
                return Action::fold();
            }
            return Action::check();
        }
        else if (action_idx == 1) {  // Check/Call
            if (to_call > 0) {
                if (stack <= 0) {
                    return Action::check();
                }
                if (stack <= to_call) {
                    return Action::all_in(stack);
                }
                return Action::call();
            }
            return Action::check();
        }
        else if (action_idx == NUM_ACTIONS - 1) {  // All-in
            if (stack <= 0) {
                return to_call > 0 ? Action::fold() : Action::check();
            }
            return Action::all_in(stack);
        }
        else if (action_idx == NUM_ACTIONS - 2) {  // Min-raise
            if (stack <= 0) {
                return to_call > 0 ? Action::fold() : Action::check();
            }
            float min_raise = state_.min_raise();
            if (min_raise >= stack) {
                return Action::all_in(stack);
            }
            if (to_call > 0) {
                return Action::raise_to(min_raise);
            }
            return Action::bet(std::max(min_raise, 1.0f));
        }
        else {  // Raise sizes (2 to NUM_ACTIONS-3)
            if (stack <= 0) {
                return to_call > 0 ? Action::fold() : Action::check();
            }

            int raise_idx = action_idx - 2;
            float frac = (raise_idx < (int)raise_fractions_.size())
                ? raise_fractions_[raise_idx] : 1.0f;

            float pot_after_call = state_.pot + to_call;
            float raise_amount = to_call + (pot_after_call * frac);

            // Ensure minimum raise
            float min_raise = state_.min_raise();
            raise_amount = std::max(raise_amount, min_raise);
            raise_amount = std::min(raise_amount, stack);

            if (raise_amount >= stack) {
                return Action::all_in(stack);
            }
            if (to_call > 0) {
                return Action::raise_to(raise_amount);
            }
            return Action::bet(raise_amount);
        }
    }

    void maybe_advance_street() {
        while (state_.should_advance_street() &&
               !state_.is_terminal() &&
               state_.street < 3) {

            int new_street = state_.street + 1;

            // Deal new cards
            if (new_street == 1) {  // Flop
                auto flop = deck_.deal_flop();
                state_.board[0] = flop[0];
                state_.board[1] = flop[1];
                state_.board[2] = flop[2];
                state_.board_size = 3;
            } else {  // Turn or River
                state_.board[state_.board_size++] = deck_.deal();
            }

            // OOP acts first postflop
            int first_to_act = 1 - state_.button;

            state_.street = new_street;
            state_.hero_invested = 0;
            state_.villain_invested = 0;
            state_.current_player = first_to_act;
            state_.action_history += '/';
        }
    }

    void maybe_opponent_acts() {
        while (!state_.is_terminal() && state_.current_player != 0) {
            auto mask = get_opponent_mask();

            int action;
            if (opponent_policy_) {
                action = opponent_policy_(state_, mask);
            } else {
                // Random action
                std::vector<int> valid;
                for (int i = 0; i < NUM_ACTIONS; i++) {
                    if (mask[i] > 0) valid.push_back(i);
                }
                std::uniform_int_distribution<int> dist(0, valid.size() - 1);
                action = valid[dist(*rng_)];
            }

            // Apply action (from opponent's perspective)
            Action poker_action = opponent_action_to_poker(action);
            state_ = state_.apply_action(poker_action);
            maybe_advance_street();
        }
    }

    std::array<float, NUM_ACTIONS> get_opponent_mask() const {
        std::array<float, NUM_ACTIONS> mask{};
        mask.fill(0.0f);

        float to_call = state_.to_call();
        float stack = state_.villain_stack;  // Opponent is villain

        if (stack <= 0) {
            mask[1] = 1.0f;
            return mask;
        }

        if (to_call > 0) {
            mask[0] = 1.0f;
            mask[1] = 1.0f;
            if (stack > to_call) {
                for (int i = 2; i < NUM_ACTIONS; i++) {
                    mask[i] = 1.0f;
                }
            }
        } else {
            mask[1] = 1.0f;
            if (stack > 0) {
                for (int i = 2; i < NUM_ACTIONS; i++) {
                    mask[i] = 1.0f;
                }
            }
        }

        return mask;
    }

    Action opponent_action_to_poker(int action_idx) const {
        // Same logic as hero but uses villain's stack
        float to_call = state_.to_call();
        float stack = state_.villain_stack;

        if (action_idx == 0) {
            return to_call > 0 ? Action::fold() : Action::check();
        }
        else if (action_idx == 1) {
            if (to_call > 0) {
                if (stack <= 0) return Action::check();
                if (stack <= to_call) return Action::all_in(stack);
                return Action::call();
            }
            return Action::check();
        }
        else if (action_idx == NUM_ACTIONS - 1) {
            if (stack <= 0) {
                return to_call > 0 ? Action::fold() : Action::check();
            }
            return Action::all_in(stack);
        }
        else if (action_idx == NUM_ACTIONS - 2) {
            if (stack <= 0) {
                return to_call > 0 ? Action::fold() : Action::check();
            }
            float min_raise = state_.min_raise();
            if (min_raise >= stack) return Action::all_in(stack);
            if (to_call > 0) return Action::raise_to(min_raise);
            return Action::bet(std::max(min_raise, 1.0f));
        }
        else {
            if (stack <= 0) {
                return to_call > 0 ? Action::fold() : Action::check();
            }
            int raise_idx = action_idx - 2;
            float frac = (raise_idx < (int)raise_fractions_.size())
                ? raise_fractions_[raise_idx] : 1.0f;
            float pot_after_call = state_.pot + to_call;
            float raise_amount = to_call + (pot_after_call * frac);
            float min_raise = state_.min_raise();
            raise_amount = std::max(raise_amount, min_raise);
            raise_amount = std::min(raise_amount, stack);
            if (raise_amount >= stack) return Action::all_in(stack);
            if (to_call > 0) return Action::raise_to(raise_amount);
            return Action::bet(raise_amount);
        }
    }

    float starting_stack_;
    GameState state_;
    Deck deck_;
    std::mt19937* rng_;
    bool owns_rng_;
    std::array<float, 10> raise_fractions_;
    std::function<int(const GameState&, const std::array<float, NUM_ACTIONS>&)> opponent_policy_;
};

}  // namespace rl_env
