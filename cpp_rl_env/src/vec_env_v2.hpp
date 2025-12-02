#pragma once

#include "game_state.hpp"
#include "cards.hpp"
#include <vector>
#include <array>
#include <random>
#include <cstring>

namespace rl_env {

// Default raise fractions matching Python env.py
constexpr std::array<float, 10> RAISE_FRACTIONS = {
    0.33f, 0.5f, 0.67f, 0.75f, 1.0f,
    1.25f, 1.5f, 2.0f, 2.5f, 3.0f
};

// Vectorized environment with batched opponent inference support
// This version doesn't handle opponents internally - Python controls both players
class VectorizedEnvV2 {
public:
    static constexpr int OBS_SHAPE_0 = 50;
    static constexpr int OBS_SHAPE_1 = 4;
    static constexpr int OBS_SHAPE_2 = 13;
    static constexpr int OBS_SIZE = OBS_SHAPE_0 * OBS_SHAPE_1 * OBS_SHAPE_2;
    static constexpr int NUM_ACTIONS = 14;

    VectorizedEnvV2(int num_envs, float starting_stack = 100.0f, int seed = -1)
        : num_envs_(num_envs)
        , starting_stack_(starting_stack)
    {
        if (seed < 0) {
            rng_ = std::mt19937(std::random_device{}());
        } else {
            rng_ = std::mt19937(seed);
        }

        states_.resize(num_envs);
        decks_.resize(num_envs);
        dones_.resize(num_envs, false);

        // Pre-allocate buffers
        obs_buffer_.resize(num_envs * OBS_SIZE);
        mask_buffer_.resize(num_envs * NUM_ACTIONS);
        opp_obs_buffer_.resize(num_envs * OBS_SIZE);
        opp_mask_buffer_.resize(num_envs * NUM_ACTIONS);
    }

    int num_envs() const { return num_envs_; }

    // Reset all environments - returns hero observations
    void reset(float* obs_out, float* masks_out) {
        for (int i = 0; i < num_envs_; i++) {
            reset_env(i);

            // Let opponent act first if it's their turn (when hero is BB)
            while (!states_[i].is_terminal() && states_[i].current_player == 1) {
                int opp_action = random_action(i, 1);
                apply_action(i, opp_action, 1);
                maybe_advance_street(i);
            }

            encode_hero_obs(i, obs_out + i * OBS_SIZE);
            get_hero_mask(i, masks_out + i * NUM_ACTIONS);
        }
    }

    // Step with both hero and opponent actions
    // This is the main entry point - handles full step including opponent
    // opponent_policy: 0 = random, 1 = use provided opp_actions
    void step_with_opponent(
        const int* hero_actions,
        const int* opp_actions,  // Can be nullptr for random opponent
        bool use_nn_opponent,
        float* obs_out,
        float* masks_out,
        float* rewards_out,
        bool* dones_out
    ) {
        for (int i = 0; i < num_envs_; i++) {
            if (dones_[i]) {
                // Reset and output new state
                reset_env(i);

                // Let opponent act first if it's their turn (when hero is BB)
                while (!states_[i].is_terminal() && states_[i].current_player == 1) {
                    int opp_action = random_action(i, 1);
                    apply_action(i, opp_action, 1);
                    maybe_advance_street(i);
                }

                encode_hero_obs(i, obs_out + i * OBS_SIZE);
                get_hero_mask(i, masks_out + i * NUM_ACTIONS);
                rewards_out[i] = 0.0f;
                dones_out[i] = false;
                continue;
            }

            // Apply hero action
            if (!states_[i].is_terminal() && states_[i].current_player == 0) {
                apply_action(i, hero_actions[i], 0);
                maybe_advance_street(i);
            }

            // Apply opponent actions until hero's turn or terminal
            while (!states_[i].is_terminal() && states_[i].current_player == 1) {
                int opp_action;
                if (use_nn_opponent && opp_actions != nullptr) {
                    opp_action = opp_actions[i];
                } else {
                    opp_action = random_action(i, 1);
                }
                apply_action(i, opp_action, 1);
                maybe_advance_street(i);
            }

            // Get results
            if (states_[i].is_terminal()) {
                rewards_out[i] = states_[i].get_payoff(0);
                dones_out[i] = true;
                dones_[i] = true;
            } else {
                rewards_out[i] = 0.0f;
                dones_out[i] = false;
            }

            encode_hero_obs(i, obs_out + i * OBS_SIZE);
            get_hero_mask(i, masks_out + i * NUM_ACTIONS);
        }
    }

    // Get opponent observations for envs where opponent needs to act
    // Returns number of envs needing opponent action
    int get_opponent_obs(float* opp_obs_out, float* opp_masks_out, int* env_indices_out) {
        int count = 0;
        for (int i = 0; i < num_envs_; i++) {
            if (!dones_[i] && !states_[i].is_terminal() && states_[i].current_player == 1) {
                encode_opponent_obs(i, opp_obs_out + count * OBS_SIZE);
                get_opponent_mask(i, opp_masks_out + count * NUM_ACTIONS);
                env_indices_out[count] = i;
                count++;
            }
        }
        return count;
    }

    // Two-phase stepping for NN opponent:
    // Phase 1: Apply hero actions, stop when opponent needs to act
    void step_hero_only(
        const int* hero_actions,
        float* rewards_out,
        bool* dones_out,
        bool* needs_opponent_out  // Which envs need opponent action
    ) {
        for (int i = 0; i < num_envs_; i++) {
            needs_opponent_out[i] = false;
            rewards_out[i] = 0.0f;
            dones_out[i] = false;

            if (dones_[i]) {
                reset_env(i);
                continue;
            }

            // Apply hero action if it's hero's turn
            if (!states_[i].is_terminal() && states_[i].current_player == 0) {
                apply_action(i, hero_actions[i], 0);
                maybe_advance_street(i);
            }

            // Check state after hero action
            if (states_[i].is_terminal()) {
                rewards_out[i] = states_[i].get_payoff(0);
                dones_out[i] = true;
                dones_[i] = true;
            } else if (states_[i].current_player == 1) {
                needs_opponent_out[i] = true;
            }
        }
    }

    // Phase 2: Apply opponent actions for envs that need them
    void step_opponent(
        const int* opp_actions,
        const int* env_indices,
        int num_opp_actions,
        float* rewards_out,
        bool* dones_out
    ) {
        // Apply opponent actions
        for (int j = 0; j < num_opp_actions; j++) {
            int i = env_indices[j];
            apply_action(i, opp_actions[j], 1);
            maybe_advance_street(i);

            // Continue with opponent actions until hero's turn or terminal
            while (!states_[i].is_terminal() && states_[i].current_player == 1) {
                // This shouldn't happen if opponent only acts once per street
                // But handle it with random action just in case
                int action = random_action(i, 1);
                apply_action(i, action, 1);
                maybe_advance_street(i);
            }
        }

        // Update all results
        for (int i = 0; i < num_envs_; i++) {
            if (states_[i].is_terminal() && !dones_[i]) {
                rewards_out[i] = states_[i].get_payoff(0);
                dones_out[i] = true;
                dones_[i] = true;
            }
        }
    }

    // Get current observations and masks for all envs
    void get_obs_and_masks(float* obs_out, float* masks_out) {
        for (int i = 0; i < num_envs_; i++) {
            encode_hero_obs(i, obs_out + i * OBS_SIZE);
            get_hero_mask(i, masks_out + i * NUM_ACTIONS);
        }
    }

private:
    void reset_env(int i) {
        decks_[i].shuffle(rng_);
        auto hero_cards = decks_[i].deal_hole();
        auto villain_cards = decks_[i].deal_hole();

        std::uniform_int_distribution<int> dist(0, 1);
        int button = dist(rng_);

        states_[i] = GameState();
        states_[i].hero_hole = hero_cards;
        states_[i].villain_hole = villain_cards;
        states_[i].street = 0;
        states_[i].button = button;

        float sb = 0.5f, bb = 1.0f;
        if (button == 0) {
            states_[i].hero_stack = starting_stack_ - sb;
            states_[i].villain_stack = starting_stack_ - bb;
            states_[i].hero_invested = sb;
            states_[i].villain_invested = bb;
            states_[i].current_player = 0;
        } else {
            states_[i].hero_stack = starting_stack_ - bb;
            states_[i].villain_stack = starting_stack_ - sb;
            states_[i].hero_invested = bb;
            states_[i].villain_invested = sb;
            states_[i].current_player = 1;
        }

        states_[i].pot = sb + bb;
        states_[i].terminal = false;
        states_[i].folded = false;
        states_[i].folder = -1;
        dones_[i] = false;
    }

    void apply_action(int env_idx, int action_idx, int player) {
        auto& state = states_[env_idx];
        float stack = (player == 0) ? state.hero_stack : state.villain_stack;
        float to_call = state.to_call();

        Action action;

        if (action_idx == 0) {  // Fold
            action = (to_call > 0) ? Action::fold() : Action::check();
        }
        else if (action_idx == 1) {  // Check/Call
            if (to_call > 0) {
                if (stack <= 0) action = Action::check();
                else if (stack <= to_call) action = Action::all_in(stack);
                else action = Action::call();
            } else {
                action = Action::check();
            }
        }
        else if (action_idx == NUM_ACTIONS - 1) {  // All-in
            if (stack <= 0) {
                action = (to_call > 0) ? Action::fold() : Action::check();
            } else {
                action = Action::all_in(stack);
            }
        }
        else if (action_idx == NUM_ACTIONS - 2) {  // Min-raise
            if (stack <= 0) {
                action = (to_call > 0) ? Action::fold() : Action::check();
            } else {
                float min_raise = state.min_raise();
                if (min_raise >= stack) action = Action::all_in(stack);
                else if (to_call > 0) action = Action::raise_to(min_raise);
                else action = Action::bet(std::max(min_raise, 1.0f));
            }
        }
        else {  // Raise sizes
            if (stack <= 0) {
                action = (to_call > 0) ? Action::fold() : Action::check();
            } else {
                int raise_idx = action_idx - 2;
                float frac = (raise_idx < 10) ? RAISE_FRACTIONS[raise_idx] : 1.0f;
                float pot_after_call = state.pot + to_call;
                float raise_amount = to_call + (pot_after_call * frac);
                float min_raise = state.min_raise();
                raise_amount = std::max(raise_amount, min_raise);
                raise_amount = std::min(raise_amount, stack);

                if (raise_amount >= stack) action = Action::all_in(stack);
                else if (to_call > 0) action = Action::raise_to(raise_amount);
                else action = Action::bet(raise_amount);
            }
        }

        state = state.apply_action(action);
    }

    void maybe_advance_street(int env_idx) {
        auto& state = states_[env_idx];
        while (state.should_advance_street() && !state.is_terminal() && state.street < 3) {
            int new_street = state.street + 1;

            if (new_street == 1) {
                auto flop = decks_[env_idx].deal_flop();
                state.board[0] = flop[0];
                state.board[1] = flop[1];
                state.board[2] = flop[2];
                state.board_size = 3;
            } else {
                state.board[state.board_size++] = decks_[env_idx].deal();
            }

            state.street = new_street;
            state.hero_invested = 0;
            state.villain_invested = 0;
            state.current_player = 1 - state.button;  // OOP acts first
            state.action_history += '/';
        }
    }

    int random_action(int env_idx, int player) {
        auto mask = get_action_mask(env_idx, player);
        std::vector<int> valid;
        for (int j = 0; j < NUM_ACTIONS; j++) {
            if (mask[j] > 0) valid.push_back(j);
        }
        if (valid.empty()) return 1;  // Default to check/call
        std::uniform_int_distribution<int> dist(0, valid.size() - 1);
        return valid[dist(rng_)];
    }

    std::array<float, NUM_ACTIONS> get_action_mask(int env_idx, int player) {
        std::array<float, NUM_ACTIONS> mask{};
        const auto& state = states_[env_idx];

        // Safety: if terminal, return check/call as valid (will be ignored anyway)
        if (state.is_terminal()) {
            mask[1] = 1.0f;
            return mask;
        }

        float to_call = state.to_call();
        float stack = (player == 0) ? state.hero_stack : state.villain_stack;

        if (stack <= 0) {
            mask[1] = 1.0f;
            return mask;
        }

        if (to_call > 0) {
            mask[0] = 1.0f;  // Fold
            mask[1] = 1.0f;  // Call
            if (stack > to_call) {
                for (int i = 2; i < NUM_ACTIONS; i++) mask[i] = 1.0f;
            }
        } else {
            mask[1] = 1.0f;  // Check
            for (int i = 2; i < NUM_ACTIONS; i++) mask[i] = 1.0f;
        }

        return mask;
    }

    void get_hero_mask(int env_idx, float* out) {
        auto mask = get_action_mask(env_idx, 0);
        std::copy(mask.begin(), mask.end(), out);
    }

    void get_opponent_mask(int env_idx, float* out) {
        auto mask = get_action_mask(env_idx, 1);
        std::copy(mask.begin(), mask.end(), out);
    }

    void encode_hero_obs(int env_idx, float* out) const {
        encode_obs(states_[env_idx], out, false);
    }

    void encode_opponent_obs(int env_idx, float* out) const {
        encode_obs(states_[env_idx], out, true);
    }

    void encode_obs(const GameState& state, float* out, bool opponent_pov) const {
        std::memset(out, 0, OBS_SIZE * sizeof(float));

        // Hole cards - swap if opponent POV
        if (!opponent_pov) {
            encode_card(state.hero_hole[0], out, 0);
            encode_card(state.hero_hole[1], out, 1);
        } else {
            encode_card(state.villain_hole[0], out, 0);
            encode_card(state.villain_hole[1], out, 1);
        }

        // Board
        for (int i = 0; i < state.board_size; i++) {
            encode_card(state.board[i], out, 2 + i);
        }

        // Position - from observer's perspective
        bool is_button = opponent_pov ? (state.button == 1) : (state.button == 0);
        if (is_button) {
            for (int s = 0; s < 4; s++) {
                for (int r = 0; r < 13; r++) {
                    out[37 * 4 * 13 + s * 13 + r] = 1.0f;
                }
            }
        }

        // Street
        int street_channel = 38 + state.street;
        for (int s = 0; s < 4; s++) {
            for (int r = 0; r < 13; r++) {
                out[street_channel * 4 * 13 + s * 13 + r] = 1.0f;
            }
        }

        // Pot, stacks, invested - swap if opponent POV
        float my_stack, opp_stack, my_invested, opp_invested;
        if (!opponent_pov) {
            my_stack = state.hero_stack;
            opp_stack = state.villain_stack;
            my_invested = state.hero_invested;
            opp_invested = state.villain_invested;
        } else {
            my_stack = state.villain_stack;
            opp_stack = state.hero_stack;
            my_invested = state.villain_invested;
            opp_invested = state.hero_invested;
        }

        fill_channel(out, 42, state.pot / (starting_stack_ * 2));
        fill_channel(out, 43, my_stack / starting_stack_);
        fill_channel(out, 44, opp_stack / starting_stack_);
        fill_channel(out, 45, my_invested / starting_stack_);
        fill_channel(out, 46, opp_invested / starting_stack_);
    }

    void encode_card(Card card, float* out, int channel) const {
        if (card == NO_CARD) return;
        out[channel * 4 * 13 + card_suit(card) * 13 + card_rank(card)] = 1.0f;
    }

    void fill_channel(float* out, int channel, float value) const {
        for (int s = 0; s < 4; s++) {
            for (int r = 0; r < 13; r++) {
                out[channel * 4 * 13 + s * 13 + r] = value;
            }
        }
    }

    int num_envs_;
    float starting_stack_;
    std::mt19937 rng_;
    std::vector<GameState> states_;
    std::vector<Deck> decks_;
    std::vector<bool> dones_;

    std::vector<float> obs_buffer_;
    std::vector<float> mask_buffer_;
    std::vector<float> opp_obs_buffer_;
    std::vector<float> opp_mask_buffer_;
};

}  // namespace rl_env
