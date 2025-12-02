#pragma once

#include "env.hpp"
#include <vector>
#include <memory>
#include <cstring>

namespace rl_env {

// Vectorized environment for batch stepping
class VectorizedEnv {
public:
    static constexpr int OBS_SHAPE_0 = 50;
    static constexpr int OBS_SHAPE_1 = 4;
    static constexpr int OBS_SHAPE_2 = 13;
    static constexpr int OBS_SIZE = OBS_SHAPE_0 * OBS_SHAPE_1 * OBS_SHAPE_2;
    static constexpr int NUM_ACTIONS = HeadsUpEnv::NUM_ACTIONS;

    VectorizedEnv(int num_envs, float starting_stack = 100.0f, int seed = -1)
        : num_envs_(num_envs)
        , starting_stack_(starting_stack)
    {
        // Initialize RNG
        if (seed < 0) {
            rng_ = std::mt19937(std::random_device{}());
        } else {
            rng_ = std::mt19937(seed);
        }

        // Create environments
        envs_.reserve(num_envs);
        for (int i = 0; i < num_envs; i++) {
            envs_.emplace_back(starting_stack, &rng_);
        }

        // Pre-allocate buffers
        obs_buffer_.resize(num_envs * OBS_SIZE);
        mask_buffer_.resize(num_envs * NUM_ACTIONS);
        reward_buffer_.resize(num_envs);
        done_buffer_.resize(num_envs);
        dones_.resize(num_envs, false);
    }

    int num_envs() const { return num_envs_; }

    // Reset all environments
    void reset(float* obs_out, float* masks_out) {
        for (int i = 0; i < num_envs_; i++) {
            envs_[i].reset();
            dones_[i] = false;

            // Encode observation
            encode_obs(envs_[i].state(), obs_out + i * OBS_SIZE);

            // Get action mask
            auto mask = envs_[i].get_action_mask();
            std::copy(mask.begin(), mask.end(), masks_out + i * NUM_ACTIONS);
        }
    }

    // Reset only done environments
    void reset_done_envs(float* obs_out, float* masks_out, bool* reset_mask_out) {
        for (int i = 0; i < num_envs_; i++) {
            bool needs_reset = dones_[i] || envs_[i].state().is_terminal();

            if (!needs_reset) {
                auto mask = envs_[i].get_action_mask();
                bool any_valid = false;
                for (int j = 0; j < NUM_ACTIONS; j++) {
                    if (mask[j] > 0) { any_valid = true; break; }
                }
                if (!any_valid) needs_reset = true;
            }

            reset_mask_out[i] = needs_reset;

            if (needs_reset) {
                envs_[i].reset();
                dones_[i] = false;
            }

            // Encode observation
            encode_obs(envs_[i].state(), obs_out + i * OBS_SIZE);

            // Get action mask with safety fallback
            auto mask = envs_[i].get_action_mask();
            bool any_valid = false;
            for (int j = 0; j < NUM_ACTIONS; j++) {
                if (mask[j] > 0) any_valid = true;
            }
            if (!any_valid) mask[1] = 1.0f;  // Fallback to check/call

            std::copy(mask.begin(), mask.end(), masks_out + i * NUM_ACTIONS);
        }
    }

    // Step all environments
    void step(const int* actions,
              float* obs_out, float* masks_out,
              float* rewards_out, bool* dones_out) {

        for (int i = 0; i < num_envs_; i++) {
            if (dones_[i]) {
                // Already done, just output state
                encode_obs(envs_[i].state(), obs_out + i * OBS_SIZE);
                auto mask = envs_[i].get_action_mask();
                std::copy(mask.begin(), mask.end(), masks_out + i * NUM_ACTIONS);
                rewards_out[i] = 0.0f;
                dones_out[i] = false;  // Will be reset next call
                continue;
            }

            // Check if terminal from opponent's last action
            if (envs_[i].state().is_terminal()) {
                rewards_out[i] = envs_[i].state().get_payoff(0);
                dones_out[i] = true;
                dones_[i] = true;
                encode_obs(envs_[i].state(), obs_out + i * OBS_SIZE);
                auto mask = envs_[i].get_action_mask();
                std::copy(mask.begin(), mask.end(), masks_out + i * NUM_ACTIONS);
                continue;
            }

            // Step environment
            auto [reward, done] = envs_[i].step(actions[i]);
            rewards_out[i] = reward;
            dones_out[i] = done;
            dones_[i] = done;

            // Encode observation
            encode_obs(envs_[i].state(), obs_out + i * OBS_SIZE);

            // Get action mask
            auto mask = envs_[i].get_action_mask();
            std::copy(mask.begin(), mask.end(), masks_out + i * NUM_ACTIONS);
        }
    }

    // Get raw pointers for direct buffer access
    float* obs_buffer() { return obs_buffer_.data(); }
    float* mask_buffer() { return mask_buffer_.data(); }
    float* reward_buffer() { return reward_buffer_.data(); }
    uint8_t* done_buffer() { return done_buffer_.data(); }

private:
    // Encode game state into observation tensor
    // Format: (50, 4, 13) matching AlphaHoldemEncoder
    void encode_obs(const GameState& state, float* out) const {
        std::memset(out, 0, OBS_SIZE * sizeof(float));

        // Channel layout (matching Python encoder):
        // 0-1: Hole cards
        // 2-6: Board cards (flop/turn/river)
        // 7-16: Action history by street
        // 17-26: Bet sizes
        // 27-36: Stack sizes
        // 37-46: Position/street info
        // 47-49: Pot odds, SPR, etc.

        // Encode hole cards (channels 0-1)
        encode_card(state.hero_hole[0], out, 0);
        encode_card(state.hero_hole[1], out, 1);

        // Encode board (channels 2-6)
        for (int i = 0; i < state.board_size; i++) {
            encode_card(state.board[i], out, 2 + i);
        }

        // Encode position (channel 37)
        if (state.button == 0) {
            // Hero is button - fill channel 37
            for (int s = 0; s < 4; s++) {
                for (int r = 0; r < 13; r++) {
                    out[37 * 4 * 13 + s * 13 + r] = 1.0f;
                }
            }
        }

        // Encode street (channels 38-41)
        int street_channel = 38 + state.street;
        for (int s = 0; s < 4; s++) {
            for (int r = 0; r < 13; r++) {
                out[street_channel * 4 * 13 + s * 13 + r] = 1.0f;
            }
        }

        // Encode pot size (channel 42) - normalized by starting stack
        float pot_norm = state.pot / (starting_stack_ * 2);
        for (int s = 0; s < 4; s++) {
            for (int r = 0; r < 13; r++) {
                out[42 * 4 * 13 + s * 13 + r] = pot_norm;
            }
        }

        // Encode hero stack (channel 43)
        float hero_norm = state.hero_stack / starting_stack_;
        for (int s = 0; s < 4; s++) {
            for (int r = 0; r < 13; r++) {
                out[43 * 4 * 13 + s * 13 + r] = hero_norm;
            }
        }

        // Encode villain stack (channel 44)
        float villain_norm = state.villain_stack / starting_stack_;
        for (int s = 0; s < 4; s++) {
            for (int r = 0; r < 13; r++) {
                out[44 * 4 * 13 + s * 13 + r] = villain_norm;
            }
        }

        // Encode hero invested this street (channel 45)
        float hero_inv = state.hero_invested / starting_stack_;
        for (int s = 0; s < 4; s++) {
            for (int r = 0; r < 13; r++) {
                out[45 * 4 * 13 + s * 13 + r] = hero_inv;
            }
        }

        // Encode villain invested this street (channel 46)
        float villain_inv = state.villain_invested / starting_stack_;
        for (int s = 0; s < 4; s++) {
            for (int r = 0; r < 13; r++) {
                out[46 * 4 * 13 + s * 13 + r] = villain_inv;
            }
        }
    }

    void encode_card(Card card, float* out, int channel) const {
        if (card == NO_CARD) return;
        int rank = card_rank(card);
        int suit = card_suit(card);
        out[channel * 4 * 13 + suit * 13 + rank] = 1.0f;
    }

    int num_envs_;
    float starting_stack_;
    std::mt19937 rng_;
    std::vector<HeadsUpEnv> envs_;
    std::vector<bool> dones_;

    // Pre-allocated buffers
    std::vector<float> obs_buffer_;
    std::vector<float> mask_buffer_;
    std::vector<float> reward_buffer_;
    std::vector<uint8_t> done_buffer_;  // Use uint8_t instead of bool for .data()
};

}  // namespace rl_env
