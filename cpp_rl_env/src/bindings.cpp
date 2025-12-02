#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "vec_env.hpp"
#include "vec_env_v2.hpp"

namespace py = pybind11;
using namespace rl_env;

// Python wrapper for VectorizedEnv with numpy interface
class PyVectorizedEnv {
public:
    PyVectorizedEnv(int num_envs, float starting_stack = 100.0f, int seed = -1)
        : env_(num_envs, starting_stack, seed)
        , num_envs_(num_envs)
    {
        // Initialize hand evaluator
        HandEvaluator::initialize();
    }

    int num_envs() const { return num_envs_; }
    static constexpr int num_actions() { return VectorizedEnv::NUM_ACTIONS; }

    // Reset all environments
    // Returns: (observations, action_masks)
    std::tuple<py::array_t<float>, py::array_t<float>> reset() {
        // Allocate output arrays
        auto obs = py::array_t<float>({num_envs_, VectorizedEnv::OBS_SHAPE_0,
                                        VectorizedEnv::OBS_SHAPE_1, VectorizedEnv::OBS_SHAPE_2});
        auto masks = py::array_t<float>({num_envs_, VectorizedEnv::NUM_ACTIONS});

        // Get pointers
        float* obs_ptr = obs.mutable_data();
        float* masks_ptr = masks.mutable_data();

        // Reset environments
        env_.reset(obs_ptr, masks_ptr);

        return std::make_tuple(obs, masks);
    }

    // Reset only done environments
    // Returns: (observations, action_masks, reset_mask)
    std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<bool>>
    reset_done_envs() {
        auto obs = py::array_t<float>({num_envs_, VectorizedEnv::OBS_SHAPE_0,
                                        VectorizedEnv::OBS_SHAPE_1, VectorizedEnv::OBS_SHAPE_2});
        auto masks = py::array_t<float>({num_envs_, VectorizedEnv::NUM_ACTIONS});
        auto reset_mask = py::array_t<bool>(num_envs_);

        float* obs_ptr = obs.mutable_data();
        float* masks_ptr = masks.mutable_data();
        bool* reset_ptr = reset_mask.mutable_data();

        env_.reset_done_envs(obs_ptr, masks_ptr, reset_ptr);

        return std::make_tuple(obs, masks, reset_mask);
    }

    // Step all environments
    // Returns: (observations, action_masks, rewards, dones)
    std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>, py::array_t<bool>>
    step(py::array_t<int> actions) {
        // Validate input
        if (actions.size() != num_envs_) {
            throw std::runtime_error("Actions array size must match num_envs");
        }

        auto obs = py::array_t<float>({num_envs_, VectorizedEnv::OBS_SHAPE_0,
                                        VectorizedEnv::OBS_SHAPE_1, VectorizedEnv::OBS_SHAPE_2});
        auto masks = py::array_t<float>({num_envs_, VectorizedEnv::NUM_ACTIONS});
        auto rewards = py::array_t<float>(num_envs_);
        auto dones = py::array_t<bool>(num_envs_);

        const int* actions_ptr = actions.data();
        float* obs_ptr = obs.mutable_data();
        float* masks_ptr = masks.mutable_data();
        float* rewards_ptr = rewards.mutable_data();
        bool* dones_ptr = dones.mutable_data();

        env_.step(actions_ptr, obs_ptr, masks_ptr, rewards_ptr, dones_ptr);

        return std::make_tuple(obs, masks, rewards, dones);
    }

private:
    VectorizedEnv env_;
    int num_envs_;
};


// V2 wrapper with batched opponent inference support
class PyVectorizedEnvV2 {
public:
    PyVectorizedEnvV2(int num_envs, float starting_stack = 100.0f, int seed = -1)
        : env_(num_envs, starting_stack, seed)
        , num_envs_(num_envs)
    {
        HandEvaluator::initialize();
    }

    int num_envs() const { return num_envs_; }
    static constexpr int num_actions() { return VectorizedEnvV2::NUM_ACTIONS; }

    std::tuple<py::array_t<float>, py::array_t<float>> reset() {
        auto obs = py::array_t<float>({num_envs_, 50, 4, 13});
        auto masks = py::array_t<float>({num_envs_, 14});
        env_.reset(obs.mutable_data(), masks.mutable_data());
        return std::make_tuple(obs, masks);
    }

    // Step with batched opponent inference
    // hero_actions: actions for hero (player 0)
    // opp_actions: actions for opponent (player 1), can be None for random
    std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>, py::array_t<bool>>
    step(py::array_t<int> hero_actions, py::object opp_actions_obj) {
        auto obs = py::array_t<float>({num_envs_, 50, 4, 13});
        auto masks = py::array_t<float>({num_envs_, 14});
        auto rewards = py::array_t<float>(num_envs_);
        auto dones = py::array_t<bool>(num_envs_);

        const int* hero_ptr = hero_actions.data();
        const int* opp_ptr = nullptr;
        bool use_nn = false;

        if (!opp_actions_obj.is_none()) {
            auto opp_actions = opp_actions_obj.cast<py::array_t<int>>();
            opp_ptr = opp_actions.data();
            use_nn = true;
        }

        env_.step_with_opponent(
            hero_ptr, opp_ptr, use_nn,
            obs.mutable_data(), masks.mutable_data(),
            rewards.mutable_data(), dones.mutable_data()
        );

        return std::make_tuple(obs, masks, rewards, dones);
    }

    // Get opponent observations for envs needing opponent action
    // Returns: (opp_obs, opp_masks, env_indices, count)
    std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<int>, int>
    get_opponent_obs() {
        auto opp_obs = py::array_t<float>({num_envs_, 50, 4, 13});
        auto opp_masks = py::array_t<float>({num_envs_, 14});
        auto env_indices = py::array_t<int>(num_envs_);

        int count = env_.get_opponent_obs(
            opp_obs.mutable_data(),
            opp_masks.mutable_data(),
            env_indices.mutable_data()
        );

        return std::make_tuple(opp_obs, opp_masks, env_indices, count);
    }

    // Two-phase stepping for proper batched opponent inference
    // Phase 1: Apply hero actions, return which envs need opponent
    std::tuple<py::array_t<float>, py::array_t<bool>, py::array_t<bool>>
    step_hero(py::array_t<int> hero_actions) {
        auto rewards = py::array_t<float>(num_envs_);
        auto dones = py::array_t<bool>(num_envs_);
        auto needs_opponent = py::array_t<bool>(num_envs_);

        env_.step_hero_only(
            hero_actions.data(),
            rewards.mutable_data(),
            dones.mutable_data(),
            needs_opponent.mutable_data()
        );

        return std::make_tuple(rewards, dones, needs_opponent);
    }

    // Phase 2: Apply opponent actions
    void step_opponent(py::array_t<int> opp_actions, py::array_t<int> env_indices, int count,
                       py::array_t<float> rewards, py::array_t<bool> dones) {
        env_.step_opponent(
            opp_actions.data(),
            env_indices.data(),
            count,
            rewards.mutable_data(),
            dones.mutable_data()
        );
    }

    // Get current obs and masks
    std::tuple<py::array_t<float>, py::array_t<float>> get_obs_and_masks() {
        auto obs = py::array_t<float>({num_envs_, 50, 4, 13});
        auto masks = py::array_t<float>({num_envs_, 14});
        env_.get_obs_and_masks(obs.mutable_data(), masks.mutable_data());
        return std::make_tuple(obs, masks);
    }

private:
    VectorizedEnvV2 env_;
    int num_envs_;
};


PYBIND11_MODULE(cpp_vec_env, m) {
    m.doc() = "C++ Vectorized Poker Environment for RL Training";

    py::class_<PyVectorizedEnv>(m, "VectorizedEnv")
        .def(py::init<int, float, int>(),
             py::arg("num_envs"),
             py::arg("starting_stack") = 100.0f,
             py::arg("seed") = -1,
             "Create vectorized environment with num_envs parallel games")
        .def("num_envs", &PyVectorizedEnv::num_envs, "Number of parallel environments")
        .def_property_readonly_static("num_actions",
            [](py::object) { return PyVectorizedEnv::num_actions(); },
            "Number of discrete actions")
        .def("reset", &PyVectorizedEnv::reset,
             "Reset all environments. Returns (observations, action_masks)")
        .def("reset_done_envs", &PyVectorizedEnv::reset_done_envs,
             "Reset only done environments. Returns (observations, action_masks, reset_mask)")
        .def("step", &PyVectorizedEnv::step,
             py::arg("actions"),
             "Step all environments. Returns (observations, action_masks, rewards, dones)");

    // V2 environment with batched opponent support
    py::class_<PyVectorizedEnvV2>(m, "VectorizedEnvV2")
        .def(py::init<int, float, int>(),
             py::arg("num_envs"),
             py::arg("starting_stack") = 100.0f,
             py::arg("seed") = -1)
        .def("num_envs", &PyVectorizedEnvV2::num_envs)
        .def("reset", &PyVectorizedEnvV2::reset)
        .def("step", &PyVectorizedEnvV2::step,
             py::arg("hero_actions"),
             py::arg("opp_actions") = py::none(),
             "Step with hero actions and optional opponent actions")
        .def("get_opponent_obs", &PyVectorizedEnvV2::get_opponent_obs)
        .def("step_hero", &PyVectorizedEnvV2::step_hero)
        .def("step_opponent", &PyVectorizedEnvV2::step_opponent)
        .def("get_obs_and_masks", &PyVectorizedEnvV2::get_obs_and_masks);

    // Constants
    m.attr("OBS_SHAPE") = py::make_tuple(50, 4, 13);
    m.attr("NUM_ACTIONS") = VectorizedEnvV2::NUM_ACTIONS;
}
