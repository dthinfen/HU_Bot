#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "vec_env.hpp"

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

    // Constants
    m.attr("OBS_SHAPE") = py::make_tuple(50, 4, 13);
    m.attr("NUM_ACTIONS") = VectorizedEnv::NUM_ACTIONS;
}
