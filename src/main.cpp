#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn_models/operations_cpu.h>

#include "../include/my_pendulum/my_pendulum.h"
#include "../include/my_pendulum/operations_generic.h"
#include "../include/my_pendulum/operations_cpu.h" // JSON conversion functions for the rl::loop::steps::save_trajectories step (stored according to the experiment tracking specification: https://docs.rl.tools/10-Experiment%20Tracking.html)

#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/extrack/operations_cpu.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/save_trajectories/operations_cpu.h>

namespace rlt = rl_tools;


using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using T = float;
using TI = typename DEVICE::index_t;
using PENDULUM_SPEC = MyPendulumSpecification<T, TI, MyPendulumParameters<T>>;
using ENVIRONMENT = MyPendulum<PENDULUM_SPEC>;
struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{

    static constexpr TI N_ENVIRONMENTS = 8;
    static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 128;
    static constexpr TI BATCH_SIZE = 128;
    static constexpr TI TOTAL_STEP_LIMIT = 500000;
    static constexpr TI ACTOR_HIDDEN_DIM = 32;
    static constexpr TI CRITIC_HIDDEN_DIM = 32;
    static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
    static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
    static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT/(ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS) + 1;
    static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
    struct OPTIMIZER_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<T>{
        static constexpr T ALPHA = 0.01;
    };

    struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<T, TI, BATCH_SIZE>{
        static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.0;
        static constexpr TI N_EPOCHS = 1;
        static constexpr bool NORMALIZE_OBSERVATIONS = true;
        static constexpr T GAMMA = 0.9;
        static constexpr T INITIAL_ACTION_STD = 2.0;
    };
};
using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS>;
#ifndef BENCHMARK
using LOOP_EXTRACK_CONFIG = rlt::rl::loop::steps::extrack::Config<LOOP_CORE_CONFIG>; // Sets up the experiment tracking structure (https://docs.rl.tools/10-Experiment%20Tracking.html)
template <typename NEXT>
struct LOOP_EVAL_PARAMETERS: rlt::rl::loop::steps::evaluation::Parameters<T, TI, NEXT>{
    static constexpr TI EVALUATION_INTERVAL = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / 5;
    static constexpr TI NUM_EVALUATION_EPISODES = 10;
    static constexpr TI N_EVALUATIONS = NEXT::CORE_PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL;
};
using LOOP_EVALUATION_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_EXTRACK_CONFIG, LOOP_EVAL_PARAMETERS<LOOP_EXTRACK_CONFIG>>; // Evaluates the policy in a fixed interval and logs the return
struct LOOP_SAVE_TRAJECTORIES_PARAMETERS: rlt::rl::loop::steps::save_trajectories::Parameters<T, TI, LOOP_EVALUATION_CONFIG>{
    static constexpr TI INTERVAL_TEMP = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / 3;
    static constexpr TI INTERVAL = INTERVAL_TEMP == 0 ? 1 : INTERVAL_TEMP;
    static constexpr TI NUM_EPISODES = 10;
};
using LOOP_SAVE_TRAJECTORIES_CONFIG = rlt::rl::loop::steps::save_trajectories::Config<LOOP_EVALUATION_CONFIG, LOOP_SAVE_TRAJECTORIES_PARAMETERS>; // Saves trajectories for replay with the extrack UI
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_SAVE_TRAJECTORIES_CONFIG>;
using LOOP_CONFIG = LOOP_TIMING_CONFIG;

#else
using LOOP_CONFIG = LOOP_CORE_CONFIG;
#endif
using LOOP_STATE = typename LOOP_CONFIG::template State<LOOP_CONFIG>;

// just for measuring the time
#include <chrono>
#include <iostream>

int main(){
    DEVICE device;
    TI seed = 2;
    LOOP_STATE ls;
#ifndef BENCHMARK
    // Set experiment tracking info
    ls.extrack_name = "example";
#endif
    rlt::malloc(device, ls);
    rlt::init(device, ls, seed);
    auto start_time = std::chrono::high_resolution_clock::now();
    while(!rlt::step(device, ls)){
        // do what ever you want here, e.g. poor man's learning rate scheduler:
        // if(ls.step % 1 == 0){
        //     ls.actor_optimizer.parameters.alpha *= 0.9;
        //     ls.critic_optimizer.parameters.alpha *= 0.9;
        // }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time-start_time;
    std::cout << "Training time: " << diff.count() << " s" << std::endl;
}