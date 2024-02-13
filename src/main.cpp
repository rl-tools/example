#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn_models/operations_cpu.h>

#include "../include/my_pendulum/my_pendulum.h"
#include "../include/my_pendulum/operations_generic.h"

#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>

namespace rlt = rl_tools;


using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using T = float;
using TI = typename DEVICE::index_t;
using PENDULUM_SPEC = MyPendulumSpecification<T, TI, MyPendulumParameters<T>>;
using ENVIRONMENT = MyPendulum<PENDULUM_SPEC>;
struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::Parameters<T, TI, ENVIRONMENT>{
    struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<T, TI>{
        static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.0;
        static constexpr TI N_EPOCHS = 2;
    };

    static constexpr TI N_ENVIRONMENTS = 4;
    static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 1024;
    static constexpr TI BATCH_SIZE = 256;
    static constexpr TI TOTAL_STEP_LIMIT = 300000;
    static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT/(ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS) + 1;
    static constexpr TI EPISODE_STEP_LIMIT = 200;
};
using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS>;
template <typename NEXT>
struct LOOP_EVAL_PARAMETERS: rlt::rl::loop::steps::evaluation::Parameters<T, TI, NEXT>{
    static constexpr TI EVALUATION_INTERVAL = 4;
    static constexpr TI NUM_EVALUATION_EPISODES = 10;
    static constexpr TI N_EVALUATIONS = NEXT::CORE_PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL;
};
#ifndef BENCHMARK
using LOOP_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CORE_CONFIG, LOOP_EVAL_PARAMETERS<LOOP_CORE_CONFIG>>;
#else
using LOOP_CONFIG = LOOP_CORE_CONFIG;
#endif
using LOOP_STATE = typename LOOP_CONFIG::template State<LOOP_CONFIG>;

// just for measuring the time
#include <chrono>
#include <iostream>

int main(){
    DEVICE device;
    TI seed = 1337;
    LOOP_STATE ls;
    rlt::malloc(device, ls);
    rlt::init(device, ls, seed);
    ls.actor_optimizer.parameters.alpha = 1e-2;
    ls.critic_optimizer.parameters.alpha = 1e-2;
    auto start_time = std::chrono::high_resolution_clock::now();
    while(!rlt::step(device, ls)){
        // do what ever you want here, e.g. poor man's learning rate scheduler:
        if(ls.step % 1 == 0){
            ls.actor_optimizer.parameters.alpha *= 0.9;
            ls.critic_optimizer.parameters.alpha *= 0.9;
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time-start_time;
    std::cout << "Training time: " << diff.count() << std::endl;
}