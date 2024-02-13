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
    static constexpr TI EPISODE_STEP_LIMIT = 200;
    static constexpr TI TOTAL_STEP_LIMIT = 300000;
    static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT/(ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS) + 1; // number of PPO steps
};
using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS>;
template <typename NEXT>
struct LOOP_EVAL_PARAMETERS: rlt::rl::loop::steps::evaluation::Parameters<T, TI, NEXT>{
    static constexpr TI EVALUATION_INTERVAL = 4;
    static constexpr TI NUM_EVALUATION_EPISODES = 10;
    static constexpr TI N_EVALUATIONS = NEXT::CORE_PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL;
};
using LOOP_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CORE_CONFIG, LOOP_EVAL_PARAMETERS<LOOP_CORE_CONFIG>>;
using LOOP_STATE = typename LOOP_CONFIG::template State<LOOP_CONFIG>;

int main(){
    DEVICE device;
    TI seed = 1337;
    LOOP_STATE ls;
    rlt::malloc(device, ls);
    rlt::init(device, ls, seed);
    ls.actor_optimizer.parameters.alpha = 1e-3; // increasing the learning rate leads to faster training of the Pendulum-v1 environment
    ls.critic_optimizer.parameters.alpha = 1e-3;
    while(!rlt::step(device, ls)){
    }
}