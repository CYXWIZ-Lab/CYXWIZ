// The Engine's PPO trainer (core/rl/ppo_trainer) learns, headless:
// CartPole-v1 (built in, discrete) reaches a mean return of 475 over 100
// episodes and stops there, the same seed gives the same first updates,
// normalised observations and rewards train, bad setups are refused with the
// reason, and every update says how it ran (fused kernels on a provider, or
// ArrayFire operations); a provider failing mid-run moves the run to ArrayFire
// operations and it trains on. With a MuJoCo plugin folder as the argument it also
// trains InvertedPendulum-v5 (continuous) to 950. Stop is checked through
// RLTrainingRun (test_rl_graph_training).
//
//   test_ppo_trainer [<MuJoCo plugin folder>]
//
// CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|cpu picks the device.

#include "../src/core/executable_model.h"
#include "../src/core/rl/ppo_trainer.h"
#include "../src/plugin/plugin_manager.h"
#include "../src/plugin/registries/plugin_environment_registry.h"
#include "algorithms/arrayfire_backend_utils.h"
#include "computation_truth/test_device_selection.h"

#include <cyxwiz/neural_provider.h>
#include <cyxwiz/sequential.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

using namespace cyxwiz;
using cyxwiz::plugin::EnvironmentBatch;
using cyxwiz::plugin::PluginEnvironmentRegistry;

// Stable-Baselines3's MlpPolicy: two hidden layers of 64 with tanh,
// orthogonal weights (gain sqrt 2, the output layer output_gain: 0.01 for the
// policy, 1 for the value) and zero biases.
std::unique_ptr<IExecutableModel> Mlp(size_t in, size_t out, float output_gain) {
    auto model = std::make_shared<SequentialModel>();
    model->Add<LinearModule>(in, 64);
    model->Add<TanhModule>();
    model->Add<LinearModule>(64, 64);
    model->Add<TanhModule>();
    model->Add<LinearModule>(64, out);
    auto executable = std::make_unique<SequentialExecutableModel>(model);
    rl::InitializeLikeStableBaselines3(*executable, output_gain);
    return executable;
}

// The registry's batch as the trainer's environment
rl::PpoEnvironment Adapt(std::shared_ptr<EnvironmentBatch> batch) {
    rl::PpoEnvironment env;
    env.info = batch->Info();
    env.count = batch->Count();
    env.reset = [batch](uint64_t seed, float* obs, std::string& error) { return batch->Reset(seed, obs, error); };
    env.step = [batch](const float* actions, const cyxwiz::plugin::PluginEnvironmentStepBuffers& out,
                       std::string& error) { return batch->Step(actions, out, error); };
    return env;
}

std::shared_ptr<EnvironmentBatch> Make(const std::string& key, int count) {
    std::string error;
    auto batch = PluginEnvironmentRegistry::Instance().Create(key, {}, count, error);
    Check(batch != nullptr, key + ": " + error);
    return std::shared_ptr<EnvironmentBatch>(std::move(batch));
}

struct Outcome {
    bool ok = false;
    std::string error;
    std::vector<rl::PpoUpdateMetrics> updates;
};

Outcome Train(const std::string& key, int copies, rl::PpoConfig config, float target, uint64_t init_seed,
              const std::function<void(const rl::PpoUpdateMetrics&)>& after_update = {}) {
    Outcome outcome;
    config.target_return = target;
    auto batch = Make(key, copies);
    if (!batch) return outcome;
    SeedCurrentArrayFireRandomEngine(init_seed);
    const auto& info = batch->Info();
    const size_t outputs = info.discrete_actions ? static_cast<size_t>(info.action_count) : static_cast<size_t>(info.action_size);
    rl::PpoTrainer trainer(Adapt(batch), Mlp(info.observation_size, outputs, 0.01f), Mlp(info.observation_size, 1, 1.0f),
                           config);
    outcome.ok = trainer.Train(
        [&](const rl::PpoUpdateMetrics& m) {
            outcome.updates.push_back(m);
            if (after_update) after_update(m);
            if (m.update % 25 == 0 || m.mean_episode_return >= target)
                std::printf("  %s update %4d  steps %7lld  return %7.1f  len %6.1f  pg %.4f  v %.4f  kl %.4f  ev %.3f  %6.0f steps/s\n",
                            key.c_str(), m.update, static_cast<long long>(m.timesteps), m.mean_episode_return,
                            m.mean_episode_length, m.policy_loss, m.value_loss, m.approx_kl, m.explained_variance,
                            m.steps_per_second);
        },
        outcome.error);
    return outcome;
}

// RL Zoo's tuned CartPole-v1 PPO (lr and clip constant instead of linear)
rl::PpoConfig CartPoleConfig(uint64_t seed) {
    rl::PpoConfig config;
    config.total_timesteps = 200000;
    config.n_steps = 32;
    config.batch_size = 256;
    config.n_epochs = 20;
    config.learning_rate = 1e-3f;
    config.gamma = 0.98f;
    config.gae_lambda = 0.8f;
    config.loss.ent_coef = 0.0f;
    config.seed = seed;
    return config;
}

void CheckCartPole() {
    const std::string key = "cyxwiz.builtin:CartPole-v1";
    const auto start = std::chrono::steady_clock::now();
    const auto outcome = Train(key, 8, CartPoleConfig(1), 475.0f, 1);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    Check(outcome.ok, "CartPole trains: " + outcome.error);
    const bool reached = !outcome.updates.empty() && outcome.updates.back().reached_target &&
                         outcome.updates.back().mean_episode_return >= 475.0f && outcome.updates.back().episodes >= 100;
    Check(reached, "CartPole reaches a mean return of 475 within 200000 steps, and the run stops there");
    Check(!outcome.updates.empty() && !outcome.updates.back().compute_path.empty(),
          "an update says how it ran");
    if (!outcome.updates.empty()) std::printf("CartPole-v1 ran as %s\n", outcome.updates.back().compute_path.c_str());
    if (!outcome.updates.empty())
        std::printf("CartPole-v1: mean return %.1f after %lld steps (%d updates, %.0f s)\n",
                    outcome.updates.back().mean_episode_return, static_cast<long long>(outcome.updates.back().timesteps),
                    outcome.updates.back().update, seconds);

    // The same seeds give the same first updates
    auto short_config = CartPoleConfig(5);
    short_config.total_timesteps = 3 * 8 * 32;
    const auto a = Train(key, 8, short_config, 1e9f, 9);
    const auto b = Train(key, 8, short_config, 1e9f, 9);
    bool same = a.updates.size() == 3 && b.updates.size() == 3;
    for (size_t i = 0; same && i < 3; ++i)
        same = a.updates[i].policy_loss == b.updates[i].policy_loss && a.updates[i].value_loss == b.updates[i].value_loss;
    Check(same, "the same seeds give the same first three updates");

    // Normalised observations and rewards (VecNormalize) train too
    auto normalised = short_config;
    normalised.normalize_observations = normalised.normalize_rewards = true;
    const auto c = Train(key, 8, normalised, 1e9f, 9);
    bool finite = c.ok && c.updates.size() == 3;
    for (const auto& u : c.updates) finite = finite && std::isfinite(u.policy_loss) && std::isfinite(u.value_loss);
    Check(finite, "with normalised observations and rewards the first three updates are finite: " + c.error);
    Check(c.ok && c.updates.back().value_loss != b.updates.back().value_loss,
          "normalised rewards change what the value network learns");
}

// The device's provider failing mid-run (every call fails from the third
// update on): the run moves to ArrayFire operations, says so in each update's
// compute path, and trains on. Needs the fused kernels (CUDA with the NVIDIA
// provider); elsewhere there is nothing to fail.
void CheckFallbackMidRun() {
    auto config = CartPoleConfig(7);
    config.total_timesteps = 6 * 8 * 32;
    const auto outcome = Train("cyxwiz.builtin:CartPole-v1", 8, config, 1e9f, 7, [](const rl::PpoUpdateMetrics& m) {
        if (m.update == 2) SetNeuralProvidersDisabledForTesting(true);
    });
    SetNeuralProvidersDisabledForTesting(false);
    if (outcome.updates.empty() || outcome.updates.front().compute_path.rfind("fused", 0) != 0) {
        std::printf("mid-run provider failure not checked: the run does not use the fused kernels here\n");
        return;
    }
    Check(outcome.ok && outcome.updates.size() == 6, "the run goes on after its provider fails: " + outcome.error);
    if (outcome.updates.size() != 6) return;
    const std::string& before = outcome.updates[1].compute_path;
    const std::string& after = outcome.updates.back().compute_path;
    Check(before.rfind("fused", 0) == 0, "before the failure the run uses the fused kernels: " + before);
    Check(after.find("ArrayFire operations (the fused kernels failed") != std::string::npos &&
              after.find("optimizers per tensor") != std::string::npos,
          "after it, each update says it runs on ArrayFire operations: " + after);
    bool finite = true;
    for (const auto& u : outcome.updates) finite = finite && std::isfinite(u.policy_loss) && std::isfinite(u.value_loss);
    Check(finite, "every update's losses stay finite across the fallback");
}

void CheckRefusals() {
    auto batch = Make("cyxwiz.builtin:CartPole-v1", 2);
    if (!batch) return;
    std::string error;
    rl::PpoTrainer wrong_head(Adapt(batch), Mlp(4, 3, 0.01f), Mlp(4, 1, 1.0f), CartPoleConfig(0));
    Check(!wrong_head.Train(nullptr, error) && error.find("needs 2 (one per choice)") != std::string::npos,
          "a policy with the wrong number of outputs is refused: " + error);
    auto bad = CartPoleConfig(0);
    bad.batch_size = 0;
    rl::PpoTrainer no_batch(Adapt(batch), Mlp(4, 2, 0.01f), Mlp(4, 1, 1.0f), bad);
    Check(!no_batch.Train(nullptr, error) && error == "Rollout steps, batch size and epochs must each be at least 1.",
          "a batch size of 0 is refused");
}

void CheckPendulum(const std::filesystem::path& plugin) {
    auto& manager = cyxwiz::plugin::PluginManager::Instance();
    std::string error;
    Check(manager.LoadPlugin(plugin, &error), "the MuJoCo plugin loads: " + error);
    if (!manager.InitializePlugin("com.cyxwiz.simulation.mujoco")) {
        Check(false, "the MuJoCo plugin starts");
        return;
    }
    // RL Zoo's InvertedPendulum PPO settings (SB3 defaults otherwise)
    rl::PpoConfig config;
    config.total_timesteps = 300000;
    config.n_steps = 512;
    config.batch_size = 64;
    config.n_epochs = 10;
    config.gamma = 0.999f;
    config.gae_lambda = 0.9f;
    config.learning_rate = 3e-4f;
    config.seed = 3;
    const auto start = std::chrono::steady_clock::now();
    const auto outcome = Train("com.cyxwiz.simulation.mujoco:InvertedPendulum-v5", 4, config, 950.0f, 3);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    Check(outcome.ok, "InvertedPendulum trains: " + outcome.error);
    Check(!outcome.updates.empty() && outcome.updates.back().mean_episode_return >= 950.0f,
          "InvertedPendulum reaches a mean return of 950 within 300000 steps");
    if (!outcome.updates.empty())
        std::printf("InvertedPendulum-v5: mean return %.1f after %lld steps (%.0f s)\n",
                    outcome.updates.back().mean_episode_return,
                    static_cast<long long>(outcome.updates.back().timesteps), seconds);
    manager.UnloadPlugin("com.cyxwiz.simulation.mujoco");
}

}  // namespace

int main(int argc, char** argv) {
    if (!cyxwiz::test::SelectTestDeviceFromEnvironment()) return EXIT_FAILURE;
    CheckRefusals();
    CheckCartPole();
    CheckFallbackMidRun();
    if (argc > 1) CheckPendulum(std::filesystem::absolute(argv[1]));
    if (failures == 0) std::printf("test_ppo_trainer: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
