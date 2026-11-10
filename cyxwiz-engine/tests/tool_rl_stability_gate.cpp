// The stability gate's runs (TOFIX195): one task and seed per process, trained
// the way Studio's Train does it (the PPO (MLP) pattern with the PPO node's
// "Tuned for" settings, the RL compile, RLTrainingRun). Writes the run's
// updates and outcome as JSON; the gate compares them with Stable-Baselines3
// on the same environments (resources/rl_reference/ppo_sb3_curves.json).
//
//   cyxwiz-rl-stability-gate <task> <seed> <output.json> [<MuJoCo plugin folder>]
//
// task: CartPole-v1 (built in), InvertedPendulum-v5 or HalfCheetah-v5 (the
// MuJoCo plugin). CartPole and InvertedPendulum stop at their target return;
// HalfCheetah runs its 1,000,000 steps with RL Zoo's two layers of 256 and ReLU.
// CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|cpu picks the device (default
// CPU). The run takes it the Engine's way (TOFIX206): the pending device
// selection and a Strict next-run policy, as Preferences > Devices sets them,
// over routes certified the installer's way; the report records the device
// the run actually trained on. CYXWIZ_GATE_NO_PROVIDERS=1 turns the neural
// providers off (the networks run as ArrayFire operations) for comparisons.

#include "../src/core/execution_device_preferences.h"
#include "../src/core/graph_compiler.h"
#include <cyxwiz/neural_provider.h>
#include "../src/core/rl/rl_training_run.h"
#include "../src/core/rl_graph_compile.h"
#include "../src/core/rl_graph_presentation.h"
#include "../src/plugin/plugin_manager.h"
#include "../src/plugin/registries/plugin_environment_registry.h"
#include "rl_graph_test_support.h"
#include "route_qualification_test_fixture.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

namespace {

nlohmann::json Number(float v) { return std::isfinite(v) ? nlohmann::json(v) : nlohmann::json(nullptr); }

int Fail(const std::string& why) {
    std::fprintf(stderr, "cyxwiz-rl-stability-gate: %s\n", why.c_str());
    return EXIT_FAILURE;
}

// The route asked for in CYXWIZ_TEST_ARRAYFIRE_BACKEND, chosen as Preferences
// > Devices chooses it; Strict, so a route that cannot start fails the run
bool SelectRequestedRoute() {
    const char* backend = std::getenv("CYXWIZ_TEST_ARRAYFIRE_BACKEND");
    const std::string name = backend != nullptr && backend[0] != '\0' ? backend : "cpu";
    cyxwiz::DeviceType type = cyxwiz::DeviceType::CPU;
    int id = 0;
    if (name == "cuda") {
        type = cyxwiz::DeviceType::CUDA;
    } else if (name == "opencl") {
        type = cyxwiz::DeviceType::OPENCL;
        if (const char* index = std::getenv("CYXWIZ_OPENCL_TEST_DEVICE")) id = std::atoi(index);
    } else if (name != "cpu") {
        return false;
    }
    cyxwiz::test::InstallQualifiedRouteSnapshot();
    cyxwiz::ClearPendingExecutionDeviceSelection();
    cyxwiz::SetPendingExecutionDeviceSelection(type, id);
    cyxwiz::ClearNextRunExecutionPolicy();
    cyxwiz::SetNextRunExecutionPolicy(cyxwiz::ExecutionPolicy::Strict);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) return Fail("usage: cyxwiz-rl-stability-gate <task> <seed> <output.json> [<MuJoCo plugin folder>]");
    const std::string task = argv[1];
    const std::string seed = argv[2];
    const std::filesystem::path output = argv[3];
    if (!SelectRequestedRoute()) return Fail("CYXWIZ_TEST_ARRAYFIRE_BACKEND names no device route (cpu, cuda, opencl)");
    // CYXWIZ_GATE_NO_PROVIDERS=1: no neural provider (the networks as ArrayFire operations), to compare
    if (const char* off = std::getenv("CYXWIZ_GATE_NO_PROVIDERS"); off && off[0] == '1')
        cyxwiz::SetNeuralProvidersDisabledForTesting(true);
    const bool mujoco = task != "CartPole-v1";
    cyxwiz::SetRLEnvironmentLookup([](const std::string& key, cyxwiz::plugin::PluginEnvironmentInfo& info) {
        cyxwiz::plugin::PluginEnvironmentRegistry::Environment e;
        if (!cyxwiz::plugin::PluginEnvironmentRegistry::Instance().Find(key, e)) return false;
        info = e.info;
        return true;
    });
    auto& manager = cyxwiz::plugin::PluginManager::Instance();
    if (mujoco) {
        if (argc < 5) return Fail(task + " needs the MuJoCo plugin folder");
        std::string error;
        if (!manager.LoadPlugin(std::filesystem::absolute(argv[4]), &error)) return Fail("the MuJoCo plugin: " + error);
        if (!manager.InitializePlugin("com.cyxwiz.simulation.mujoco")) return Fail("the MuJoCo plugin did not start");
    }

    const auto preset = cyxwiz::rl_graph::TunedPpoSettings(task);
    if (!preset) return Fail("no tuned settings for " + task);
    const bool cheetah = task == "HalfCheetah-v5";
    rl_graph_test::Ids id{};
    auto g = cheetah ? rl_graph_test::Pattern(id, 256, gui::NodeType::ReLU) : rl_graph_test::Pattern(id);
    g.At(id.env).parameters["environment"] =
        (mujoco ? std::string("com.cyxwiz.simulation.mujoco:") : std::string("cyxwiz.builtin:")) + task;
    for (const auto& [k, v] : preset->environment) g.At(id.env).parameters[k] = v;
    for (const auto& [k, v] : preset->ppo) g.At(id.ppo).parameters[k] = v;
    for (const auto& [k, v] : preset->policy_head) g.At(id.phead).parameters[k] = v;
    g.At(id.ppo).parameters["seed"] = seed;
    if (!cheetah) {  // the budgets of the SB3 reference
        g.At(id.ppo).parameters["total_timesteps"] = task == "CartPole-v1" ? "200000" : "300000";
        g.At(id.ppo).parameters["stop_at_return"] = "true";
    }
    cyxwiz::GraphCompiler compiler;
    const auto compiled = compiler.Compile(g.nodes, g.links, true);
    if (!compiled.rl) return Fail("the graph does not compile: " + compiled.error_message);
    const auto& config = *compiled.rl;
    const float target = cheetah ? 4800.0f : (task == "CartPole-v1" ? 475.0f : 950.0f);

    cyxwiz::rl::RLTrainingRun run;
    std::string error;
    const auto start = std::chrono::steady_clock::now();
    if (!run.Start(compiled.rl, rl_graph_test::Json(g), error)) return Fail(error);
    nlohmann::json rows = nlohmann::json::array();
    std::string device;  // where the run trains, once its preflight is done
    long long reached = -1;
    float last_return = std::numeric_limits<float>::quiet_NaN();
    // The report as it stands: written every few updates as well as at the end,
    // so a long run that is cut off still leaves its curve ("outcome": "Running")
    std::filesystem::create_directories(std::filesystem::absolute(output).parent_path());
    const auto write = [&](const std::string& outcome) {
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const nlohmann::json report = {
            {"task", task},
            {"seed", std::stoll(seed)},
            {"copies", config.copies},
            {"total_timesteps", config.ppo.total_timesteps},
            {"target", target},
            {"layers", cheetah ? "256-256 ReLU" : "64-64 Tanh"},
            {"device", device},
            {"outcome", outcome},
            {"reached_target_at", reached < 0 ? nlohmann::json(nullptr) : nlohmann::json(reached)},
            {"final_mean_return", Number(last_return)},
            {"steps", rows.empty() ? 0 : rows.back()[0].get<long long>()},
            {"seconds", std::round(seconds * 10.0) / 10.0},
            {"columns", {"timesteps", "mean_return", "policy_loss", "value_loss", "approx_kl", "clip_fraction",
                         "explained_variance", "action_std", "steps_per_second"}},
            {"updates", rows},
        };
        const auto partial = std::filesystem::path(output).concat(".part");
        std::ofstream(partial) << report.dump() << "\n";
        std::filesystem::rename(partial, output);  // a reader never sees half a file
    };
    const auto drain = [&]() {
        if (device.empty()) {
            const auto context = run.Device();
            if (context.valid) {
                device = context.effective_backend + " " + std::to_string(context.effective_device_id) +
                         (context.device_name.empty() ? "" : " (" + context.device_name + ")");
                std::printf("device: %s, policy %s%s\n", device.c_str(), context.ExecutionPolicyName().c_str(),
                            context.selection_fallback_applied ? ", fallback" : "");
                std::fflush(stdout);
            }
        }
        for (const auto& m : run.TakeUpdates()) {
            rows.push_back({m.timesteps, Number(m.mean_episode_return), Number(m.policy_loss), Number(m.value_loss),
                            Number(m.approx_kl), Number(m.clip_fraction), Number(m.explained_variance),
                            Number(m.std), Number(m.steps_per_second)});
            if (std::isfinite(m.mean_episode_return)) last_return = m.mean_episode_return;
            if (reached < 0 && m.episodes >= 100 && m.mean_episode_return >= target) reached = m.timesteps;
            if (m.update % 50 == 0) {
                std::printf("%s seed %s: update %d, %lld steps, mean return %.1f, %.0f steps/s\n", task.c_str(),
                            seed.c_str(), m.update, static_cast<long long>(m.timesteps), m.mean_episode_return,
                            m.steps_per_second);
                std::fflush(stdout);
            }
            if (m.update % 10 == 0) write("Running");
        }
    };
    while (run.Active()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        drain();
    }
    drain();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const bool finished = run.GetState() == cyxwiz::rl::RLTrainingRun::State::Finished;
    const long long steps = rows.empty() ? 0 : rows.back()[0].get<long long>();
    std::printf("%s seed %s: %s, target %.0f reached at %lld steps, final mean return %.1f after %lld steps (%.0f s)\n",
                task.c_str(), seed.c_str(), run.Message().c_str(), target, reached, last_return, steps, seconds);
    write(run.Message());
    if (mujoco) manager.UnloadPlugin("com.cyxwiz.simulation.mujoco");
    return finished ? EXIT_SUCCESS : EXIT_FAILURE;
}
