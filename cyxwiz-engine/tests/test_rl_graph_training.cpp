// Train on an RL graph, as the Engine's Train does it (TOFIX195 S4.3): the
// PPO (MLP 64-64) pattern on the built-in CartPole-v1, PPO "Tuned for
// CartPole-v1", compiled against the plugin environment registry and run by
// RLTrainingRun until the last 100 episodes average 475. Also: Stop ends a
// run, a second Start while one runs is refused, an environment no plugin
// offers fails with the reason.

#include "../src/core/graph_compiler.h"
#include "../src/core/rl/rl_training_run.h"
#include "../src/core/rl_graph_compile.h"
#include "../src/core/rl_graph_presentation.h"
#include "../src/plugin/registries/plugin_environment_registry.h"
#include "rl_graph_test_support.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

std::shared_ptr<const cyxwiz::RLTrainingConfiguration> Compile(const rl_graph_test::Graph& g) {
    cyxwiz::GraphCompiler compiler;
    const auto config = compiler.Compile(g.nodes, g.links, true);
    Check(config.is_valid && config.rl, "the graph compiles: " + config.error_message);
    return config.rl;
}

std::vector<cyxwiz::rl::PpoUpdateMetrics> Wait(cyxwiz::rl::RLTrainingRun& run, double seconds) {
    std::vector<cyxwiz::rl::PpoUpdateMetrics> all;
    const auto start = std::chrono::steady_clock::now();
    while (run.Active() &&
           std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < seconds) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        for (const auto& m : run.TakeUpdates()) all.push_back(m);
    }
    for (const auto& m : run.TakeUpdates()) all.push_back(m);
    return all;
}

void CheckCartPole() {
    rl_graph_test::Ids id{};
    auto g = rl_graph_test::Pattern(id);
    const auto preset = cyxwiz::rl_graph::TunedPpoSettings("CartPole-v1");
    for (const auto& [k, v] : preset->ppo) g.At(id.ppo).parameters[k] = v;
    for (const auto& [k, v] : preset->environment) g.At(id.env).parameters[k] = v;
    g.At(id.ppo).parameters["total_timesteps"] = "200000";
    const auto config = Compile(g);
    if (!config) return;

    cyxwiz::rl::RLTrainingRun run;
    std::string error;
    const auto start = std::chrono::steady_clock::now();
    Check(run.Start(config, error), "the run starts: " + error);
    Check(!run.Start(config, error) && error.find("already training") != std::string::npos,
          "a second start while it runs is refused: " + error);
    const auto updates = Wait(run, 900.0);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    Check(run.GetState() == cyxwiz::rl::RLTrainingRun::State::Finished, "the run finishes: " + run.Message());
    Check(!updates.empty() && updates.back().reached_target && updates.back().mean_episode_return >= 475.0f,
          "the graph's PPO reaches CartPole's 475 and stops there");
    if (!updates.empty())
        std::printf("CartPole-v1 from the graph: mean return %.1f after %lld steps (%d updates, %.0f s)\n",
                    updates.back().mean_episode_return, static_cast<long long>(updates.back().timesteps),
                    updates.back().update, seconds);
}

void CheckStopAndFailure() {
    rl_graph_test::Ids id{};
    auto g = rl_graph_test::Pattern(id);
    g.At(id.ppo).parameters["n_steps"] = "64";
    const auto config = Compile(g);
    if (!config) return;
    cyxwiz::rl::RLTrainingRun run;
    std::string error;
    Check(run.Start(config, error), "a long run starts: " + error);
    run.Pause();
    Check(run.Paused(), "it pauses");
    run.Resume();
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    run.Stop();
    Wait(run, 60.0);
    Check(run.GetState() == cyxwiz::rl::RLTrainingRun::State::Stopped, "Stop ends it: " + run.Message());

    // An environment the registry no longer offers (its plugin went after the compile)
    auto copy = std::make_shared<cyxwiz::RLTrainingConfiguration>(*config);
    copy->environment_key = "com.example.gone:Nope-v0";
    Check(run.Start(copy, error), "a run with a vanished environment starts: " + error);
    Wait(run, 30.0);
    Check(run.GetState() == cyxwiz::rl::RLTrainingRun::State::Failed && !run.Message().empty(),
          "it fails with the reason: " + run.Message());
}

}  // namespace

int main() {
    cyxwiz::SetRLEnvironmentLookup([](const std::string& key, cyxwiz::plugin::PluginEnvironmentInfo& info) {
        cyxwiz::plugin::PluginEnvironmentRegistry::Environment e;
        if (!cyxwiz::plugin::PluginEnvironmentRegistry::Instance().Find(key, e)) return false;
        info = e.info;
        return true;
    });
    CheckStopAndFailure();
    CheckCartPole();
    if (failures == 0) std::printf("test_rl_graph_training: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
