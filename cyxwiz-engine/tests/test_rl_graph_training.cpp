// Train on an RL graph, as the Engine's Train does it (TOFIX195 S4.3): the
// PPO (MLP 64-64) pattern on the built-in CartPole-v1, PPO "Tuned for
// CartPole-v1", compiled against the plugin environment registry and run by
// RLTrainingRun until the last 100 episodes average 475. Also: Stop ends a
// run, a second Start while one runs is refused, an environment no plugin
// offers fails with the reason. The trained policy (TOFIX195 S5) stays after
// the run, plays CartPole with its best actions, saves to a .cyxmodel and
// loads back acting the same; a graph whose layers differ is refused. With a
// MuJoCo plugin folder as the argument, a short InvertedPendulum run with
// normalised observations saves and loads the same way (continuous actions).
//
//   test_rl_graph_training [<MuJoCo plugin folder>]

#include "../src/core/graph_compiler.h"
#include "../src/core/rl/rl_training_run.h"
#include "../src/core/rl/trained_policy.h"
#include "../src/plugin/plugin_manager.h"
#include "../src/core/rl_graph_compile.h"
#include "../src/core/rl_graph_presentation.h"
#include "../src/plugin/registries/plugin_environment_registry.h"
#include "rl_graph_test_support.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
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

// Plays `episodes` episodes of one copy with the policy's best actions; the returns
std::vector<double> Play(cyxwiz::rl::TrainedPolicy& policy, int episodes) {
    std::string error;
    auto batch = cyxwiz::plugin::PluginEnvironmentRegistry::Instance().Create(
        policy.config.environment_key, policy.config.environment_settings, 1, error);
    Check(batch != nullptr, "a copy to play in: " + error);
    std::vector<double> returns;
    if (!batch) return returns;
    const auto& info = batch->Info();
    std::vector<float> obs(info.observation_size), final_obs(info.observation_size);
    float reward = 0.0f;
    uint8_t terminated = 0, truncated = 0;
    const cyxwiz::plugin::PluginEnvironmentStepBuffers out{obs.data(), &reward, &terminated, &truncated,
                                                           final_obs.data()};
    Check(batch->Reset(11, obs.data(), error), "the copy resets: " + error);
    double total = 0.0;
    while (static_cast<int>(returns.size()) < episodes) {
        const auto action = cyxwiz::rl::PolicyActions(policy, obs, 1, true, 0);
        if (!batch->Step(action.data(), out, error)) {
            Check(false, "the copy steps: " + error);
            break;
        }
        total += reward;
        if (terminated || truncated) {
            returns.push_back(total);
            total = 0.0;
        }
    }
    return returns;
}

// Random observations in the environment's range of the first rows played
std::vector<float> SampleObservations(size_t rows, size_t width) {
    std::mt19937 rng(5);
    std::normal_distribution<float> d(0.0f, 0.5f);
    std::vector<float> out(rows * width);
    for (auto& v : out) v = d(rng);
    return out;
}

// Save, load, and the same actions; the optimizer states and updates come back
void CheckSaveAndLoad(cyxwiz::rl::TrainedPolicy& trained, const std::string& file) {
    const auto path = std::filesystem::temp_directory_path() / file;
    std::string error;
    Check(cyxwiz::rl::SaveTrainedPolicy(trained, path, error), "the policy saves: " + error);
    auto loaded = cyxwiz::rl::LoadTrainedPolicy(path, error);
    Check(loaded != nullptr, "the policy loads: " + error);
    if (loaded) {
        const size_t width = trained.config.observation_size;
        const auto obs = SampleObservations(32, width);
        Check(cyxwiz::rl::PolicyActions(trained, obs, 32, true, 0) == cyxwiz::rl::PolicyActions(*loaded, obs, 32, true, 0),
              "the loaded policy takes the same best actions");
        Check(cyxwiz::rl::PolicyActions(trained, obs, 32, false, 9) == cyxwiz::rl::PolicyActions(*loaded, obs, 32, false, 9),
              "and the same sampled actions for the same seed");
        Check(loaded->config.environment_key == trained.config.environment_key, "it names the same environment");
        Check(loaded->updates.size() == trained.updates.size() && loaded->steps == trained.steps,
              "its PPO updates and steps come back");
        bool same_optimizers = loaded->optimizers.size() == trained.optimizers.size();
        for (const auto& [key, state] : trained.optimizers) {
            const auto it = loaded->optimizers.find(key);
            same_optimizers = same_optimizers && it != loaded->optimizers.end() &&
                              it->second.step_count == state.step_count &&
                              it->second.tensors.size() == state.tensors.size();
        }
        Check(same_optimizers, "the optimizers' states come back");
        Check((loaded->normalizer != nullptr) == (trained.normalizer != nullptr), "the observation statistics come back");
    }
    std::filesystem::remove(path);
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
    Check(run.Start(config, rl_graph_test::Json(g), error), "the run starts: " + error);
    Check(!run.Start(config, "", error) && error.find("already training") != std::string::npos,
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

    // The policy stays, plays and saves
    const auto trained = run.Trained();
    Check(trained != nullptr, "the finished run keeps its policy");
    if (!trained) return;
    Check(!updates.empty() && trained->steps == updates.back().timesteps && trained->updates.size() == updates.size(),
          "the kept policy knows its steps and updates");
    const auto returns = Play(*trained, 3);
    bool balanced = returns.size() == 3;
    for (const double r : returns) balanced = balanced && r >= 475.0;
    Check(balanced, "its best actions balance CartPole (475 or more in each of 3 episodes)");
    for (const double r : returns) std::printf("  played: return %.0f\n", r);
    CheckSaveAndLoad(*trained, "cyxwiz_rl_cartpole_policy_test.cyxmodel");

    // A graph whose layers differ from the saved weights is refused with what differs
    const std::string kept = trained->graph_json;
    auto narrower = g;
    narrower.At(id.p1).parameters["units"] = "32";
    trained->graph_json = rl_graph_test::Json(narrower);
    const auto path = std::filesystem::temp_directory_path() / "cyxwiz_rl_mismatch_test.cyxmodel";
    std::string save_error;
    Check(cyxwiz::rl::SaveTrainedPolicy(*trained, path, save_error), "a mismatched package saves: " + save_error);
    trained->graph_json = kept;
    std::string load_error;
    Check(cyxwiz::rl::LoadTrainedPolicy(path, load_error) == nullptr &&
              load_error.find("the graph builds") != std::string::npos,
          "a graph with other layer widths is refused with the shapes: " + load_error);
    std::filesystem::remove(path);
    Check(cyxwiz::rl::LoadTrainedPolicy(std::filesystem::temp_directory_path() / "cyxwiz_no_such_policy.cyxmodel",
                                        load_error) == nullptr && !load_error.empty(),
          "a missing file is refused with the reason: " + load_error);
}

void CheckContinuous(const std::filesystem::path& plugin) {
    auto& manager = cyxwiz::plugin::PluginManager::Instance();
    std::string error;
    Check(manager.LoadPlugin(plugin, &error), "the MuJoCo plugin loads: " + error);
    if (!manager.InitializePlugin("com.cyxwiz.simulation.mujoco")) {
        Check(false, "the MuJoCo plugin starts");
        return;
    }
    rl_graph_test::Ids id{};
    auto g = rl_graph_test::Pattern(id);
    g.At(id.env).parameters["environment"] = "com.cyxwiz.simulation.mujoco:InvertedPendulum-v5";
    g.At(id.env).parameters["copies"] = "2";
    g.At(id.ppo).parameters["n_steps"] = "128";
    g.At(id.ppo).parameters["total_timesteps"] = "768";
    g.At(id.ppo).parameters["normalize_observations"] = "true";
    const auto config = Compile(g);
    if (config) {
        cyxwiz::rl::RLTrainingRun run;
        Check(run.Start(config, rl_graph_test::Json(g), error), "the InvertedPendulum run starts: " + error);
        Wait(run, 300.0);
        Check(run.GetState() == cyxwiz::rl::RLTrainingRun::State::Finished, "it finishes: " + run.Message());
        const auto trained = run.Trained();
        Check(trained != nullptr && trained->normalizer != nullptr && trained->log_std.NumElements() == 1,
              "it keeps a Gaussian policy with its observation statistics");
        if (trained) {
            const auto& info = trained->config.environment;
            const auto actions = cyxwiz::rl::PolicyActions(*trained, SampleObservations(16, 4), 16, false, 3);
            bool bounded = actions.size() == 16;
            for (const float a : actions) bounded = bounded && a >= info.action_low[0] && a <= info.action_high[0];
            Check(bounded, "sampled actions stay within the bounds");
            CheckSaveAndLoad(*trained, "cyxwiz_rl_pendulum_policy_test.cyxmodel");
        }
    }
    manager.UnloadPlugin("com.cyxwiz.simulation.mujoco");
}

void CheckStopAndFailure() {
    rl_graph_test::Ids id{};
    auto g = rl_graph_test::Pattern(id);
    g.At(id.ppo).parameters["n_steps"] = "64";
    const auto config = Compile(g);
    if (!config) return;
    cyxwiz::rl::RLTrainingRun run;
    std::string error;
    Check(run.Start(config, "", error), "a long run starts: " + error);
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
    Check(run.Start(copy, "", error), "a run with a vanished environment starts: " + error);
    Wait(run, 30.0);
    Check(run.GetState() == cyxwiz::rl::RLTrainingRun::State::Failed && !run.Message().empty(),
          "it fails with the reason: " + run.Message());
}

}  // namespace

int main(int argc, char** argv) {
    cyxwiz::SetRLEnvironmentLookup([](const std::string& key, cyxwiz::plugin::PluginEnvironmentInfo& info) {
        cyxwiz::plugin::PluginEnvironmentRegistry::Environment e;
        if (!cyxwiz::plugin::PluginEnvironmentRegistry::Instance().Find(key, e)) return false;
        info = e.info;
        return true;
    });
    CheckStopAndFailure();
    CheckCartPole();
    if (argc > 1) CheckContinuous(std::filesystem::absolute(argv[1]));
    if (failures == 0) std::printf("test_rl_graph_training: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
