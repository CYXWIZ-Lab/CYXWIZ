// Train on an RL graph, as the Engine's Train does it (TOFIX195 S4.3): the
// PPO (MLP 64-64) pattern on the built-in CartPole-v1, PPO "Tuned for
// CartPole-v1", compiled against the plugin environment registry and run by
// RLTrainingRun until the last 100 episodes average 475. Also: Stop ends a
// run, a second Start while one runs is refused, an environment no plugin
// offers fails with the reason. The trained policy (TOFIX195 S5) stays after
// the run, plays CartPole with its best actions, saves to a .cyxmodel and
// loads back acting the same; a graph whose layers differ is refused. With a
// MuJoCo plugin folder as the argument, a short InvertedPendulum run with
// normalised observations saves and loads the same way (continuous actions),
// and so does a SAC run of the SAC pattern (TOFIX216 S4): its reports, its
// replay buffer, a live preview of its actor, Stop, its five networks and
// entropy weight saved and loaded, playback, ghost and Stress test.
// The device (TOFIX206): every run takes the route chosen the Engine's way
// (the pending selection + the next-run policy, here the ArrayFire CPU), and
// reports it; Strict refuses a route that cannot start, Compatibility falls
// back to the CPU and says so.
//
//   test_rl_graph_training [<MuJoCo plugin folder>]

#include "../src/core/execution_device_preferences.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/formats/cyxmodel_format.h"
#include "../src/core/training_trace_collector.h"
#include <cyxwiz/neural_provider.h>
#include "../src/core/rl/policy_playback.h"
#include "../src/core/rl/policy_stress_test.h"
#include "../src/core/rl/rl_training_run.h"
#include "../src/core/rl/training_watch.h"
#include "../src/core/rl/trained_policy.h"
#include "../src/plugin/plugin_manager.h"
#include "../src/core/rl_graph_compile.h"
#include "../src/core/rl_graph_presentation.h"
#include "../src/plugin/registries/plugin_environment_registry.h"
#include "route_qualification_test_fixture.h"
#include "rl_graph_test_support.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <thread>

namespace {

int failures = 0;

void Check(bool ok, const std::string& what);
void CheckSac(const rl_graph_test::Graph& ppo_graph);

// The route for the next run, chosen as Preferences > Devices chooses it
void SelectRoute(cyxwiz::DeviceType type, int device_id, cyxwiz::ExecutionPolicy policy) {
    cyxwiz::ClearPendingExecutionDeviceSelection();
    cyxwiz::SetPendingExecutionDeviceSelection(type, device_id);
    cyxwiz::ClearNextRunExecutionPolicy();
    cyxwiz::SetNextRunExecutionPolicy(policy);
}

void SelectCpu() { SelectRoute(cyxwiz::DeviceType::CPU, 0, cyxwiz::ExecutionPolicy::Strict); }

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
    SelectCpu();  // loading follows the Engine's device choice, here the CPU the run trained on
    auto loaded = cyxwiz::rl::LoadTrainedPolicy(path, error);
    Check(loaded != nullptr, "the policy loads: " + error);
    Check(loaded == nullptr || (loaded->device_type == cyxwiz::DeviceType::CPU && loaded->device_id == 0),
          "the loaded policy acts on the chosen device");
    if (loaded) {
        const size_t width = trained.config.observation_size;
        const auto obs = SampleObservations(32, width);
        Check(cyxwiz::rl::PolicyActions(trained, obs, 32, true, 0) == cyxwiz::rl::PolicyActions(*loaded, obs, 32, true, 0),
              "the loaded policy takes the same best actions");
        Check(cyxwiz::rl::PolicyActions(trained, obs, 32, false, 9) == cyxwiz::rl::PolicyActions(*loaded, obs, 32, false, 9),
              "and the same sampled actions for the same seed");
        Check(loaded->config.environment_key == trained.config.environment_key, "it names the same environment");
        Check(loaded->updates.size() == trained.updates.size() &&
                  loaded->sac_reports.size() == trained.sac_reports.size() && loaded->steps == trained.steps &&
                  loaded->IsSac() == trained.IsSac(),
              "its learner, its PPO updates or SAC reports, and its steps come back");
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

// A ghost beside the played policy (TOFIX212 P3d): the same policy on the
// same seed plays the same episode step for step; removing it leaves the
// policy playing. in_view: the environment's view draws it (MuJoCo).
void CheckGhostPlayback(const std::shared_ptr<cyxwiz::rl::TrainedPolicy>& policy, bool in_view) {
    using cyxwiz::rl::PolicyPlayback;
    PolicyPlayback playback;
    playback.SetGhost(policy, "ghost", "played");
    std::string error;
    Check(playback.Start(policy, cyxwiz::rl::PlaybackSpeed::Fastest, true, error),
          "a playback with a ghost starts: " + error);
    for (int i = 0; i < 3000; ++i) {  // into the second episode: the finished returns compare too
        const auto s = playback.Get();
        if (s.state == PolicyPlayback::State::Failed || (s.episode >= 2 && s.step >= 10)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    playback.Pause();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));  // the step under way ends
    auto s = playback.Get();
    Check(s.state == PolicyPlayback::State::Paused && s.ghost && s.ghost_in_view == in_view,
          "the ghost plays beside the policy" + std::string(in_view ? ", drawn in the view: " : ": ") + s.ghost_note +
              s.message);
    Check(s.ghost_episode == s.episode && s.ghost_step == s.step && s.ghost_return == s.episode_return &&
              s.ghost_returns == s.returns,
          "the same policy on the same seed plays the same steps: step " + std::to_string(s.step) + " / " +
              std::to_string(s.ghost_step) + ", return " + std::to_string(s.episode_return) + " / " +
              std::to_string(s.ghost_return));
    playback.NewEpisode();
    playback.Resume();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    playback.Pause();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    s = playback.Get();
    Check(s.ghost_episode == s.episode && s.ghost_step == s.step, "New episode starts both together");
    playback.RemoveGhost();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    Check(!playback.Get().ghost, "Remove the ghost stops it, also while paused");
    playback.Resume();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    Check(playback.Get().state == PolicyPlayback::State::Playing, "the policy plays on");
    playback.Stop();
}

// The ghost's settings: the played copy's, except those that change what
// the ghost observes
void CheckGhostSettings() {
    using cyxwiz::rl::GhostSettings;
    const auto size = [](const std::map<std::string, std::string>& settings) {
        const auto it = settings.find("look_ahead");
        return 17 + (it != settings.end() && it->second == "1" ? 5 : 0);
    };
    std::string error;
    const auto mixed = GhostSettings({{"ground", "rough"}, {"look_ahead", "0"}},
                                     {{"ground", "flat"}, {"look_ahead", "1"}}, 22, size, error);
    Check(mixed && mixed->at("ground") == "rough" && mixed->at("look_ahead") == "1",
          "the played ground, the ghost's own observation: " + error);
    const auto same = GhostSettings({{"ground", "rough"}}, {{"ground", "flat"}}, 17, size, error);
    Check(same && same->at("ground") == "rough", "the same observation: the played settings whole");
    Check(!GhostSettings({{"ground", "rough"}}, {{"ground", "flat"}}, 30, size, error) &&
              error == "The ghost observes 30 values; this environment gives 17.",
          "no mix fits: why: " + error);
}

void CheckCartPole() {
    rl_graph_test::Ids id{};
    auto g = rl_graph_test::Pattern(id);
    const auto preset = cyxwiz::rl_graph::TunedPpoSettings("CartPole-v1");
    for (const auto& [k, v] : preset->learner) g.At(id.ppo).parameters[k] = v;
    for (const auto& [k, v] : preset->environment) g.At(id.env).parameters[k] = v;
    g.At(id.ppo).parameters["total_timesteps"] = "200000";
    const auto config = Compile(g);
    if (!config) return;

    // Compile's prediction of the fused small-network kernels: never on the
    // CPU; on CUDA exactly when a provider serves CUDA; not for a layer
    // wider than the kernels take
    Check(!cyxwiz::FusedNetworksExpected(*config, cyxwiz::DeviceType::CPU, 0), "no fused kernels on the CPU");
    const bool cuda_provider = !cyxwiz::NeuralProviderRegistry::Instance().ListServing({cyxwiz::DeviceType::CUDA, 0}).empty();
    Check(cyxwiz::FusedNetworksExpected(*config, cyxwiz::DeviceType::CUDA, 0) == cuda_provider,
          "CartPole's 64-64 networks are expected fused on CUDA exactly when a provider serves CUDA");
    auto wide = *config;
    wide.policy_layers.front().units = 300;
    Check(!cyxwiz::FusedNetworksExpected(wide, cyxwiz::DeviceType::CUDA, 0), "a 300-unit layer is not expected fused");
    if (cuda_provider) {
        auto mixed = *config;  // two LeakyReLU slopes in one network: FusedMlp refuses them
        for (auto& layer : mixed.policy_layers)
            if (layer.type != gui::NodeType::Dense) layer.type = gui::NodeType::LeakyReLU;
        mixed.policy_layers.back().negative_slope = 0.2f;
        Check(!cyxwiz::FusedNetworksExpected(mixed, cyxwiz::DeviceType::CUDA, 0),
              "LeakyReLU layers with different slopes are not expected fused");
        auto huge = *config;  // a minibatch past the backward's rows
        huge.ppo.batch_size = 70000;
        Check(!cyxwiz::FusedNetworksExpected(huge, cyxwiz::DeviceType::CUDA, 0),
              "a minibatch past the backward's 65,536 rows is not expected fused");
    }

    cyxwiz::rl::RLTrainingRun run;
    std::string error;
    const auto start = std::chrono::steady_clock::now();
    SelectCpu();
    Check(run.Start(config, rl_graph_test::Json(g), error), "the run starts: " + error);
    Check(!run.Start(config, "", error) && error.find("already training") != std::string::npos,
          "a second start while it runs is refused: " + error);
    const auto updates = Wait(run, 900.0);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    Check(run.GetState() == cyxwiz::rl::RLTrainingRun::State::Finished, "the run finishes: " + run.Message());
    // Its trace says how it computed (the path its updates report)
    bool path_traced = false;
    for (const auto& event : cyxwiz::TrainingTraceCollector::Instance().Snapshot().recent_events)
        path_traced |= event.stage == "PPO.ComputePath" && !updates.empty() && event.message == updates.back().compute_path;
    Check(path_traced, "the run's trace records how it computes");
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
    CheckGhostPlayback(trained, false);  // the built-in CartPole has no view: numbers only

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
    Check(cyxwiz::rl::PolicyMismatch(*trained, *config).empty(), "the policy fits the graph it was trained from");
    const auto narrow_config = Compile(narrower);
    const std::string mismatch = narrow_config ? cyxwiz::rl::PolicyMismatch(*trained, *narrow_config) : "";
    Check(mismatch == "The policy was trained on 64-64 policy and 64-64 value layers; this graph has 32-64 and 64-64.",
          "a narrower graph is named: " + mismatch);
    Check(cyxwiz::rl::LoadTrainedPolicy(std::filesystem::temp_directory_path() / "cyxwiz_no_such_policy.cyxmodel",
                                        load_error) == nullptr && !load_error.empty(),
          "a missing file is refused with the reason: " + load_error);
}

// Watching a run (TOFIX212 P5): with the preview on, the run copies its
// networks for the preview, the last copy is the trained networks, and the
// run learns exactly what it learns unwatched (same seed, same updates)
void CheckWatchLeavesTrainingAlone(gui::NodeType activation) {
    rl_graph_test::Ids id{};
    auto g = rl_graph_test::Pattern(id, 64, activation);
    g.At(id.ppo).parameters["n_steps"] = "64";
    g.At(id.ppo).parameters["total_timesteps"] = "1024";
    const auto config = Compile(g);
    if (!config) return;
    struct Outcome {
        std::vector<cyxwiz::rl::PpoUpdateMetrics> updates;
        std::shared_ptr<cyxwiz::rl::TrainedPolicy> trained, preview;
        int preview_update = 0;
    };
    const auto train = [&config](bool watch) {
        Outcome o;
        cyxwiz::rl::RLTrainingRun run;
        std::string error;
        SelectCpu();
        run.SetWatch(watch ? cyxwiz::rl::RLTrainingRun::Watch::Preview : cyxwiz::rl::RLTrainingRun::Watch::Off, 1);
        Check(run.Start(config, "", error), "a short run starts: " + error);
        o.updates = Wait(run, 300.0);
        o.trained = run.Trained();
        o.preview = run.PreviewPolicy();
        long long steps = 0;
        if (o.preview) run.RefreshPreview(*o.preview, o.preview_update, steps);
        return o;
    };
    const Outcome plain = train(false);
    const Outcome watched = train(true);
    Check(!plain.preview, "no preview networks when nobody watches");
    bool same = plain.updates.size() == watched.updates.size() && !plain.updates.empty();
    for (size_t i = 0; same && i < plain.updates.size(); ++i)
        same = plain.updates[i].policy_loss == watched.updates[i].policy_loss &&
               plain.updates[i].value_loss == watched.updates[i].value_loss;
    Check(same, "watching does not change what the run learns");
    Check(watched.preview != nullptr && watched.preview_update == static_cast<int>(watched.updates.size()),
          "the preview's last copy is of the last update");
    if (watched.preview && watched.trained) {
        const cyxwiz::rl::PolicyDeviceScope on_its_device(*watched.trained);
        auto trained = watched.trained->policy->GetParameters();
        auto copied = watched.preview->policy->GetParameters();
        bool equal = !trained.empty() && trained.size() == copied.size();
        for (auto& [name, tensor] : trained) {
            if (!equal) break;
            const auto it = copied.find(name);
            equal = it != copied.end() && it->second.NumElements() == tensor.NumElements();
            if (!equal) break;
            const float* a = tensor.ReadData<float>();
            const float* b = it->second.ReadData<float>();
            for (size_t k = 0; equal && k < tensor.NumElements(); ++k) equal = a[k] == b[k];
        }
        Check(equal, "the preview plays the trained networks after the last copy");
    }
}

// The preview plays in the plugin's view while the run trains, and stops with it
void CheckWatchInView(const std::shared_ptr<const cyxwiz::RLTrainingConfiguration>& config) {
    cyxwiz::rl::RLTrainingRun run;
    cyxwiz::rl::TrainingWatch watch;
    std::string error;
    SelectCpu();
    Check(run.Start(config, "", error), "a watched run starts: " + error);
    bool previewed = false;
    int update = 0;
    for (int i = 0; i < 6000 && run.Active(); ++i) {
        watch.Update(&run, cyxwiz::rl::RLTrainingRun::Watch::Preview, 1);
        const auto s = watch.Get();
        previewed = previewed || s.previewing;
        update = std::max(update, s.networks_update);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    watch.Update(&run, cyxwiz::rl::RLTrainingRun::Watch::Preview, 1);
    Check(previewed && update > 0, "the preview played the run's networks while it trained");
    Check(!watch.Get().previewing, "the preview stops when the run ends");
}

// A stress test of a trained policy, run whole and stopped early
void CheckStressRun(const std::shared_ptr<cyxwiz::rl::TrainedPolicy>& policy) {
    using namespace cyxwiz::rl;
    Check(SupportsStress(policy->config.environment), "a MuJoCo task takes the stress test's settings");
    Check(SupportsStressKind(policy->config.environment, StressKind::Thrown), "and the thrown objects'");
    StressTestSpec spec;
    spec.kind = StressKind::Kicks;
    spec.levels = 2;
    spec.episodes = 2;
    PolicyStressTest test;
    std::string error;
    Check(test.Start({{"a", policy}, {"b", policy}}, spec, error), "the stress test starts: " + error);
    for (int i = 0; i < 6000 && test.Get().state == PolicyStressTest::State::Running; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const auto done = test.Get();
    Check(done.state == PolicyStressTest::State::Finished, "it finishes: " + done.message);
    bool shaped = done.results.size() == 2 && done.done == 4 && done.total == 4;
    for (const auto& r : done.results) {
        shaped = shaped && r.levels.size() == 2;
        for (const auto& level : r.levels)
            shaped = shaped && level.returns.size() == 2 && level.falls >= 0 && level.falls <= 2;
    }
    Check(shaped, "a result for each policy and level: two returns, at most two falls");
    // The same policy, the same seeds: the same returns
    Check(shaped && done.results[0].levels[1].returns == done.results[1].levels[1].returns,
          "both columns meet the same kicks");

    spec.levels = 21;
    spec.episodes = 100;
    Check(test.Start({{"a", policy}}, spec, error), "a long stress test starts: " + error);
    test.Stop();
    Check(test.Get().state == PolicyStressTest::State::Stopped, "Stop ends it");
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
        SelectCpu();
        Check(run.Start(config, rl_graph_test::Json(g), error), "the InvertedPendulum run starts: " + error);
        const auto updates = Wait(run, 300.0);
        Check(run.GetState() == cyxwiz::rl::RLTrainingRun::State::Finished, "it finishes: " + run.Message());
        // The batch's own numbers reach the dashboard with each update
        bool falls_reported = false;
        for (const auto& m : updates)
            for (const auto& stat : m.environment_stats) falls_reported = falls_reported || stat.label == "Falls";
        Check(falls_reported, "the environment's Falls reaches the update metrics");
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
            CheckStressRun(trained);
            CheckGhostPlayback(trained, true);
        }
        auto longer = std::make_shared<cyxwiz::RLTrainingConfiguration>(*config);
        longer->ppo.total_timesteps = 2560;  // long enough to watch a few updates
        CheckWatchInView(longer);
        CheckSac(g);
    }
    manager.UnloadPlugin("com.cyxwiz.simulation.mujoco");
}

std::vector<cyxwiz::rl::OffPolicyUpdateMetrics> WaitSac(cyxwiz::rl::RLTrainingRun& run, double seconds) {
    std::vector<cyxwiz::rl::OffPolicyUpdateMetrics> all;
    const auto start = std::chrono::steady_clock::now();
    while (run.Active() &&
           std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < seconds) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        for (const auto& m : run.TakeSacReports()) all.push_back(m);
    }
    for (const auto& m : run.TakeSacReports()) all.push_back(m);
    return all;
}

// Two tensors the same: type, shape and every byte (the weights are float64
// on ArrayFire's CPU backend)
bool SameTensor(const cyxwiz::Tensor& a, const cyxwiz::Tensor& b) {
    return a.GetDataType() == b.GetDataType() && a.Shape() == b.Shape() && a.NumBytes() == b.NumBytes() &&
           std::memcmp(a.ReadData(), b.ReadData(), a.NumBytes()) == 0;
}

// Every parameter of two networks the same
bool SameParameters(cyxwiz::IExecutableModel& a, cyxwiz::IExecutableModel& b) {
    const auto pa = a.GetParameters(), pb = b.GetParameters();
    if (pa.size() != pb.size()) return false;
    for (const auto& [name, t] : pa) {
        const auto it = pb.find(name);
        if (it == pb.end() || !SameTensor(t, it->second)) return false;
    }
    return true;
}

// The SAC pattern on InvertedPendulum (the MuJoCo plugin loaded), trained the
// Engine's way (TOFIX216 S4)
void CheckSac(const rl_graph_test::Graph& ppo_graph) {
    rl_graph_test::Ids id{};
    auto g = rl_graph_test::Pattern(id, 64, gui::NodeType::ReLU, gui::NodeType::SAC);
    g.At(id.env).parameters["environment"] = "com.cyxwiz.simulation.mujoco:InvertedPendulum-v5";
    g.At(id.env).parameters["copies"] = "1";
    auto& sac = g.At(id.ppo).parameters;
    sac["total_timesteps"] = "3000";
    sac["learning_starts"] = "500";
    sac["buffer_size"] = "2000";  // smaller than the run: the ring wraps
    sac["batch_size"] = "64";
    const auto config = Compile(g);
    if (!config) return;
    Check(config->algorithm == cyxwiz::RLAlgorithm::SAC, "the SAC pattern compiles to SAC");
    auto run_config = std::make_shared<cyxwiz::RLTrainingConfiguration>(*config);
    run_config->sac.report_every = 500;
    std::string error;
    cyxwiz::rl::RLTrainingRun run;
    SelectCpu();
    run.SetWatch(cyxwiz::rl::RLTrainingRun::Watch::Preview, 1);
    Check(run.Start(run_config, rl_graph_test::Json(g), error), "the SAC run starts: " + error);
    const auto reports = WaitSac(run, 600.0);
    Check(run.GetState() == cyxwiz::rl::RLTrainingRun::State::Finished, "the SAC run finishes: " + run.Message());
    Check(run.TakeUpdates().empty(), "a SAC run reports no PPO updates");
    Check(reports.size() == 6 && reports.back().timesteps == 3000 && reports.back().report == 6,
          "a report every 500 steps to the end: " + std::to_string(reports.size()));
    if (!reports.empty()) {
        const auto& last = reports.back();
        Check(!reports.front().learning && last.learning && last.updates == 2500,
              "random steps first, then a gradient step a step: " + std::to_string(last.updates));
        Check(last.buffer_size == 2000 && last.buffer_capacity == 2000 &&
                  last.buffer_bytes == cyxwiz::rl::ReplayBytes(2000, 4, 1),
              "the replay buffer fills to its size");
        Check(std::isfinite(last.critic_loss) && std::isfinite(last.actor_loss) && last.ent_coef > 0.0f &&
                  last.ent_coef < 1.0f,
              "the losses are numbers and the learned entropy weight has fallen");
        bool falls = false;
        for (const auto& stat : last.environment_stats) falls = falls || stat.label == "Falls";
        Check(falls, "the environment's Falls reaches the SAC reports");
    }
    const auto preview = run.PreviewPolicy();
    Check(preview && preview->IsSac() && preview->policy && !preview->q1, "the live preview holds a copy of the actor");
    const auto trained = run.Trained();
    Check(trained && trained->IsSac() && trained->policy && trained->q1 && trained->q2 && trained->q1_target &&
              trained->q2_target && trained->log_ent_coef.NumElements() == 1 && !trained->value,
          "the run keeps the actor, the twin critics, their targets and log alpha");
    if (!trained) return;
    Check(trained->sac_reports.size() == reports.size() && trained->steps == 3000 && trained->optimizers.size() == 4,
          "with its reports, steps and the four optimizers (actor, q1, q2, entropy)");
    const auto& info = trained->config.environment;
    const auto obs = SampleObservations(16, 4);
    const auto sampled = cyxwiz::rl::PolicyActions(*trained, obs, 16, false, 3);
    const auto best = cyxwiz::rl::PolicyActions(*trained, obs, 16, true, 0);
    bool bounded = sampled.size() == 16 && best.size() == 16;
    for (const float a : sampled) bounded = bounded && a >= info.action_low[0] && a <= info.action_high[0];
    for (const float a : best) bounded = bounded && a >= info.action_low[0] && a <= info.action_high[0];
    Check(bounded, "SAC's actions are scaled into the bounds");
    Check(best == cyxwiz::rl::PolicyActions(*trained, obs, 16, true, 99) && sampled != best,
          "its best actions are deterministic, its sampled ones are not");
    CheckSaveAndLoad(*trained, "cyxwiz_rl_sac_policy_test.cyxmodel");
    {
        const auto path = std::filesystem::temp_directory_path() / "cyxwiz_rl_sac_networks_test.cyxmodel";
        Check(cyxwiz::rl::SaveTrainedPolicy(*trained, path, error), "the SAC policy saves: " + error);
        // The package names its learner (Open Model File reads it)
        const cyxwiz::ProbeResult probe = cyxwiz::formats::CyxModelFormat().Probe(path.string());
        const auto algorithm = probe.custom_metadata.find("rl_algorithm");
        Check(probe.has_rl_policy && algorithm != probe.custom_metadata.end() && algorithm->second == "SAC",
              "the SAC package names SAC as its learner");
        auto loaded = cyxwiz::rl::LoadTrainedPolicy(path, error);
        // Both live on the CPU route the run trained on: read them there, whatever this thread is on
        const cyxwiz::rl::PolicyDeviceScope on_its_device(*trained);
        Check(loaded && loaded->IsSac() && SameParameters(*loaded->q1, *trained->q1) &&
                  SameParameters(*loaded->q2_target, *trained->q2_target) &&
                  SameTensor(loaded->log_ent_coef, trained->log_ent_coef),
              "the critics, targets and entropy weight load back as saved: " + error);
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
    std::printf("SAC: saved and loaded; the policy against a PPO graph\n");
    // A SAC policy does not fit a PPO graph of the same task
    Check(cyxwiz::rl::PolicyMismatch(*trained, *config).empty(), "the SAC policy fits its own graph");
    cyxwiz::GraphCompiler compiler;
    const auto ppo = compiler.Compile(ppo_graph.nodes, ppo_graph.links, true);
    if (ppo.rl)
        Check(cyxwiz::rl::PolicyMismatch(*trained, *ppo.rl).find("trained by SAC") != std::string::npos,
              "a PPO graph refuses the SAC policy: " + cyxwiz::rl::PolicyMismatch(*trained, *ppo.rl));
    std::printf("SAC: the Stress test\n");
    CheckStressRun(trained);
    std::printf("SAC: a ghost beside the policy\n");
    CheckGhostPlayback(trained, true);
    std::printf("SAC: Stop\n");

    // Stop: the run says where it stopped, and keeps what it trained
    auto longer = std::make_shared<cyxwiz::RLTrainingConfiguration>(*run_config);
    longer->sac.total_timesteps = 1000000;
    cyxwiz::rl::RLTrainingRun stopped;
    SelectCpu();
    Check(stopped.Start(longer, rl_graph_test::Json(g), error), "a long SAC run starts: " + error);
    std::vector<cyxwiz::rl::OffPolicyUpdateMetrics> seen;
    const auto start = std::chrono::steady_clock::now();
    while (stopped.Active() && std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < 300.0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        for (const auto& m : stopped.TakeSacReports()) seen.push_back(m);
        if (!seen.empty() && seen.back().updates > 0) break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    stopped.Stop();
    for (const auto& m : WaitSac(stopped, 60.0)) seen.push_back(m);
    Check(stopped.GetState() == cyxwiz::rl::RLTrainingRun::State::Stopped, "Stop ends the SAC run: " + stopped.Message());
    const auto kept = stopped.Trained();
    Check(kept && !seen.empty() && kept->steps == seen.back().timesteps && seen.back().timesteps % 500 != 0,
          "a stopped SAC run reports where it stopped and keeps its policy");
}

void CheckStopAndFailure() {
    rl_graph_test::Ids id{};
    auto g = rl_graph_test::Pattern(id);
    g.At(id.ppo).parameters["n_steps"] = "64";
    const auto config = Compile(g);
    if (!config) return;
    cyxwiz::rl::RLTrainingRun run;
    std::string error;
    SelectCpu();
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
    SelectCpu();
    Check(run.Start(copy, "", error), "a run with a vanished environment starts: " + error);
    Wait(run, 30.0);
    Check(run.GetState() == cyxwiz::rl::RLTrainingRun::State::Failed && !run.Message().empty(),
          "it fails with the reason: " + run.Message());
}

// The run's device is the chosen route, reported as such; Strict refuses a
// route that cannot start; Compatibility falls back to the CPU and says so
void CheckDevice() {
    rl_graph_test::Ids id{};
    auto g = rl_graph_test::Pattern(id);
    g.At(id.ppo).parameters["n_steps"] = "64";
    g.At(id.ppo).parameters["total_timesteps"] = "1024";
    g.At(id.ppo).parameters["stop_at_return"] = "false";
    const auto config = Compile(g);
    if (!config) return;
    std::string error;

    cyxwiz::rl::RLTrainingRun run;
    SelectCpu();
    Check(run.Start(config, rl_graph_test::Json(g), error), "a CPU run starts: " + error);
    Wait(run, 120.0);
    const auto device = run.Device();
    Check(run.GetState() == cyxwiz::rl::RLTrainingRun::State::Finished, "the CPU run finishes: " + run.Message());
    Check(device.valid && device.effective_backend == "arrayfire_cpu" && device.requested_backend == "arrayfire_cpu" &&
              !device.selection_fallback_applied,
          "the run reports the chosen CPU route: " + device.Describe());
    const auto trained = run.Trained();
    Check(trained && trained->device_type == cyxwiz::DeviceType::CPU, "its policy lives on the CPU");
    Check(cyxwiz::TrainingTraceCollector::LatestTrace().effective_backend == "arrayfire_cpu",
          "the training trace records the run's device");
    Check(!cyxwiz::HasActiveExecutionDeviceContext(), "the device is free once the run ends");

    // A route that cannot start (no CPU device 7)
    cyxwiz::rl::RLTrainingRun strict;
    SelectRoute(cyxwiz::DeviceType::CPU, 7, cyxwiz::ExecutionPolicy::Strict);
    Check(strict.Start(config, "", error), "a Strict run on a missing route starts: " + error);
    Wait(strict, 60.0);
    Check(strict.GetState() == cyxwiz::rl::RLTrainingRun::State::Failed &&
              strict.Message().find("The training device did not start") != std::string::npos,
          "Strict refuses it with the reason: " + strict.Message());

    cyxwiz::rl::RLTrainingRun compatible;
    SelectRoute(cyxwiz::DeviceType::CPU, 7, cyxwiz::ExecutionPolicy::Compatibility);
    Check(compatible.Start(config, "", error), "a Compatibility run on a missing route starts: " + error);
    Wait(compatible, 120.0);
    const auto fallback = compatible.Device();
    Check(compatible.GetState() == cyxwiz::rl::RLTrainingRun::State::Finished &&
              fallback.selection_fallback_applied && fallback.effective_backend == "arrayfire_cpu" &&
              !fallback.error.empty(),
          "Compatibility falls back to the CPU and says so: " + fallback.Describe());
}

// With a CUDA device: a run on the CUDA route chosen the Engine's way keeps
// its policy there; this thread (not on CUDA) acts and saves through the
// policy's device; loaded on the CPU route it acts there. Skipped (said so)
// without a CUDA device.
void CheckGpuPolicy() {
    bool cuda = false;
    for (const auto& d : cyxwiz::Device::GetAvailableDevices()) cuda = cuda || d.type == cyxwiz::DeviceType::CUDA;
    if (!cuda) {
        std::printf("no CUDA device: the GPU policy checks are skipped\n");
        return;
    }
    rl_graph_test::Ids id{};
    auto g = rl_graph_test::Pattern(id);
    g.At(id.ppo).parameters["n_steps"] = "64";
    g.At(id.ppo).parameters["total_timesteps"] = "2048";
    g.At(id.ppo).parameters["stop_at_return"] = "false";
    const auto config = Compile(g);
    if (!config) return;
    std::string error;
    cyxwiz::rl::RLTrainingRun run;
    SelectRoute(cyxwiz::DeviceType::CUDA, 0, cyxwiz::ExecutionPolicy::Strict);
    Check(run.Start(config, rl_graph_test::Json(g), error), "a CUDA run starts: " + error);
    Wait(run, 300.0);
    Check(run.GetState() == cyxwiz::rl::RLTrainingRun::State::Finished, "the CUDA run finishes: " + run.Message());
    const auto trained = run.Trained();
    Check(trained && trained->device_type == cyxwiz::DeviceType::CUDA, "its policy lives on CUDA");
    if (!trained) return;
    // This thread acts and saves whatever device it is on: the policy's scope switches to CUDA and back
    const auto obs = SampleObservations(8, trained->config.observation_size);
    const auto actions = cyxwiz::rl::PolicyActions(*trained, obs, 8, true, 0);
    Check(actions.size() == 8, "a CUDA policy acts from another thread's device");
    const auto path = std::filesystem::temp_directory_path() / "cyxwiz_rl_gpu_policy.cyxmodel";
    Check(cyxwiz::rl::SaveTrainedPolicy(*trained, path, error), "a CUDA policy saves: " + error);
    SelectCpu();
    auto loaded = cyxwiz::rl::LoadTrainedPolicy(path, error);
    Check(loaded && loaded->device_type == cyxwiz::DeviceType::CPU, "it loads on the CPU route: " + error);
    if (loaded) {
        const auto cpu_actions = cyxwiz::rl::PolicyActions(*loaded, obs, 8, true, 0);
        Check(cpu_actions.size() == 8, "and acts there");
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

}  // namespace

// The stress test's conditions (board 6): its settings over the policy's own,
// levels, seeds, the CSV, and refusing an environment without grounds
void CheckStressTestConditions() {
    using namespace cyxwiz::rl;
    const std::map<std::string, std::string> own{{"look_ahead", "1"}, {"grounds", "rough,steps"}, {"curriculum", "1500"},
                                                 {"friction_percent", "20"}, {"frame_skip", "5"}};
    auto steps = StressSettings(own, StressKind::Steps, 0.75);
    Check(steps["grounds"] == "steps" && steps["difficulty_from"] == "0.75" && steps["difficulty_to"] == "0.75" &&
              steps["curriculum"] == "0" && steps["friction_percent"] == "0" && steps["push_every"] == "0" &&
              steps["look_ahead"] == "1" && steps["frame_skip"] == "5",
          "a ground level: that ground at that difficulty, no random physics, the policy's own Look ahead kept");
    auto kicks = StressSettings(own, StressKind::Kicks, 0.5);
    Check(kicks["grounds"].empty() && kicks["push_every"] == "1.5" && kicks["push_speed"] == "2",
          "kicks: no ground, a push every 1.5 s of up to 4 m/s times the difficulty");
    Check(StressSettings(own, StressKind::Kicks, 0.0)["push_every"] == "0", "kicks at 0: none");
    // Thrown objects (TOFIX216 T3): a throw every 1.5 s of up to 5 kg at up to 10 m/s times the difficulty
    auto thrown = StressSettings(own, StressKind::Thrown, 0.5);
    Check(thrown["grounds"].empty() && thrown["push_every"] == "0" && thrown["throw_every"] == "1.5" &&
              thrown["throw_object"] == "any" && thrown["throw_mass_from"] == "2.5" && thrown["throw_mass_to"] == "2.5" &&
              thrown["throw_speed_from"] == "5" && thrown["throw_speed_to"] == "5" && thrown["throw_size_to"] == "0.2",
          "thrown objects: no ground, a throw every 1.5 s, 2.5 kg at 5 m/s at difficulty 0.5");
    Check(StressSettings(own, StressKind::Thrown, 0.0)["throw_every"] == "0", "thrown objects at 0: none");
    auto trained_with_throws = own;
    trained_with_throws["throw_every"] = "2";
    Check(StressSettings(trained_with_throws, StressKind::Steps, 0.5)["throw_every"] == "0" &&
              !StressSettings(own, StressKind::Steps, 0.5).count("throw_every"),
          "a policy trained with throws is tested on the disturbance only; one without keeps no throw key");
    Check(std::string(StressKindName(StressKind::Thrown)) == "Thrown objects" && kStressKindCount == 6,
          "six disturbances, the last Thrown objects");
    StressTestSpec spec;
    spec.levels = 5;
    Check(StressDifficulty(spec, 0) == 0.0 && StressDifficulty(spec, 4) == 1.0 && StressDifficulty(spec, 1) == 0.25,
          "five levels: 0, 0.25 ... 1");
    Check(StressSeed(spec, 2) == StressSeed(spec, 2) && StressSeed(spec, 1) != StressSeed(spec, 2), "a seed a level");
    StressPolicyResult a{"walk", {{0.0, {100.0, 300.0}, 0}, {1.0, {10.0, 30.0}, 2}}};
    StressPolicyResult b{"robust", {{0.0, {200.0, 200.0}, 0}}};
    Check(StressResultsCsv({a, b}, 2) ==
              "difficulty,\"walk mean return\",\"walk falls\",\"robust mean return\",\"robust falls\",episodes\n"
              "0,200.00,0,200.00,0,2\n1,20.00,2,,,2\n",
          "the CSV: a row a level, each policy's mean return and falls");
    cyxwiz::plugin::PluginEnvironmentRegistry::Environment cartpole;
    Check(cyxwiz::plugin::PluginEnvironmentRegistry::Instance().Find("cyxwiz.builtin:CartPole-v1", cartpole) &&
              !SupportsStress(cartpole.info),
          "CartPole has no grounds or pushes: no stress test");
}

int main(int argc, char** argv) {
    // Unbuffered: what ran last stays in a redirected log if the run crashes
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    cyxwiz::SetRLEnvironmentLookup([](const std::string& key, const std::map<std::string, std::string>&,
                                      cyxwiz::plugin::PluginEnvironmentInfo& info, std::string&) {
        cyxwiz::plugin::PluginEnvironmentRegistry::Environment e;
        if (!cyxwiz::plugin::PluginEnvironmentRegistry::Instance().Find(key, e)) return false;
        info = e.info;
        return true;
    });
    // Routes certified as the installer's qualification would certify them
    cyxwiz::test::InstallQualifiedRouteSnapshot();
    CheckStressTestConditions();
    CheckGhostSettings();
    CheckWatchLeavesTrainingAlone(gui::NodeType::Tanh);
    // Dropout draws its masks from the default random engine while it trains
    CheckWatchLeavesTrainingAlone(gui::NodeType::Dropout);
    CheckDevice();
    CheckStopAndFailure();
    CheckCartPole();
    CheckGpuPolicy();
    if (argc > 1) CheckContinuous(std::filesystem::absolute(argv[1]));
    if (failures == 0) std::printf("test_rl_graph_training: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
