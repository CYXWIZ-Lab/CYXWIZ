// The stability gate's runs (TOFIX195 PPO, TOFIX216 SAC, TD3 and DQN): one task and seed
// per process, trained the way Studio's Train does it (the learner's pattern
// with its node's "Tuned for" settings, the RL compile, RLTrainingRun). Writes
// the run's reports and outcome as JSON; the gate compares them with
// Stable-Baselines3 on the same environments
// (resources/rl_reference/ppo_sb3_curves.json, sac_sb3_curves.json, td3_sb3_curves.json,
// dqn_sb3_curves.json).
//
//   cyxwiz-rl-stability-gate [--sac | --td3 | --dqn] <task> <seed> <output.json> [<environment plugin folder>]
//
// With a plugin folder the task is that plugin's (its id is read from its plugin.json): the MuJoCo
// plugin's, or another plugin's that offers the same task ids, as CyxSim's gates run its environments.
// Without one the task is a built-in environment; a task no built-in offers needs the folder.
//
// A learner's tasks, each task's target return and its step budget are the ones its SB3 reference
// file records ("target", "total_timesteps" of the task), found as the Engine finds the file
// (rl_reference.h, ReadReferenceFile): they are written nowhere else.
// So the tool needs those files: resources/rl_reference beside the executable,
// or the source tree's cyxwiz-engine/resources/rl_reference three folders
// above it, as in a build tree (<build>/bin/Release). A copy of the
// executable alone, elsewhere, stops and names the folders it looked in.
//
// PPO: task CartPole-v1, InvertedPendulum-v5 or HalfCheetah-v5. CartPole and
// InvertedPendulum stop at their target return within their budget;
// HalfCheetah runs all its steps with RL Zoo's two layers of 256 and ReLU.
// --sac: InvertedPendulum-v5 (SB3's SAC defaults, one copy, stops at its target)
// or HalfCheetah-v5 (RL Zoo, all its steps: its file records no target); both
// with SAC's two layers of 256 and ReLU.
// --td3: the same tasks and budgets with the TD3 node's defaults (SB3's, RL
// Zoo's noise 0.1) on InvertedPendulum and RL Zoo's settings on HalfCheetah;
// both with TD3's layers of 400 and 300 and ReLU.
// --dqn: CartPole-v1 with the DQN node's Tuned for (RL Zoo's settings and
// steps) and RL Zoo's Q layers of 256 and 256 and ReLU; it runs all its
// steps, recording when the mean return of the last 100 episodes first
// reaches the target.
// The report's "target" is the mark "reached_target_at" is measured against.
// Where the learner's file records none (SAC's and TD3's HalfCheetah) it is
// PPO's recorded target for the same task, and a note says so.
// CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|cpu picks the device (default
// CPU). The run takes it the Engine's way (TOFIX206): the pending device
// selection and a Strict session execution policy, as Preferences > Devices sets them,
// over routes certified the installer's way; the report records the device
// the run actually trained on. CYXWIZ_GATE_NO_PROVIDERS=1 turns the neural
// providers off (the networks run as ArrayFire operations) for comparisons;
// CYXWIZ_GATE_STEPS=<n> caps the run at n steps for measurements.
// CYXWIZ_EXPECT_PATH=nvidia|opencl|oneapi|<provider id> requires the path the
// gate measures (computation_truth/expected_compute_path.h): the run FAILS,
// naming the op and the reason, when its networks, its Adam step or one of
// PPO's ops did not run on that provider. A gate measures a path; without
// this a CUDA run on ArrayFire alone passes as a CUDA result. The report
// records the path either way ("compute_path", "compute_parts").
// CYXWIZ_GATE_PRINT_CONFIG=1 prints the run's configuration as JSON (the
// graph with every node's settings, the compiled budget, stop and layers) and
// trains nothing: to check that a change leaves the gate's protocol as it was.
// Standard output then carries that JSON alone; every log line goes to
// standard error. tests/check_rl_gate_configs.py (ctest rl_gate_configs)
// compares it, for every task and learner, with tests/data/rl_gate_configs.json.

#include "../src/core/backend_placement_capabilities.h"
#include "../src/core/execution_device_preferences.h"
#include "../src/core/graph_compiler.h"
#include <cyxwiz/neural_provider.h>
#include "../src/core/rl/rl_reference.h"
#include "../src/core/rl/rl_training_run.h"
#include "../src/core/rl_graph_compile.h"
#include "../src/core/rl_graph_presentation.h"
#include "../src/plugin/plugin_manager.h"
#include "../src/plugin/registries/plugin_environment_registry.h"
#include "computation_truth/expected_compute_path.h"
#include "rl_graph_test_support.h"
#include "route_qualification_test_fixture.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <thread>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

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
    cyxwiz::ClearSessionExecutionPolicy();
    cyxwiz::SetSessionExecutionPolicy(cyxwiz::ExecutionPolicy::Strict);
    return true;
}

// What a learner's SB3 reference file records for a task
struct GateReference {
    std::optional<float> target;    // none: the file records no target (null)
    long long total_timesteps = 0;  // the recorded runs' step budget
};

// The task's entry in the reference file `file` of the Engine's
// resources/rl_reference; nullopt with the reason when the file is not found,
// is not a reference file, or does not record the task (not_recorded)
std::optional<GateReference> ReadGateReference(const std::string& file, const std::string& task, std::string& error,
                                               bool* not_recorded = nullptr) {
    if (not_recorded) *not_recorded = false;
    const std::string text = cyxwiz::rl::ReadReferenceFile(file);
    if (text.empty()) {
        error = "the SB3 reference " + file + " was not found; the gate takes its tasks, targets and step budgets "
                "from it. It looked in:";
        for (const auto& folder : cyxwiz::rl::ReferenceFileFolders()) error += "\n  " + folder.string();
        error += "\nRun the tool from its build tree, or put the Engine's resources/rl_reference beside it.";
        return std::nullopt;
    }
    try {
        const auto document = nlohmann::json::parse(text);
        for (const auto& entry : document.at("tasks")) {
            if (entry.at("id").get<std::string>() != task) continue;
            GateReference reference;
            reference.total_timesteps = entry.at("total_timesteps").get<long long>();
            if (reference.total_timesteps <= 0) throw std::runtime_error("total_timesteps is not positive");
            const auto& target = entry.at("target");
            if (!target.is_null()) reference.target = target.get<float>();
            return reference;
        }
    } catch (const std::exception& e) {
        error = "the SB3 reference " + file + " cannot be read for " + task + ": " + e.what();
        return std::nullopt;
    }
    if (not_recorded) *not_recorded = true;
    error = file + " records no task " + task;
    return std::nullopt;
}

// A whole number as the nodes' settings write it ("950"), else its shortest form
std::string SettingText(float value) {
    if (value == std::floor(value) && std::abs(value) < 1e9f) return std::to_string(static_cast<long long>(value));
    char text[32];
    std::snprintf(text, sizeof(text), "%.9g", static_cast<double>(value));
    return text;
}

// CYXWIZ_GATE_PRINT_CONFIG: standard output is kept for the configuration.
// Whatever else writes to it (the Engine's and the plugins' log lines, from
// any library) goes to standard error from here on; the returned descriptor
// is the real standard output, or -1 if it could not be kept.
int KeepStandardOutputForTheConfiguration() {
    std::fflush(stdout);
#ifdef _WIN32
    const int kept = _dup(_fileno(stdout));
    if (kept < 0 || _dup2(_fileno(stderr), _fileno(stdout)) != 0) return -1;
#else
    const int kept = dup(fileno(stdout));
    if (kept < 0 || dup2(fileno(stderr), fileno(stdout)) < 0) return -1;
#endif
    return kept;
}

bool WriteToDescriptor(int descriptor, const std::string& text) {
    std::size_t done = 0;
    while (done < text.size()) {
#ifdef _WIN32
        const auto written = _write(descriptor, text.data() + done, static_cast<unsigned int>(text.size() - done));
#else
        const auto written = write(descriptor, text.data() + done, text.size() - done);
#endif
        if (written <= 0) return false;
        done += static_cast<std::size_t>(written);
    }
    return true;
}

// A difference between the gate's run and its reference file: said, never applied silently
void Note(const std::string& what) {
    std::fprintf(stderr, "cyxwiz-rl-stability-gate: note: %s\n", what.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    const bool sac = argc > 1 && std::string(argv[1]) == "--sac";
    const bool td3 = argc > 1 && std::string(argv[1]) == "--td3";
    const bool dqn = argc > 1 && std::string(argv[1]) == "--dqn";
    const bool off_policy = sac || td3 || dqn;
    if (off_policy) {
        --argc;
        ++argv;
    }
    if (argc < 4)
        return Fail("usage: cyxwiz-rl-stability-gate [--sac | --td3 | --dqn] <task> <seed> <output.json> "
                    "[<environment plugin folder>]");
    const char* print = std::getenv("CYXWIZ_GATE_PRINT_CONFIG");
    const bool print_configuration = print && print[0] == '1';
    int configuration_output = -1;
    if (print_configuration) {  // before anything logs
        configuration_output = KeepStandardOutputForTheConfiguration();
        if (configuration_output < 0) return Fail("standard output could not be kept for the configuration");
    }
    const std::string task = argv[1];
    const std::string seed = argv[2];
    const std::filesystem::path output = argv[3];
    if (!SelectRequestedRoute()) return Fail("CYXWIZ_TEST_ARRAYFIRE_BACKEND names no device route (cpu, cuda, opencl)");
    // CYXWIZ_GATE_NO_PROVIDERS=1: no neural provider (the networks as ArrayFire operations), to compare
    if (const char* off = std::getenv("CYXWIZ_GATE_NO_PROVIDERS"); off && off[0] == '1')
        cyxwiz::SetNeuralProvidersDisabledForTesting(true);
    // The learner's gate tasks, targets and budgets are its reference file's
    const std::string learner_name = sac ? "SAC" : td3 ? "TD3" : dqn ? "DQN" : "PPO";
    const std::string reference_file =
        sac ? "sac_sb3_curves.json" : td3 ? "td3_sb3_curves.json" : dqn ? "dqn_sb3_curves.json" : "ppo_sb3_curves.json";
    std::string reference_error;
    bool not_recorded = false;
    const auto reference = ReadGateReference(reference_file, task, reference_error, &not_recorded);
    if (!reference) return Fail(not_recorded ? "no " + learner_name + " gate task " + task : reference_error);
    // A plugin folder given: the task is that plugin's; none: a built-in environment
    const bool from_plugin = argc >= 5;
    cyxwiz::SetRLEnvironmentLookup([](const std::string& key, const std::map<std::string, std::string>&,
                                      cyxwiz::plugin::PluginEnvironmentInfo& info, std::string&) {
        cyxwiz::plugin::PluginEnvironmentRegistry::Environment e;
        if (!cyxwiz::plugin::PluginEnvironmentRegistry::Instance().Find(key, e)) return false;
        info = e.info;
        return true;
    });
    auto& manager = cyxwiz::plugin::PluginManager::Instance();
    // The plugin that offers the task: the Engine's MuJoCo plugin, or any plugin offering the same
    // task ids (CyxSim); its id is read from the folder's plugin.json
    std::string plugin_id;
    if (!from_plugin) {
        cyxwiz::plugin::PluginEnvironmentRegistry::Environment builtin;
        if (!cyxwiz::plugin::PluginEnvironmentRegistry::Instance().Find("cyxwiz.builtin:" + task, builtin))
            return Fail(task + " needs the folder of a plugin that offers it (the MuJoCo plugin's)");
    } else {
        const std::filesystem::path folder = std::filesystem::absolute(argv[4]);
        std::ifstream manifest(folder / "plugin.json");
        const nlohmann::json info = nlohmann::json::parse(manifest, nullptr, false);
        if (info.is_discarded() || !info.contains("id") || !info["id"].is_string())
            return Fail("no plugin id in " + (folder / "plugin.json").string());
        plugin_id = info["id"].get<std::string>();
        std::string error;
        if (!manager.LoadPlugin(folder, &error)) return Fail("the plugin " + plugin_id + ": " + error);
        if (!manager.InitializePlugin(plugin_id)) return Fail("the plugin " + plugin_id + " did not start");
        cyxwiz::plugin::PluginEnvironmentRegistry::Environment offered;
        if (!cyxwiz::plugin::PluginEnvironmentRegistry::Instance().Find(plugin_id + ":" + task, offered)) {
            manager.UnloadPlugin(plugin_id);
            return Fail("the plugin " + plugin_id + " offers no " + task +
                        " (without a plugin folder the task is a built-in environment)");
        }
    }

    const bool cheetah = task == "HalfCheetah-v5";
    // SAC's and TD3's InvertedPendulum have no RL Zoo entry: the node's defaults
    const auto preset = sac   ? cyxwiz::rl_graph::TunedSacSettings(task)
                        : td3 ? cyxwiz::rl_graph::TunedTd3Settings(task)
                        : dqn ? cyxwiz::rl_graph::TunedDqnSettings(task)
                              : cyxwiz::rl_graph::TunedPpoSettings(task);
    if (!preset && !((sac || td3) && !cheetah)) return Fail("no tuned settings for " + task);
    rl_graph_test::Ids id{};
    auto g = sac       ? rl_graph_test::Pattern(id, 256, gui::NodeType::ReLU, gui::NodeType::SAC)
             : td3     ? rl_graph_test::Pattern(id, 400, gui::NodeType::ReLU, gui::NodeType::TD3)
             : dqn     ? rl_graph_test::DqnPattern(id, 256, gui::NodeType::ReLU)
             : cheetah ? rl_graph_test::Pattern(id, 256, gui::NodeType::ReLU)
                       : rl_graph_test::Pattern(id);
    if (td3)  // TD3's second layer is 300 (SB3's and RL Zoo's [400, 300])
        for (const int dense : {id.p2, id.v2}) g.At(dense).parameters["units"] = "300";
    g.At(id.env).parameters["environment"] =
        (from_plugin ? plugin_id + ":" : std::string("cyxwiz.builtin:")) + task;
    if (preset) {
        for (const auto& [k, v] : preset->environment) g.At(id.env).parameters[k] = v;
        for (const auto& [k, v] : preset->learner) g.At(id.ppo).parameters[k] = v;
        for (const auto& [k, v] : preset->policy_head) g.At(id.phead).parameters[k] = v;
    }
    g.At(id.ppo).parameters["seed"] = seed;
    // The mark the report's "reached_target_at" is measured against: the
    // learner's recorded target, else (its file records none) PPO's for the task
    float target = std::numeric_limits<float>::quiet_NaN();
    if (reference->target) {
        target = *reference->target;
    } else {
        std::string ppo_error;
        const auto ppo = ReadGateReference("ppo_sb3_curves.json", task, ppo_error);
        if (ppo && ppo->target) {
            target = *ppo->target;
            Note(reference_file + " records no target for " + task + ": the run goes its whole budget, and the "
                 "report marks the reach of PPO's recorded target " + SettingText(target));
        } else {
            Note(reference_file + " records no target for " + task + ": the run goes its whole budget, and the "
                 "report marks no reach");
        }
    }
    if (dqn) {  // RL Zoo's steps, all of them (the reach of the target is recorded, not a stop)
        g.At(id.ppo).parameters["stop_at_return"] = "false";
    } else if (!cheetah) {  // the budget of the SB3 reference; HalfCheetah keeps its tuned settings' steps
        g.At(id.ppo).parameters["total_timesteps"] = std::to_string(reference->total_timesteps);
        g.At(id.ppo).parameters["stop_at_return"] = "true";
        if (off_policy) {  // one copy, as the SB3 reference (the Environment node's default is more)
            g.At(id.env).parameters["copies"] = "1";
            if (!reference->target) return Fail(reference_file + " records no target for " + task + " to stop at");
            g.At(id.ppo).parameters["target_return"] = SettingText(*reference->target);  // the nodes' own default is 4000
        }
    }
    // CYXWIZ_GATE_STEPS=<n>: a shorter run, for measurements (not the gate)
    const char* step_cap = std::getenv("CYXWIZ_GATE_STEPS");
    const bool capped = step_cap && std::atoll(step_cap) > 0;
    if (capped) g.At(id.ppo).parameters["total_timesteps"] = step_cap;
    cyxwiz::GraphCompiler compiler;
    const auto compiled = compiler.Compile(g.nodes, g.links, true);
    if (!compiled.rl) return Fail("the graph does not compile: " + compiled.error_message);
    const auto& config = *compiled.rl;
    // Where the tuned settings, not this tool, set the budget or the stop (HalfCheetah's and DQN's
    // steps, PPO's target): a value other than the reference file's is said, and the run keeps its own
    const long long budget = sac   ? config.sac.total_timesteps
                             : td3 ? config.td3.total_timesteps
                             : dqn ? config.dqn.total_timesteps
                                   : config.ppo.total_timesteps;
    const float stops_at = sac   ? config.sac.target_return
                           : td3 ? config.td3.target_return
                           : dqn ? config.dqn.target_return
                                 : config.ppo.target_return;
    if (!capped && budget != reference->total_timesteps)
        Note("the run's " + std::to_string(budget) + " steps differ from the " +
             std::to_string(reference->total_timesteps) + " that " + reference_file + " records for " + task);
    if (std::isfinite(stops_at) && (!reference->target || stops_at != *reference->target))
        Note("the run stops at a mean return of " + SettingText(stops_at) + ", but " + reference_file + " records " +
             (reference->target ? "a target of " + SettingText(*reference->target) : std::string("no target")) +
             " for " + task);

    // The policy's hidden layers as compiled: "64 Tanh-64 Tanh"
    std::string layers;
    for (const auto& layer : config.policy_layers) {
        if (layer.units > 0) layers += (layers.empty() ? "" : "-") + std::to_string(layer.units);
        else layers += std::string(" ") + cyxwiz::backend_placement::LayerTypeName(layer.type);
    }
    if (print_configuration) {
        const nlohmann::json configuration = {
            {"task", task},
            {"learner", cyxwiz::RLAlgorithmName(config.algorithm)},
            {"copies", config.copies},
            {"total_timesteps", budget},
            {"stops_at_return", Number(stops_at)},  // null: the run goes its whole budget
            {"target", Number(target)},
            {"layers", layers},
            {"graph", nlohmann::json::parse(rl_graph_test::Json(g))},
        };
        const bool written = WriteToDescriptor(configuration_output, configuration.dump(2) + "\n");
        if (from_plugin) manager.UnloadPlugin(plugin_id);
        return written ? EXIT_SUCCESS : Fail("the configuration could not be written to standard output");
    }

    const cyxwiz::test::ExpectedComputePath expected = cyxwiz::test::ReadExpectedComputePath();
    cyxwiz::rl::RLTrainingRun run;
    std::string error;
    const auto start = std::chrono::steady_clock::now();
    if (!run.Start(compiled.rl, rl_graph_test::Json(g), error)) return Fail(error);
    nlohmann::json rows = nlohmann::json::array();
    std::string device;        // where the run trains, once its preflight is done
    std::string compute_path;  // how the last update ran (fused kernels or ArrayFire)
    std::vector<cyxwiz::rl::ComputePathPart> compute_parts;  // the same, part by part
    const auto follow_path = [&](const std::string& path, const std::vector<cyxwiz::rl::ComputePathPart>& parts) {
        if (path != compute_path) {  // at the first update, and when it changes
            std::printf("compute_path: %s\n", path.c_str());
            std::fflush(stdout);
        }
        compute_path = path;
        compute_parts = parts;
    };
    const auto parts_json = [&]() {
        nlohmann::json parts = nlohmann::json::array();
        for (const auto& part : compute_parts)
            parts.push_back({{"what", part.what},
                             {"provider", part.provider_id},  // empty: ArrayFire
                             {"mixed", part.mixed},
                             {"reason", part.reason},
                             {"detail", part.detail}});
        return parts;
    };
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
            {"learner", cyxwiz::RLAlgorithmName(config.algorithm)},
            {"environment", g.At(id.env).parameters["environment"]},  // which plugin's task ran
            {"total_timesteps", budget},
            {"target", Number(target)},
            {"layers", layers},
            {"device", device},
            {"compute_path", compute_path},
            {"compute_parts", parts_json()},
            {"expected_path", expected.set ? nlohmann::json(expected.provider_id) : nlohmann::json(nullptr)},
            {"outcome", outcome},
            {"reached_target_at", reached < 0 ? nlohmann::json(nullptr) : nlohmann::json(reached)},
            {"final_mean_return", Number(last_return)},
            {"steps", rows.empty() ? 0 : rows.back()[0].get<long long>()},
            {"seconds", std::round(seconds * 10.0) / 10.0},
            {"columns", dqn ? nlohmann::json{"timesteps", "mean_return", "loss", "exploration_rate", "mean_q",
                                             "steps_per_second"}
                        : off_policy ? nlohmann::json{"timesteps", "mean_return", "critic_loss", "actor_loss",
                                                      "ent_coef", "mean_q", "steps_per_second"}
                            : nlohmann::json{"timesteps", "mean_return", "policy_loss", "value_loss", "approx_kl",
                                             "clip_fraction", "explained_variance", "action_std", "steps_per_second"}},
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
            follow_path(m.compute_path, m.compute_parts);
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
        for (const auto& m : run.TakeOffPolicyReports()) {
            follow_path(m.compute_path, m.compute_parts);
            if (dqn)
                rows.push_back({m.timesteps, Number(m.mean_episode_return), Number(m.critic_loss),
                                Number(m.exploration_rate), Number(m.mean_q), Number(m.steps_per_second)});
            else
                rows.push_back({m.timesteps, Number(m.mean_episode_return), Number(m.critic_loss),
                                Number(m.actor_loss), Number(m.ent_coef), Number(m.mean_q), Number(m.steps_per_second)});
            if (std::isfinite(m.mean_episode_return)) last_return = m.mean_episode_return;
            if (reached < 0 && m.episodes >= 100 && m.mean_episode_return >= target) reached = m.timesteps;
            if (m.report % 50 == 0) {
                std::printf("%s seed %s: %lld steps, mean return %.1f, %.0f steps/s\n", task.c_str(), seed.c_str(),
                            static_cast<long long>(m.timesteps), m.mean_episode_return, m.steps_per_second);
                std::fflush(stdout);
            }
            if (m.report % 10 == 0) write("Running");
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
    // CYXWIZ_EXPECT_PATH: a run that took another path than the one the gate
    // measures is a failed run, whatever its returns
    const std::vector<std::string> violations =
        expected.set ? cyxwiz::test::ComputePathViolations(compute_parts, expected.provider_id)
                     : std::vector<std::string>{};
    const bool on_path = cyxwiz::test::ReportComputePathViolations(expected, violations, "cyxwiz-rl-stability-gate");
    write(on_path ? run.Message() : "Wrong compute path: " + violations.front());
    if (from_plugin) manager.UnloadPlugin(plugin_id);
    return finished && on_path ? EXIT_SUCCESS : EXIT_FAILURE;
}
