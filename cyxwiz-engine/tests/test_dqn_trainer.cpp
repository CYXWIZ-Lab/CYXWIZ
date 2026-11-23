// The Engine's DQN (core/rl/dqn_trainer on the off-policy loop; TOFIX216
// DQN V3), headless:
// - DqnLearner replays four gradient steps and two target updates of a real
//   Stable-Baselines3 DQN (fixture dqn_step_sb3.json from
//   generate_dqn_step_fixture.py: the gradients' norm clipped, a target
//   moved halfway) and ends with SB3's weights, target weights and losses;
// - DqnTrainer learns a small task with a choice of actions (pick the side
//   an observation is on), epsilon falls as SB3's schedule says, the same
//   seed gives the same run, copies count as SB3 counts them, bad setups are
//   refused with the reason, and what a run leaves can be taken and copied;
// - with --cartpole it trains the built-in CartPole-v1 at RL Zoo's settings.
//
//   test_dqn_trainer [--cartpole]
//
// CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|cpu picks the device.

#include "../src/core/executable_model.h"
#include "../src/core/rl/dqn_trainer.h"
#include "../src/core/rl/replay_buffer.h"
#include "../src/plugin/registries/plugin_environment_registry.h"
#include "algorithms/arrayfire_backend_utils.h"
#include "computation_truth/test_device_selection.h"

#include <cyxwiz/rl/dqn.h>
#include <cyxwiz/sequential.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
using namespace cyxwiz;
using cyxwiz::plugin::EnvironmentBatch;
using cyxwiz::plugin::PluginEnvironmentRegistry;

int failures = 0;

void Check(bool ok, const std::string& what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

std::vector<float> Host(const Tensor& t) {
    const float* data = t.ReadData<float>();
    return std::vector<float>(data, data + t.NumElements());
}

Tensor FromFixture(const json& f) {
    std::vector<size_t> shape = f.at("shape").get<std::vector<size_t>>();
    const std::vector<float> values = f.at("values").get<std::vector<float>>();
    if (shape.empty()) shape = {1};
    return Tensor(shape, values.data());
}

// Dense(in, hidden) - ReLU [- Dense(hidden, hidden) - ReLU] - Dense(hidden, out)
std::unique_ptr<IExecutableModel> Mlp(size_t in, size_t hidden, size_t out, int layers) {
    auto model = std::make_shared<SequentialModel>();
    size_t width = in;
    for (int i = 0; i < layers; ++i) {
        model->Add<LinearModule>(width, hidden);
        model->Add<ReLUModule>();
        width = hidden;
    }
    model->Add<LinearModule>(width, out);
    auto executable = std::make_unique<SequentialExecutableModel>(model);
    rl::InitializeLikeTorchLinear(*executable);
    return executable;
}

rl::DqnNetworks Networks(size_t obs, size_t choices, size_t hidden, int layers) {
    rl::DqnNetworks n;
    n.q = Mlp(obs, hidden, choices, layers);
    n.q_target = Mlp(obs, hidden, choices, layers);
    return n;
}

// ------------------------------------------------------- SB3 four steps

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" / "dqn_step_sb3.json";
    if (std::filesystem::exists(beside)) return beside;
    return CYXWIZ_DQN_STEP_FIXTURE;
}

void SetFrom(IExecutableModel& model, const json& weights, const std::string& prefix) {
    auto params = model.GetParameters();
    for (auto& [name, tensor] : params) {
        const auto it = weights.find(prefix + "." + name);
        if (it == weights.end()) {
            Check(false, "the fixture has " + prefix + "." + name);
            continue;
        }
        const Tensor t = FromFixture(*it);
        Check(t.Shape() == tensor.Shape(), prefix + "." + name + " has the network's shape");
        tensor = t;
    }
    model.SetParameters(params);
}

double Worst(IExecutableModel& model, const json& weights, const std::string& prefix) {
    double worst = 0.0;
    for (const auto& [name, tensor] : model.GetParameters()) {
        const auto want = weights.at(prefix + "." + name).at("values").get<std::vector<float>>();
        const auto got = Host(tensor);
        if (got.size() != want.size()) return 1e9;
        for (size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::abs(double(got[i]) - want[i]));
    }
    return worst;
}

void CheckSb3Steps(const char* argv0) {
    std::ifstream in(FixturePath(argv0));
    Check(in.good(), "dqn_step_sb3.json is readable");
    if (!in.good()) return;
    const json f = json::parse(in);
    const auto& s = f.at("settings");
    const size_t obs = s.at("observations"), choices = s.at("choices"), hidden = s.at("hidden");
    rl::DqnConfig config;
    config.learning_rate = s.at("learning_rate").get<float>();
    config.gamma = s.at("gamma").get<float>();
    config.tau = s.at("tau").get<float>();
    config.max_grad_norm = s.at("max_grad_norm").get<float>();
    rl::DqnNetworks networks = Networks(obs, choices, hidden, 1);
    SetFrom(*networks.q, f.at("before"), "q");
    rl::DqnLearner learner(std::move(networks), obs, choices, config, 1);
    Check(Worst(*learner.Networks().q_target, f.at("before"), "target_q") == 0.0,
          "the target starts as the Q network's copy, as SB3's");
    const size_t calls = s.at("calls"), per_call = s.at("steps_per_call");
    for (size_t call = 0; call < calls; ++call) {
        double loss = 0.0;
        for (size_t k = 0; k < per_call; ++k) {
            const auto& b = f.at("batches")[call * per_call + k];
            rl::ReplayBuffer::Batch batch;
            batch.observations = FromFixture(b.at("observations"));
            batch.actions = FromFixture(b.at("actions"));
            batch.rewards = FromFixture(b.at("rewards"));
            batch.next_observations = FromFixture(b.at("next_observations"));
            batch.dones = FromFixture(b.at("dones"));
            loss += Host(learner.Step(batch).loss)[0];
        }
        learner.UpdateTarget();
        const double want = f.at("logged_loss")[call].get<double>();
        const double mean = loss / static_cast<double>(per_call);
        Check(std::abs(mean - want) <= 1e-5 * std::max(1.0, std::abs(want)),
              "SB3's logged loss of train call " + std::to_string(call + 1) + ": " + std::to_string(mean) + " vs " +
                  std::to_string(want));
    }
    Check(learner.Updates() == 4 && learner.TargetUpdates() == 2, "four gradient steps, two target updates");
    // Adam moves each weight by about the learning rate per step: a wrong
    // gradient, clip or target shows at the 1e-3 scale
    const double tolerance = 2e-6;
    const double q = Worst(*learner.Networks().q, f.at("after"), "q");
    const double target = Worst(*learner.Networks().q_target, f.at("after"), "target_q");
    Check(q <= tolerance, "after four steps the Q network matches SB3: largest difference " + std::to_string(q));
    Check(target <= tolerance, "after two target updates the target matches SB3: largest difference " +
                                   std::to_string(target));
    std::printf("SB3 four steps: losses %.8f %.8f, Q off by %.2g, target by %.2g (%s)\n",
                f.at("logged_loss")[0].get<double>(), f.at("logged_loss")[1].get<double>(), q, target,
                learner.NetworkPath().c_str());
}

// ------------------------------------------------------- A small task

// Pick: the observation is one value x in [-1, 1]; choice 1 says x > 0,
// choice 0 says it is not; a right pick earns 1; 10 steps an episode (a time
// limit: truncated). Random picks earn about 5; always right earns 10.
rl::TrainingEnvironment Pick(int copies) {
    struct State {
        std::vector<float> x;
        std::vector<int> t;
        uint64_t rng = 1;
        float Uniform() {
            rng = rng * 6364136223846793005ull + 1442695040888963407ull;
            return static_cast<float>((rng >> 40) & 0xFFFFFF) / float(0xFFFFFF) * 2.0f - 1.0f;
        }
    };
    auto s = std::make_shared<State>();
    s->x.resize(copies);
    s->t.resize(copies);
    rl::TrainingEnvironment env;
    env.info.id = "Pick";
    env.info.name = "Pick";
    env.info.observation_size = 1;
    env.info.discrete_actions = true;
    env.info.action_size = 1;
    env.info.action_count = 2;
    env.info.max_episode_steps = 10;
    env.count = copies;
    env.reset = [s](uint64_t seed, float* obs, std::string&) {
        s->rng = seed * 2654435761ull + 1;
        for (size_t i = 0; i < s->x.size(); ++i) {
            s->x[i] = s->Uniform();
            s->t[i] = 0;
            obs[i] = s->x[i];
        }
        return true;
    };
    env.step = [s](const float* actions, const plugin::PluginEnvironmentStepBuffers& out, std::string& error) {
        for (size_t i = 0; i < s->x.size(); ++i) {
            if (actions[i] != 0.0f && actions[i] != 1.0f) {
                error = "Pick takes choice 0 or 1, got " + std::to_string(actions[i]);
                return false;
            }
            out.rewards[i] = (actions[i] == 1.0f) == (s->x[i] > 0.0f) ? 1.0f : 0.0f;
            out.terminated[i] = 0;
            out.truncated[i] = ++s->t[i] >= 10;
            if (out.truncated[i]) {
                out.final_observations[i] = s->x[i];
                s->t[i] = 0;
            }
            s->x[i] = s->Uniform();
            out.observations[i] = s->x[i];
        }
        return true;
    };
    return env;
}

rl::DqnConfig PickConfig() {
    rl::DqnConfig c;
    c.total_timesteps = 6000;
    c.buffer_size = 10000;
    c.learning_starts = 500;
    c.batch_size = 64;
    c.learning_rate = 1e-3f;
    c.train_freq = 1;
    c.gradient_steps = 1;
    c.target_update_interval = 100;
    c.exploration_fraction = 0.2f;
    c.exploration_final_eps = 0.02f;
    c.report_every = 500;
    c.seed = 7;
    return c;
}

struct Run {
    bool ok = false;
    std::string error;
    std::vector<rl::OffPolicyUpdateMetrics> reports;
};

Run TrainPick(rl::DqnConfig config, int copies, rl::DqnTrainedState* trained = nullptr) {
    SeedCurrentArrayFireRandomEngine(5);
    rl::DqnTrainer trainer(Pick(copies), Networks(1, 2, 32, 2), config);
    Run run;
    run.ok = trainer.Train(
        [&](const rl::OffPolicyUpdateMetrics& m) {
            run.reports.push_back(m);
            if (m.report == 2) {
                rl::ActorCopy copy;
                std::string why;
                Check(trainer.CopyPolicy(copy, why) && !copy.parameters.empty() && copy.normalizer,
                      "the Q network can be copied between reports: " + why);
            }
        },
        run.error);
    if (trained && run.ok) Check(trainer.TakeTrained(*trained, run.error), "a run's networks can be taken: " + run.error);
    return run;
}

// SB3's epsilon after `timesteps` of a run
float Epsilon(const rl::DqnConfig& c, int64_t timesteps) {
    return rl::DqnExplorationRate(c.exploration_initial_eps, c.exploration_final_eps, c.exploration_fraction,
                                  1.0 - static_cast<double>(timesteps) / static_cast<double>(c.total_timesteps));
}

void CheckPick() {
    rl::DqnTrainedState trained;
    const rl::DqnConfig config = PickConfig();
    const Run run = TrainPick(config, 1, &trained);
    Check(run.ok, "DQN trains on Pick: " + run.error);
    if (run.reports.empty()) return;
    const auto& first = run.reports.front();
    const auto& last = run.reports.back();
    std::printf("Pick: return %.6f at %lld steps -> %.6f at %lld (%lld updates, loss %.8f, mean Q %.6f, epsilon %.4f, "
                "%.0f steps/s, %s)\n",
                first.mean_episode_return, static_cast<long long>(first.timesteps), last.mean_episode_return,
                static_cast<long long>(last.timesteps), static_cast<long long>(last.updates), last.critic_loss,
                last.mean_q, last.exploration_rate, last.steps_per_second, last.compute_path.c_str());
    Check(!first.learning && first.updates == 0, "no update before learning starts");
    Check(last.timesteps == 6000 && last.updates == 6000 - 500, "one gradient step a step after learning starts");
    Check(last.mean_episode_return > 9.0f, "DQN learns to pick the side: mean return " +
                                               std::to_string(last.mean_episode_return));
    Check(first.exploration_rate == Epsilon(config, first.timesteps) && last.exploration_rate == 0.02f &&
              std::abs(first.exploration_rate - 0.5916667f) < 1e-5f,
          "epsilon falls as SB3's schedule: " + std::to_string(first.exploration_rate) + " at 500 steps, " +
              std::to_string(last.exploration_rate) + " at the end");
    Check(last.buffer_size == 6000 && last.buffer_capacity == 10000 && last.buffer_bytes == 10000ull * 5 * 4,
          "the report says how full the buffer is (the choice kept as one value)");
    Check(last.actor_loss == 0.0f && last.ent_coef == 0.0f, "DQN reports no actor or entropy weight");
    Check(trained.networks.q && trained.networks.q_target && trained.normalizer && trained.optimizers.count("q"),
          "what a run leaves: the networks, the statistics and the optimizer");
    if (trained.networks.q) {
        const std::vector<float> xs = {-0.9f, -0.2f, 0.3f, 0.8f};
        const auto greedy = Host(rl::DqnGreedyActions(trained.networks.q->Forward(Tensor({4, 1}, xs.data()))));
        Check(greedy == std::vector<float>({0.0f, 0.0f, 1.0f, 1.0f}), "the trained Q network picks each side");
    }

    rl::DqnConfig short_config = PickConfig();
    short_config.total_timesteps = 1500;
    const Run a = TrainPick(short_config, 1), b = TrainPick(short_config, 1);
    Check(a.ok && b.ok && !a.reports.empty() && a.reports.size() == b.reports.size() &&
              a.reports.back().critic_loss == b.reports.back().critic_loss &&
              a.reports.back().mean_episode_return == b.reports.back().mean_episode_return,
          "one seed, one run");
    rl::DqnConfig other_seed = short_config;
    other_seed.seed = 8;
    const Run c = TrainPick(other_seed, 1);
    Check(c.ok && !c.reports.empty() && c.reports.back().critic_loss != a.reports.back().critic_loss,
          "another seed is another run");

    rl::DqnConfig copies_config = PickConfig();
    copies_config.total_timesteps = 2000;
    const Run four = TrainPick(copies_config, 4);
    Check(four.ok && !four.reports.empty() && four.reports.back().timesteps == 2000 &&
              four.reports.back().updates == (2000 - 500) / 4,
          "four copies: a step is four transitions, one gradient step each: " + four.error);

    rl::DqnConfig normalized = PickConfig();
    normalized.total_timesteps = 1500;
    normalized.normalize_observations = true;
    const Run norm = TrainPick(normalized, 1);
    Check(norm.ok && !norm.reports.empty() && std::isfinite(norm.reports.back().critic_loss),
          "normalised observations train: " + norm.error);
}

void CheckRefusals() {
    const auto refused = [](rl::TrainingEnvironment env, rl::DqnConfig config, const std::string& says) {
        rl::DqnTrainer trainer(std::move(env), Networks(1, 2, 8, 1), config);
        std::string error;
        const bool ok = trainer.Train({}, error);
        Check(!ok && error.find(says) != std::string::npos, "refused (" + says + "): " + error);
    };
    rl::TrainingEnvironment continuous = Pick(1);
    continuous.info.discrete_actions = false;
    continuous.info.action_low = {-1.0f};
    continuous.info.action_high = {1.0f};
    refused(continuous, PickConfig(), "DQN chooses among actions");
    rl::DqnConfig no_target = PickConfig();
    no_target.target_update_interval = 0;
    refused(Pick(1), no_target, "at least 1");
    rl::DqnConfig high_end = PickConfig();
    high_end.exploration_final_eps = 2.0f;
    refused(Pick(1), high_end, "from 0 to 1");
    rl::DqnConfig no_clip = PickConfig();
    no_clip.max_grad_norm = 0.0f;
    refused(Pick(1), no_clip, "largest gradient norm must be above 0");
    rl::DqnConfig huge = PickConfig();
    huge.buffer_size = 2'000'000'000'000;  // 40 TB
    refused(Pick(1), huge, "does not fit on the training device");
}

// ------------------------------------------------------- CartPole

rl::TrainingEnvironment Adapt(std::shared_ptr<EnvironmentBatch> batch) {
    rl::TrainingEnvironment env;
    env.info = batch->Info();
    env.count = batch->Count();
    env.reset = [batch](uint64_t seed, float* obs, std::string& error) { return batch->Reset(seed, obs, error); };
    env.step = [batch](const float* actions, const cyxwiz::plugin::PluginEnvironmentStepBuffers& out,
                       std::string& error) { return batch->Step(actions, out, error); };
    return env;
}

// RL Zoo's dqn.yml CartPole-v1, stopping at 475
void CheckCartPole() {
    std::string error;
    auto made = PluginEnvironmentRegistry::Instance().Create("cyxwiz.builtin:CartPole-v1", {}, 1, error);
    Check(made != nullptr, "CartPole: " + error);
    if (!made) return;
    std::shared_ptr<EnvironmentBatch> batch(std::move(made));
    rl::DqnConfig c;
    c.total_timesteps = 50000;
    c.buffer_size = 100000;
    c.learning_starts = 1000;
    c.batch_size = 64;
    c.learning_rate = 2.3e-3f;
    c.train_freq = 256;
    c.gradient_steps = 128;
    c.target_update_interval = 10;
    c.exploration_fraction = 0.16f;
    c.exploration_final_eps = 0.04f;
    c.target_return = 475.0f;
    c.report_every = 2500;
    c.seed = 1;
    SeedCurrentArrayFireRandomEngine(1);
    rl::DqnTrainer trainer(Adapt(batch), Networks(4, 2, 256, 2), c);
    float best = 0.0f;
    const bool ok = trainer.Train(
        [&](const rl::OffPolicyUpdateMetrics& m) {
            best = std::max(best, m.mean_episode_return);
            std::printf("  CartPole %6lld steps  return %8.2f  length %7.2f  loss %.6f  mean Q %8.3f  epsilon %.3f  "
                        "%5.0f steps/s\n",
                        static_cast<long long>(m.timesteps), m.mean_episode_return, m.mean_episode_length,
                        m.critic_loss, m.mean_q, m.exploration_rate, m.steps_per_second);
        },
        error);
    Check(ok, "CartPole trains: " + error);
    Check(best > 200.0f, "DQN learns CartPole: the best mean return " + std::to_string(best));
}

}  // namespace

int main(int argc, char** argv) {
    // The device CYXWIZ_TEST_ARRAYFIRE_BACKEND names
    if (!cyxwiz::test::SelectTestDeviceFromEnvironment()) return EXIT_FAILURE;
    try {
        CheckSb3Steps(argv[0]);
        CheckPick();
        CheckRefusals();
        if (argc > 1 && std::strcmp(argv[1], "--cartpole") == 0) CheckCartPole();
    } catch (const std::exception& e) {
        Check(false, std::string("threw: ") + e.what());
    }
    if (failures == 0) std::printf("test_dqn_trainer: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
