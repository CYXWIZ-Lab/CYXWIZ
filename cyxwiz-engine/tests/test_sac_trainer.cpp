// The Engine's SAC (core/rl/sac_trainer, replay_buffer; TOFIX216 S3), headless:
// - the replay buffer keeps the last `capacity` steps as a ring, samples only
//   stored rows, one seed one draw, and keeps a step's columns together;
// - SacLearner replays three gradient steps of a real Stable-Baselines3 SAC
//   (fixture sac_step_sb3.json from generate_sac_step_fixture.py) and ends
//   with SB3's weights, target weights, entropy weight and logged losses;
// - SacTrainer learns a small continuous task (reach a target on a line),
//   the same seed gives the same run, bad setups are refused with the reason,
//   and what a run leaves can be taken and copied;
// - with a MuJoCo plugin folder as the argument it trains InvertedPendulum-v5.
//
//   test_sac_trainer [<MuJoCo plugin folder>]
//
// CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|cpu picks the device.

#include "../src/core/executable_model.h"
#include "../src/core/rl/replay_buffer.h"
#include "../src/core/rl/sac_trainer.h"
#include "../src/plugin/plugin_manager.h"
#include "../src/plugin/registries/plugin_environment_registry.h"
#include "algorithms/arrayfire_backend_utils.h"
#include "computation_truth/test_device_selection.h"

#include <cyxwiz/rl/sac.h>
#include <cyxwiz/sequential.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
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
std::unique_ptr<IExecutableModel> Mlp(size_t in, size_t hidden, size_t out, int layers = 1) {
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

rl::SacNetworks Networks(size_t obs, size_t act, size_t hidden, int layers) {
    rl::SacNetworks n;
    n.actor = Mlp(obs, hidden, 2 * act, layers);
    n.q1 = Mlp(obs + act, hidden, 1, layers);
    n.q2 = Mlp(obs + act, hidden, 1, layers);
    n.q1_target = Mlp(obs + act, hidden, 1, layers);
    n.q2_target = Mlp(obs + act, hidden, 1, layers);
    return n;
}

// ------------------------------------------------------------ Replay buffer

void CheckReplayBuffer() {
    const size_t obs = 2, act = 1;
    rl::ReplayBuffer buffer(4, obs, act);
    Check(buffer.Size() == 0 && buffer.Capacity() == 4 && buffer.Bytes() == 4 * (2 * obs + act + 2) * sizeof(float),
          "an empty buffer of 4 steps, its bytes from the width");
    // Step k: observation (k, 10k), action k/10, reward k, next observation (k+1, 10k+10), done on odd k
    const auto add = [&](int first, int count) {
        std::vector<float> o, a, r, n;
        std::vector<uint8_t> d;
        for (int k = first; k < first + count; ++k) {
            o.insert(o.end(), {float(k), 10.0f * k});
            a.push_back(k / 10.0f);
            r.push_back(float(k));
            n.insert(n.end(), {float(k + 1), 10.0f * k + 10.0f});
            d.push_back(k % 2 == 1);
        }
        buffer.Add(o.data(), a.data(), r.data(), n.data(), d.data(), static_cast<size_t>(count));
    };
    add(0, 3);
    Check(buffer.Size() == 3, "three steps stored");
    add(3, 3);  // wraps: steps 2..5 remain
    Check(buffer.Size() == 4, "the buffer holds its capacity");
    const auto batch = buffer.Sample(64, 11);
    const auto o = Host(batch.observations), a = Host(batch.actions), r = Host(batch.rewards),
               n = Host(batch.next_observations), d = Host(batch.dones);
    std::set<int> seen;
    bool consistent = true;
    for (size_t i = 0; i < 64; ++i) {
        const int k = static_cast<int>(std::lround(r[i]));
        seen.insert(k);
        consistent = consistent && o[2 * i] == float(k) && o[2 * i + 1] == 10.0f * k && std::abs(a[i] - k / 10.0f) < 1e-6f &&
                     n[2 * i] == float(k + 1) && n[2 * i + 1] == 10.0f * k + 10.0f && d[i] == (k % 2 == 1 ? 1.0f : 0.0f);
    }
    Check(consistent, "a sampled row keeps its step's columns together");
    Check(seen == std::set<int>({2, 3, 4, 5}), "only the stored steps are drawn, the overwritten ones never");
    Check(Host(buffer.Sample(64, 11).rewards) == r, "one seed, one draw");
    Check(Host(buffer.Sample(64, 12).rewards) != r, "another seed, another draw");
    add(10, 9);  // more than the capacity at once: the last four survive
    std::set<int> last;
    for (const float v : Host(buffer.Sample(64, 3).rewards)) last.insert(static_cast<int>(std::lround(v)));
    Check(last == std::set<int>({15, 16, 17, 18}), "a batch larger than the buffer keeps its last rows");
    bool threw = false;
    try {
        rl::ReplayBuffer empty(8, obs, act);
        empty.Sample(4, 1);
    } catch (const std::logic_error&) {
        threw = true;
    }
    Check(threw, "an empty buffer is not sampled");
}

// ------------------------------------------------------- SB3 three steps

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" / "sac_step_sb3.json";
    if (std::filesystem::exists(beside)) return beside;
    return CYXWIZ_SAC_STEP_FIXTURE;
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
        for (size_t i = 0; i < got.size() && i < want.size(); ++i)
            worst = std::max(worst, std::abs(double(got[i]) - want[i]));
    }
    return worst;
}

void CheckSb3Steps(const char* argv0) {
    std::ifstream in(FixturePath(argv0));
    Check(in.good(), "sac_step_sb3.json is readable");
    if (!in.good()) return;
    const json f = json::parse(in);
    const auto& s = f.at("settings");
    const size_t obs = s.at("observations"), act = s.at("actions"), hidden = s.at("hidden");
    rl::SacConfig config;
    config.learning_rate = s.at("learning_rate").get<float>();
    config.gamma = s.at("gamma").get<float>();
    config.tau = s.at("tau").get<float>();
    config.target_entropy = s.at("target_entropy").get<float>();
    config.ent_coef = 1.0f;  // "auto": log alpha starts at 0
    rl::SacNetworks networks = Networks(obs, act, hidden, 1);
    const json& before = f.at("before");
    SetFrom(*networks.actor, before, "actor");
    SetFrom(*networks.q1, before, "q1");
    SetFrom(*networks.q2, before, "q2");
    rl::SacLearner learner(std::move(networks), obs, act, config);
    double critic = 0.0, actor = 0.0, alpha = 0.0, alpha_loss = 0.0;
    const size_t steps = f.at("batches").size();
    for (size_t k = 0; k < steps; ++k) {
        const auto& b = f.at("batches")[k];
        rl::ReplayBuffer::Batch batch;
        batch.observations = FromFixture(b.at("observations"));
        batch.actions = FromFixture(b.at("actions"));
        batch.rewards = FromFixture(b.at("rewards"));
        batch.next_observations = FromFixture(b.at("next_observations"));
        batch.dones = FromFixture(b.at("dones"));
        const auto l = learner.Step(batch, FromFixture(f.at("noise")[k].at("pi")), FromFixture(f.at("noise")[k].at("next")),
                                    true);
        critic += Host(l.critic_loss)[0];
        actor += Host(l.actor_loss)[0];
        alpha += Host(l.ent_coef)[0];
        alpha_loss += Host(l.ent_coef_loss)[0];
    }
    const auto& logged = f.at("logged");
    const auto close = [&](double got, const char* key, double tolerance) {
        const double want = logged.at(key).get<double>();
        Check(std::abs(got / steps - want) <= tolerance * std::max(1.0, std::abs(want)),
              std::string("SB3's logged ") + key + ": " + std::to_string(got / steps) + " vs " + std::to_string(want));
    };
    close(critic, "critic_loss", 1e-5);
    close(actor, "actor_loss", 1e-5);
    close(alpha, "ent_coef", 1e-6);
    close(alpha_loss, "ent_coef_loss", 1e-4);
    const json& after = f.at("after");
    auto& n = learner.Networks();
    // Adam moves each weight by about the learning rate per step: a wrong
    // gradient sign or size shows at the 1e-4 scale
    const double tolerance = 2e-6;
    const struct { IExecutableModel* model; const char* prefix; } nets[] = {
        {n.actor.get(), "actor"}, {n.q1.get(), "q1"}, {n.q2.get(), "q2"},
        {n.q1_target.get(), "target_q1"}, {n.q2_target.get(), "target_q2"}};
    for (const auto& net : nets) {
        const double worst = Worst(*net.model, after, net.prefix);
        Check(worst <= tolerance, std::string("after three steps ") + net.prefix + " matches SB3: largest difference " +
                                      std::to_string(worst));
    }
    const float log_alpha = Host(learner.LogEntCoef())[0];
    const float want_log_alpha = after.at("log_ent_coef").at("values")[0].get<float>();
    Check(std::abs(log_alpha - want_log_alpha) <= 1e-7f,
          "the log entropy weight matches SB3: " + std::to_string(log_alpha) + " vs " + std::to_string(want_log_alpha));
}

// ------------------------------------------------------- A small task

// Reach: the observation is (x, target) with x, target in [-1, 1]; the action
// moves x by up to 0.2; the reward is -|x - target|; 20 steps an episode
// (a time limit: truncated). Random play earns about -9; reaching and
// holding the target earns above -3.
rl::TrainingEnvironment Reach(int copies) {
    struct State {
        std::vector<float> x, target;
        std::vector<int> t;
        uint64_t rng = 1;
        float Uniform() {
            rng = rng * 6364136223846793005ull + 1442695040888963407ull;
            return static_cast<float>((rng >> 40) & 0xFFFFFF) / float(0xFFFFFF) * 2.0f - 1.0f;
        }
        void Start(size_t i) {
            x[i] = Uniform();
            target[i] = Uniform();
            t[i] = 0;
        }
    };
    auto s = std::make_shared<State>();
    s->x.resize(copies);
    s->target.resize(copies);
    s->t.resize(copies);
    rl::TrainingEnvironment env;
    env.info.id = "Reach";
    env.info.name = "Reach";
    env.info.observation_size = 2;
    env.info.action_size = 1;
    env.info.action_low = {-1.0f};
    env.info.action_high = {1.0f};
    env.info.max_episode_steps = 20;
    env.count = copies;
    env.reset = [s](uint64_t seed, float* obs, std::string&) {
        s->rng = seed * 2654435761ull + 1;
        for (size_t i = 0; i < s->x.size(); ++i) {
            s->Start(i);
            obs[2 * i] = s->x[i];
            obs[2 * i + 1] = s->target[i];
        }
        return true;
    };
    env.step = [s](const float* actions, const plugin::PluginEnvironmentStepBuffers& out, std::string&) {
        for (size_t i = 0; i < s->x.size(); ++i) {
            s->x[i] = std::clamp(s->x[i] + 0.2f * std::clamp(actions[i], -1.0f, 1.0f), -1.5f, 1.5f);
            out.rewards[i] = -std::abs(s->x[i] - s->target[i]);
            out.terminated[i] = 0;
            out.truncated[i] = ++s->t[i] >= 20;
            if (out.truncated[i]) {
                out.final_observations[2 * i] = s->x[i];
                out.final_observations[2 * i + 1] = s->target[i];
                s->Start(i);
            }
            out.observations[2 * i] = s->x[i];
            out.observations[2 * i + 1] = s->target[i];
        }
        return true;
    };
    return env;
}

rl::SacConfig ReachConfig() {
    rl::SacConfig c;
    c.total_timesteps = 5000;
    c.buffer_size = 10000;
    c.learning_starts = 500;
    c.batch_size = 64;
    c.learning_rate = 1e-3f;
    c.target_entropy = -1.0f;
    c.report_every = 500;
    c.seed = 7;
    return c;
}

struct Run {
    bool ok = false;
    std::string error;
    std::vector<rl::SacUpdateMetrics> reports;
};

Run TrainReach(rl::SacConfig config, int copies, rl::SacTrainedState* trained = nullptr) {
    SeedCurrentArrayFireRandomEngine(5);
    rl::SacTrainer trainer(Reach(copies), Networks(2, 1, 32, 2), config);
    Run run;
    run.ok = trainer.Train([&](const rl::SacUpdateMetrics& m) { run.reports.push_back(m); }, run.error);
    if (trained && run.ok) Check(trainer.TakeTrained(*trained, run.error), "a run's networks can be taken: " + run.error);
    return run;
}

void CheckReach() {
    rl::SacTrainedState trained;
    const Run run = TrainReach(ReachConfig(), 1, &trained);
    Check(run.ok, "SAC trains on Reach: " + run.error);
    if (run.reports.empty()) return;
    const auto& first = run.reports.front();
    const auto& last = run.reports.back();
    std::printf("Reach: return %.2f at %lld steps -> %.2f at %lld (%lld updates, alpha %.3f, critic %.4f, %.0f steps/s)\n",
                first.mean_episode_return, static_cast<long long>(first.timesteps), last.mean_episode_return,
                static_cast<long long>(last.timesteps), static_cast<long long>(last.updates), last.ent_coef,
                last.critic_loss, last.steps_per_second);
    Check(!first.learning && first.updates == 0, "no update before learning starts");
    Check(last.timesteps == 5000 && last.updates == 5000 - 500, "one gradient step a step after learning starts");
    Check(last.mean_episode_return > -3.0f, "SAC learns to reach the target: mean return " +
                                                std::to_string(last.mean_episode_return));
    // a row: observation 2, action 1, reward, next observation 2, done = 7 floats
    Check(last.buffer_size == 5000 && last.buffer_capacity == 10000 && last.buffer_bytes == 10000ull * 7 * 4,
          "the report says how full the buffer is");
    Check(last.ent_coef < 1.0f && last.ent_coef > 0.0f, "the entropy weight was learned down from 1");
    Check(trained.networks.actor && trained.networks.q1_target && trained.normalizer &&
              trained.optimizers.count("actor") && trained.optimizers.count("ent_coef"),
          "what a run leaves: the networks, the statistics and the optimizers");

    rl::SacConfig short_config = ReachConfig();
    short_config.total_timesteps = 1500;
    const Run a = TrainReach(short_config, 1), b = TrainReach(short_config, 1);
    Check(a.ok && b.ok && !a.reports.empty() && a.reports.size() == b.reports.size() &&
              a.reports.back().critic_loss == b.reports.back().critic_loss &&
              a.reports.back().mean_episode_return == b.reports.back().mean_episode_return,
          "one seed, one run");

    rl::SacConfig copies_config = ReachConfig();
    copies_config.total_timesteps = 2000;
    const Run four = TrainReach(copies_config, 4);
    Check(four.ok && !four.reports.empty() && four.reports.back().timesteps == 2000 &&
              four.reports.back().updates == (2000 - 500) / 4,
          "four copies: a step is four transitions, one gradient step each");

    rl::SacConfig normalized = ReachConfig();
    normalized.total_timesteps = 1500;
    normalized.normalize_observations = true;
    const Run norm = TrainReach(normalized, 1);
    Check(norm.ok && !norm.reports.empty() && std::isfinite(norm.reports.back().critic_loss),
          "normalised observations train: " + norm.error);
}

void CheckRefusals() {
    const auto refused = [](rl::TrainingEnvironment env, rl::SacConfig config, const std::string& says) {
        rl::SacTrainer trainer(std::move(env), Networks(2, 1, 8, 1), config);
        std::string error;
        const bool ok = trainer.Train({}, error);
        Check(!ok && error.find(says) != std::string::npos, "refused (" + says + "): " + error);
    };
    rl::TrainingEnvironment discrete = Reach(1);
    discrete.info.discrete_actions = true;
    refused(discrete, ReachConfig(), "continuous actions");
    rl::TrainingEnvironment unbounded = Reach(1);
    unbounded.info.action_high = {std::numeric_limits<float>::infinity()};
    refused(unbounded, ReachConfig(), "bounded actions");
    rl::SacConfig no_batch = ReachConfig();
    no_batch.batch_size = 0;
    refused(Reach(1), no_batch, "at least 1");
    rl::SacConfig huge = ReachConfig();
    huge.buffer_size = 2'000'000'000'000;  // 64 TB
    refused(Reach(1), huge, "does not fit on the training device");
}

// ------------------------------------------------------- InvertedPendulum

rl::TrainingEnvironment Adapt(std::shared_ptr<EnvironmentBatch> batch) {
    rl::TrainingEnvironment env;
    env.info = batch->Info();
    env.count = batch->Count();
    env.reset = [batch](uint64_t seed, float* obs, std::string& error) { return batch->Reset(seed, obs, error); };
    env.step = [batch](const float* actions, const cyxwiz::plugin::PluginEnvironmentStepBuffers& out,
                       std::string& error) { return batch->Step(actions, out, error); };
    return env;
}

void CheckPendulum(const std::filesystem::path& plugin) {
    auto& manager = plugin::PluginManager::Instance();
    std::string error;
    Check(manager.LoadPlugin(plugin, &error), "the MuJoCo plugin loads: " + error);
    if (!manager.InitializePlugin("com.cyxwiz.simulation.mujoco")) {
        Check(false, "the MuJoCo plugin starts");
        return;
    }
    {
        auto made = PluginEnvironmentRegistry::Instance().Create("com.cyxwiz.simulation.mujoco:InvertedPendulum-v5", {}, 1,
                                                                 error);
        Check(made != nullptr, "InvertedPendulum: " + error);
        if (!made) return;
        std::shared_ptr<EnvironmentBatch> batch(std::move(made));
        // SB3 defaults with a smaller buffer. The 100-episode mean is slow to
        // show it here (100 balanced episodes are 100,000 steps), so the
        // trained actor plays: tanh(mean), no noise, as playback acts
        rl::SacConfig c;
        c.total_timesteps = 25000;
        c.buffer_size = 100000;
        c.learning_starts = 1000;
        c.target_entropy = -1.0f;
        c.report_every = 2500;
        c.seed = 1;
        SeedCurrentArrayFireRandomEngine(1);
        rl::SacTrainer trainer(Adapt(batch), Networks(4, 1, 256, 2), c);
        const bool ok = trainer.Train(
            [&](const rl::SacUpdateMetrics& m) {
                std::printf("  InvertedPendulum %6lld steps  return %7.1f  length %6.1f  critic %.3f  alpha %.4f  %5.0f steps/s\n",
                            static_cast<long long>(m.timesteps), m.mean_episode_return, m.mean_episode_length,
                            m.critic_loss, m.ent_coef, m.steps_per_second);
            },
            error);
        Check(ok, "InvertedPendulum trains: " + error);
        rl::SacTrainedState trained;
        if (!ok || !trainer.TakeTrained(trained, error)) return;
        const auto& info = batch->Info();
        std::vector<float> obs(4), final_obs(4), reward(1), action(1);
        std::vector<uint8_t> terminated(1), truncated(1);
        const plugin::PluginEnvironmentStepBuffers out{obs.data(), reward.data(), terminated.data(), truncated.data(),
                                                       final_obs.data()};
        for (uint64_t episode = 0; episode < 3; ++episode) {
            Check(batch->Reset(100 + episode, obs.data(), error), "the pendulum resets: " + error);
            double total = 0.0;
            for (int step = 0; step < 1000; ++step) {
                const Tensor head = trained.networks.actor->Forward(Tensor({1, 4}, obs.data()));
                const float a = Host(rl::SacDeterministicAction(head.Slice(1, 0, 1)))[0];
                action[0] = info.action_low[0] + 0.5f * (a + 1.0f) * (info.action_high[0] - info.action_low[0]);
                if (!batch->Step(action.data(), out, error)) break;
                total += reward[0];
                if (terminated[0] || truncated[0]) break;
            }
            std::printf("  played: return %.0f\n", total);
            Check(total >= 950.0, "the trained actor balances the pendulum: return " + std::to_string(total));
        }
    }
    manager.UnloadPlugin("com.cyxwiz.simulation.mujoco");
}

}  // namespace

int main(int argc, char** argv) {
    if (!cyxwiz::test::SelectTestDeviceFromEnvironment()) return EXIT_FAILURE;
    try {
        CheckReplayBuffer();
        CheckSb3Steps(argv[0]);
        CheckRefusals();
        CheckReach();
        if (argc > 1) CheckPendulum(std::filesystem::absolute(argv[1]));
    } catch (const std::exception& e) {
        Check(false, std::string("threw: ") + e.what());
    }
    if (failures == 0) std::printf("test_sac_trainer: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
