#include "off_policy_trainer.h"

#include "../optimizer_step.h"

#include <cyxwiz/fused_mlp.h>
#include "algorithms/arrayfire_host_materialization.h"
#include <cyxwiz/optimizers/adam.h>
#include <cyxwiz/optimizers/gradient_clipping.h>
#include <cyxwiz/rl/sac.h>
#include <cyxwiz/sequential.h>

#include <arrayfire.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <map>
#include <stdexcept>
#include <thread>

namespace cyxwiz::rl {

namespace {

std::vector<float> Host(const Tensor& t) {
    const float* data = t.ReadData<float>();
    return std::vector<float>(data, data + t.NumElements());
}

// Observations normalised with the statistics as they are now, as SB3's
// replay buffer does at sample time with a VecNormalize
Tensor NormalizeOnDevice(const Tensor& observations, const VecNormalize& normalizer) {
    const auto& stats = normalizer.ObservationStats();
    const auto& s = normalizer.Settings();
    const size_t rows = observations.Shape()[0], width = observations.Shape()[1];
    std::vector<float> mean(width), scale(width);
    for (size_t i = 0; i < width; ++i) {
        mean[i] = static_cast<float>(stats.mean[i]);
        scale[i] = static_cast<float>(1.0 / std::sqrt(stats.var[i] + s.epsilon));
    }
    const unsigned r = static_cast<unsigned>(rows);
    const af::array m = af::tile(af::array(1, static_cast<dim_t>(width), mean.data()), r, 1);
    const af::array k = af::tile(af::array(1, static_cast<dim_t>(width), scale.data()), r, 1);
    af::array out = af::clamp((observations.GetSemanticArray() - m) * k, -static_cast<double>(s.clip_observations),
                              static_cast<double>(s.clip_observations));
    out.eval();
    return Tensor::FromSemanticArray(out, {rows, width});
}

}  // namespace

// ---------------------------------------------------------------- shared pieces

void InitializeLikeTorchLinear(IExecutableModel& model) {
    auto params = model.GetParameters();
    // fan_in of each layer: its weight's columns ([out, in]); "layer<K>.weight" / "layer<K>.bias"
    std::map<std::string, size_t> fan_in;
    for (const auto& [name, tensor] : params) {
        const auto dot = name.rfind('.');
        if (dot != std::string::npos && name.substr(dot + 1) == "weight" && tensor.Shape().size() == 2)
            fan_in[name.substr(0, dot)] = tensor.Shape()[1];
    }
    for (auto& [name, tensor] : params) {
        const auto dot = name.rfind('.');
        const auto it = fan_in.find(dot == std::string::npos ? name : name.substr(0, dot));
        if (it == fan_in.end() || it->second == 0) continue;
        const float bound = 1.0f / std::sqrt(static_cast<float>(it->second));
        af::array u = (af::randu(af::dim4(static_cast<dim_t>(tensor.NumElements())), f32) * 2.0f - 1.0f) * bound;
        u.eval();
        std::vector<float> host(tensor.NumElements());
        MaterializeArrayFireToHost(u, host.data(), ArrayFireHostSyncCategory::LayoutConversion,
                                   "InitializeLikeTorchLinear", "arrayfire_flat_vector",
                                   "weight_initialization", "uniform fan-in draw into a parameter");
        tensor = Tensor(tensor.Shape(), host.data());
    }
    model.SetParameters(params);
}

std::unique_ptr<Optimizer> OffPolicyAdam(float learning_rate) {
    return std::make_unique<AdamOptimizer>(learning_rate, 0.9, 0.999, 1e-8);
}

void CopyParameters(IExecutableModel& from, IExecutableModel& to) {
    std::map<std::string, Tensor> copy;
    for (const auto& [name, tensor] : from.GetParameters()) copy[name] = tensor.Clone();
    to.SetParameters(copy);
}

void PolyakUpdate(IExecutableModel& network, IExecutableModel& target, float tau) {
    auto params = network.GetParameters();
    auto targets = target.GetParameters();
    for (auto& [name, t] : targets) {
        const auto it = params.find(name);
        if (it == params.end())
            throw std::runtime_error("a target network's parameter " + name + " has no counterpart in its network");
        t = PolyakAverage(it->second, t, tau);
    }
    target.SetParameters(targets);
}

float FirstValue(const Tensor& t) { return Host(t).at(0); }

void StepOffPolicyOptimizers(const std::vector<OffPolicyUpdate>& updates, AdamRoute& adam, const char* learner,
                              float max_grad_norm) {
    if (adam.together) {
        std::vector<std::map<std::string, Tensor>> params(updates.size());
        std::vector<AdamOptimizer::Group> groups;
        bool plain_adam = true;
        for (size_t i = 0; i < updates.size(); ++i) {
            const OffPolicyUpdate& u = updates[i];
            params[i] = u.model ? u.model->GetParameters() : *u.parameters;
            auto* optimizer = dynamic_cast<AdamOptimizer*>(u.optimizer);
            plain_adam = plain_adam && optimizer != nullptr;
            groups.push_back({optimizer, &params[i], u.gradients});
        }
        std::string why = "only plain Adam steps together";
        if (plain_adam && AdamOptimizer::StepTogether(groups, max_grad_norm, why)) {
            for (size_t i = 0; i < updates.size(); ++i) {
                if (updates[i].model) updates[i].model->SetParameters(params[i]);
                else *updates[i].parameters = params[i];
            }
            return;
        }
        adam.together = false;
        adam.why = why;
        spdlog::info("{}: the optimizers step per tensor ({})", learner, why);
    }
    // One global norm over every gradient, as one optimizer over all of them clips
    if (max_grad_norm > 0.0f) {
        std::map<std::string, Tensor> all;
        for (size_t i = 0; i < updates.size(); ++i)
            for (const auto& [k, g] : *updates[i].gradients) all[std::to_string(i) + ":" + k] = g;
        ClipGradientsByGlobalNormOnDevice(all, max_grad_norm);
        for (size_t i = 0; i < updates.size(); ++i)
            for (auto& [k, g] : *updates[i].gradients) g = all.at(std::to_string(i) + ":" + k);
    }
    for (const OffPolicyUpdate& u : updates) {
        if (u.model) ApplyOptimizerStep(*u.model, *u.optimizer, *u.gradients, {});
        else u.optimizer->Step(*u.parameters, *u.gradients);
    }
}

// ---------------------------------------------------------------- the loop

bool CheckOffPolicySetup(const TrainingEnvironment& environment, const OffPolicySettings& s,
                         const std::string& algorithm, std::string& error, bool choices) {
    const auto& info = environment.info;
    const auto finite = [](const std::vector<float>& v) {
        return std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
    };
    if (environment.count < 1 || !environment.reset || !environment.step) error = algorithm + " needs an environment.";
    else if (!choices && info.discrete_actions)
        error = algorithm + " learns continuous actions; " + info.name + " has a choice of actions.";
    else if (choices && !info.discrete_actions)
        error = algorithm + " chooses among actions; " + info.name + " has continuous actions.";
    else if (info.observation_size < 1 || info.action_size < 1 || (choices && info.action_count < 1))
        error = "The environment observes or acts on nothing.";
    else if (choices && info.action_size != 1)
        error = algorithm + " takes one choice a step; " + info.name + " asks for " + std::to_string(info.action_size) +
                ".";
    else if (!choices && (info.action_low.size() != static_cast<size_t>(info.action_size) ||
             info.action_high.size() != static_cast<size_t>(info.action_size) || !finite(info.action_low) ||
             !finite(info.action_high)))
        error = algorithm + " needs bounded actions; " + info.name + " has actions without finite bounds.";
    else if (s.total_timesteps < 1 || s.buffer_size < 1 || s.learning_starts < 0 || s.batch_size < 1 ||
             s.train_freq < 1 || s.gradient_steps < 1 || s.report_every < 1)
        error = algorithm + "'s counts must each be at least 1 (the random steps first at least 0).";
    return error.empty();
}

OffPolicyLoop::OffPolicyLoop(TrainingEnvironment environment, OffPolicySettings settings, std::string algorithm)
    : environment_(std::move(environment)), settings_(std::move(settings)), algorithm_(std::move(algorithm)) {}

bool OffPolicyLoop::Run(OffPolicyLearner& learner, const std::function<void(const OffPolicyUpdateMetrics&)>& on_report,
                        std::string& error) {
    error.clear();
    const auto& info = environment_.info;
    const auto& s = settings_;
    const size_t n = static_cast<size_t>(environment_.count);
    const size_t obs_size = static_cast<size_t>(info.observation_size);
    const size_t act_size = static_cast<size_t>(info.action_size);  // 1 for a choice of actions
    const bool discrete = info.discrete_actions;
    const size_t batch = static_cast<size_t>(s.batch_size);
    SplitMix64 random(s.seed ^ 0x5AC5AC5AC5ull);
    const OffPolicyLearner::NextSeed next_seed = [&random]() { return random.Next(); };

    std::vector<float> obs(n * obs_size), final_obs(n * obs_size), rewards(n), env_actions(n * act_size);
    std::vector<uint8_t> terminated(n), truncated(n);
    const plugin::PluginEnvironmentStepBuffers out{obs.data(), rewards.data(), terminated.data(), truncated.data(),
                                                   final_obs.data()};
    std::vector<float> raw_obs(n * obs_size), raw_next(n * obs_size);
    std::vector<double> episode_return(n, 0.0);
    std::vector<int> episode_length(n, 0);
    std::deque<std::pair<float, int>> recent;  // the last 100 episodes
    OffPolicyUpdateMetrics metrics;

    // What the run's provider calls did, for the compute path each report
    // carries; the run's own observer (its trace) still sees them all
    const NeuralDispatchTally tally;
    // The device family its calls ask for: the run's bound device
    const DeviceType device = CaptureCurrentNeuralDeviceTarget().platform;
    try {
        spdlog::info("{}: the networks run as {}", algorithm_, learner.NetworkPath());
        ReplayBuffer buffer(s.buffer_size, obs_size, act_size, algorithm_);
        metrics.buffer_capacity = buffer.Capacity();
        metrics.buffer_bytes = buffer.Bytes();
        VecNormalizeSettings norm;
        norm.observations = s.normalize_observations;
        norm.rewards = false;  // observations only (RL Zoo normalises none for SAC and TD3)
        norm.gamma = s.gamma;
        normalizer_ = std::make_unique<VecNormalize>(n, obs_size, norm);
        spdlog::info("{}: training on {} x {} for {} steps; replay buffer {} steps ({:.0f} MB on the device)",
                     algorithm_, info.id, n, s.total_timesteps, buffer.Capacity(),
                     static_cast<double>(buffer.Bytes()) / 1e6);

        if (!environment_.reset(s.seed, obs.data(), error)) return false;
        raw_obs = obs;
        normalizer_->Reset(obs.data());

        int64_t vec_steps = 0, last_report = 0;
        auto report_start = std::chrono::steady_clock::now();
        const auto report = [&]() {
            ++metrics.report;
            learner.Report(metrics);
            {
                // The joint Adam step, also when it was refused before any
                // provider call; before the first gradient step nothing is known
                std::vector<ComputePathPart> ops;
                if (const auto adam = AdamPart(tally, device, learner.Adam())) ops.push_back(*adam);
                metrics.compute_path = FormatComputePath(learner.Route(), ops);
                metrics.compute_parts = ComputePathParts(learner.Route(), ops);
                metrics.missing_provider = learner.Route().missing_provider;
            }
            if (recent.empty()) {
                metrics.mean_episode_return = metrics.mean_episode_length = std::numeric_limits<float>::quiet_NaN();
            } else {
                double r = 0.0, l = 0.0;
                for (const auto& [ret, len] : recent) {
                    r += ret;
                    l += len;
                }
                metrics.mean_episode_return = static_cast<float>(r / recent.size());
                metrics.mean_episode_length = static_cast<float>(l / recent.size());
            }
            metrics.buffer_size = buffer.Size();
            const auto now = std::chrono::steady_clock::now();
            const double seconds = std::chrono::duration<double>(now - report_start).count();
            metrics.steps_per_second =
                seconds > 0.0 ? static_cast<float>(static_cast<double>(metrics.timesteps - last_report) / seconds) : 0.0f;
            report_start = now;
            last_report = metrics.timesteps;
            metrics.reached_target = recent.size() >= 100 && metrics.mean_episode_return >= s.target_return;
            if (on_report) on_report(metrics);
        };
        while (metrics.timesteps < s.total_timesteps && !stop_) {
            while (paused_ && !stop_) std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if (stop_) break;
            // ---- Act: uniform random until learning starts (a random choice for
            // each copy when the actions are choices), then the learner's
            std::vector<float> actions;
            if (metrics.timesteps < s.learning_starts && discrete) {
                actions.resize(n);
                for (auto& a : actions) a = static_cast<float>(random.Next() % static_cast<uint64_t>(info.action_count));
            } else if (metrics.timesteps < s.learning_starts) {
                actions = Host(learner.Warmup(SacUniformActions(n, act_size, random.Next()), next_seed));
            } else {
                actions = Host(learner.Explore(Tensor({n, obs_size}, obs.data()), next_seed));
            }
            if (actions.size() != n * act_size)
                throw std::runtime_error("the learner gave " + std::to_string(actions.size()) + " action values for " +
                                         std::to_string(n * act_size));
            for (size_t i = 0; i < n * act_size; ++i) {  // [-1, 1] to the environment's bounds; a choice as it is
                const size_t j = i % act_size;
                env_actions[i] = discrete ? actions[i]
                                          : info.action_low[j] + 0.5f * (actions[i] + 1.0f) *
                                                                     (info.action_high[j] - info.action_low[j]);
            }
            if (!environment_.step(env_actions.data(), out, error)) return false;

            // ---- The transitions, raw: an ended copy's next observation is its final one
            for (size_t i = 0; i < n; ++i) {
                const bool done = terminated[i] || truncated[i];
                std::copy_n((done ? final_obs.begin() : obs.begin()) + i * obs_size, obs_size,
                            raw_next.begin() + i * obs_size);
                episode_return[i] += rewards[i];
                ++episode_length[i];
                if (done) {
                    recent.emplace_back(static_cast<float>(episode_return[i]), episode_length[i]);
                    if (recent.size() > 100) recent.pop_front();
                    ++metrics.episodes;
                    episode_return[i] = 0.0;
                    episode_length[i] = 0;
                }
            }
            buffer.Add(raw_obs.data(), actions.data(), rewards.data(), raw_next.data(), terminated.data(), n);
            raw_obs = obs;  // the next step's raw observations
            normalizer_->Step(obs.data(), rewards.data(), terminated.data(), truncated.data(), final_obs.data());
            metrics.timesteps += static_cast<int64_t>(n);
            ++vec_steps;
            learner.AfterStep(metrics.timesteps, vec_steps);

            // ---- Train every train_freq steps once past learning_starts
            metrics.learning = metrics.timesteps > s.learning_starts;
            if (metrics.learning && vec_steps % s.train_freq == 0) {
                for (int g = 0; g < s.gradient_steps && !stop_; ++g) {
                    ReplayBuffer::Batch b = buffer.Sample(batch, random.Next());
                    if (s.normalize_observations) {
                        b.observations = NormalizeOnDevice(b.observations, *normalizer_);
                        b.next_observations = NormalizeOnDevice(b.next_observations, *normalizer_);
                    }
                    learner.Learn(b, g, next_seed);
                    ++metrics.updates;
                }
            }

            // ---- Report
            const bool last = metrics.timesteps >= s.total_timesteps;
            if (!last && metrics.timesteps - last_report < s.report_every) continue;
            report();
            if (metrics.reached_target) break;
        }
        // A Stop reports where it stopped, so the run's last numbers say so
        if (stop_ && metrics.timesteps > last_report) report();
    } catch (const std::exception& e) {
        error = algorithm_ + " training failed: " + e.what();
        return false;
    }
    return true;
}

}  // namespace cyxwiz::rl
