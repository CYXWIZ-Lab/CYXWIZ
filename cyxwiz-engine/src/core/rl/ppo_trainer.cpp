#include "ppo_trainer.h"

#include "../optimizer_step.h"

#include <cyxwiz/layers/linear.h>
#include <cyxwiz/optimizers/adam.h>
#include <cyxwiz/optimizers/gradient_clipping.h>
#include <cyxwiz/sequential.h>

#include <arrayfire.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <thread>
#include <vector>

namespace cyxwiz::rl {

namespace {

// SplitMix64 for the trainer's own draws (sample seeds, minibatch order)
class Random {
public:
    explicit Random(uint64_t seed) : state_(seed) {}
    uint64_t Next() {
        uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

private:
    uint64_t state_;
};

std::vector<float> Host(const Tensor& t) {
    const float* data = t.ReadData<float>();
    return std::vector<float>(data, data + t.NumElements());
}

Tensor Rows(const std::vector<float>& values, size_t rows, size_t width) {
    return width == 0 ? Tensor({rows}, values.data()) : Tensor({rows, width}, values.data());
}

// 1 - Var(returns - values) / Var(returns), as Stable-Baselines3 reports it
float ExplainedVariance(const std::vector<float>& values, const std::vector<float>& returns) {
    const auto variance = [](const std::vector<double>& x) {
        const double mean = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
        double sum = 0.0;
        for (const double v : x) sum += (v - mean) * (v - mean);
        return sum / static_cast<double>(x.size());
    };
    std::vector<double> y(returns.begin(), returns.end()), residual(returns.size());
    for (size_t i = 0; i < returns.size(); ++i) residual[i] = static_cast<double>(returns[i]) - values[i];
    const double var_y = variance(y);
    return var_y == 0.0 ? std::numeric_limits<float>::quiet_NaN() : static_cast<float>(1.0 - variance(residual) / var_y);
}

}  // namespace

void InitializeLikeStableBaselines3(IExecutableModel& model, float output_gain) {
    auto params = model.GetParameters();
    // the output layer: the weight matrix of the highest layer index ("layer<K>.weight")
    int last = -1;
    const auto layer_index = [](const std::string& name) {
        if (name.rfind("layer", 0) != 0) return -1;
        try {
            return std::stoi(name.substr(5));
        } catch (...) {
            return -1;
        }
    };
    for (const auto& [name, tensor] : params)
        if (tensor.Shape().size() == 2) last = std::max(last, layer_index(name));
    for (auto& [name, tensor] : params) {
        const auto shape = tensor.Shape();
        if (shape.size() == 2)
            tensor = OrthogonalWeights(shape[0], shape[1], layer_index(name) == last ? output_gain : std::sqrt(2.0f));
        else
            tensor = Tensor::Zeros(shape, DataType::Float32);
    }
    model.SetParameters(params);
}

PpoTrainer::PpoTrainer(PpoEnvironment environment, std::unique_ptr<IExecutableModel> policy,
                       std::unique_ptr<IExecutableModel> value, PpoConfig config)
    : environment_(std::move(environment)), policy_(std::move(policy)), value_(std::move(value)),
      config_(std::move(config)) {
    const size_t act = static_cast<size_t>(std::max(environment_.info.action_size, 1));
    const std::vector<float> init(act, config_.log_std_init);
    log_std_ = Tensor({act}, init.data());
    const auto adam = [this]() {
        return std::make_unique<AdamOptimizer>(config_.learning_rate, 0.9, 0.999, config_.adam_epsilon);
    };
    policy_optimizer_ = adam();
    value_optimizer_ = adam();
    log_std_optimizer_ = adam();
}

PpoTrainer::~PpoTrainer() = default;

bool PpoTrainer::TakeTrained(PpoTrainedState& out, std::string& error) {
    if (!policy_ || !value_ || !normalizer_) {
        error = "PPO has not trained a policy.";
        return false;
    }
    const auto state = [&error](const Optimizer& optimizer, OptimizerState& s) {
        return optimizer.ExportState(s, error);
    };
    std::map<std::string, OptimizerState> optimizers;
    if (!state(*policy_optimizer_, optimizers["policy"]) || !state(*value_optimizer_, optimizers["value"]) ||
        (!environment_.info.discrete_actions && !state(*log_std_optimizer_, optimizers["log_std"])))
        return false;
    out.optimizers = std::move(optimizers);
    policy_fused_.reset();  // they read the networks that move out
    value_fused_.reset();
    out.policy = std::move(policy_);
    out.value = std::move(value_);
    out.log_std = log_std_;
    out.normalizer = std::move(normalizer_);
    return true;
}

bool PpoTrainer::CheckSetup(std::string& error) const {
    const auto& info = environment_.info;
    if (!policy_ || !value_) error = "PPO needs a policy network and a value network.";
    else if (environment_.count < 1 || !environment_.reset || !environment_.step) error = "PPO needs an environment.";
    else if (config_.n_steps < 1 || config_.batch_size < 1 || config_.n_epochs < 1)
        error = "Rollout steps, batch size and epochs must each be at least 1.";
    else if (!(config_.learning_rate > 0.0f)) error = "The learning rate must be above 0.";
    else if (!(config_.gamma > 0.0f && config_.gamma <= 1.0f) || !(config_.gae_lambda >= 0.0f && config_.gae_lambda <= 1.0f))
        error = "Gamma must be in (0, 1] and the GAE lambda in [0, 1].";
    else if (!(config_.clip_observations > 0.0f) || !(config_.clip_rewards > 0.0f))
        error = "The observation and reward clips must be above 0.";
    else if (info.observation_size < 1 || info.action_size < 1) error = "The environment observes or acts on nothing.";
    return error.empty();
}

bool PpoTrainer::Train(const std::function<void(const PpoUpdateMetrics&)>& on_update, std::string& error) {
    error.clear();
    if (!CheckSetup(error)) return false;
    const auto& info = environment_.info;
    const bool discrete = info.discrete_actions;
    const size_t n = static_cast<size_t>(environment_.count);
    const size_t obs_size = static_cast<size_t>(info.observation_size);
    const size_t act_size = static_cast<size_t>(info.action_size);
    const size_t policy_outputs = discrete ? static_cast<size_t>(info.action_count) : act_size;
    const size_t steps = static_cast<size_t>(config_.n_steps);
    const size_t rollout = steps * n;
    Random random(config_.seed ^ 0x5DEECE66Dull);

    std::vector<float> obs(n * obs_size), final_obs(n * obs_size), rewards(n), env_actions(n * act_size);
    std::vector<uint8_t> terminated(n), truncated(n);
    const plugin::PluginEnvironmentStepBuffers out{obs.data(), rewards.data(), terminated.data(), truncated.data(),
                                                   final_obs.data()};
    std::vector<float> obs_buf(rollout * obs_size), act_buf(rollout * act_size), logp_buf(rollout),
        value_buf(rollout), reward_buf(rollout), done_buf(rollout);
    std::vector<double> episode_return(n, 0.0);
    std::vector<int> episode_length(n, 0);
    std::deque<std::pair<float, int>> recent;  // the last 100 episodes
    PpoUpdateMetrics metrics;
    VecNormalizeSettings norm;
    norm.observations = config_.normalize_observations;
    norm.rewards = config_.normalize_rewards;
    norm.clip_observations = config_.clip_observations;
    norm.clip_rewards = config_.clip_rewards;
    norm.gamma = config_.gamma;
    normalizer_ = std::make_unique<VecNormalize>(n, obs_size, norm);

    // Small Dense/activation networks run whole on the device's provider: a
    // few launches per forward and backward instead of several per layer
    // (TOFIX206 3.2). Otherwise, or when a fused call fails, the layers run
    // as ArrayFire operations.
    const auto fuse = [](IExecutableModel& model, std::string& why) -> std::unique_ptr<FusedMlp> {
        SequentialModel* sequential = model.AsSequentialModel();
        if (!sequential) {
            why = "not a sequential network";
            return nullptr;
        }
        return FusedMlp::For(*sequential, why);
    };
    std::string policy_why, value_why;
    policy_fused_ = fuse(*policy_, policy_why);
    value_fused_ = fuse(*value_, value_why);
    if (policy_fused_ && value_fused_) {
        network_path_ = "fused small-network kernels (" + policy_fused_->ProviderId() + ")";
    } else {
        policy_fused_.reset();  // both or neither: one path to report
        value_fused_.reset();
        network_path_ = "ArrayFire operations (" + (policy_why.empty() ? "value: " + value_why : "policy: " + policy_why) + ")";
    }
    spdlog::info("PPO: the networks run as {}", network_path_);
    const auto forward = [this](IExecutableModel& model, std::unique_ptr<FusedMlp>& fused, const Tensor& x,
                                bool keep) -> Tensor {
        if (fused) {
            try {
                return fused->Forward(x, keep);
            } catch (const std::exception& e) {
                spdlog::warn("PPO: the fused network failed ({}); ArrayFire operations from now on", e.what());
                fused.reset();  // this network only: its forward and backward stay on one path
                network_path_ = std::string("ArrayFire operations (the fused kernels failed: ") + e.what() + ")";
            }
        }
        return model.Forward(x);
    };
    // One Adam pass (and clip) for the policy, the value network and log_std
    // together when the device's provider runs it (TOFIX206 3.2 C); else, or
    // once it fails, each optimizer steps per tensor as before.
    auto* policy_adam = dynamic_cast<AdamOptimizer*>(policy_optimizer_.get());
    auto* value_adam = dynamic_cast<AdamOptimizer*>(value_optimizer_.get());
    auto* log_std_adam = dynamic_cast<AdamOptimizer*>(log_std_optimizer_.get());
    bool adam_together = policy_adam && value_adam && log_std_adam;
    // x: the minibatch's input, to run the layers' own forward when the
    // fused backward fails (the layers have no forward of it cached)
    const auto backward = [this](IExecutableModel& model, std::unique_ptr<FusedMlp>& fused, const Tensor& x,
                                 const Tensor& grad) -> std::map<std::string, Tensor> {
        if (fused) {
            try {
                return fused->Backward(grad);
            } catch (const std::exception& e) {
                spdlog::warn("PPO: the fused network failed ({}); ArrayFire operations from now on", e.what());
                fused.reset();
                network_path_ = std::string("ArrayFire operations (the fused kernels failed: ") + e.what() + ")";
                model.Forward(x);
            }
        }
        model.Backward(grad);
        return model.GetGradients();
    };
    // Both networks over the same rows (the rollout's observations, a
    // minibatch): one launch forward and two backward for the two when both
    // are fused (TOFIX206 3.2 B2). A failure moves both to ArrayFire
    // operations.
    const auto drop_fused = [this](const std::exception& e) {
        spdlog::warn("PPO: the fused networks failed ({}); ArrayFire operations from now on", e.what());
        policy_fused_.reset();
        value_fused_.reset();
        network_path_ = std::string("ArrayFire operations (the fused kernels failed: ") + e.what() + ")";
    };
    const auto forward_both = [&](const Tensor& x, bool keep) -> std::pair<Tensor, Tensor> {
        if (policy_fused_ && value_fused_) {
            try {
                return FusedMlp::ForwardPair(*policy_fused_, *value_fused_, x, keep);
            } catch (const std::exception& e) {
                drop_fused(e);
            }
        }
        Tensor head = forward(*policy_, policy_fused_, x, keep);
        return {std::move(head), forward(*value_, value_fused_, x, keep)};
    };
    const auto backward_both = [&](const Tensor& x, const Tensor& policy_grad, const Tensor& value_grad)
        -> std::pair<FusedMlp::Gradients, FusedMlp::Gradients> {
        if (policy_fused_ && value_fused_) {
            try {
                return FusedMlp::BackwardPair(*policy_fused_, policy_grad, *value_fused_, value_grad);
            } catch (const std::exception& e) {
                drop_fused(e);
                policy_->Forward(x);  // the layers' own forward of the minibatch, for their backward
                value_->Forward(x);
            }
        }
        auto policy_grads = backward(*policy_, policy_fused_, x, policy_grad);
        return {std::move(policy_grads), backward(*value_, value_fused_, x, value_grad)};
    };

    try {
        if (!environment_.reset(config_.seed, obs.data(), error)) return false;
        normalizer_->Reset(obs.data());
        while (metrics.timesteps < config_.total_timesteps && !stop_) {
            const auto update_start = std::chrono::steady_clock::now();
            policy_->SetTraining(false);
            value_->SetTraining(false);
            // ---- Collect n_steps from every copy
            for (size_t t = 0; t < steps; ++t) {
                while (paused_ && !stop_) std::this_thread::sleep_for(std::chrono::milliseconds(50));
                if (stop_) break;
                const Tensor obs_t = Rows(obs, n, obs_size);
                const auto [head, value] = forward_both(obs_t, false);
                if (head.Shape() != std::vector<size_t>{n, policy_outputs}) {
                    error = "The policy network gives " + std::to_string(head.NumElements() / std::max<size_t>(n, 1)) +
                            " outputs per observation; " + info.name + " needs " + std::to_string(policy_outputs) +
                            (discrete ? " (one per choice)." : " (one per action).");
                    return false;
                }
                if (value.NumElements() != n) {
                    error = "The value network must give one value per observation.";
                    return false;
                }
                const uint64_t sample_seed = random.Next();
                // One read for the step: the values, log-probabilities and actions (one launch on a
                // provider, TOFIX206 3.2 B)
                const auto step_host = Host(PpoSampleStep(head, value, log_std_, discrete, sample_seed));
                const float* values = step_host.data();
                const float* logps = values + n;
                const float* actions = logps + n;
                for (size_t i = 0; i < n * act_size; ++i) {
                    float a = actions[i];
                    if (!discrete) a = std::clamp(a, info.action_low[i % act_size], info.action_high[i % act_size]);
                    env_actions[i] = a;
                }
                const size_t row = t * n;
                std::copy(obs.begin(), obs.end(), obs_buf.begin() + row * obs_size);
                std::copy_n(actions, n * act_size, act_buf.begin() + row * act_size);
                std::copy_n(logps, n, logp_buf.begin() + row);
                std::copy_n(values, n, value_buf.begin() + row);

                if (!environment_.step(env_actions.data(), out, error)) return false;
                for (size_t i = 0; i < n; ++i) {  // what the environment paid, for the episode's return
                    episode_return[i] += rewards[i];
                    ++episode_length[i];
                }
                normalizer_->Step(obs.data(), rewards.data(), terminated.data(), truncated.data(), final_obs.data());
                // A cut episode is worth what its last observation is worth
                std::vector<size_t> cut;
                for (size_t i = 0; i < n; ++i)
                    if (truncated[i] && !terminated[i]) cut.push_back(i);  // a time limit, as SB3 bootstraps
                if (!cut.empty()) {
                    std::vector<float> last(cut.size() * obs_size);
                    for (size_t k = 0; k < cut.size(); ++k)
                        std::copy_n(final_obs.begin() + cut[k] * obs_size, obs_size, last.begin() + k * obs_size);
                    const auto last_values = Host(forward(*value_, value_fused_, Rows(last, cut.size(), obs_size), false));
                    for (size_t k = 0; k < cut.size(); ++k) rewards[cut[k]] += config_.gamma * last_values[k];
                }
                for (size_t i = 0; i < n; ++i) {
                    const bool done = terminated[i] || truncated[i];
                    reward_buf[row + i] = rewards[i];
                    done_buf[row + i] = done ? 1.0f : 0.0f;
                    if (done) {
                        recent.emplace_back(static_cast<float>(episode_return[i]), episode_length[i]);
                        if (recent.size() > 100) recent.pop_front();
                        ++metrics.episodes;
                        episode_return[i] = 0.0;
                        episode_length[i] = 0;
                    }
                }
                metrics.timesteps += static_cast<int64_t>(n);
            }
            if (stop_) break;

            // ---- Advantages and returns
            const auto last_values = Host(forward(*value_, value_fused_, Rows(obs, n, obs_size), false));
            Tensor advantages, returns;
            ComputeGae(Rows(reward_buf, steps, n), Rows(value_buf, steps, n), Rows(done_buf, steps, n),
                       Tensor({n}, last_values.data()), config_.gamma, config_.gae_lambda, advantages, returns);
            const auto returns_host = Host(returns);
            // The rollout as one [rollout, width] table on the device (the
            // host's columns go up in one piece)
            const size_t host_width = obs_size + act_size + 2;
            const size_t width = host_width + 2;
            std::vector<float> host_table(rollout * host_width);
            for (size_t r = 0; r < rollout; ++r) {
                float* row = host_table.data() + r * host_width;
                std::copy_n(obs_buf.begin() + r * obs_size, obs_size, row);
                std::copy_n(act_buf.begin() + r * act_size, act_size, row + obs_size);
                row[obs_size + act_size] = logp_buf[r];
                row[obs_size + act_size + 1] = value_buf[r];
            }
            const Tensor table = Tensor::Cat({Rows(host_table, rollout, host_width), advantages.Reshape({rollout, 1}),
                                              returns.Reshape({rollout, 1})},
                                             1);
            const af::array table_array = table.GetSemanticArray();
            // Every epoch's shuffled order, uploaded once per update (TOFIX206 3.2 B2)
            const size_t epochs = static_cast<size_t>(std::max(config_.n_epochs, 0));
            std::vector<unsigned> orders(rollout * epochs);
            for (size_t e = 0; e < epochs; ++e) {
                unsigned* order = orders.data() + e * rollout;
                std::iota(order, order + rollout, 0u);
                for (size_t i = rollout; i > 1; --i) std::swap(order[i - 1], order[random.Next() % i]);
            }
            const af::array orders_device = orders.empty() ? af::array() : af::array(static_cast<dim_t>(orders.size()), orders.data());

            // ---- K epochs of minibatches
            policy_->SetTraining(true);
            value_->SetTraining(true);
            Tensor statistics_sum, last_statistics;  // on the device until the update ends
            int minibatches = 0;
            for (size_t epoch = 0; epoch < epochs && !stop_; ++epoch) {
                for (size_t start = 0; start < rollout && !stop_; start += static_cast<size_t>(config_.batch_size)) {
                    const size_t end = std::min(rollout, start + static_cast<size_t>(config_.batch_size));
                    // The minibatch's rows gathered into a table of their own: each
                    // of its columns is a contiguous run the provider reads in place
                    const dim_t at = static_cast<dim_t>(epoch * rollout + start);
                    af::array rows_array =
                        af::lookup(table_array, orders_device(af::seq(static_cast<double>(at), static_cast<double>(at + static_cast<dim_t>(end - start) - 1))), 0);
                    rows_array.eval();
                    const Tensor minibatch = Tensor::FromSemanticArray(rows_array, {end - start, width});
                    const auto column = [&minibatch](size_t from, size_t count) {
                        return minibatch.Slice(1, static_cast<int>(from), static_cast<int>(from + count));
                    };
                    PpoBatch batch;
                    batch.actions = column(obs_size, act_size);
                    batch.old_log_prob = column(obs_size + act_size, 1);
                    batch.old_values = column(obs_size + act_size + 1, 1);
                    batch.advantages = column(obs_size + act_size + 2, 1);
                    batch.returns = column(width - 1, 1);
                    const Tensor mb_obs = column(0, obs_size);
                    const auto [head, value] = forward_both(mb_obs, true);
                    const PpoLossResult r = discrete ? PpoCategoricalLoss(head, value, batch, config_.loss)
                                                     : PpoGaussianLoss(head, log_std_, value, batch, config_.loss);
                    auto [policy_grads, value_grads] = backward_both(mb_obs, r.grad_policy_output, r.grad_value_output);
                    std::map<std::string, Tensor> log_std_grads;
                    if (!discrete) log_std_grads["log_std"] = r.grad_log_std;
                    bool stepped = false;
                    if (adam_together) {
                        auto policy_params = policy_->GetParameters();
                        auto value_params = value_->GetParameters();
                        std::map<std::string, Tensor> log_std_params{{"log_std", log_std_}};
                        std::vector<AdamOptimizer::Group> groups{{policy_adam, &policy_params, &policy_grads},
                                                                 {value_adam, &value_params, &value_grads}};
                        if (!discrete) groups.push_back({log_std_adam, &log_std_params, &log_std_grads});
                        std::string why;
                        if (AdamOptimizer::StepTogether(groups, config_.max_grad_norm, why)) {
                            policy_->SetParameters(policy_params);
                            value_->SetParameters(value_params);
                            if (!discrete) log_std_ = log_std_params.at("log_std");
                            stepped = true;
                        } else {
                            adam_together = false;
                            spdlog::info("PPO: the optimizers step per tensor ({})", why);
                        }
                    }
                    // One global norm over every parameter, as one optimizer over all of them clips
                    if (!stepped && config_.max_grad_norm > 0.0f) {
                        std::map<std::string, Tensor> all;
                        for (auto& [k, g] : policy_grads) all["p:" + k] = g;
                        for (auto& [k, g] : value_grads) all["v:" + k] = g;
                        for (auto& [k, g] : log_std_grads) all["s:" + k] = g;
                        ClipGradientsByGlobalNormOnDevice(all, config_.max_grad_norm);
                        for (auto& [k, g] : policy_grads) g = all["p:" + k];
                        for (auto& [k, g] : value_grads) g = all["v:" + k];
                        for (auto& [k, g] : log_std_grads) g = all["s:" + k];
                    }
                    if (!stepped) {
                        ApplyOptimizerStep(*policy_, *policy_optimizer_, policy_grads, {});
                        ApplyOptimizerStep(*value_, *value_optimizer_, value_grads, {});
                        if (!discrete) {
                            std::map<std::string, Tensor> params{{"log_std", log_std_}};
                            log_std_optimizer_->Step(params, log_std_grads);
                            log_std_ = params.at("log_std");
                        }
                    }
                    statistics_sum = minibatches == 0 ? r.statistics : statistics_sum + r.statistics;
                    last_statistics = r.statistics;
                    ++minibatches;
                }
            }

            if (stop_) break;  // a stopped update is not reported as one
            ++metrics.update;
            metrics.compute_path = network_path_ + (adam_together ? "" : ", optimizers per tensor");
            if (minibatches > 0) {  // the minibatches' mean, and the last one's loss
                const PpoLossStatistics mean = ReadPpoLossStatistics(statistics_sum, static_cast<float>(minibatches));
                metrics.policy_loss = mean.policy_loss;
                metrics.value_loss = mean.value_loss;
                metrics.entropy_loss = mean.entropy_loss;
                metrics.approx_kl = mean.approx_kl;
                metrics.clip_fraction = mean.clip_fraction;
                metrics.loss = ReadPpoLossStatistics(last_statistics).loss;
            }
            metrics.explained_variance = ExplainedVariance(value_buf, returns_host);
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
            if (!discrete) {
                const auto ls = Host(log_std_);
                double s = 0.0;
                for (const float v : ls) s += std::exp(v);
                metrics.std = static_cast<float>(s / ls.size());
            }
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - update_start).count();
            metrics.steps_per_second = seconds > 0.0 ? static_cast<float>(rollout / seconds) : 0.0f;
            metrics.reached_target = recent.size() >= 100 && metrics.mean_episode_return >= config_.target_return;
            if (on_update) on_update(metrics);
            if (metrics.reached_target) break;
        }
    } catch (const std::exception& e) {
        error = std::string("PPO training failed: ") + e.what();
        return false;
    }
    return true;
}

}  // namespace cyxwiz::rl
