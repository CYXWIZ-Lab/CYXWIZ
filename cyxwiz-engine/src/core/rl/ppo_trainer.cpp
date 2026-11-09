#include "ppo_trainer.h"

#include "../optimizer_step.h"

#include <cyxwiz/layers/linear.h>
#include <cyxwiz/optimizers/adam.h>
#include <cyxwiz/optimizers/gradient_clipping.h>

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
                const Tensor head = policy_->Forward(obs_t);
                if (head.Shape() != std::vector<size_t>{n, policy_outputs}) {
                    error = "The policy network gives " + std::to_string(head.NumElements() / std::max<size_t>(n, 1)) +
                            " outputs per observation; " + info.name + " needs " + std::to_string(policy_outputs) +
                            (discrete ? " (one per choice)." : " (one per action).");
                    return false;
                }
                const Tensor value = value_->Forward(obs_t);
                if (value.NumElements() != n) {
                    error = "The value network must give one value per observation.";
                    return false;
                }
                const uint64_t sample_seed = random.Next();
                const Tensor action = discrete ? SampleCategorical(head, sample_seed) : SampleGaussian(head, log_std_, sample_seed);
                const Tensor logp = discrete ? CategoricalLogProb(head, action) : GaussianLogProb(head, log_std_, action);
                // One read for the step: the values, log-probabilities and actions
                const auto step_host =
                    Host(Tensor::Cat({value.Reshape({n}), logp.Reshape({n}), action.Reshape({n * act_size})}, 0));
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
                    if (truncated[i]) cut.push_back(i);
                if (!cut.empty()) {
                    std::vector<float> last(cut.size() * obs_size);
                    for (size_t k = 0; k < cut.size(); ++k)
                        std::copy_n(final_obs.begin() + cut[k] * obs_size, obs_size, last.begin() + k * obs_size);
                    const auto last_values = Host(value_->Forward(Rows(last, cut.size(), obs_size)));
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
            const auto last_values = Host(value_->Forward(Rows(obs, n, obs_size)));
            Tensor advantages, returns;
            ComputeGae(Rows(reward_buf, steps, n), Rows(value_buf, steps, n), Rows(done_buf, steps, n),
                       Tensor({n}, last_values.data()), config_.gamma, config_.gae_lambda, advantages, returns);
            const auto returns_host = Host(returns);
            // The rollout as one [rollout, width] table on the device: each
            // epoch shuffles it with one gather, each minibatch is a run of rows
            const size_t width = obs_size + act_size + 4;
            const Tensor table = Tensor::Cat({Rows(obs_buf, rollout, obs_size), Rows(act_buf, rollout, act_size),
                                              Rows(logp_buf, rollout, 1), Rows(value_buf, rollout, 1),
                                              advantages.Reshape({rollout, 1}), returns.Reshape({rollout, 1})},
                                             1);

            // ---- K epochs of minibatches
            policy_->SetTraining(true);
            value_->SetTraining(true);
            std::vector<int> order(rollout);
            Tensor statistics_sum, last_statistics;  // on the device until the update ends
            int minibatches = 0;
            for (int epoch = 0; epoch < config_.n_epochs && !stop_; ++epoch) {
                std::iota(order.begin(), order.end(), 0);
                for (size_t i = rollout; i > 1; --i) std::swap(order[i - 1], order[random.Next() % i]);
                const Tensor shuffled = table.IndexSelect(0, order);
                const auto column = [&shuffled](size_t from, size_t count) {
                    return shuffled.Slice(1, static_cast<int>(from), static_cast<int>(from + count));
                };
                const Tensor epoch_obs = column(0, obs_size);
                const Tensor epoch_actions = column(obs_size, act_size);
                const Tensor epoch_logp = column(obs_size + act_size, 1);
                const Tensor epoch_values = column(obs_size + act_size + 1, 1);
                const Tensor epoch_adv = column(obs_size + act_size + 2, 1);
                const Tensor epoch_ret = column(width - 1, 1);
                for (size_t start = 0; start < rollout && !stop_; start += static_cast<size_t>(config_.batch_size)) {
                    const size_t end = std::min(rollout, start + static_cast<size_t>(config_.batch_size));
                    const auto rows = [start, end](const Tensor& t) {
                        return t.Slice(0, static_cast<int>(start), static_cast<int>(end));
                    };
                    PpoBatch batch;
                    batch.actions = rows(epoch_actions);
                    batch.old_log_prob = rows(epoch_logp);
                    batch.advantages = rows(epoch_adv);
                    batch.returns = rows(epoch_ret);
                    batch.old_values = rows(epoch_values);
                    const Tensor mb_obs = rows(epoch_obs);
                    const Tensor head = policy_->Forward(mb_obs);
                    const Tensor value = value_->Forward(mb_obs);
                    const PpoLossResult r = discrete ? PpoCategoricalLoss(head, value, batch, config_.loss)
                                                     : PpoGaussianLoss(head, log_std_, value, batch, config_.loss);
                    policy_->Backward(r.grad_policy_output);
                    value_->Backward(r.grad_value_output);
                    auto policy_grads = policy_->GetGradients();
                    auto value_grads = value_->GetGradients();
                    std::map<std::string, Tensor> log_std_grads;
                    if (!discrete) log_std_grads["log_std"] = r.grad_log_std;
                    // One global norm over every parameter, as one optimizer over all of them clips
                    if (config_.max_grad_norm > 0.0f) {
                        std::map<std::string, Tensor> all;
                        for (auto& [k, g] : policy_grads) all["p:" + k] = g;
                        for (auto& [k, g] : value_grads) all["v:" + k] = g;
                        for (auto& [k, g] : log_std_grads) all["s:" + k] = g;
                        ClipGradientsByGlobalNormOnDevice(all, config_.max_grad_norm);
                        for (auto& [k, g] : policy_grads) g = all["p:" + k];
                        for (auto& [k, g] : value_grads) g = all["v:" + k];
                        for (auto& [k, g] : log_std_grads) g = all["s:" + k];
                    }
                    ApplyOptimizerStep(*policy_, *policy_optimizer_, policy_grads, {});
                    ApplyOptimizerStep(*value_, *value_optimizer_, value_grads, {});
                    if (!discrete) {
                        std::map<std::string, Tensor> params{{"log_std", log_std_}};
                        log_std_optimizer_->Step(params, log_std_grads);
                        log_std_ = params.at("log_std");
                    }
                    statistics_sum = minibatches == 0 ? r.statistics : statistics_sum + r.statistics;
                    last_statistics = r.statistics;
                    ++minibatches;
                }
            }

            ++metrics.update;
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
