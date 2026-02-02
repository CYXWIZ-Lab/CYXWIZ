#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ppo_agent.h"
#include <numeric>
#include <cstring>

namespace cyxwiz::plugin::rl {

// =============================================================================
// LinearLayer
// =============================================================================

void LinearLayer::Init(int in, int out, std::mt19937& rng) {
    in_features = in;
    out_features = out;

    // Xavier/Glorot initialization
    float scale = std::sqrt(2.0f / (in + out));
    std::normal_distribution<float> dist(0.0f, scale);

    weights.resize(out * in);
    bias.resize(out, 0.0f);
    grad_weights.resize(out * in, 0.0f);
    grad_bias.resize(out, 0.0f);

    for (auto& w : weights) w = dist(rng);
}

std::vector<float> LinearLayer::Forward(const std::vector<float>& input) const {
    assert(static_cast<int>(input.size()) == in_features);
    std::vector<float> output(out_features);

    for (int o = 0; o < out_features; ++o) {
        float sum = bias[o];
        const float* row = &weights[o * in_features];
        for (int i = 0; i < in_features; ++i) {
            sum += row[i] * input[i];
        }
        output[o] = sum;
    }
    return output;
}

void LinearLayer::ZeroGrad() {
    std::fill(grad_weights.begin(), grad_weights.end(), 0.0f);
    std::fill(grad_bias.begin(), grad_bias.end(), 0.0f);
}

// =============================================================================
// Activation functions
// =============================================================================

static std::vector<float> Tanh(const std::vector<float>& x) {
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = std::tanh(x[i]);
    return out;
}

// =============================================================================
// PPOAgent
// =============================================================================

PPOAgent::PPOAgent(int obs_dim, int act_dim, const PPOConfig& config)
    : obs_dim_(obs_dim), act_dim_(act_dim), config_(config), rng_(42)
{
    // Build policy network: obs -> hidden1 -> hidden2 -> ... -> mean_action
    auto sizes = config_.hidden_sizes;
    int prev = obs_dim;
    for (int h : sizes) {
        LinearLayer layer;
        layer.Init(prev, h, rng_);
        policy_layers_.push_back(std::move(layer));
        prev = h;
    }
    // Output layer (no activation — raw mean)
    {
        LinearLayer out_layer;
        out_layer.Init(prev, act_dim, rng_);
        // Small init for output layer
        float scale = 0.01f;
        for (auto& w : out_layer.weights) w *= scale / std::sqrt(2.0f / (prev + act_dim));
        policy_layers_.push_back(std::move(out_layer));
    }

    // Log standard deviation (learnable, per action dim)
    log_std_.assign(act_dim, 0.0f);  // init std = exp(0) = 1.0
    log_std_grad_.assign(act_dim, 0.0f);

    // Build value network (same architecture as policy)
    prev = obs_dim;
    for (int h : sizes) {
        LinearLayer layer;
        layer.Init(prev, h, rng_);
        value_layers_.push_back(std::move(layer));
        prev = h;
    }
    // Value output: single scalar
    {
        LinearLayer out_layer;
        out_layer.Init(prev, 1, rng_);
        value_layers_.push_back(std::move(out_layer));
    }

    // Initialize Adam states
    for (auto& layer : policy_layers_) {
        AdamState s;
        s.m.assign(layer.weights.size() + layer.bias.size(), 0.0f);
        s.v.assign(layer.weights.size() + layer.bias.size(), 0.0f);
        policy_adam_.push_back(std::move(s));
    }
    for (auto& layer : value_layers_) {
        AdamState s;
        s.m.assign(layer.weights.size() + layer.bias.size(), 0.0f);
        s.v.assign(layer.weights.size() + layer.bias.size(), 0.0f);
        value_adam_.push_back(std::move(s));
    }
    log_std_adam_.m.assign(act_dim, 0.0f);
    log_std_adam_.v.assign(act_dim, 0.0f);
}

std::vector<float> PPOAgent::PolicyForward(const std::vector<float>& obs) const {
    std::vector<float> x = obs;
    for (size_t i = 0; i + 1 < policy_layers_.size(); ++i) {
        x = policy_layers_[i].Forward(x);
        x = Tanh(x);
    }
    // Last layer: no activation (mean of Gaussian)
    x = policy_layers_.back().Forward(x);
    return x;
}

float PPOAgent::ValueForward(const std::vector<float>& obs) const {
    std::vector<float> x = obs;
    for (size_t i = 0; i + 1 < value_layers_.size(); ++i) {
        x = value_layers_[i].Forward(x);
        x = Tanh(x);
    }
    x = value_layers_.back().Forward(x);
    return x[0];
}

float PPOAgent::GaussianLogProb(const std::vector<float>& action,
                                 const std::vector<float>& mean) const {
    float log_prob = 0.0f;
    for (int i = 0; i < act_dim_; ++i) {
        float std = std::exp(log_std_[i]);
        float diff = action[i] - mean[i];
        log_prob += -0.5f * (diff * diff) / (std * std) - log_std_[i] - 0.5f * std::log(2.0f * static_cast<float>(M_PI));
    }
    return log_prob;
}

PPOAgent::ActionResult PPOAgent::SelectAction(const std::vector<float>& obs) {
    ActionResult result;

    // Forward pass through policy network
    std::vector<float> mean = PolicyForward(obs);

    // Sample from Gaussian
    result.action.resize(act_dim_);
    for (int i = 0; i < act_dim_; ++i) {
        float std = std::exp(log_std_[i]);
        std::normal_distribution<float> dist(mean[i], std);
        result.action[i] = dist(rng_);
    }

    result.log_prob = GaussianLogProb(result.action, mean);
    result.value = ValueForward(obs);

    return result;
}

std::vector<float> PPOAgent::GetMeanAction(const std::vector<float>& obs) const {
    return PolicyForward(obs);
}

float PPOAgent::GetValue(const std::vector<float>& obs) const {
    return ValueForward(obs);
}

// =============================================================================
// GAE Computation
// =============================================================================

void PPOAgent::ComputeGAE(std::vector<Transition>& rollout, float last_value,
                           std::vector<float>& advantages, std::vector<float>& returns) {
    int n = static_cast<int>(rollout.size());
    advantages.resize(n);
    returns.resize(n);

    float gae = 0.0f;
    float next_value = last_value;

    for (int t = n - 1; t >= 0; --t) {
        float next_non_terminal = rollout[t].done ? 0.0f : 1.0f;
        float delta = rollout[t].reward + config_.gamma * next_value * next_non_terminal - rollout[t].value;
        gae = delta + config_.gamma * config_.gae_lambda * next_non_terminal * gae;
        advantages[t] = gae;
        returns[t] = gae + rollout[t].value;
        next_value = rollout[t].value;
    }

    // Normalize advantages
    float mean = 0.0f, var = 0.0f;
    for (float a : advantages) mean += a;
    mean /= n;
    for (float a : advantages) var += (a - mean) * (a - mean);
    var /= n;
    float std = std::sqrt(var + 1e-8f);
    for (float& a : advantages) a = (a - mean) / std;
}

// =============================================================================
// PPO Update
// =============================================================================

PPOUpdateResult PPOAgent::Update(std::vector<Transition>& rollout, float last_value) {
    if (rollout.empty()) return {};

    int n = static_cast<int>(rollout.size());

    // Compute advantages and returns
    std::vector<float> advantages, returns;
    ComputeGAE(rollout, last_value, advantages, returns);

    // Store old log probs
    std::vector<float> old_log_probs(n);
    for (int i = 0; i < n; ++i) {
        old_log_probs[i] = rollout[i].log_prob;
    }

    PPOUpdateResult total_result;
    int num_updates = 0;

    // Multiple epochs over the data
    for (int epoch = 0; epoch < config_.n_epochs; ++epoch) {
        // Shuffle indices
        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng_);

        // Mini-batch updates
        for (int start = 0; start + config_.batch_size <= n; start += config_.batch_size) {
            std::vector<int> batch_indices(indices.begin() + start,
                                            indices.begin() + start + config_.batch_size);

            // Zero gradients
            for (auto& l : policy_layers_) l.ZeroGrad();
            for (auto& l : value_layers_) l.ZeroGrad();
            std::fill(log_std_grad_.begin(), log_std_grad_.end(), 0.0f);

            PPOUpdateResult batch_result;
            SGDStep(batch_indices, rollout, advantages, returns, old_log_probs, batch_result);

            // Apply gradients
            ApplyGradients();

            total_result.policy_loss += batch_result.policy_loss;
            total_result.value_loss += batch_result.value_loss;
            total_result.entropy += batch_result.entropy;
            num_updates++;
        }
    }

    if (num_updates > 0) {
        total_result.policy_loss /= num_updates;
        total_result.value_loss /= num_updates;
        total_result.entropy /= num_updates;
    }

    // Compute explained variance
    float var_returns = 0.0f, mean_returns = 0.0f;
    for (int i = 0; i < n; ++i) mean_returns += returns[i];
    mean_returns /= n;
    for (int i = 0; i < n; ++i) var_returns += (returns[i] - mean_returns) * (returns[i] - mean_returns);
    var_returns /= n;

    float var_unexplained = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = returns[i] - rollout[i].value;
        var_unexplained += diff * diff;
    }
    var_unexplained /= n;

    total_result.explained_variance = (var_returns > 1e-8f) ? 1.0f - var_unexplained / var_returns : 0.0f;

    return total_result;
}

void PPOAgent::SGDStep(const std::vector<int>& indices,
                        const std::vector<Transition>& rollout,
                        const std::vector<float>& advantages,
                        const std::vector<float>& returns,
                        const std::vector<float>& old_log_probs,
                        PPOUpdateResult& accum) {
    int bs = static_cast<int>(indices.size());

    for (int idx : indices) {
        const auto& tr = rollout[idx];

        // ---- Policy loss (numerical gradient approximation) ----
        // Forward: compute new log prob and ratio
        std::vector<float> mean = PolicyForward(tr.obs);
        float new_log_prob = GaussianLogProb(tr.action, mean);
        float ratio = std::exp(new_log_prob - old_log_probs[idx]);
        float adv = advantages[idx];

        float clipped_ratio = std::clamp(ratio, 1.0f - config_.clip_range, 1.0f + config_.clip_range);
        float policy_loss = -std::min(ratio * adv, clipped_ratio * adv);
        accum.policy_loss += policy_loss / bs;

        // Entropy bonus (Gaussian entropy = 0.5 * ln(2*pi*e*sigma^2))
        float entropy = 0.0f;
        for (int i = 0; i < act_dim_; ++i) {
            entropy += log_std_[i] + 0.5f * std::log(2.0f * static_cast<float>(M_PI) * std::exp(1.0f));
        }
        accum.entropy += entropy / bs;

        // ---- Value loss ----
        float value = ValueForward(tr.obs);
        float value_loss = (value - returns[idx]) * (value - returns[idx]);
        accum.value_loss += value_loss / bs;

        // ---- Compute gradients via finite differences (simplified) ----
        // For a production implementation, we'd use autograd. Here we use
        // a simple parameter perturbation approach for the policy gradient.

        // Policy gradient: d(loss)/d(param) ≈ (loss(param+eps) - loss(param-eps)) / (2*eps)
        // This is expensive but correct for small networks. For the initial
        // implementation we use the analytical gradient for Gaussian policy.

        // Analytical gradient for Gaussian policy:
        // d(log_prob)/d(mean_i) = (action_i - mean_i) / std_i^2
        // d(log_prob)/d(log_std_i) = (action_i - mean_i)^2 / std_i^2 - 1

        // Effective gradient = -ratio * advantage * d(log_prob)/d(params) when not clipped
        bool clipped = (ratio < 1.0f - config_.clip_range) || (ratio > 1.0f + config_.clip_range);
        float grad_scale = clipped ? 0.0f : -adv;  // 0 if clipped (no gradient)
        grad_scale /= bs;

        // Backprop through policy network
        // d(loss)/d(mean) = grad_scale * d(log_prob)/d(mean)
        std::vector<float> d_mean(act_dim_);
        for (int i = 0; i < act_dim_; ++i) {
            float std_val = std::exp(log_std_[i]);
            d_mean[i] = grad_scale * (tr.action[i] - mean[i]) / (std_val * std_val);

            // log_std gradient
            float d_log_std = grad_scale * ((tr.action[i] - mean[i]) * (tr.action[i] - mean[i]) / (std_val * std_val) - 1.0f);
            d_log_std -= config_.entropy_coeff / bs;  // entropy bonus encourages exploration
            log_std_grad_[i] += d_log_std;
        }

        // Backprop through MLP layers (policy)
        std::vector<float> grad_out = d_mean;
        // We need activations from forward pass for backprop
        std::vector<std::vector<float>> activations;
        {
            std::vector<float> x = tr.obs;
            activations.push_back(x);
            for (size_t l = 0; l + 1 < policy_layers_.size(); ++l) {
                x = policy_layers_[l].Forward(x);
                x = Tanh(x);
                activations.push_back(x);
            }
        }

        // Backprop from output layer to first layer
        for (int l = static_cast<int>(policy_layers_.size()) - 1; l >= 0; --l) {
            auto& layer = policy_layers_[l];
            const auto& input = activations[l];

            // Gradient w.r.t. weights and bias
            for (int o = 0; o < layer.out_features; ++o) {
                layer.grad_bias[o] += grad_out[o];
                for (int i = 0; i < layer.in_features; ++i) {
                    layer.grad_weights[o * layer.in_features + i] += grad_out[o] * input[i];
                }
            }

            if (l > 0) {
                // Gradient w.r.t. input (for next layer backward)
                std::vector<float> grad_input(layer.in_features, 0.0f);
                for (int o = 0; o < layer.out_features; ++o) {
                    for (int i = 0; i < layer.in_features; ++i) {
                        grad_input[i] += grad_out[o] * layer.weights[o * layer.in_features + i];
                    }
                }
                // Tanh derivative: 1 - tanh^2(x)
                for (int i = 0; i < layer.in_features; ++i) {
                    float a = activations[l][i];  // tanh output
                    grad_input[i] *= (1.0f - a * a);
                }
                grad_out = grad_input;
            }
        }

        // ---- Value network gradient (MSE loss) ----
        float d_value = 2.0f * (value - returns[idx]) / bs;

        // Backprop through value network
        std::vector<std::vector<float>> val_activations;
        {
            std::vector<float> x = tr.obs;
            val_activations.push_back(x);
            for (size_t l = 0; l + 1 < value_layers_.size(); ++l) {
                x = value_layers_[l].Forward(x);
                x = Tanh(x);
                val_activations.push_back(x);
            }
        }

        std::vector<float> vgrad = {d_value * config_.value_coeff};
        for (int l = static_cast<int>(value_layers_.size()) - 1; l >= 0; --l) {
            auto& layer = value_layers_[l];
            const auto& input = val_activations[l];

            for (int o = 0; o < layer.out_features; ++o) {
                layer.grad_bias[o] += vgrad[o];
                for (int i = 0; i < layer.in_features; ++i) {
                    layer.grad_weights[o * layer.in_features + i] += vgrad[o] * input[i];
                }
            }

            if (l > 0) {
                std::vector<float> grad_input(layer.in_features, 0.0f);
                for (int o = 0; o < layer.out_features; ++o) {
                    for (int i = 0; i < layer.in_features; ++i) {
                        grad_input[i] += vgrad[o] * layer.weights[o * layer.in_features + i];
                    }
                }
                for (int i = 0; i < layer.in_features; ++i) {
                    float a = val_activations[l][i];
                    grad_input[i] *= (1.0f - a * a);
                }
                vgrad = grad_input;
            }
        }
    }
}

void PPOAgent::ApplyGradients() {
    adam_step_++;
    float lr = config_.learning_rate;
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    float bc1 = 1.0f - std::pow(beta1, adam_step_);
    float bc2 = 1.0f - std::pow(beta2, adam_step_);

    auto adam_update = [&](float* param, float grad, float* m, float* v) {
        *m = beta1 * (*m) + (1 - beta1) * grad;
        *v = beta2 * (*v) + (1 - beta2) * grad * grad;
        float m_hat = *m / bc1;
        float v_hat = *v / bc2;
        *param -= lr * m_hat / (std::sqrt(v_hat) + eps);
    };

    // Update policy layers
    for (size_t l = 0; l < policy_layers_.size(); ++l) {
        auto& layer = policy_layers_[l];
        auto& state = policy_adam_[l];
        int nw = static_cast<int>(layer.weights.size());
        int nb = static_cast<int>(layer.bias.size());

        for (int i = 0; i < nw; ++i)
            adam_update(&layer.weights[i], layer.grad_weights[i], &state.m[i], &state.v[i]);
        for (int i = 0; i < nb; ++i)
            adam_update(&layer.bias[i], layer.grad_bias[i], &state.m[nw + i], &state.v[nw + i]);
    }

    // Update log_std
    for (int i = 0; i < act_dim_; ++i) {
        adam_update(&log_std_[i], log_std_grad_[i], &log_std_adam_.m[i], &log_std_adam_.v[i]);
    }

    // Update value layers
    for (size_t l = 0; l < value_layers_.size(); ++l) {
        auto& layer = value_layers_[l];
        auto& state = value_adam_[l];
        int nw = static_cast<int>(layer.weights.size());
        int nb = static_cast<int>(layer.bias.size());

        for (int i = 0; i < nw; ++i)
            adam_update(&layer.weights[i], layer.grad_weights[i], &state.m[i], &state.v[i]);
        for (int i = 0; i < nb; ++i)
            adam_update(&layer.bias[i], layer.grad_bias[i], &state.m[nw + i], &state.v[nw + i]);
    }
}

void PPOAgent::Reset() {
    // Re-initialize with same config
    *this = PPOAgent(obs_dim_, act_dim_, config_);
}

} // namespace cyxwiz::plugin::rl
