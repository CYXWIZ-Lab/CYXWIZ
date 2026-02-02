#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <algorithm>

namespace cyxwiz::plugin::rl {

// Simple MLP layer (dense + activation)
struct LinearLayer {
    std::vector<float> weights;  // [out_features x in_features] row-major
    std::vector<float> bias;     // [out_features]
    int in_features = 0;
    int out_features = 0;

    // Gradient accumulators
    std::vector<float> grad_weights;
    std::vector<float> grad_bias;

    void Init(int in, int out, std::mt19937& rng);
    std::vector<float> Forward(const std::vector<float>& input) const;
    void ZeroGrad();
};

// Transition stored in rollout buffer
struct Transition {
    std::vector<float> obs;
    std::vector<float> action;
    float reward = 0.0f;
    float value = 0.0f;
    float log_prob = 0.0f;
    bool done = false;
};

struct PPOConfig {
    float learning_rate = 3e-4f;
    float gamma = 0.99f;
    float gae_lambda = 0.95f;
    float clip_range = 0.2f;
    int n_steps = 2048;        // rollout buffer size
    int batch_size = 64;
    int n_epochs = 10;
    std::vector<int> hidden_sizes = {64, 64};
    float entropy_coeff = 0.01f;
    float value_coeff = 0.5f;
    float max_grad_norm = 0.5f;
};

struct PPOUpdateResult {
    float policy_loss = 0.0f;
    float value_loss = 0.0f;
    float entropy = 0.0f;
    float explained_variance = 0.0f;
};

/**
 * PPOAgent - Lightweight Proximal Policy Optimization implementation.
 *
 * Uses simple MLP for both policy (actor) and value (critic) networks.
 * Continuous action space with diagonal Gaussian policy.
 * No external ML framework dependency — pure C++ with manual backprop.
 */
class PPOAgent {
public:
    PPOAgent(int obs_dim, int act_dim, const PPOConfig& config = {});

    // Forward pass: get action, log probability, and value estimate
    struct ActionResult {
        std::vector<float> action;
        float log_prob;
        float value;
    };
    ActionResult SelectAction(const std::vector<float>& obs);

    // Deterministic action (for evaluation/inference)
    std::vector<float> GetMeanAction(const std::vector<float>& obs) const;

    // Value estimate only
    float GetValue(const std::vector<float>& obs) const;

    // PPO update from collected rollout
    PPOUpdateResult Update(std::vector<Transition>& rollout, float last_value);

    // Access dimensions
    int GetObsDim() const { return obs_dim_; }
    int GetActDim() const { return act_dim_; }

    // Get policy network weights (for ONNX export)
    const std::vector<LinearLayer>& GetPolicyLayers() const { return policy_layers_; }
    const std::vector<float>& GetLogStd() const { return log_std_; }

    void Reset();

private:
    // MLP forward pass
    std::vector<float> PolicyForward(const std::vector<float>& obs) const;
    float ValueForward(const std::vector<float>& obs) const;

    // Gaussian log probability
    float GaussianLogProb(const std::vector<float>& action,
                          const std::vector<float>& mean) const;

    // Compute GAE advantages
    void ComputeGAE(std::vector<Transition>& rollout, float last_value,
                    std::vector<float>& advantages, std::vector<float>& returns);

    // Single SGD step on a mini-batch
    void SGDStep(const std::vector<int>& indices,
                 const std::vector<Transition>& rollout,
                 const std::vector<float>& advantages,
                 const std::vector<float>& returns,
                 const std::vector<float>& old_log_probs,
                 PPOUpdateResult& accum);

    // Apply gradients with Adam-like update
    void ApplyGradients();

    int obs_dim_;
    int act_dim_;
    PPOConfig config_;
    std::mt19937 rng_;

    // Policy network (actor): obs -> mean action
    std::vector<LinearLayer> policy_layers_;
    std::vector<float> log_std_;  // learnable log std dev per action dim

    // Value network (critic): obs -> scalar value
    std::vector<LinearLayer> value_layers_;

    // Adam optimizer state
    struct AdamState {
        std::vector<float> m;  // first moment
        std::vector<float> v;  // second moment
    };
    std::vector<AdamState> policy_adam_;
    std::vector<AdamState> value_adam_;
    AdamState log_std_adam_;
    int adam_step_ = 0;

    // Gradient for log_std
    std::vector<float> log_std_grad_;
};

} // namespace cyxwiz::plugin::rl
