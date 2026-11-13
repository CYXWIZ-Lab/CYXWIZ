// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

// Proximal Policy Optimization on the device, with hand-written gradients
// like every CyxWiz loss. The policy network outputs the mean of a Gaussian
// (continuous actions, with a learned state-independent log standard
// deviation) or the logits of a categorical (discrete actions); the value
// network outputs one value per row. The loss is Stable-Baselines3's:
//
//   ratio = exp(log_prob - old_log_prob)
//   policy_loss  = -mean(min(A * ratio, A * clip(ratio, 1 - c, 1 + c)))
//   value_loss   = mean((returns - v')^2), v' = v, or old_v + clip(v - old_v, -cv, cv)
//   entropy_loss = -mean(entropy)
//   loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss
//
// with A the minibatch's advantages, normalised to mean 0 and (unbiased)
// standard deviation 1 when asked. Gradients are those of `loss`.

#include "../api_export.h"
#include "../tensor.h"

#include <cstdint>

namespace cyxwiz::rl {

struct CYXWIZ_API PpoLossSettings {
    float clip_range = 0.2f;
    float clip_range_vf = 0.0f;  // 0: the value is not clipped
    float ent_coef = 0.0f;
    float vf_coef = 0.5f;
    bool normalize_advantage = true;
};

// What a minibatch's loss came to, read from PpoLossResult::statistics.
struct CYXWIZ_API PpoLossStatistics {
    float loss = 0.0f;
    float policy_loss = 0.0f;
    float value_loss = 0.0f;
    float entropy_loss = 0.0f;
    float approx_kl = 0.0f;       // mean((ratio - 1) - log(ratio))
    float clip_fraction = 0.0f;   // share of rows with |ratio - 1| > clip_range
};

struct CYXWIZ_API PpoLossResult {
    // [6] on the device: loss, policy_loss, value_loss, entropy_loss,
    // approx_kl, clip_fraction. A trainer sums them over minibatches and
    // reads them once (ReadPpoLossStatistics), so a minibatch never waits
    // for the device.
    Tensor statistics;
    Tensor log_prob;              // [B], under the current policy
    // d loss / d the networks' outputs and the log standard deviation
    Tensor grad_policy_output;    // [B, A] means or [B, K] logits
    Tensor grad_log_std;          // [A]; empty for a categorical policy
    Tensor grad_value_output;     // [B, 1]
};

// The minibatch: what was done and what it was worth.
struct CYXWIZ_API PpoBatch {
    Tensor actions;        // [B, A] Gaussian; [B] or [B, 1] action indices (as floats) categorical
    Tensor old_log_prob;   // [B]
    Tensor advantages;     // [B]
    Tensor returns;        // [B]
    Tensor old_values;     // [B]
};

// The host copy of a statistics tensor, or of a sum of `count` of them as their mean.
CYXWIZ_API PpoLossStatistics ReadPpoLossStatistics(const Tensor& statistics, float count = 1.0f);

// mean [B, A], log_std [A], value [B, 1]
CYXWIZ_API PpoLossResult PpoGaussianLoss(const Tensor& mean, const Tensor& log_std, const Tensor& value,
                                         const PpoBatch& batch, const PpoLossSettings& settings);
// logits [B, K], value [B, 1]
CYXWIZ_API PpoLossResult PpoCategoricalLoss(const Tensor& logits, const Tensor& value, const PpoBatch& batch,
                                            const PpoLossSettings& settings);

// Log-probabilities [B] of actions under the policy (for the rollout).
CYXWIZ_API Tensor GaussianLogProb(const Tensor& mean, const Tensor& log_std, const Tensor& actions);
CYXWIZ_API Tensor CategoricalLogProb(const Tensor& logits, const Tensor& actions);

// Sampling: the same seed gives the same actions on the same device.
CYXWIZ_API Tensor SampleGaussian(const Tensor& mean, const Tensor& log_std, uint64_t seed);   // [B, A]
CYXWIZ_API Tensor SampleCategorical(const Tensor& logits, uint64_t seed);                     // [B] indices as floats

// One rollout step's draw for B rows, ready for one host read: [B * (2 + A)]
// on the device, the values, the log-probabilities, then the actions row by
// row (A = 1, an index, for a categorical policy; log_std is then unused).
// One launch on the device's neural provider when it has one, its draws from
// a counter-based hash of seed; else the ArrayFire path (SampleCategorical /
// SampleGaussian and the log-probability), whose draws differ (TOFIX206 3.2 B).
CYXWIZ_API Tensor PpoSampleStep(const Tensor& head, const Tensor& value, const Tensor& log_std, bool discrete,
                                uint64_t seed);

// Test-only: how many PPO ops (sampling, loss, GAE) ran on a neural provider,
// and how many fell back to ArrayFire after the provider failed, since the
// process started. After a failure that op stays on ArrayFire.
struct CYXWIZ_API PpoProviderCounts {
    long long ran = 0;
    long long failed = 0;
};
CYXWIZ_API PpoProviderCounts PpoProviderCountsForTesting();

// Generalized advantage estimation over a rollout of T steps of N copies
// (row t = step t). dones[t, i] = 1 when copy i's episode ended at step t
// (terminated or truncated; a truncated step's reward should already hold
// gamma x V(its final observation), as Stable-Baselines3 does).
// last_values [N] = V of the observation after the last step.
// Writes advantages and returns (= advantages + values), both [T, N].
CYXWIZ_API void ComputeGae(const Tensor& rewards, const Tensor& values, const Tensor& dones,
                           const Tensor& last_values, float gamma, float gae_lambda, Tensor& advantages,
                           Tensor& returns);

}  // namespace cyxwiz::rl
