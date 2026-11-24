// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

// Soft Actor-Critic on the device, with hand-written gradients like every
// CyxWiz loss: the pieces of Stable-Baselines3 2.7's SAC.train between the
// networks (TOFIX216 S2). A trainer runs the networks and, for each gradient
// step, in SB3's order:
//
//   alpha = exp(log_alpha)                      (before this step's update)
//   a, logp = SacSquash(actor(obs), noise)      (reparameterised)
//   alpha loss = -mean(log_alpha * (logp + target_entropy))
//   y = SacTarget(min target Q(next_obs, a'), logp', r, done, alpha, gamma)
//   critic loss = 0.5 * sum over the twins of mean((Q(obs, action) - y)^2)
//   actor loss = mean(alpha * logp - min(Q1, Q2)(obs, a))
//   targets = (1 - tau) * targets + tau * critics
//
// The actor's distribution is SB3's SquashedDiagGaussian: u = mean + std *
// noise with log std clamped to [-20, 2] and taken from the state, a =
// tanh(u), log pi(a) = sum(log N(u; mean, std)) - sum(log(1 - a^2 + 1e-6)).
// Actions are in [-1, 1]; scaling to the environment's bounds is the
// trainer's.

#include "../api_export.h"
#include "../tensor.h"

#include <cstddef>
#include <cstdint>

namespace cyxwiz::rl {

inline constexpr float kSacLogStdMin = -20.0f;
inline constexpr float kSacLogStdMax = 2.0f;
inline constexpr float kSacSquashEpsilon = 1e-6f;

// Standard normal noise [rows, actions]: the same draws for a seed on every device.
CYXWIZ_API Tensor SacNoise(size_t rows, size_t actions, uint64_t seed);
// Uniform actions in [-1, 1) [rows, actions] (the random steps before learning starts).
CYXWIZ_API Tensor SacUniformActions(size_t rows, size_t actions, uint64_t seed);

// The actor's two outputs, mean [B, A] and log std [B, A] (raw, before the
// clamp), with the noise [B, A]: actions tanh(u) [B, A] and log pi [B].
struct CYXWIZ_API SacSquashed {
    Tensor actions;
    Tensor log_prob;
};
CYXWIZ_API SacSquashed SacSquash(const Tensor& mean, const Tensor& log_std, const Tensor& noise);
// The deterministic action, tanh(mean) [B, A] (playback).
CYXWIZ_API Tensor SacDeterministicAction(const Tensor& mean);

// Gradients through SacSquash to the actor's outputs from d loss / d actions
// [B, A] (or empty: none) and d loss / d log pi [B] (or empty). The log std's
// is zero where the clamp held it.
struct CYXWIZ_API SacSquashGradients {
    Tensor grad_mean;     // [B, A]
    Tensor grad_log_std;  // [B, A]
};
CYXWIZ_API SacSquashGradients SacSquashBackward(const Tensor& mean, const Tensor& log_std, const Tensor& noise,
                                                const Tensor& grad_actions, const Tensor& grad_log_prob);

// The critics' target [B, 1]: r + (1 - done) * gamma * (min(q1', q2') - alpha * log pi').
// next_q1, next_q2, rewards, dones: B values each; next_log_prob [B];
// ent_coef: alpha, one value on the device.
CYXWIZ_API Tensor SacTarget(const Tensor& next_q1, const Tensor& next_q2, const Tensor& next_log_prob,
                            const Tensor& rewards, const Tensor& dones, const Tensor& ent_coef, float gamma);

// 0.5 * (mean((q1 - y)^2) + mean((q2 - y)^2)) and d / d q1, q2 [B, 1].
struct CYXWIZ_API SacCriticLoss {
    Tensor loss;  // [1] on the device
    Tensor grad_q1;
    Tensor grad_q2;
};
CYXWIZ_API SacCriticLoss SacCriticLossAndGradients(const Tensor& q1, const Tensor& q2, const Tensor& target);

// mean(alpha * log pi - min(q1, q2)) and its gradients. The minimum's
// gradient goes to q1 where q1 <= q2, else to q2 (torch.min over dim 1 picks
// the first on a tie).
struct CYXWIZ_API SacActorLoss {
    Tensor loss;           // [1]
    Tensor mean_min_q;     // [1] mean(min(q1, q2)), for the dashboard
    Tensor grad_q1;        // [B, 1]
    Tensor grad_q2;        // [B, 1]
    Tensor grad_log_prob;  // [B]
};
CYXWIZ_API SacActorLoss SacActorLossAndGradients(const Tensor& q1_pi, const Tensor& q2_pi, const Tensor& log_prob,
                                                 const Tensor& ent_coef);

// -mean(log_alpha * (log pi + target_entropy)) with log pi held fixed, and d / d log_alpha.
struct CYXWIZ_API SacEntCoefLoss {
    Tensor loss;                // [1]
    Tensor grad_log_ent_coef;   // [1]
};
CYXWIZ_API SacEntCoefLoss SacEntCoefLossAndGradient(const Tensor& log_ent_coef, const Tensor& log_prob,
                                                    float target_entropy);

// One target parameter moved toward its critic parameter: (1 - tau) * target + tau * param.
CYXWIZ_API Tensor PolyakAverage(const Tensor& param, const Tensor& target, float tau);

}  // namespace cyxwiz::rl
