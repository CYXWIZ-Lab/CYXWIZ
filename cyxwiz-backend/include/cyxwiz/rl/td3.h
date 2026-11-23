// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

// Twin Delayed DDPG on the device, with hand-written gradients like every
// CyxWiz loss: the pieces of Stable-Baselines3 2.7's TD3.train between the
// networks (TOFIX216 TD3). A trainer runs the networks and, for each
// gradient step, in SB3's order:
//
//   a' = Td3TargetActions(actor_target(next_obs), noise)   (smoothed, no gradient)
//   y = Td3Target(min target Q(next_obs, a'), r, done, gamma)
//   critic loss = mean((Q1(obs, action) - y)^2) + mean((Q2(obs, action) - y)^2)
//   every policy_delay-th step:
//     actor loss = -mean(Q1(obs, actor(obs)))
//     targets (critics and actor) = (1 - tau) * targets + tau * networks
//
// The actor ends in tanh (SB3's squash_output), so its outputs are actions
// in [-1, 1]; scaling to the environment's bounds is the trainer's.
// Exploration adds Gaussian noise to the actor's actions, clipped to [-1, 1]
// (NormalActionNoise). Standard normal draws come from SacNoise, Polyak
// averaging from PolyakAverage (cyxwiz/rl/sac.h).

#include "../api_export.h"
#include "../tensor.h"

#include <cstddef>

namespace cyxwiz::rl {

// Target policy smoothing: clamp(actions + clamp(std * noise, -clip, clip), -1, 1)
// for the target actor's actions [B, A] and standard normal noise [B, A].
CYXWIZ_API Tensor Td3TargetActions(const Tensor& actions, const Tensor& noise, float target_policy_noise,
                                   float target_noise_clip);

// Exploration: clamp(actions + std * noise, -1, 1) for the actor's actions
// [N, A] and standard normal noise [N, A].
CYXWIZ_API Tensor Td3ExploreActions(const Tensor& actions, const Tensor& noise, float noise_std);

// The critics' target [B, 1]: r + (1 - done) * gamma * min(q1', q2').
// next_q1, next_q2, rewards, dones: B values each.
CYXWIZ_API Tensor Td3Target(const Tensor& next_q1, const Tensor& next_q2, const Tensor& rewards, const Tensor& dones,
                            float gamma);

// mean((q1 - y)^2) + mean((q2 - y)^2) (SB3's sum of the twins' MSE, no 0.5)
// and d / d q1, q2 [B, 1].
struct CYXWIZ_API Td3CriticLoss {
    Tensor loss;  // [1] on the device
    Tensor grad_q1;
    Tensor grad_q2;
};
CYXWIZ_API Td3CriticLoss Td3CriticLossAndGradients(const Tensor& q1, const Tensor& q2, const Tensor& target);

// -mean(q1(obs, actor(obs))) and d / d q1 [B, 1] (TD3's actor learns from the
// first Q network only).
struct CYXWIZ_API Td3ActorLoss {
    Tensor loss;     // [1]
    Tensor mean_q;   // [1] mean(q1), for the dashboard
    Tensor grad_q1;  // [B, 1]
};
CYXWIZ_API Td3ActorLoss Td3ActorLossAndGradient(const Tensor& q1_pi);

}  // namespace cyxwiz::rl
