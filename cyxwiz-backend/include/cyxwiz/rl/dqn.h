// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

// Deep Q-Network on the device, with hand-written gradients like every
// CyxWiz loss: the pieces of Stable-Baselines3 2.7's DQN.train between the
// networks (TOFIX216 DQN). A trainer runs the Q network and its target and,
// for each gradient step, in SB3's order:
//
//   y = DqnTarget(q_target(next_obs), r, done, gamma)   (max over the choices, no gradient)
//   loss = smooth_l1(Q(obs)[action], y)                 (Huber, beta 1, the batch's mean)
//   the gradients' global norm clipped to max_grad_norm, then Adam
//   every target_update_interval environment steps: target = (1 - tau) * target + tau * Q
//
// Acting is epsilon-greedy: with probability DqnExplorationRate a random
// choice, else DqnGreedyActions' argmax.

#include "../api_export.h"
#include "../tensor.h"

#include <cstddef>

namespace cyxwiz::rl {

// The target [B, 1]: r + (1 - done) * gamma * max_a next_q[b, a], for the
// target network's next_q [B, A] and B rewards and dones.
CYXWIZ_API Tensor DqnTarget(const Tensor& next_q, const Tensor& rewards, const Tensor& dones, float gamma);

// smooth_l1(q[b, action_b], y_b) averaged over the batch (Huber with beta 1:
// 0.5 d^2 within 1, |d| - 0.5 past it) and its gradient d / d q [B, A]: the
// clipped difference / B in each row's chosen column, 0 elsewhere. actions:
// B choice indices in [0, A) held as floats (the replay buffer's). They are
// not checked, so a step reads nothing back from the device: an index
// outside [0, A) or not whole selects no column, and its row's Q counts as 0.
struct CYXWIZ_API DqnLoss {
    Tensor loss;    // [1] on the device
    Tensor mean_q;  // [1] mean of the chosen Q values, for the dashboard
    Tensor grad_q;  // [B, A]
};
CYXWIZ_API DqnLoss DqnLossAndGradient(const Tensor& q, const Tensor& actions, const Tensor& target);

// The greedy choice of each row of q [N, A]: N indices (as floats), the
// first one on ties (torch.argmax's).
CYXWIZ_API Tensor DqnGreedyActions(const Tensor& q);

// The epsilon of SB3's LinearSchedule(initial, final, fraction) at
// progress_remaining (1 at the start, 0 at the end): initial falling linearly
// to final over the first `fraction` of the run, final after it (and when
// fraction is 0).
CYXWIZ_API float DqnExplorationRate(float initial_eps, float final_eps, float fraction, double progress_remaining);

}  // namespace cyxwiz::rl
