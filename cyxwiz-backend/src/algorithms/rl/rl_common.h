#pragma once

// Helpers the RL learners (PPO, SAC, TD3, DQN) share. Internal to the backend:
// not part of the SDK headers. `learner` is the error prefix ("SAC: ") so each
// learner keeps its own messages.

#include "cyxwiz/tensor.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <arrayfire.h>

namespace cyxwiz::rl::detail {

inline constexpr double kLogSqrtTwoPi = 0.91893853320467274178;  // log(sqrt(2 pi))

inline af::array Array(const Tensor& t) { return t.GetSemanticArray(); }

inline Tensor Make(const af::array& a, std::vector<size_t> shape) {
    af::array evaluated = a;
    evaluated.eval();
    return Tensor::FromSemanticArray(evaluated, std::move(shape));
}

inline void Require(bool ok, const std::string& what) {
    if (!ok) throw std::invalid_argument(what);
}

// A [B] or [B, 1] tensor holds B values
inline void RequireRows(const Tensor& t, size_t rows, const char* learner, const char* name) {
    Require(t.NumElements() == rows, std::string(learner) + name + " needs " + std::to_string(rows) +
                                         " values, got " + std::to_string(t.NumElements()));
}

// B values ([B] or [B, 1]) as an ArrayFire column of B rows
inline af::array Column(const Tensor& t, size_t rows, const char* learner, const char* name) {
    RequireRows(t, rows, learner, name);
    return af::moddims(Array(t), static_cast<dim_t>(rows));
}

// A sum as a [1] array on the device (no host read)
inline af::array Total(const af::array& a) { return af::sum(af::flat(a)); }

// The one-step target r + (1 - done) * gamma * next_value as a [B, 1] tensor.
// The learners differ only in next_value (DQN's best Q, TD3's twin minimum,
// SAC's twin minimum less the entropy term).
inline Tensor BellmanTarget(const af::array& next_value, const Tensor& rewards, const Tensor& dones, float gamma,
                            size_t rows, const char* learner) {
    const af::array y = Column(rewards, rows, learner, "rewards") +
                        (1.0f - Column(dones, rows, learner, "dones")) * gamma * next_value;
    return Make(af::moddims(y, static_cast<dim_t>(rows), 1), {rows, 1});
}

}  // namespace cyxwiz::rl::detail
