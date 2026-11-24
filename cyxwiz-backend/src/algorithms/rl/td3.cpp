#include "cyxwiz/rl/td3.h"

#include "rl_common.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <arrayfire.h>

// Windows headers define these; ArrayFire's min/max must win.
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz::rl {

namespace {

using detail::Array;
using detail::Make;
using detail::Require;
using detail::Total;

constexpr const char* kLearner = "TD3: ";

af::array Column(const Tensor& t, size_t rows, const char* name) { return detail::Column(t, rows, kLearner, name); }

void RequireActions(const Tensor& actions, const Tensor& noise) {
    Require(actions.Shape().size() == 2, "TD3: the actions must be [B, A]");
    Require(noise.Shape() == actions.Shape(), "TD3: one noise value per action");
}

}  // namespace

Tensor Td3TargetActions(const Tensor& actions, const Tensor& noise, float target_policy_noise,
                        float target_noise_clip) {
    RequireActions(actions, noise);
    Require(std::isfinite(target_policy_noise) && target_policy_noise >= 0.0f &&
                std::isfinite(target_noise_clip) && target_noise_clip >= 0.0f,
            "TD3: the target smoothing noise and its clip are at least 0");
    const double clip = target_noise_clip;
    const af::array smoothing = af::clamp(target_policy_noise * Array(noise), -clip, clip);
    return Make(af::clamp(Array(actions) + smoothing, -1.0, 1.0), actions.Shape());
}

Tensor Td3ExploreActions(const Tensor& actions, const Tensor& noise, float noise_std) {
    RequireActions(actions, noise);
    Require(std::isfinite(noise_std) && noise_std >= 0.0f, "TD3: the noise std is at least 0");
    return Make(af::clamp(Array(actions) + noise_std * Array(noise), -1.0, 1.0), actions.Shape());
}

Tensor Td3Target(const Tensor& next_q1, const Tensor& next_q2, const Tensor& rewards, const Tensor& dones,
                 float gamma) {
    const size_t rows = next_q1.NumElements();
    Require(rows > 0, "TD3: the target needs at least one row");
    const af::array next_q = af::min(Column(next_q1, rows, "next Q1"), Column(next_q2, rows, "next Q2"));
    return detail::BellmanTarget(next_q, rewards, dones, gamma, rows, kLearner);
}

Td3CriticLoss Td3CriticLossAndGradients(const Tensor& q1, const Tensor& q2, const Tensor& target) {
    const size_t rows = q1.NumElements();
    Require(rows > 0, "TD3: the critic loss needs at least one row");
    const float b = static_cast<float>(rows);
    const af::array y = Column(target, rows, "the target");
    const af::array e1 = Column(q1, rows, "Q1") - y;
    const af::array e2 = Column(q2, rows, "Q2") - y;
    Td3CriticLoss out;
    out.loss = Make((Total(e1 * e1) + Total(e2 * e2)) / b, {1});
    out.grad_q1 = Make(af::moddims(2.0f * e1 / b, static_cast<dim_t>(rows), 1), {rows, 1});
    out.grad_q2 = Make(af::moddims(2.0f * e2 / b, static_cast<dim_t>(rows), 1), {rows, 1});
    return out;
}

Td3ActorLoss Td3ActorLossAndGradient(const Tensor& q1_pi) {
    const size_t rows = q1_pi.NumElements();
    Require(rows > 0, "TD3: the actor loss needs at least one row");
    const float b = static_cast<float>(rows);
    const af::array q1 = Column(q1_pi, rows, "Q1");
    Td3ActorLoss out;
    out.mean_q = Make(Total(q1) / b, {1});
    out.loss = Make(-Total(q1) / b, {1});
    out.grad_q1 = Make(af::constant(-1.0f / b, static_cast<dim_t>(rows), 1), {rows, 1});
    return out;
}

}  // namespace cyxwiz::rl
