#include "cyxwiz/rl/sac.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <arrayfire.h>

#include "../device_random.h"

// Windows headers define these; ArrayFire's min/max must win.
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz::rl {

namespace {

constexpr double kLogSqrtTwoPi = 0.91893853320467274178;  // log(sqrt(2 pi))

af::array Array(const Tensor& t) { return t.GetSemanticArray(); }

Tensor Make(const af::array& a, std::vector<size_t> shape) {
    af::array evaluated = a;
    evaluated.eval();
    return Tensor::FromSemanticArray(evaluated, std::move(shape));
}

void Require(bool ok, const std::string& what) {
    if (!ok) throw std::invalid_argument(what);
}

// B values ([B] or [B, 1]) as an ArrayFire column of B rows
af::array Column(const Tensor& t, size_t rows, const char* name) {
    Require(t.NumElements() == rows,
            std::string("SAC: ") + name + " needs " + std::to_string(rows) + " values, got " +
                std::to_string(t.NumElements()));
    return af::moddims(Array(t), static_cast<dim_t>(rows));
}

// One value on the device as a [1] array
af::array Scalar(const Tensor& t, const char* name) {
    Require(t.NumElements() == 1, std::string("SAC: ") + name + " is one value");
    return af::moddims(Array(t), 1);
}

// The actor's outputs, checked: [B, A] each
void RequirePair(const Tensor& mean, const Tensor& log_std, const Tensor& noise) {
    Require(mean.Shape().size() == 2, "SAC: the means must be [B, A]");
    Require(log_std.Shape() == mean.Shape(), "SAC: one log standard deviation per mean");
    Require(noise.Shape() == mean.Shape(), "SAC: one noise value per mean");
}

// What SacSquash and its backward share
struct Squash {
    af::array log_std;  // clamped [B, A]
    af::array std_dev;
    af::array u;        // mean + std * noise
    af::array a;        // tanh(u)
};

Squash Forward(const Tensor& mean, const Tensor& log_std, const Tensor& noise) {
    Squash s;
    s.log_std = af::clamp(Array(log_std), static_cast<double>(kSacLogStdMin), static_cast<double>(kSacLogStdMax));
    s.std_dev = af::exp(s.log_std);
    s.u = Array(mean) + s.std_dev * Array(noise);
    s.a = af::tanh(s.u);
    return s;
}

af::array Total(const af::array& a) { return af::sum(af::flat(a)); }

// An optional input given (a default Tensor has no shape)
bool Given(const Tensor& t) { return !t.Shape().empty() && t.NumElements() > 0; }

}  // namespace

Tensor SacNoise(size_t rows, size_t actions, uint64_t seed) {
    af::randomEngine engine = cyxwiz::DeviceRandomEngine(seed);
    return Make(cyxwiz::DeviceNormal(af::dim4(static_cast<dim_t>(rows), static_cast<dim_t>(actions)), f32, engine),
                {rows, actions});
}

Tensor SacUniformActions(size_t rows, size_t actions, uint64_t seed) {
    af::randomEngine engine = cyxwiz::DeviceRandomEngine(seed);
    const af::array u = af::randu(af::dim4(static_cast<dim_t>(rows), static_cast<dim_t>(actions)), f32, engine);
    return Make(2.0f * u - 1.0f, {rows, actions});
}

SacSquashed SacSquash(const Tensor& mean, const Tensor& log_std, const Tensor& noise) {
    RequirePair(mean, log_std, noise);
    const size_t rows = mean.Shape()[0];
    const Squash s = Forward(mean, log_std, noise);
    // N(u; mean, std) as torch computes it, from u - mean (at a tiny std that
    // rounds to 0 in float32, as it does in SB3)
    const af::array diff = s.u - Array(mean);
    const af::array gaussian =
        -(diff * diff) / (2.0f * s.std_dev * s.std_dev) - af::log(s.std_dev) - static_cast<float>(kLogSqrtTwoPi);
    const af::array correction = af::log(1.0f - s.a * s.a + kSacSquashEpsilon);
    SacSquashed out;
    out.actions = Make(s.a, mean.Shape());
    out.log_prob = Make(af::moddims(af::sum(gaussian - correction, 1), static_cast<dim_t>(rows)), {rows});
    return out;
}

Tensor SacDeterministicAction(const Tensor& mean) {
    Require(mean.Shape().size() == 2, "SAC: the means must be [B, A]");
    return Make(af::tanh(Array(mean)), mean.Shape());
}

SacSquashGradients SacSquashBackward(const Tensor& mean, const Tensor& log_std, const Tensor& noise,
                                     const Tensor& grad_actions, const Tensor& grad_log_prob) {
    RequirePair(mean, log_std, noise);
    const size_t rows = mean.Shape()[0];
    const unsigned act = static_cast<unsigned>(mean.Shape()[1]);
    const Squash s = Forward(mean, log_std, noise);
    const af::array one_minus = 1.0f - s.a * s.a;
    // d u: from the actions (d tanh = 1 - a^2) and from log pi (the squash
    // correction's d/du = 2 a (1 - a^2) / (1 - a^2 + eps); the Gaussian term
    // does not depend on u once u - mean = std * noise)
    af::array grad_u = af::constant(0.0f, s.u.dims());
    af::array grad_lp_rows = af::constant(0.0f, s.u.dims());
    if (Given(grad_actions)) {
        Require(grad_actions.Shape() == mean.Shape(), "SAC: d loss / d actions must be [B, A]");
        grad_u += Array(grad_actions) * one_minus;
    }
    if (Given(grad_log_prob)) {
        grad_lp_rows = af::tile(Column(grad_log_prob, rows, "d loss / d log pi"), 1, act);
        grad_u += grad_lp_rows * 2.0f * s.a * one_minus / (one_minus + kSacSquashEpsilon);
    }
    // u = mean + exp(log std) * noise; log pi has -log std directly
    af::array grad_ls = grad_u * s.std_dev * Array(noise) - grad_lp_rows;
    // The clamp passes the gradient inside [-20, 2], ends included (torch.clamp)
    const af::array raw = Array(log_std);
    grad_ls *= ((raw >= kSacLogStdMin) && (raw <= kSacLogStdMax)).as(f32);
    SacSquashGradients out;
    out.grad_mean = Make(grad_u, mean.Shape());
    out.grad_log_std = Make(grad_ls, mean.Shape());
    return out;
}

Tensor SacTarget(const Tensor& next_q1, const Tensor& next_q2, const Tensor& next_log_prob, const Tensor& rewards,
                 const Tensor& dones, const Tensor& ent_coef, float gamma) {
    const size_t rows = next_q1.NumElements();
    Require(rows > 0, "SAC: the target needs at least one row");
    const af::array alpha = af::tile(Scalar(ent_coef, "the entropy weight"), static_cast<unsigned>(rows));
    const af::array next_q = af::min(Column(next_q1, rows, "next Q1"), Column(next_q2, rows, "next Q2")) -
                             alpha * Column(next_log_prob, rows, "next log pi");
    const af::array y = Column(rewards, rows, "rewards") +
                        (1.0f - Column(dones, rows, "dones")) * gamma * next_q;
    return Make(af::moddims(y, static_cast<dim_t>(rows), 1), {rows, 1});
}

SacCriticLoss SacCriticLossAndGradients(const Tensor& q1, const Tensor& q2, const Tensor& target) {
    const size_t rows = q1.NumElements();
    Require(rows > 0, "SAC: the critic loss needs at least one row");
    const float b = static_cast<float>(rows);
    const af::array y = Column(target, rows, "the target");
    const af::array e1 = Column(q1, rows, "Q1") - y;
    const af::array e2 = Column(q2, rows, "Q2") - y;
    SacCriticLoss out;
    out.loss = Make(0.5f * (Total(e1 * e1) + Total(e2 * e2)) / b, {1});
    out.grad_q1 = Make(af::moddims(e1 / b, static_cast<dim_t>(rows), 1), {rows, 1});
    out.grad_q2 = Make(af::moddims(e2 / b, static_cast<dim_t>(rows), 1), {rows, 1});
    return out;
}

SacActorLoss SacActorLossAndGradients(const Tensor& q1_pi, const Tensor& q2_pi, const Tensor& log_prob,
                                      const Tensor& ent_coef) {
    const size_t rows = q1_pi.NumElements();
    Require(rows > 0, "SAC: the actor loss needs at least one row");
    const float b = static_cast<float>(rows);
    const af::array q1 = Column(q1_pi, rows, "Q1");
    const af::array q2 = Column(q2_pi, rows, "Q2");
    const af::array lp = Column(log_prob, rows, "log pi");
    const af::array alpha = Scalar(ent_coef, "the entropy weight");
    const af::array first = (q1 <= q2).as(f32);  // the minimum's gradient, the first on a tie
    const af::array min_q = af::min(q1, q2);
    SacActorLoss out;
    out.loss = Make((alpha * Total(lp) - Total(min_q)) / b, {1});
    out.mean_min_q = Make(Total(min_q) / b, {1});
    out.grad_q1 = Make(af::moddims(-first / b, static_cast<dim_t>(rows), 1), {rows, 1});
    out.grad_q2 = Make(af::moddims(-(1.0f - first) / b, static_cast<dim_t>(rows), 1), {rows, 1});
    out.grad_log_prob = Make(af::tile(alpha, static_cast<unsigned>(rows)) / b, {rows});
    return out;
}

SacEntCoefLoss SacEntCoefLossAndGradient(const Tensor& log_ent_coef, const Tensor& log_prob, float target_entropy) {
    const size_t rows = log_prob.NumElements();
    Require(rows > 0, "SAC: the entropy weight's loss needs at least one row");
    const af::array mean_term = Total(Column(log_prob, rows, "log pi") + target_entropy) / static_cast<float>(rows);
    const af::array log_alpha = Scalar(log_ent_coef, "the log entropy weight");
    SacEntCoefLoss out;
    out.loss = Make(-log_alpha * mean_term, {1});
    out.grad_log_ent_coef = Make(-mean_term, {1});
    return out;
}

Tensor PolyakAverage(const Tensor& param, const Tensor& target, float tau) {
    Require(param.Shape() == target.Shape(), "SAC: a target parameter has its critic parameter's shape");
    Require(std::isfinite(tau) && tau >= 0.0f && tau <= 1.0f, "SAC: tau is in [0, 1]");
    return Make((1.0f - tau) * Array(target) + tau * Array(param), target.Shape());
}

}  // namespace cyxwiz::rl
