#include "cyxwiz/rl/ppo.h"

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

constexpr double kLogSqrtTwoPi = 0.91893853320467274178;  // log(sqrt(2 pi))

af::array Array(const Tensor& t) { return t.GetSemanticArray(); }

Tensor Make(const af::array& a, std::vector<size_t> shape) {
    af::array evaluated = a;
    evaluated.eval();
    return Tensor::FromSemanticArray(evaluated, std::move(shape));
}

size_t Rows(const Tensor& t) { return t.Shape().empty() ? 0 : t.Shape()[0]; }

void Require(bool ok, const std::string& what) {
    if (!ok) throw std::invalid_argument(what);
}

// A [B] or [B, 1] tensor as an ArrayFire column of B rows.
af::array Column(const Tensor& t, size_t rows, const char* name) {
    Require(t.NumElements() == rows, std::string("PPO: ") + name + " needs " + std::to_string(rows) +
                                         " values, got " + std::to_string(t.NumElements()));
    return af::moddims(Array(t), static_cast<dim_t>(rows));
}

float Scalar(const af::array& a) { return af::sum<float>(a); }

af::array LogSoftmaxRows(const af::array& logits) {
    const af::array shifted = logits - af::tile(af::max(logits, 1), 1, static_cast<unsigned>(logits.dims(1)));
    const af::array log_norm = af::log(af::sum(af::exp(shifted), 1));
    return shifted - af::tile(log_norm, 1, static_cast<unsigned>(logits.dims(1)));
}

af::array OneHot(const af::array& indices, dim_t rows, dim_t classes) {
    const af::array cols = af::tile(af::range(af::dim4(1, classes), 1, f32), static_cast<unsigned>(rows), 1);
    return (cols == af::tile(indices, 1, static_cast<unsigned>(classes))).as(f32);
}

// The shared part: advantages, ratio, the clipped surrogate and the value
// loss with their gradients. dloss_dlogp is d(policy_loss)/d(log_prob).
struct Common {
    af::array dloss_dlogp;
    af::array grad_value;
    float policy_loss = 0.0f;
    float value_loss = 0.0f;
    float approx_kl = 0.0f;
    float clip_fraction = 0.0f;
};

Common SurrogateAndValue(const af::array& log_prob, const af::array& value, const PpoBatch& batch,
                         const PpoLossSettings& settings, size_t rows) {
    const float b = static_cast<float>(rows);
    af::array adv = Column(batch.advantages, rows, "advantages");
    if (settings.normalize_advantage && rows > 1) {
        const af::array mean = af::mean(adv);
        const af::array centered = adv - af::tile(mean, static_cast<unsigned>(rows));
        const af::array std_unbiased = af::sqrt(af::sum(centered * centered) / (b - 1.0f));
        adv = centered / af::tile(std_unbiased + 1e-8f, static_cast<unsigned>(rows));
    }
    const af::array old_log_prob = Column(batch.old_log_prob, rows, "old_log_prob");
    const af::array log_ratio = log_prob - old_log_prob;
    const af::array ratio = af::exp(log_ratio);
    const float c = settings.clip_range;
    const af::array clipped = af::clamp(ratio, 1.0 - c, 1.0 + c);
    const af::array s1 = adv * ratio;
    const af::array s2 = adv * clipped;
    Common out;
    out.policy_loss = -Scalar(af::min(s1, s2)) / b;
    // torch.minimum sends a tie's gradient half to each side; clamp passes
    // it inside [1 - c, 1 + c], ends included
    const af::array tie = (s1 == s2).as(f32);
    const af::array w1 = (s1 < s2).as(f32) + 0.5f * tie;
    const af::array w2 = (s2 < s1).as(f32) + 0.5f * tie;
    const af::array inside = ((ratio >= 1.0 - c) && (ratio <= 1.0 + c)).as(f32);
    out.dloss_dlogp = -(adv * (w1 + w2 * inside) * ratio) / b;
    out.approx_kl = Scalar((ratio - 1.0f) - log_ratio) / b;
    out.clip_fraction = Scalar((af::abs(ratio - 1.0f) > c).as(f32)) / b;

    const af::array returns = Column(batch.returns, rows, "returns");
    af::array predicted = value;
    af::array value_pass = af::constant(1.0f, static_cast<dim_t>(rows));
    if (settings.clip_range_vf > 0.0f) {
        const af::array old_values = Column(batch.old_values, rows, "old_values");
        const af::array delta = value - old_values;
        predicted = old_values + af::clamp(delta, -settings.clip_range_vf, settings.clip_range_vf);
        value_pass = ((delta >= -settings.clip_range_vf) && (delta <= settings.clip_range_vf)).as(f32);
    }
    const af::array error = predicted - returns;
    out.value_loss = Scalar(error * error) / b;
    out.grad_value = settings.vf_coef * 2.0f * error * value_pass / b;
    return out;
}

void CheckSettings(const PpoLossSettings& s) {
    Require(std::isfinite(s.clip_range) && s.clip_range > 0.0f, "PPO: the clip range must be above 0");
    Require(std::isfinite(s.clip_range_vf) && s.clip_range_vf >= 0.0f, "PPO: the value clip range must be 0 or more");
    Require(std::isfinite(s.ent_coef) && std::isfinite(s.vf_coef), "PPO: the coefficients must be numbers");
}

}  // namespace

Tensor GaussianLogProb(const Tensor& mean, const Tensor& log_std, const Tensor& actions) {
    Require(mean.Shape().size() == 2 && actions.Shape() == mean.Shape(), "PPO: actions must match the means [B, A]");
    const dim_t rows = static_cast<dim_t>(mean.Shape()[0]);
    const dim_t act = static_cast<dim_t>(mean.Shape()[1]);
    Require(log_std.NumElements() == static_cast<size_t>(act), "PPO: one log standard deviation per action");
    const af::array ls = af::tile(af::moddims(Array(log_std), 1, act), static_cast<unsigned>(rows), 1);
    const af::array z = (Array(actions) - Array(mean)) / af::exp(ls);
    const af::array lp = af::sum(-0.5f * z * z - ls - static_cast<float>(kLogSqrtTwoPi), 1);
    return Make(af::moddims(lp, rows), {static_cast<size_t>(rows)});
}

Tensor CategoricalLogProb(const Tensor& logits, const Tensor& actions) {
    Require(logits.Shape().size() == 2, "PPO: logits must be [B, K]");
    const dim_t rows = static_cast<dim_t>(logits.Shape()[0]);
    const dim_t classes = static_cast<dim_t>(logits.Shape()[1]);
    const af::array index = Column(actions, static_cast<size_t>(rows), "actions");
    const af::array lp = af::sum(LogSoftmaxRows(Array(logits)) * OneHot(index, rows, classes), 1);
    return Make(af::moddims(lp, rows), {static_cast<size_t>(rows)});
}

PpoLossResult PpoGaussianLoss(const Tensor& mean, const Tensor& log_std, const Tensor& value, const PpoBatch& batch,
                              const PpoLossSettings& settings) {
    CheckSettings(settings);
    Require(mean.Shape().size() == 2, "PPO: means must be [B, A]");
    const size_t rows = Rows(mean);
    const dim_t act = static_cast<dim_t>(mean.Shape()[1]);
    Require(batch.actions.Shape() == mean.Shape(), "PPO: actions must match the means [B, A]");
    Require(log_std.NumElements() == static_cast<size_t>(act), "PPO: one log standard deviation per action");
    const unsigned r = static_cast<unsigned>(rows);
    const af::array ls_row = af::moddims(Array(log_std), 1, act);
    const af::array ls = af::tile(ls_row, r, 1);
    const af::array inv_var = af::exp(-2.0f * ls);
    const af::array diff = Array(batch.actions) - Array(mean);
    const af::array lp = af::moddims(af::sum(-0.5f * diff * diff * inv_var - ls - static_cast<float>(kLogSqrtTwoPi), 1),
                                     static_cast<dim_t>(rows));
    const af::array v = Column(value, rows, "values");
    const Common common = SurrogateAndValue(lp, v, batch, settings, rows);

    // Entropy per row = sum over actions of 0.5 + log(sqrt(2 pi)) + log_std
    const float entropy = Scalar(0.5f + static_cast<float>(kLogSqrtTwoPi) + ls_row);
    PpoLossResult out;
    out.policy_loss = common.policy_loss;
    out.value_loss = common.value_loss;
    out.entropy_loss = -entropy;
    out.loss = out.policy_loss + settings.ent_coef * out.entropy_loss + settings.vf_coef * out.value_loss;
    out.approx_kl = common.approx_kl;
    out.clip_fraction = common.clip_fraction;
    out.log_prob = Make(lp, {rows});
    const af::array d = af::tile(common.dloss_dlogp, 1, static_cast<unsigned>(act));
    out.grad_policy_output = Make(d * diff * inv_var, {rows, static_cast<size_t>(act)});
    const af::array grad_ls = af::sum(d * (diff * diff * inv_var - 1.0f), 0) - settings.ent_coef;
    out.grad_log_std = Make(af::moddims(grad_ls, act), {static_cast<size_t>(act)});
    out.grad_value_output = Make(af::moddims(common.grad_value, static_cast<dim_t>(rows), 1), {rows, 1});
    return out;
}

PpoLossResult PpoCategoricalLoss(const Tensor& logits, const Tensor& value, const PpoBatch& batch,
                                 const PpoLossSettings& settings) {
    CheckSettings(settings);
    Require(logits.Shape().size() == 2, "PPO: logits must be [B, K]");
    const size_t rows = Rows(logits);
    const dim_t classes = static_cast<dim_t>(logits.Shape()[1]);
    const unsigned k = static_cast<unsigned>(classes);
    const af::array log_p = LogSoftmaxRows(Array(logits));
    const af::array p = af::exp(log_p);
    const af::array index = Column(batch.actions, rows, "actions");
    const af::array one_hot = OneHot(index, static_cast<dim_t>(rows), classes);
    const af::array lp = af::moddims(af::sum(log_p * one_hot, 1), static_cast<dim_t>(rows));
    const af::array v = Column(value, rows, "values");
    const Common common = SurrogateAndValue(lp, v, batch, settings, rows);

    const af::array entropy = -af::sum(p * log_p, 1);  // [B, 1]
    const float b = static_cast<float>(rows);
    PpoLossResult out;
    out.policy_loss = common.policy_loss;
    out.value_loss = common.value_loss;
    out.entropy_loss = -Scalar(entropy) / b;
    out.loss = out.policy_loss + settings.ent_coef * out.entropy_loss + settings.vf_coef * out.value_loss;
    out.approx_kl = common.approx_kl;
    out.clip_fraction = common.clip_fraction;
    out.log_prob = Make(lp, {rows});
    // d log_prob / d logits = one_hot - p; d entropy / d logits = -p (log p + H)
    const af::array d_entropy = -p * (log_p + af::tile(entropy, 1, k));
    const af::array grad = af::tile(common.dloss_dlogp, 1, k) * (one_hot - p) - settings.ent_coef * d_entropy / b;
    out.grad_policy_output = Make(grad, {rows, static_cast<size_t>(classes)});
    out.grad_value_output = Make(af::moddims(common.grad_value, static_cast<dim_t>(rows), 1), {rows, 1});
    return out;
}

Tensor SampleGaussian(const Tensor& mean, const Tensor& log_std, uint64_t seed) {
    Require(mean.Shape().size() == 2, "PPO: means must be [B, A]");
    const dim_t rows = static_cast<dim_t>(mean.Shape()[0]);
    const dim_t act = static_cast<dim_t>(mean.Shape()[1]);
    Require(log_std.NumElements() == static_cast<size_t>(act), "PPO: one log standard deviation per action");
    af::randomEngine engine(AF_RANDOM_ENGINE_DEFAULT, seed);
    const af::array noise = af::randn(af::dim4(rows, act), f32, engine);
    const af::array std_dev = af::tile(af::exp(af::moddims(Array(log_std), 1, act)), static_cast<unsigned>(rows), 1);
    return Make(Array(mean) + noise * std_dev, {static_cast<size_t>(rows), static_cast<size_t>(act)});
}

Tensor SampleCategorical(const Tensor& logits, uint64_t seed) {
    Require(logits.Shape().size() == 2, "PPO: logits must be [B, K]");
    const dim_t rows = static_cast<dim_t>(logits.Shape()[0]);
    const dim_t classes = static_cast<dim_t>(logits.Shape()[1]);
    af::randomEngine engine(AF_RANDOM_ENGINE_DEFAULT, seed);
    // Gumbel-max: argmax(log p - log(-log u)) is a draw from p
    const af::array u = af::clamp(af::randu(af::dim4(rows, classes), f32, engine), 1e-20, 1.0 - 1e-7);
    af::array best_value, best_index;
    af::max(best_value, best_index, LogSoftmaxRows(Array(logits)) - af::log(-af::log(u)), 1);
    return Make(af::moddims(best_index.as(f32), rows), {static_cast<size_t>(rows)});
}

void ComputeGae(const Tensor& rewards, const Tensor& values, const Tensor& dones, const Tensor& last_values,
                float gamma, float gae_lambda, Tensor& advantages, Tensor& returns) {
    Require(rewards.Shape().size() == 2 && values.Shape() == rewards.Shape() && dones.Shape() == rewards.Shape(),
            "GAE: rewards, values and dones must all be [T, N]");
    const dim_t steps = static_cast<dim_t>(rewards.Shape()[0]);
    const dim_t copies = static_cast<dim_t>(rewards.Shape()[1]);
    Require(last_values.NumElements() == static_cast<size_t>(copies), "GAE: one last value per copy");
    const af::array r = Array(rewards);
    const af::array v = Array(values);
    const af::array not_done = 1.0f - Array(dones);
    af::array next_value = af::moddims(Array(last_values), 1, copies);
    af::array gae = af::constant(0.0f, 1, copies);
    af::array adv = af::constant(0.0f, steps, copies);
    for (int t = static_cast<int>(steps) - 1; t >= 0; --t) {
        const af::array delta = r.row(t) + gamma * next_value * not_done.row(t) - v.row(t);
        gae = delta + gamma * gae_lambda * not_done.row(t) * gae;
        gae.eval();
        adv.row(t) = gae;
        next_value = v.row(t);
    }
    const std::vector<size_t> shape = {static_cast<size_t>(steps), static_cast<size_t>(copies)};
    advantages = Make(adv, shape);
    returns = Make(adv + v, shape);
}

}  // namespace cyxwiz::rl
