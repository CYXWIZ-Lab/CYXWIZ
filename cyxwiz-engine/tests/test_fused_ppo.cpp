// PPO's provider kernels (TOFIX206 3.2 B) against the ArrayFire path: the
// loss (categorical and Gaussian, with and without a value clip, advantages
// normalised or not) gives the same statistics, log-probabilities and
// gradients; GAE the same advantages and returns; the packed rollout step
// the same values and log-probabilities of its own draws. The draws
// themselves (a counter-based hash, D10) follow the policy: categorical
// frequencies match the softmax, Gaussian draws the mean and std, and the
// same seed draws the same. The PyTorch oracle (test_computation_truth_ppo)
// checks both paths against PyTorch.
//
// Runs on the device CYXWIZ_TEST_ARRAYFIRE_BACKEND names; without a provider
// for it (the CPU) there is nothing to compare and it says so.

#include <cyxwiz/neural_provider.h>
#include <cyxwiz/rl/ppo.h>
#include <cyxwiz/tensor.h>
#include "computation_truth/test_device_selection.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

cyxwiz::Tensor Random(const std::vector<size_t>& shape, unsigned seed, float scale, float offset = 0.0f) {
    size_t n = 1;
    for (size_t d : shape) n *= d;
    std::vector<float> v(n);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(offset, scale);
    for (float& f : v) f = dist(rng);
    return cyxwiz::Tensor(shape, v.data(), cyxwiz::DataType::Float32);
}

cyxwiz::Tensor Indices(size_t rows, size_t classes, unsigned seed) {
    std::vector<float> v(rows);
    std::mt19937 rng(seed);
    for (float& f : v) f = static_cast<float>(rng() % classes);
    return cyxwiz::Tensor({rows}, v.data(), cyxwiz::DataType::Float32);
}

std::vector<float> Host(const cyxwiz::Tensor& t) {
    const float* d = t.ReadData<float>();
    return std::vector<float>(d, d + t.NumElements());
}

double Difference(const cyxwiz::Tensor& got, const cyxwiz::Tensor& want) {
    if (got.NumElements() != want.NumElements()) return 1e30;
    const auto a = Host(got), b = Host(want);
    double d = 0.0, scale = 1e-6;
    for (size_t i = 0; i < a.size(); ++i) {
        d = std::max(d, static_cast<double>(std::fabs(a[i] - b[i])));
        scale = std::max(scale, static_cast<double>(std::fabs(b[i])));
    }
    return d / scale;
}

bool ProviderServes(cyxwiz::NeuralOp op, bool discrete) {
    cyxwiz::NeuralOpRequest r;
    r.op = op;
    r.target = cyxwiz::CaptureCurrentNeuralDeviceTarget();
    r.device_resident = true;
    r.batch = 64;
    r.ppo_width = 4;
    r.ppo_steps = 16;
    r.ppo_discrete = discrete;
    return cyxwiz::NeuralProviderRegistry::Instance().FindSupporting(r) != nullptr;
}

template <typename F>
auto OnArrayFire(F f) {
    cyxwiz::SetNeuralProvidersDisabledForTesting(true);
    auto out = f();
    cyxwiz::SetNeuralProvidersDisabledForTesting(false);
    return out;
}

void CompareLoss(bool discrete, float clip_vf, bool normalize, size_t rows) {
    const std::string what = std::string(discrete ? "categorical" : "Gaussian") + (clip_vf > 0.0f ? ", value clip" : "") +
                             (normalize ? "" : ", raw advantages") + " x" + std::to_string(rows);
    const size_t width = discrete ? 3 : 6;
    const auto head = Random({rows, width}, 1, 1.0f);
    const auto log_std = Random({width}, 2, 0.3f, -0.5f);
    const auto value = Random({rows, 1}, 3, 1.0f);
    cyxwiz::rl::PpoBatch batch;
    batch.actions = discrete ? Indices(rows, width, 4) : Random({rows, width}, 4, 1.0f);
    // old log-probabilities near the current ones, so some ratios clip and some do not
    const auto current = discrete ? cyxwiz::rl::CategoricalLogProb(head, batch.actions)
                                  : cyxwiz::rl::GaussianLogProb(head, log_std, batch.actions);
    auto noise = Host(Random({rows}, 5, 0.3f));
    const auto lp = Host(current);
    for (size_t i = 0; i < rows; ++i) noise[i] += lp[i];
    batch.old_log_prob = cyxwiz::Tensor({rows}, noise.data(), cyxwiz::DataType::Float32);
    batch.advantages = Random({rows}, 6, 2.0f, 0.3f);
    batch.returns = Random({rows}, 7, 1.0f);
    batch.old_values = Random({rows}, 8, 1.0f);
    cyxwiz::rl::PpoLossSettings s;
    s.clip_range = 0.2f;
    s.clip_range_vf = clip_vf;
    s.ent_coef = 0.01f;
    s.vf_coef = 0.5f;
    s.normalize_advantage = normalize;
    const auto run = [&]() {
        return discrete ? cyxwiz::rl::PpoCategoricalLoss(head, value, batch, s)
                        : cyxwiz::rl::PpoGaussianLoss(head, log_std, value, batch, s);
    };
    const auto device = run();
    const auto reference = OnArrayFire(run);
    const std::pair<const char*, double> parts[] = {
        {"statistics", Difference(device.statistics, reference.statistics)},
        {"log_prob", Difference(device.log_prob, reference.log_prob)},
        {"grad_policy_output", Difference(device.grad_policy_output, reference.grad_policy_output)},
        {"grad_value_output", Difference(device.grad_value_output, reference.grad_value_output)},
        {"grad_log_std", discrete ? 0.0 : Difference(device.grad_log_std, reference.grad_log_std)}};
    double d = 0.0;
    for (const auto& [name, diff] : parts) {
        // grad_log_std sums the rows' mixed-sign terms: the two paths' summation
        // orders differ by up to a few 1e-5 of it (both pass the PyTorch oracle)
        const double bound = std::string(name) == "grad_log_std" ? 5e-5 : 1e-5;
        Check(diff < bound, what + ": " + name + " differs by " + std::to_string(diff));
        d = std::max(d, diff);
    }
    Check(device.grad_policy_output.Shape() == reference.grad_policy_output.Shape() &&
              device.grad_value_output.Shape() == reference.grad_value_output.Shape(),
          what + ": the same shapes");
    std::printf("loss %s: within %.2g\n", what.c_str(), d);
}

void CompareGae(size_t steps, size_t copies) {
    const auto rewards = Random({steps, copies}, 11, 1.0f);
    const auto values = Random({steps, copies}, 12, 1.0f);
    std::vector<float> d(steps * copies);
    std::mt19937 rng(13);
    for (float& f : d) f = (rng() % 7 == 0) ? 1.0f : 0.0f;
    const cyxwiz::Tensor dones({steps, copies}, d.data(), cyxwiz::DataType::Float32);
    const auto last = Random({copies}, 14, 1.0f);
    cyxwiz::Tensor adv, ret, ref_adv, ref_ret;
    cyxwiz::rl::ComputeGae(rewards, values, dones, last, 0.99f, 0.95f, adv, ret);
    OnArrayFire([&]() {
        cyxwiz::rl::ComputeGae(rewards, values, dones, last, 0.99f, 0.95f, ref_adv, ref_ret);
        return 0;
    });
    const double diff = std::max(Difference(adv, ref_adv), Difference(ret, ref_ret));
    Check(diff < 1e-5, "GAE " + std::to_string(steps) + "x" + std::to_string(copies) + " differs by " +
                           std::to_string(diff));
    std::printf("GAE %zux%zu: within %.2g\n", steps, copies, diff);
}

void CheckSampleStep() {
    // Categorical: the packed values and log-probabilities of its own draws, and frequencies
    const size_t rows = 20000, classes = 3;
    std::vector<float> logits_host(rows * classes);
    const float row_logits[3] = {0.5f, -1.0f, 1.2f};
    for (size_t r = 0; r < rows; ++r)
        for (size_t k = 0; k < classes; ++k) logits_host[r * classes + k] = row_logits[k];
    const cyxwiz::Tensor logits({rows, classes}, logits_host.data(), cyxwiz::DataType::Float32);
    const auto value = Random({rows, 1}, 21, 1.0f);
    const auto packed = Host(cyxwiz::rl::PpoSampleStep(logits, value, cyxwiz::Tensor(), true, 77));
    Check(packed.size() == rows * 3, "categorical packed size");
    const auto values = Host(value);
    double z = 0.0;
    for (float l : row_logits) z += std::exp(l);
    std::vector<double> count(classes, 0.0);
    bool values_ok = true, lp_ok = true;
    for (size_t r = 0; r < rows; ++r) {
        values_ok = values_ok && packed[r] == values[r];
        const int a = static_cast<int>(packed[2 * rows + r]);
        if (a < 0 || a >= static_cast<int>(classes)) {
            lp_ok = false;
            continue;
        }
        count[a] += 1.0;
        lp_ok = lp_ok && std::fabs(packed[rows + r] - (row_logits[a] - std::log(z))) < 1e-5;
    }
    Check(values_ok, "the packed values are the values");
    Check(lp_ok, "the packed log-probabilities are those of the drawn actions");
    for (size_t k = 0; k < classes; ++k) {
        const double p = std::exp(row_logits[k]) / z, f = count[k] / rows;
        Check(std::fabs(f - p) < 4.0 * std::sqrt(p * (1.0 - p) / rows) + 1e-3,
              "choice " + std::to_string(k) + " drawn " + std::to_string(f) + " of the time, p " + std::to_string(p));
    }
    const auto again = Host(cyxwiz::rl::PpoSampleStep(logits, value, cyxwiz::Tensor(), true, 77));
    const auto other = Host(cyxwiz::rl::PpoSampleStep(logits, value, cyxwiz::Tensor(), true, 78));
    Check(again == packed, "the same seed draws the same");
    Check(other != packed, "another seed draws otherwise");

    // Gaussian: draws around the mean with the std, log-probabilities of the draws
    const size_t act = 2;
    std::vector<float> mean_host(rows * act);
    for (size_t r = 0; r < rows; ++r) {
        mean_host[r * act] = 0.5f;
        mean_host[r * act + 1] = -1.0f;
    }
    const cyxwiz::Tensor mean({rows, act}, mean_host.data(), cyxwiz::DataType::Float32);
    const float ls_host[2] = {-0.5f, 0.2f};
    const cyxwiz::Tensor log_std({act}, ls_host, cyxwiz::DataType::Float32);
    const auto g = Host(cyxwiz::rl::PpoSampleStep(mean, value, log_std, false, 91));
    Check(g.size() == rows * (2 + act), "Gaussian packed size");
    for (size_t j = 0; j < act; ++j) {
        double sum = 0.0, sq = 0.0;
        for (size_t r = 0; r < rows; ++r) {
            const double a = g[2 * rows + r * act + j];
            sum += a;
            sq += a * a;
        }
        const double m = sum / rows, sd = std::sqrt(sq / rows - m * m), want_sd = std::exp(ls_host[j]);
        Check(std::fabs(m - mean_host[j]) < 4.0 * want_sd / std::sqrt(static_cast<double>(rows)),
              "action " + std::to_string(j) + " mean " + std::to_string(m));
        Check(std::fabs(sd - want_sd) < 0.02 * want_sd, "action " + std::to_string(j) + " std " + std::to_string(sd));
    }
    bool glp = true;
    for (size_t r = 0; r < rows; r += 997) {
        double lp = 0.0;
        for (size_t j = 0; j < act; ++j) {
            const double zj = (g[2 * rows + r * act + j] - mean_host[j]) / std::exp(ls_host[j]);
            lp += -0.5 * zj * zj - ls_host[j] - 0.91893853320467274178;
        }
        glp = glp && std::fabs(g[rows + r] - lp) < 1e-4;
    }
    Check(glp, "the Gaussian log-probabilities are those of the draws");
    std::printf("sampling: categorical %.3f / %.3f / %.3f; Gaussian checked\n", count[0] / rows, count[1] / rows,
                count[2] / rows);
}

}  // namespace

int main() {
    if (!cyxwiz::test::SelectTestDeviceFromEnvironment()) return EXIT_FAILURE;
    const bool served = ProviderServes(cyxwiz::NeuralOp::PpoLoss, true) && ProviderServes(cyxwiz::NeuralOp::PpoLoss, false) &&
                        ProviderServes(cyxwiz::NeuralOp::PpoSample, true) && ProviderServes(cyxwiz::NeuralOp::PpoGae, true);
    if (!served) {
        std::printf("no provider runs PPO's kernels on this device: nothing to compare\n");
        std::printf("test_fused_ppo: all checks passed\n");
        return EXIT_SUCCESS;
    }
    for (const bool discrete : {true, false})
        for (const float clip_vf : {0.0f, 0.2f})
            for (const bool normalize : {true, false}) CompareLoss(discrete, clip_vf, normalize, 64);
    CompareLoss(true, 0.0f, true, 1);
    CompareLoss(false, 0.0f, true, 1000);
    CompareGae(32, 8);
    CompareGae(512, 1);
    CompareGae(7, 33);
    CheckSampleStep();
    if (failures == 0) std::printf("test_fused_ppo: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
