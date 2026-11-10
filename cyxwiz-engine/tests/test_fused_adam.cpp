// One Adam step for several optimizers together (TOFIX206 3.2 C) against the
// existing path: the gradients of all groups clipped by one global norm
// (ClipGradientsByGlobalNorm), then each optimizer's Step. The parameters
// and the exported optimizer states after three steps agree, with and
// without the clip, for PPO's groups (a policy and a value network of six
// tensors each, a log standard deviation). Also: what StepTogether refuses
// (AdamW, different learning rates, parameters at different steps).
//
// Runs on the device CYXWIZ_TEST_ARRAYFIRE_BACKEND names; without a provider
// for it (the CPU), StepTogether says so and the comparison is not run.

#include <cyxwiz/optimizers/adam.h>
#include <cyxwiz/optimizers/gradient_clipping.h>
#include <cyxwiz/tensor.h>
#include "computation_truth/test_device_selection.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
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

cyxwiz::Tensor Random(const std::vector<size_t>& shape, unsigned seed, float scale) {
    size_t n = 1;
    for (size_t d : shape) n *= d;
    std::vector<float> v(n);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, scale);
    for (float& f : v) f = dist(rng);
    return cyxwiz::Tensor(shape, v.data(), cyxwiz::DataType::Float32);
}

using Params = std::map<std::string, cyxwiz::Tensor>;

Params Network(unsigned seed, size_t in, size_t hidden, size_t out) {
    return {{"layer0.weight", Random({hidden, in}, seed, 0.5f)},     {"layer0.bias", Random({hidden}, seed + 1, 0.1f)},
            {"layer2.weight", Random({hidden, hidden}, seed + 2, 0.2f)}, {"layer2.bias", Random({hidden}, seed + 3, 0.1f)},
            {"layer4.weight", Random({out, hidden}, seed + 4, 0.2f)},  {"layer4.bias", Random({out}, seed + 5, 0.1f)}};
}

Params Gradients(const Params& params, unsigned seed, float scale) {
    Params g;
    for (const auto& [name, p] : params) g[name] = Random(p.Shape(), seed++, scale);
    return g;
}

Params Copy(const Params& p) {
    Params out;
    for (const auto& [name, t] : p) out[name] = t.Clone();
    return out;
}

double Difference(const cyxwiz::Tensor& a, const cyxwiz::Tensor& b) {
    if (a.NumElements() != b.NumElements()) return 1e30;
    const float* x = a.ReadData<float>();
    const float* y = b.ReadData<float>();
    double d = 0.0, scale = 1e-6;
    for (size_t i = 0; i < a.NumElements(); ++i) {
        d = std::max(d, static_cast<double>(std::fabs(x[i] - y[i])));
        scale = std::max(scale, static_cast<double>(std::fabs(y[i])));
    }
    return d / scale;
}

double StateDifference(const cyxwiz::AdamOptimizer& a, const cyxwiz::AdamOptimizer& b) {
    cyxwiz::OptimizerState sa, sb;
    std::string error;
    if (!a.ExportState(sa, error) || !b.ExportState(sb, error) || sa.step_count != sb.step_count ||
        sa.tensors.size() != sb.tensors.size())
        return 1e30;
    double d = 0.0;
    for (const auto& [key, t] : sa.tensors) {
        const auto it = sb.tensors.find(key);
        if (it == sb.tensors.end()) return 1e30;
        if (t.GetDataType() == cyxwiz::DataType::Float32) d = std::max(d, Difference(t, it->second));
    }
    return d;
}

// PPO's three groups, three steps, both ways
void Compare(float max_norm, float gradient_scale) {
    const std::string what = max_norm > 0.0f ? "clipped at " + std::to_string(max_norm) : "no clip";
    Params policy = Network(1, 17, 256, 6), value = Network(11, 17, 256, 1);
    Params log_std{{"log_std", Random({6}, 21, 0.3f)}};
    Params ref_policy = Copy(policy), ref_value = Copy(value), ref_log_std = Copy(log_std);
    cyxwiz::AdamOptimizer p(3e-4, 0.9, 0.999, 1e-5), v(3e-4, 0.9, 0.999, 1e-5), s(3e-4, 0.9, 0.999, 1e-5);
    cyxwiz::AdamOptimizer rp(3e-4, 0.9, 0.999, 1e-5), rv(3e-4, 0.9, 0.999, 1e-5), rs(3e-4, 0.9, 0.999, 1e-5);
    for (int step = 0; step < 3; ++step) {
        const unsigned seed = 100 + 10 * static_cast<unsigned>(step);
        const Params gp = Gradients(policy, seed, gradient_scale), gv = Gradients(value, seed + 3, gradient_scale);
        const Params gs = Gradients(log_std, seed + 6, gradient_scale);
        std::string why;
        const bool together = cyxwiz::AdamOptimizer::StepTogether({{&p, &policy, &gp}, {&v, &value, &gv}, {&s, &log_std, &gs}},
                                                                   max_norm, why);
        Check(together, what + ": StepTogether runs (" + why + ")");
        if (!together) return;
        // The existing path: one global norm over every gradient, then each Step
        Params all;
        for (const auto& [k, g] : gp) all["p:" + k] = g.Clone();
        for (const auto& [k, g] : gv) all["v:" + k] = g.Clone();
        for (const auto& [k, g] : gs) all["s:" + k] = g.Clone();
        if (max_norm > 0.0f) cyxwiz::ClipGradientsByGlobalNorm(all, max_norm);
        Params rgp, rgv, rgs;
        for (const auto& [k, g] : gp) rgp[k] = all.at("p:" + k);
        for (const auto& [k, g] : gv) rgv[k] = all.at("v:" + k);
        for (const auto& [k, g] : gs) rgs[k] = all.at("s:" + k);
        rp.Step(ref_policy, rgp);
        rv.Step(ref_value, rgv);
        rs.Step(ref_log_std, rgs);
    }
    double d = 0.0;
    for (const auto& [k, t] : policy) d = std::max(d, Difference(t, ref_policy.at(k)));
    for (const auto& [k, t] : value) d = std::max(d, Difference(t, ref_value.at(k)));
    d = std::max(d, Difference(log_std.at("log_std"), ref_log_std.at("log_std")));
    Check(d < 1e-5, what + ": parameters differ by " + std::to_string(d));
    const double sd = std::max({StateDifference(p, rp), StateDifference(v, rv), StateDifference(s, rs)});
    Check(sd < 1e-5, what + ": optimizer states differ by " + std::to_string(sd));
    std::printf("%s: parameters within %.2g, states within %.2g after 3 steps\n", what.c_str(), d, sd);
}

void CheckRefusals(bool provider) {
    Params a = Network(31, 4, 8, 2), b = Network(41, 4, 8, 2);
    const Params ga = Gradients(a, 51, 0.1f), gb = Gradients(b, 61, 0.1f);
    std::string why;
    cyxwiz::AdamWOptimizer w(1e-3);
    cyxwiz::AdamOptimizer x(1e-3), y(2e-3);
    Check(!cyxwiz::AdamOptimizer::StepTogether({{&w, &a, &ga}}, 0.5f, why) && why.find("plain Adam") != std::string::npos,
          "AdamW is refused: " + why);
    Check(!cyxwiz::AdamOptimizer::StepTogether({{&x, &a, &ga}, {&y, &b, &gb}}, 0.5f, why) &&
              why.find("hyperparameters") != std::string::npos,
          "different learning rates are refused: " + why);
    if (!provider) return;
    cyxwiz::AdamOptimizer one(1e-3), two(1e-3);
    one.Step(a, ga);  // a at step 1, b at step 0
    Check(!cyxwiz::AdamOptimizer::StepTogether({{&one, &a, &ga}, {&two, &b, &gb}}, 0.5f, why) &&
              why.find("different steps") != std::string::npos,
          "parameters at different steps are refused: " + why);
}

}  // namespace

int main() {
    if (!cyxwiz::test::SelectTestDeviceFromEnvironment()) return EXIT_FAILURE;
    Params probe{{"w", Random({3}, 1, 1.0f)}};
    const Params probe_grad{{"w", Random({3}, 2, 1.0f)}};
    cyxwiz::AdamOptimizer o(1e-3);
    std::string why;
    const bool provider = cyxwiz::AdamOptimizer::StepTogether({{&o, &probe, &probe_grad}}, 0.0f, why);
    CheckRefusals(provider);
    if (provider) {
        Compare(0.0f, 0.1f);
        Compare(0.5f, 0.1f);   // the norm is far above 0.5: every step clips
        Compare(500.0f, 0.1f); // a clip that does not bite
    } else {
        std::printf("StepTogether comparison not run: %s\n", why.c_str());
    }
    if (failures == 0) std::printf("test_fused_adam: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
