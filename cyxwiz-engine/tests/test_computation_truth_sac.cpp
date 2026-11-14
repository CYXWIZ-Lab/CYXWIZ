// The backend's SAC math (cyxwiz/rl/sac.h) against PyTorch and
// Stable-Baselines3 2.7: the squashed Gaussian (actions, log pi and the
// gradients to the actor's mean and log std, through the clamp and near
// saturation), the critics' target, the critic, actor and entropy-weight
// losses with their gradients (ties in the minimum), Polyak averaging, and
// the seeded draws. Fixture: computation_truth/fixtures/sac_pytorch.json
// (reference/generate_sac_fixtures.py, TOFIX216 S2).

#include <cyxwiz/rl/sac.h>
#include <cyxwiz/tensor.h>

#include "computation_truth/test_device_selection.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
using cyxwiz::Tensor;
namespace rl = cyxwiz::rl;

int failures = 0;

void Check(bool ok, const std::string& what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" / "sac_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return CYXWIZ_SAC_FIXTURE;
}

Tensor FromFixture(const json& f) {
    std::vector<size_t> shape = f.at("shape").get<std::vector<size_t>>();
    const std::vector<float> values = f.at("values").get<std::vector<float>>();
    if (shape.empty()) shape = {1};
    return Tensor(shape, values.data());
}

std::vector<float> Values(const Tensor& t) {
    const float* data = t.ReadData<float>();
    return std::vector<float>(data, data + t.NumElements());
}

// Largest |a - b| relative to max(1, |b|)
double Worst(const std::vector<float>& got, const std::vector<float>& want) {
    if (got.size() != want.size()) return 1e9;
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i)
        worst = std::max(worst, std::abs(double(got[i]) - want[i]) / std::max(1.0, std::abs(double(want[i]))));
    return worst;
}

void Close(const Tensor& got, const json& want, double tolerance, const std::string& what) {
    const double worst = Worst(Values(got), want.at("values").get<std::vector<float>>());
    Check(worst <= tolerance, what + ": largest difference " + std::to_string(worst));
}

void Close(const Tensor& got, double want, double tolerance, const std::string& what) {
    const float value = Values(got).at(0);
    const double diff = std::abs(value - want) / std::max(1.0, std::abs(want));
    Check(diff <= tolerance, what + ": " + std::to_string(value) + " vs PyTorch " + std::to_string(want));
}

constexpr double kTolerance = 2e-5;

void CheckSquash(const json& fixture) {
    for (const auto& c : fixture.at("squash")) {
        const std::string name = c.at("name");
        const auto& in = c.at("inputs");
        const auto& want = c.at("expected");
        const Tensor mean = FromFixture(in.at("mean"));
        const Tensor log_std = FromFixture(in.at("log_std"));
        const Tensor noise = FromFixture(in.at("noise"));
        const auto squashed = rl::SacSquash(mean, log_std, noise);
        Close(squashed.actions, want.at("actions"), kTolerance, name + " actions");
        // log pi near saturation sums logs of small numbers: relative to its size
        Close(squashed.log_prob, want.at("log_prob"), 1e-4, name + " log pi");
        Close(rl::SacDeterministicAction(mean), want.at("deterministic"), kTolerance, name + " deterministic action");
        const auto grads = rl::SacSquashBackward(mean, log_std, noise, FromFixture(in.at("grad_actions")),
                                                 FromFixture(in.at("grad_log_prob")));
        Close(grads.grad_mean, want.at("grad_mean"), 1e-4, name + " d / d mean");
        Close(grads.grad_log_std, want.at("grad_log_std"), 1e-4, name + " d / d log std");
        // Each upstream gradient alone adds up to both together
        const auto only_actions = rl::SacSquashBackward(mean, log_std, noise, FromFixture(in.at("grad_actions")), Tensor());
        const auto only_log_prob = rl::SacSquashBackward(mean, log_std, noise, Tensor(), FromFixture(in.at("grad_log_prob")));
        const auto a = Values(only_actions.grad_mean), b = Values(only_log_prob.grad_mean), both = Values(grads.grad_mean);
        double worst = 0.0;
        for (size_t i = 0; i < both.size(); ++i) worst = std::max(worst, std::abs(double(a[i]) + b[i] - both[i]));
        Check(worst < 1e-5, name + ": the two upstream gradients add up");
    }
}

void CheckLosses(const json& fixture) {
    for (const auto& c : fixture.at("losses")) {
        const std::string name = c.at("name");
        const auto& in = c.at("inputs");
        const auto& want = c.at("expected");
        const Tensor ent_coef = FromFixture(in.at("ent_coef"));
        const Tensor log_prob = FromFixture(in.at("log_prob"));

        const auto alpha = rl::SacEntCoefLossAndGradient(FromFixture(in.at("log_ent_coef")), log_prob,
                                                         c.at("target_entropy").get<float>());
        Close(alpha.loss, want.at("ent_coef_loss").get<double>(), kTolerance, name + " entropy weight loss");
        Close(alpha.grad_log_ent_coef, want.at("grad_log_ent_coef"), kTolerance, name + " d / d log alpha");

        const Tensor target = rl::SacTarget(FromFixture(in.at("next_q1")), FromFixture(in.at("next_q2")),
                                            FromFixture(in.at("next_log_prob")), FromFixture(in.at("rewards")),
                                            FromFixture(in.at("dones")), ent_coef, c.at("gamma").get<float>());
        Close(target, want.at("target"), kTolerance, name + " target");
        Check(target.Shape() == std::vector<size_t>({target.NumElements(), 1}), name + ": the target is [B, 1]");

        const auto critic = rl::SacCriticLossAndGradients(FromFixture(in.at("q1")), FromFixture(in.at("q2")), target);
        Close(critic.loss, want.at("critic_loss").get<double>(), kTolerance, name + " critic loss");
        Close(critic.grad_q1, want.at("grad_q1"), kTolerance, name + " d / d Q1");
        Close(critic.grad_q2, want.at("grad_q2"), kTolerance, name + " d / d Q2");

        const auto actor = rl::SacActorLossAndGradients(FromFixture(in.at("q1_pi")), FromFixture(in.at("q2_pi")),
                                                        log_prob, ent_coef);
        Close(actor.loss, want.at("actor_loss").get<double>(), kTolerance, name + " actor loss");
        Close(actor.mean_min_q, want.at("mean_min_q").get<double>(), kTolerance, name + " mean min Q");
        Close(actor.grad_q1, want.at("grad_q1_pi"), kTolerance, name + " d / d Q1(pi)");
        Close(actor.grad_q2, want.at("grad_q2_pi"), kTolerance, name + " d / d Q2(pi)");
        Close(actor.grad_log_prob, want.at("grad_log_prob"), kTolerance, name + " d / d log pi");
    }
}

void CheckPolyak(const json& fixture) {
    const auto& p = fixture.at("polyak");
    const float tau = p.at("tau").get<float>();
    for (size_t i = 0; i < p.at("params").size(); ++i) {
        const Tensor moved = rl::PolyakAverage(FromFixture(p.at("params")[i]), FromFixture(p.at("targets")[i]), tau);
        Close(moved, p.at("expected")[i], 1e-6, "Polyak parameter " + std::to_string(i));
    }
}

void CheckDraws() {
    const auto noise = Values(rl::SacNoise(512, 6, 7));
    Check(noise == Values(rl::SacNoise(512, 6, 7)), "one seed, the same noise");
    Check(noise != Values(rl::SacNoise(512, 6, 8)), "another seed, other noise");
    double sum = 0.0, squares = 0.0;
    for (const float v : noise) {
        sum += v;
        squares += double(v) * v;
    }
    const double mean = sum / noise.size(), variance = squares / noise.size() - mean * mean;
    Check(std::abs(mean) < 0.06 && std::abs(variance - 1.0) < 0.08,
          "the noise is standard normal: mean " + std::to_string(mean) + ", variance " + std::to_string(variance));

    const auto uniform = Values(rl::SacUniformActions(512, 6, 7));
    Check(uniform == Values(rl::SacUniformActions(512, 6, 7)), "one seed, the same random actions");
    const auto [lo, hi] = std::minmax_element(uniform.begin(), uniform.end());
    double total = 0.0;
    for (const float v : uniform) total += v;
    Check(*lo >= -1.0f && *hi <= 1.0f && *lo < -0.95f && *hi > 0.95f && std::abs(total / uniform.size()) < 0.05,
          "random actions span [-1, 1] evenly");
}

void CheckErrors() {
    const Tensor mean({2, 3}, std::vector<float>(6, 0.0f).data());
    const Tensor narrow({2, 2}, std::vector<float>(4, 0.0f).data());
    bool threw = false;
    try {
        rl::SacSquash(mean, narrow, mean);
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what()) == "SAC: one log standard deviation per mean";
    }
    Check(threw, "a log std of the wrong shape is refused, saying why");
    threw = false;
    try {
        rl::PolyakAverage(mean, narrow, 0.005f);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Check(threw, "a target of another shape is refused");
}

}  // namespace

int main(int, char** argv) {
    // The device CYXWIZ_TEST_ARRAYFIRE_BACKEND names
    if (!cyxwiz::test::SelectTestDeviceFromEnvironment()) return EXIT_FAILURE;
    std::ifstream in(FixturePath(argv[0]));
    Check(in.good(), "sac_pytorch.json is readable");
    if (!in.good()) return EXIT_FAILURE;
    try {
        const json fixture = json::parse(in);
        CheckSquash(fixture);
        CheckLosses(fixture);
        CheckPolyak(fixture);
        CheckDraws();
        CheckErrors();
    } catch (const std::exception& e) {
        Check(false, std::string("threw: ") + e.what());
    }
    if (failures == 0) std::printf("test_computation_truth_sac: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
