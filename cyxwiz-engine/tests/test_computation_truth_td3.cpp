// The backend's TD3 math (cyxwiz/rl/td3.h) against PyTorch and
// Stable-Baselines3 2.7: the target policy smoothing (noise past the clip,
// actions at the bounds), exploration's clipped noise, the critics' target,
// the critic loss (the sum of the twins' MSE) and the actor loss (Q1 only)
// with their gradients, ties in the minimum. Fixture:
// computation_truth/fixtures/td3_pytorch.json
// (reference/generate_td3_fixtures.py, TOFIX216 TD3 U2).

#include <cyxwiz/rl/td3.h>
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
    const auto beside = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" / "td3_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return CYXWIZ_TD3_FIXTURE;
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

void CheckActions(const json& fixture) {
    for (const auto& c : fixture.at("actions")) {
        const std::string name = c.at("name");
        const auto& in = c.at("inputs");
        const auto& want = c.at("expected");
        const Tensor actions = FromFixture(in.at("actions"));
        const Tensor noise = FromFixture(in.at("noise"));
        const Tensor target = rl::Td3TargetActions(actions, noise, c.at("target_policy_noise").get<float>(),
                                                   c.at("target_noise_clip").get<float>());
        Close(target, want.at("target_actions"), kTolerance, name + " target actions");
        const Tensor explore = rl::Td3ExploreActions(actions, noise, c.at("noise_std").get<float>());
        Close(explore, want.at("explore"), kTolerance, name + " exploring actions");
        const auto t = Values(target), e = Values(explore);
        Check(std::all_of(t.begin(), t.end(), [](float v) { return v >= -1.0f && v <= 1.0f; }) &&
                  std::all_of(e.begin(), e.end(), [](float v) { return v >= -1.0f && v <= 1.0f; }),
              name + ": the actions stay in [-1, 1]");
    }
}

void CheckLosses(const json& fixture) {
    for (const auto& c : fixture.at("losses")) {
        const std::string name = c.at("name");
        const auto& in = c.at("inputs");
        const auto& want = c.at("expected");
        const Tensor target = rl::Td3Target(FromFixture(in.at("next_q1")), FromFixture(in.at("next_q2")),
                                            FromFixture(in.at("rewards")), FromFixture(in.at("dones")),
                                            c.at("gamma").get<float>());
        Close(target, want.at("target"), kTolerance, name + " target");
        Check(target.Shape() == std::vector<size_t>({target.NumElements(), 1}), name + ": the target is [B, 1]");

        const auto critic = rl::Td3CriticLossAndGradients(FromFixture(in.at("q1")), FromFixture(in.at("q2")), target);
        Close(critic.loss, want.at("critic_loss").get<double>(), kTolerance, name + " critic loss");
        Close(critic.grad_q1, want.at("grad_q1"), kTolerance, name + " d / d Q1");
        Close(critic.grad_q2, want.at("grad_q2"), kTolerance, name + " d / d Q2");

        const auto actor = rl::Td3ActorLossAndGradient(FromFixture(in.at("q1_pi")));
        Close(actor.loss, want.at("actor_loss").get<double>(), kTolerance, name + " actor loss");
        Close(actor.mean_q, want.at("mean_q").get<double>(), kTolerance, name + " mean Q");
        Close(actor.grad_q1, want.at("grad_q1_pi"), kTolerance, name + " d / d Q1(pi)");
    }
}

void CheckErrors() {
    const Tensor actions({2, 3}, std::vector<float>(6, 0.0f).data());
    const Tensor narrow({2, 2}, std::vector<float>(4, 0.0f).data());
    bool threw = false;
    try {
        rl::Td3TargetActions(actions, narrow, 0.2f, 0.5f);
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what()) == "TD3: one noise value per action";
    }
    Check(threw, "noise of the wrong shape is refused, saying why");
    threw = false;
    try {
        rl::Td3TargetActions(actions, actions, -0.1f, 0.5f);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Check(threw, "a negative smoothing noise is refused");
}

}  // namespace

int main(int, char** argv) {
    // The device CYXWIZ_TEST_ARRAYFIRE_BACKEND names
    if (!cyxwiz::test::SelectTestDeviceFromEnvironment()) return EXIT_FAILURE;
    std::ifstream in(FixturePath(argv[0]));
    Check(in.good(), "td3_pytorch.json is readable");
    if (!in.good()) return EXIT_FAILURE;
    try {
        const json fixture = json::parse(in);
        CheckActions(fixture);
        CheckLosses(fixture);
        CheckErrors();
    } catch (const std::exception& e) {
        Check(false, std::string("threw: ") + e.what());
    }
    if (failures == 0) std::printf("test_computation_truth_td3: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
