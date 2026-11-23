// The backend's DQN math (cyxwiz/rl/dqn.h) against PyTorch and
// Stable-Baselines3 2.7: the target (the max of the target network's next Q
// values, ties included), the Huber loss on the Q values gathered at the
// stored actions with its gradient into the chosen columns (differences
// within and past 1, of both signs, and 0), the greedy choice (the first on
// ties) and SB3's LinearSchedule for epsilon. Fixture:
// computation_truth/fixtures/dqn_pytorch.json
// (reference/generate_dqn_fixtures.py, TOFIX216 DQN V2).

#include <cyxwiz/rl/dqn.h>
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
    const auto beside = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" / "dqn_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return CYXWIZ_DQN_FIXTURE;
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

void CheckLosses(const json& fixture) {
    for (const auto& c : fixture.at("losses")) {
        const std::string name = c.at("name");
        const auto& in = c.at("inputs");
        const auto& want = c.at("expected");
        const Tensor target = rl::DqnTarget(FromFixture(in.at("next_q")), FromFixture(in.at("rewards")),
                                            FromFixture(in.at("dones")), c.at("gamma").get<float>());
        Close(target, want.at("target"), kTolerance, name + " target");
        Check(target.Shape() == std::vector<size_t>({target.NumElements(), 1}), name + ": the target is [B, 1]");

        const Tensor q = FromFixture(in.at("q"));
        const auto loss = rl::DqnLossAndGradient(q, FromFixture(in.at("actions")), target);
        Close(loss.loss, want.at("loss").get<double>(), kTolerance, name + " Huber loss");
        Close(loss.mean_q, want.at("mean_q").get<double>(), kTolerance, name + " mean chosen Q");
        Close(loss.grad_q, want.at("grad_q"), kTolerance, name + " d / d Q");
        Check(loss.grad_q.Shape() == q.Shape(), name + ": the gradient has the Q values' shape");
    }
}

void CheckGreedy(const json& fixture) {
    for (const auto& c : fixture.at("greedy")) {
        const std::string name = c.at("name");
        const Tensor greedy = rl::DqnGreedyActions(FromFixture(c.at("inputs").at("q")));
        Check(Values(greedy) == c.at("expected").at("greedy").at("values").get<std::vector<float>>(),
              name + ": the greedy choices are torch.argmax's, the first on ties");
    }
}

void CheckSchedule(const json& fixture) {
    for (const auto& c : fixture.at("schedule")) {
        const float rate = rl::DqnExplorationRate(c.at("start").get<float>(), c.at("end").get<float>(),
                                                  c.at("fraction").get<float>(),
                                                  c.at("progress_remaining").get<double>());
        const double want = c.at("rate").get<double>();
        Check(std::abs(rate - want) <= 1e-6,
              "LinearSchedule(" + c.at("start").dump() + ", " + c.at("end").dump() + ", " + c.at("fraction").dump() +
                  ") at " + c.at("progress_remaining").dump() + ": " + std::to_string(rate) + " vs SB3 " +
                  std::to_string(want));
    }
    Check(rl::DqnExplorationRate(1.0f, 0.05f, 0.0f, 1.0) == 0.05f,
          "an exploration over 0 of the run is the end value from the start");
}

void CheckErrors() {
    const Tensor flat({4}, std::vector<float>(4, 0.0f).data());
    const Tensor q({2, 3}, std::vector<float>(6, 0.0f).data());
    const Tensor three({3}, std::vector<float>(3, 0.0f).data());
    bool threw = false;
    try {
        rl::DqnGreedyActions(flat);
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what()) == "DQN: the Q values must be [B, A] with at least one row and one choice";
    }
    Check(threw, "Q values that are not [B, A] are refused, saying why");
    threw = false;
    try {
        rl::DqnLossAndGradient(q, three, Tensor({2, 1}, std::vector<float>(2, 0.0f).data()));
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what()) == "DQN: actions needs 2 values, got 3";
    }
    Check(threw, "one action per row is required");
}

}  // namespace

int main(int, char** argv) {
    // The device CYXWIZ_TEST_ARRAYFIRE_BACKEND names
    if (!cyxwiz::test::SelectTestDeviceFromEnvironment()) return EXIT_FAILURE;
    std::ifstream in(FixturePath(argv[0]));
    Check(in.good(), "dqn_pytorch.json is readable");
    if (!in.good()) return EXIT_FAILURE;
    try {
        const json fixture = json::parse(in);
        CheckLosses(fixture);
        CheckGreedy(fixture);
        CheckSchedule(fixture);
        CheckErrors();
    } catch (const std::exception& e) {
        Check(false, std::string("threw: ") + e.what());
    }
    if (failures == 0) std::printf("test_computation_truth_dqn: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
