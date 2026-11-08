// The backend's PPO math (cyxwiz/rl/ppo.h) against PyTorch: the
// Stable-Baselines3 loss and its gradients for Gaussian and categorical
// policies under three settings, log-probabilities, GAE, and the samplers'
// seeds and distributions. Fixture: computation_truth/fixtures/ppo_pytorch.json
// (reference/generate_ppo_fixtures.py, TOFIX195 S4.1).

#include <cyxwiz/rl/ppo.h>
#include <cyxwiz/tensor.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
using cyxwiz::Tensor;

int failures = 0;

void Check(bool ok, const std::string& what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" / "ppo_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return CYXWIZ_PPO_FIXTURE;
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

void Close(const std::vector<float>& got, const json& want, double tolerance, const std::string& what) {
    const double worst = Worst(got, want.at("values").get<std::vector<float>>());
    Check(worst <= tolerance, what + ": largest difference " + std::to_string(worst));
}

void Close(float got, double want, double tolerance, const std::string& what) {
    const double diff = std::abs(got - want) / std::max(1.0, std::abs(want));
    Check(diff <= tolerance, what + ": " + std::to_string(got) + " vs PyTorch " + std::to_string(want));
}

cyxwiz::rl::PpoLossSettings Settings(const json& s) {
    cyxwiz::rl::PpoLossSettings out;
    out.clip_range = s.at("clip_range").get<float>();
    out.clip_range_vf = s.at("clip_range_vf").get<float>();
    out.ent_coef = s.at("ent_coef").get<float>();
    out.vf_coef = s.at("vf_coef").get<float>();
    out.normalize_advantage = s.at("normalize_advantage").get<bool>();
    return out;
}

cyxwiz::rl::PpoBatch Batch(const json& in) {
    cyxwiz::rl::PpoBatch batch;
    batch.actions = FromFixture(in.at("actions"));
    batch.old_log_prob = FromFixture(in.at("old_log_prob"));
    batch.advantages = FromFixture(in.at("advantages"));
    batch.returns = FromFixture(in.at("returns"));
    batch.old_values = FromFixture(in.at("old_values"));
    return batch;
}

constexpr double kTolerance = 2e-5;

void CheckScalars(const cyxwiz::rl::PpoLossResult& r, const json& expected, const std::string& name) {
    Close(r.loss, expected.at("loss"), kTolerance, name + " loss");
    Close(r.policy_loss, expected.at("policy_loss"), kTolerance, name + " policy loss");
    Close(r.value_loss, expected.at("value_loss"), kTolerance, name + " value loss");
    Close(r.entropy_loss, expected.at("entropy_loss"), kTolerance, name + " entropy loss");
    Close(r.approx_kl, expected.at("approx_kl"), kTolerance, name + " approx KL");
    Close(r.clip_fraction, expected.at("clip_fraction"), 1e-6, name + " clip fraction");
}

void CheckGaussian(const json& fixture) {
    for (const auto& c : fixture.at("gaussian")) {
        const std::string name = c.at("name");
        const auto& in = c.at("inputs");
        const Tensor mean = FromFixture(in.at("mean"));
        const Tensor log_std = FromFixture(in.at("log_std"));
        const Tensor values = FromFixture(in.at("values"));
        const auto batch = Batch(in);
        Close(Values(cyxwiz::rl::GaussianLogProb(mean, log_std, batch.actions)), c.at("log_prob"), kTolerance,
              name + " log-probabilities");
        const auto r = cyxwiz::rl::PpoGaussianLoss(mean, log_std, values, batch, Settings(c.at("settings")));
        CheckScalars(r, c.at("expected"), name);
        Close(Values(r.grad_policy_output), c.at("expected").at("grad_mean"), kTolerance, name + " d loss / d mean");
        Close(Values(r.grad_log_std), c.at("expected").at("grad_log_std"), kTolerance, name + " d loss / d log_std");
        Close(Values(r.grad_value_output), c.at("expected").at("grad_values"), kTolerance, name + " d loss / d value");
    }
}

void CheckCategorical(const json& fixture) {
    for (const auto& c : fixture.at("categorical")) {
        const std::string name = c.at("name");
        const auto& in = c.at("inputs");
        const Tensor logits = FromFixture(in.at("logits"));
        const Tensor values = FromFixture(in.at("values"));
        const auto batch = Batch(in);
        Close(Values(cyxwiz::rl::CategoricalLogProb(logits, batch.actions)), c.at("log_prob"), kTolerance,
              name + " log-probabilities");
        const auto r = cyxwiz::rl::PpoCategoricalLoss(logits, values, batch, Settings(c.at("settings")));
        CheckScalars(r, c.at("expected"), name);
        Check(r.grad_log_std.Shape().empty(), name + ": a categorical policy has no log_std gradient");
        Close(Values(r.grad_policy_output), c.at("expected").at("grad_logits"), kTolerance, name + " d loss / d logits");
        Close(Values(r.grad_value_output), c.at("expected").at("grad_values"), kTolerance, name + " d loss / d value");
    }
}

void CheckGae(const json& fixture) {
    const auto& g = fixture.at("gae");
    Tensor advantages, returns;
    cyxwiz::rl::ComputeGae(FromFixture(g.at("rewards")), FromFixture(g.at("values")), FromFixture(g.at("dones")),
                           FromFixture(g.at("last_values")), g.at("gamma").get<float>(), g.at("gae_lambda").get<float>(),
                           advantages, returns);
    Check(advantages.Shape() == std::vector<size_t>({6, 3}), "GAE advantages are [T, N]");
    Close(Values(advantages), g.at("advantages"), 1e-5, "GAE advantages");
    Close(Values(returns), g.at("returns"), 1e-5, "GAE returns");
}

void CheckSampling() {
    const size_t rows = 20000;
    std::vector<float> means(rows * 2);
    for (size_t i = 0; i < rows; ++i) {
        means[2 * i] = 1.5f;
        means[2 * i + 1] = -0.5f;
    }
    const Tensor mean({rows, 2}, means.data());
    const std::vector<float> log_std_values = {std::log(0.5f), std::log(2.0f)};
    const Tensor log_std({2}, log_std_values.data());
    const auto a = Values(cyxwiz::rl::SampleGaussian(mean, log_std, 7));
    const auto b = Values(cyxwiz::rl::SampleGaussian(mean, log_std, 7));
    const auto c = Values(cyxwiz::rl::SampleGaussian(mean, log_std, 8));
    Check(a == b && a != c, "a Gaussian seed gives the same actions; another seed others");
    for (int k = 0; k < 2; ++k) {
        double sum = 0.0, sq = 0.0;
        for (size_t i = 0; i < rows; ++i) sum += a[2 * i + k];
        const double m = sum / rows;
        for (size_t i = 0; i < rows; ++i) sq += (a[2 * i + k] - m) * (a[2 * i + k] - m);
        const double sd = std::sqrt(sq / (rows - 1));
        const double want_m = means[k], want_sd = std::exp(log_std_values[k]);
        Check(std::abs(m - want_m) < 0.05 * want_sd + 0.01 && std::abs(sd - want_sd) < 0.03 * want_sd,
              "Gaussian action " + std::to_string(k) + " has mean " + std::to_string(m) + " and sd " +
                  std::to_string(sd));
    }

    const std::vector<float> probs = {0.2f, 0.5f, 0.3f};
    std::vector<float> logits(rows * 3);
    for (size_t i = 0; i < rows; ++i)
        for (int k = 0; k < 3; ++k) logits[3 * i + k] = std::log(probs[k]) + 4.0f;  // any shift
    const Tensor logit_tensor({rows, 3}, logits.data());
    const auto picks = Values(cyxwiz::rl::SampleCategorical(logit_tensor, 11));
    Check(picks == Values(cyxwiz::rl::SampleCategorical(logit_tensor, 11)), "a categorical seed gives the same picks");
    std::vector<double> counts(3, 0.0);
    bool whole = true;
    for (const float p : picks) {
        whole = whole && (p == 0.0f || p == 1.0f || p == 2.0f);
        if (p >= 0.0f && p <= 2.0f) counts[static_cast<int>(p)] += 1.0;
    }
    Check(whole, "categorical picks are the indices 0, 1, 2");
    for (int k = 0; k < 3; ++k)
        Check(std::abs(counts[k] / rows - probs[k]) < 0.015,
              "choice " + std::to_string(k) + " picked " + std::to_string(counts[k] / rows) + " of the time");
}

void CheckErrors() {
    const std::vector<float> two = {0.0f, 0.0f};
    const Tensor mean({1, 2}, two.data());
    const Tensor wrong_std({3}, std::vector<float>(3, 0.0f).data());
    bool threw = false;
    try {
        cyxwiz::rl::GaussianLogProb(mean, wrong_std, mean);
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what()) == "PPO: one log standard deviation per action";
    }
    Check(threw, "a log_std of the wrong size is refused, saying why");
}

}  // namespace

int main(int, char** argv) {
    std::ifstream in(FixturePath(argv[0]));
    Check(in.good(), "ppo_pytorch.json is readable");
    if (!in.good()) return EXIT_FAILURE;
    const json fixture = json::parse(in);
    CheckGaussian(fixture);
    CheckCategorical(fixture);
    CheckGae(fixture);
    CheckSampling();
    CheckErrors();
    if (failures == 0) std::printf("test_computation_truth_ppo: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
