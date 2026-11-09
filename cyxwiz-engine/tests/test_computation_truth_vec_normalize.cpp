// The Engine's VecNormalize (core/rl/vec_normalize) against Stable-Baselines3's:
// a recorded stream of observations, rewards and episode ends from three
// copies gives the normalised observations, the scaled and clipped rewards,
// the normalised final observations and the running statistics SB3 gives.
// Fixture: computation_truth/fixtures/vec_normalize_sb3.json
// (reference/generate_vec_normalize_fixtures.py, TOFIX195 S4.3a).

#include "../src/core/rl/vec_normalize.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

int failures = 0;
double worst = 0.0;

void Check(bool ok, const std::string& what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

std::vector<float> Flat(const json& rows) {
    std::vector<float> out;
    for (const auto& row : rows)
        for (const auto& v : row) out.push_back(v.get<float>());
    return out;
}

void Close(const float* got, const std::vector<float>& want, const std::string& what) {
    for (size_t i = 0; i < want.size(); ++i) {
        const double diff = std::abs(double(got[i]) - want[i]) / std::max(1.0, std::abs(double(want[i])));
        worst = std::max(worst, diff);
        if (diff > 1e-5) {
            Check(false, what + " [" + std::to_string(i) + "]: " + std::to_string(got[i]) + " vs SB3 " +
                             std::to_string(want[i]));
            return;
        }
    }
}

}  // namespace

int main() {
    std::ifstream in(CYXWIZ_VEC_NORMALIZE_FIXTURE);
    Check(in.good(), "vec_normalize_sb3.json is readable");
    if (!in.good()) return EXIT_FAILURE;
    const json f = json::parse(in);
    const size_t copies = f.at("copies"), width = f.at("width");

    cyxwiz::rl::VecNormalizeSettings settings;
    settings.observations = settings.rewards = true;
    settings.gamma = f.at("gamma");
    settings.clip_observations = f.at("clip_observations");
    settings.clip_rewards = f.at("clip_rewards");
    cyxwiz::rl::VecNormalize norm(copies, width, settings);

    std::vector<float> obs = Flat(f.at("inputs").at("reset"));
    norm.Reset(obs.data());
    Close(obs.data(), Flat(f.at("expected").at("reset")), "first observations");

    const auto& steps = f.at("inputs").at("steps");
    const auto& want = f.at("expected").at("steps");
    int clipped_rewards = 0, finals = 0;
    for (size_t t = 0; t < steps.size(); ++t) {
        const std::string step = "step " + std::to_string(t);
        obs = Flat(steps[t].at("obs"));
        auto rewards = steps[t].at("rewards").get<std::vector<float>>();
        std::vector<uint8_t> terminated(copies), truncated(copies, 0);
        for (size_t i = 0; i < copies; ++i) terminated[i] = static_cast<uint8_t>(steps[t].at("done")[i].get<int>());
        std::vector<float> final_obs = Flat(steps[t].at("final"));
        norm.Step(obs.data(), rewards.data(), terminated.data(), truncated.data(), final_obs.data());
        Close(obs.data(), Flat(want[t].at("obs")), step + " observations");
        const auto want_rewards = want[t].at("rewards").get<std::vector<float>>();
        Close(rewards.data(), want_rewards, step + " rewards");
        for (const float r : want_rewards) clipped_rewards += std::abs(r) == settings.clip_rewards;
        for (size_t i = 0; i < copies; ++i) {
            if (want[t].at("final")[i].is_null()) continue;
            ++finals;
            Close(final_obs.data() + i * width, want[t].at("final")[i].get<std::vector<float>>(),
                  step + " final observation of copy " + std::to_string(i));
        }
    }
    Check(clipped_rewards > 0 && finals > 0, "the stream clips rewards and ends episodes");

    // SB3 takes a float32 batch's moments in float32 (numpy) and gamma in
    // double; the Engine sums in double with a float gamma: float precision apart
    const auto& stats = f.at("final_stats");
    const auto& o = norm.ObservationStats();
    for (size_t j = 0; j < width; ++j) {
        const double m = stats.at("obs_mean")[j], v = stats.at("obs_var")[j];
        Check(std::abs(o.mean[j] - m) <= 1e-6 * std::max(1.0, std::abs(m)) &&
                  std::abs(o.var[j] - v) <= 1e-6 * std::max(1.0, v),
              "observation statistics of feature " + std::to_string(j));
    }
    Check(std::abs(o.count - stats.at("obs_count").get<double>()) < 1e-9, "observation count");
    const double rv = stats.at("ret_var");
    Check(std::abs(norm.ReturnStats().var[0] - rv) <= 1e-6 * std::max(1.0, rv), "return variance");

    std::printf("largest difference %.3g over %zu steps (%d clipped rewards, %d final observations)\n", worst,
                steps.size(), clipped_rewards, finals);
    if (failures == 0) std::printf("test_computation_truth_vec_normalize: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
