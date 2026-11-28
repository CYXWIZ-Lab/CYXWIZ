#pragma once

// CYXWIZ_EXPECT_PATH: the compute path a test run must have taken.
//
// Unset (or empty), a test behaves as it always has: on a device whose
// provider is absent it says "nothing to compare" or "not checked" and
// passes. That is right for a developer's CPU run, and wrong for a gate, a
// benchmark or a lab run, which measure a path and must not pass on another.
//
// Set to a provider, the RL tests and the RL gate tool FAIL when any of their
// RL ops did not run on that provider, and say which op and why:
//
//     CYXWIZ_EXPECT_PATH=nvidia    cyxwiz.nvidia-cublas-cell (CUDA)
//     CYXWIZ_EXPECT_PATH=opencl    cyxwiz.opencl-cell
//     CYXWIZ_EXPECT_PATH=oneapi    cyxwiz.oneapi-sycl-attention
//     CYXWIZ_EXPECT_PATH=<id>      any provider id, as given
//
// It is read beside CYXWIZ_TEST_ARRAYFIRE_BACKEND (test_device_selection.h),
// which chooses the device; this one says what must run there. Read by
// test_ppo_trainer, test_fused_ppo and cyxwiz-rl-stability-gate.

#include <cyxwiz/neural_provider.h>
#include <cyxwiz/neural_provider_dispatch.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::test {

struct ExpectedComputePath {
    bool set = false;
    std::string name;         // the variable's value
    std::string provider_id;  // the provider every RL op must have run on
};

inline ExpectedComputePath ReadExpectedComputePath() {
    ExpectedComputePath expected;
    const char* value = std::getenv("CYXWIZ_EXPECT_PATH");
    if (value == nullptr || value[0] == '\0') return expected;
    expected.set = true;
    expected.name = value;
    if (expected.name == "nvidia") expected.provider_id = "cyxwiz.nvidia-cublas-cell";
    else if (expected.name == "opencl") expected.provider_id = "cyxwiz.opencl-cell";
    else if (expected.name == "oneapi") expected.provider_id = "cyxwiz.oneapi-sycl-attention";
    else expected.provider_id = expected.name;
    std::printf("expected path: every RL op on %s (CYXWIZ_EXPECT_PATH=%s)\n", expected.provider_id.c_str(),
                expected.name.c_str());
    std::fflush(stdout);
    return expected;
}

// The parts of a run's compute path (rl::ComputePathPart: its networks, its
// Adam step, PPO's ops) that did not run wholly on the provider, each as
// "<what>: <where it ran> (<why>)". Empty when every part did. No parts at
// all is a violation too: nothing ran.
template <class Part>
std::vector<std::string> ComputePathViolations(const std::vector<Part>& parts, const std::string& provider_id) {
    std::vector<std::string> violations;
    if (parts.empty()) violations.push_back("the run reported no compute path");
    for (const auto& part : parts) {
        if (part.provider_id == provider_id && !part.mixed) continue;
        std::string where = part.provider_id.empty() ? std::string("ArrayFire")
                                                     : part.mixed ? part.provider_id + ", some calls on ArrayFire"
                                                                  : part.provider_id;
        std::string why = part.detail;
        if (!part.reason.empty()) why = part.reason + (why.empty() ? "" : ": " + why);
        violations.push_back(part.what + ": " + where + (why.empty() ? std::string() : " (" + why + ")"));
    }
    return violations;
}

// The same from a dispatch tally: each (layer, op) of `required` must have
// been asked for and have run every time on the provider. An op that was
// never dispatched is reported with the reason no provider serves it.
inline std::vector<std::string> DispatchViolations(const NeuralDispatchTally& tally, const std::string& provider_id,
                                                   const std::vector<std::pair<const char*, NeuralOp>>& required) {
    std::vector<std::string> violations;
    for (const auto& [layer, op] : required) {
        const std::string what = std::string(layer) + " " + NeuralOpName(op);
        const NeuralDeviceTarget target = CaptureCurrentNeuralDeviceTarget();
        const NeuralDispatchTally::Entry* entry = tally.Find(layer, op, target.platform);
        if (!entry) {
            const NeuralProviderAbsence absence = WhyNoNeuralProviderServes(target.platform, op);
            const bool served = !NeuralProviderRegistry::Instance().ListServing(target).empty();
            violations.push_back(what + ": never asked of a provider" +
                                 (served ? std::string()
                                         : std::string(" (") + BackendFallbackReasonName(absence.reason) + ": " +
                                               absence.Sentence() + ")"));
            continue;
        }
        const unsigned long long elsewhere = entry->declined + entry->failed + entry->not_served;
        if (entry->ran > 0 && elsewhere == 0 && entry->provider_id == provider_id) continue;
        std::string where = entry->ran == 0 ? std::string("ArrayFire")
                                            : entry->provider_id + (elsewhere ? ", some calls on ArrayFire" : "");
        std::string why = entry->reason;
        if (!entry->detail.empty()) why += (why.empty() ? "" : ": ") + entry->detail;
        violations.push_back(what + ": " + where + (why.empty() ? std::string() : " (" + why + ")"));
    }
    return violations;
}

// Prints each violation and returns whether there was none
inline bool ReportComputePathViolations(const ExpectedComputePath& expected, const std::vector<std::string>& violations,
                                        const char* who) {
    for (const auto& violation : violations)
        std::fprintf(stderr, "FAIL: %s: CYXWIZ_EXPECT_PATH=%s, but %s\n", who, expected.name.c_str(), violation.c_str());
    return violations.empty();
}

}  // namespace cyxwiz::test
