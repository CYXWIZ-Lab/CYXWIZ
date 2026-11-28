#pragma once

// The start of an oracle test that trains through the training executor
// (TOFIX228 1.1, ORA-04). The executor picks its own device for a run (the
// saved device selection, else the default route), so activating ArrayFire
// is not enough: a test asked for CPU trained on CUDA. This pins the run to
// the requested device (CPU when nothing is asked, so a training test runs
// on a known device) with the Strict policy, which fails instead of falling
// back, and TrainingDeviceProblem() checks the device the run bound.

#include "oracle_test.h"

#include "../route_qualification_test_fixture.h"
#include "../../src/core/execution_device_preferences.h"
#include "../../src/core/runtime_log_store.h"

#include <cstdio>
#include <optional>
#include <string>

namespace cyxwiz::test {

// The device training is pinned to: the requested one, or CPU 0.
inline RequestedTestDevice TrainingTestDevice() {
    std::optional<RequestedTestDevice> request;
    ReadRequestedTestDevice(request);
    if (request) return *request;
    RequestedTestDevice cpu;
    cpu.name = "cpu";
    return cpu;
}

// Every following training run uses the test device, qualified, with the
// Strict policy (no fallback). A pending selection alone covers only the next
// run (the run clears it), so the selection is committed as the saved one (in
// memory, like the Preferences device): later runs resolve to it too.
// BeginTrainingOracleTest does it once; a test that changes the selection
// between runs calls it again.
inline void PinTrainingDevice() {
    const RequestedTestDevice device = TrainingTestDevice();
    InstallQualifiedRouteSnapshot();
    CommitExecutionDeviceSelectionState(PendingExecutionDeviceSelection{device.type, device.id, {}});
    ClearSessionExecutionPolicy();
    SetSessionExecutionPolicy(ExecutionPolicy::Strict);
}

// The test and its commit, the test device active in ArrayFire (CPU when
// nothing is asked) and the training executor pinned to it.
inline bool BeginTrainingOracleTest(const char* name) {
    PrintOracleHeader(name);
    std::optional<RequestedTestDevice> request;
    if (!ReadRequestedTestDevice(request)) return false;
    const RequestedTestDevice device = TrainingTestDevice();
    if (!ActivateTestDevice(device)) return false;
    PinTrainingDevice();
    std::printf("training device: %s %d (Strict)\n", ExecutionDeviceSelectionBackendName(device.type).c_str(),
                device.id);
    std::fflush(stdout);
    return true;
}

// What is wrong with this test's training and test runs, or "" when every one
// bound the pinned device with the Strict policy (their
// "ExecutionDeviceContext.Bind" events; at least one). Prints the count when
// all is well. Call it before building the failure message from its result.
inline std::string TrainingDeviceProblem() {
    const RequestedTestDevice device = TrainingTestDevice();
    const std::string backend = ExecutionDeviceSelectionBackendName(device.type);
    RuntimeLogSnapshotRequest all;
    all.limit = RuntimeLogStore::kDefaultCapacity;
    const auto snapshot = RuntimeLogStore::Instance().Snapshot(all);
    size_t runs = 0;
    for (const auto& event : snapshot.events) {
        if (event.event_name != "ExecutionDeviceContext.Bind") continue;
        ++runs;
        const std::string run = "run " + std::to_string(runs) + " (" + event.source + ")";
        if (event.backend != backend || event.device_id != device.id)
            return "asked for " + backend + " " + std::to_string(device.id) + ", " + run + " bound " + event.backend +
                   " " + std::to_string(event.device_id);
        std::string policy;
        for (const auto& [key, value] : event.details)
            if (key == "execution_policy") policy = value;
        if (policy != ExecutionPolicyName(ExecutionPolicy::Strict))
            return run + " ran with the " + (policy.empty() ? std::string("unknown") : policy) + " policy, not strict";
    }
    if (runs == 0) return "no training run bound a device";
    std::printf("training device: every run (%zu) on %s %d, strict\n", runs, backend.c_str(), device.id);
    return {};
}

}  // namespace cyxwiz::test
