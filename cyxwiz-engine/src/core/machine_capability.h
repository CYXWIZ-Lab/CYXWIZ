#pragma once

// What this machine can train on, as measured (TOFIX118 P3 S2): each route
// the machine verified (route-qualification evidence) with its standard
// training benchmark, an environment fingerprint, and a compute score. A
// Server Node sends it at registration; S4 attaches the fingerprint to every
// job result.

#include "route_qualification_snapshot.h"
#include "training_benchmark.h"

#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

struct MachineRouteCapability {
    RouteQualificationRecord route;
    std::optional<TrainingBenchmarkResult> benchmark;  // absent: never benchmarked
    // The benchmark was measured by this build on this physical device.
    bool benchmark_current = false;
};

struct MachineEnvironment {
    std::string cyxwiz_build;
    std::string os;
    std::string route_matrix_id;
    std::string compute_contract_id;
    // SHA-256 over the build, OS, route matrix and each route's identity,
    // driver and runtime: equal fingerprints mean the same software on the
    // same devices.
    std::string fingerprint;
};

struct MachineCapability {
    std::vector<MachineRouteCapability> routes;
    MachineEnvironment environment;
    // Best current benchmark tokens/s over certified routes; 0 = not measured.
    double compute_score = 0.0;
    std::string compute_score_route;  // e.g. arrayfire_cuda:0
};

MachineCapability BuildMachineCapability(const RouteQualificationSnapshot& snapshot,
                                         const std::vector<TrainingBenchmarkResult>& benchmarks,
                                         const std::string& build);

// This machine: the installed route evidence and the cached benchmarks.
MachineCapability DetectMachineCapability();

}  // namespace cyxwiz
