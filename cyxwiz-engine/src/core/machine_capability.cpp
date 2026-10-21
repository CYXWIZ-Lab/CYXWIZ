#include "machine_capability.h"

#include "execution_device_preferences.h"
#include "sha256_digest.h"

#include <cyxwiz/cyxwiz.h>

namespace cyxwiz {
namespace {

std::string OsName() {
#if defined(_WIN32)
    return "windows";
#elif defined(__APPLE__)
    return "macos";
#elif defined(__linux__)
    return "linux";
#else
    return "unknown";
#endif
}

std::string RouteName(const RouteQualificationRecord& route) {
    return ExecutionDeviceSelectionBackendName(route.type) + ":" + std::to_string(route.device_id);
}

}  // namespace

MachineCapability BuildMachineCapability(const RouteQualificationSnapshot& snapshot,
                                         const std::vector<TrainingBenchmarkResult>& benchmarks,
                                         const std::string& build) {
    MachineCapability capability;
    capability.environment.cyxwiz_build = build;
    capability.environment.os = OsName();
    capability.environment.route_matrix_id = snapshot.matrix_id;
    capability.environment.compute_contract_id = snapshot.compute_contract_id;

    std::string identity = build + "|" + capability.environment.os + "|" + snapshot.matrix_id + "|" +
                           snapshot.compute_contract_id;
    for (const auto& route : snapshot.routes) {
        MachineRouteCapability entry;
        entry.route = route;
        const std::string backend = ExecutionDeviceSelectionBackendName(route.type);
        for (const auto& benchmark : benchmarks) {
            if (benchmark.backend == backend && benchmark.device_id == route.device_id) entry.benchmark = benchmark;
        }
        entry.benchmark_current = entry.benchmark && entry.benchmark->benchmark_id == kTrainingBenchmarkId &&
                                  entry.benchmark->build == build &&
                                  entry.benchmark->physical_fingerprint == route.physical_fingerprint;
        if (route.certified && entry.benchmark_current && entry.benchmark->ok &&
            entry.benchmark->tokens_per_second > capability.compute_score) {
            capability.compute_score = entry.benchmark->tokens_per_second;
            capability.compute_score_route = RouteName(route);
        }
        identity += "|" + RouteName(route) + "," + route.physical_fingerprint + "," + route.driver_version + "," +
                    route.runtime_version + "," + (route.certified ? "certified" : "not_certified");
        capability.routes.push_back(std::move(entry));
    }

    Sha256Hasher hasher;
    std::string error;
    if (!hasher.Update(identity, error) || !hasher.Finish(capability.environment.fingerprint, error)) {
        capability.environment.fingerprint.clear();
    }
    return capability;
}

MachineCapability DetectMachineCapability() {
    RouteQualificationSnapshot snapshot;
    if (const auto installed = GetRouteQualificationSnapshot()) snapshot = *installed;
    std::vector<TrainingBenchmarkResult> benchmarks;
    std::string error;
    LoadTrainingBenchmarkResults(GetTrainingBenchmarkCachePath(), benchmarks, error);
    return BuildMachineCapability(snapshot, benchmarks, GetVersionString());
}

}  // namespace cyxwiz
