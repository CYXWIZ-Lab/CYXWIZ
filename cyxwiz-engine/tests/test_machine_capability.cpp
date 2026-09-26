// Measured machine capability (TOFIX118 P3 S2): verified routes joined with
// their training benchmarks, the compute score and the environment
// fingerprint.
#include "../src/core/machine_capability.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++g_failures;
}

cyxwiz::RouteQualificationRecord Route(cyxwiz::DeviceType type, int id, const std::string& fingerprint,
                                       bool certified) {
    cyxwiz::RouteQualificationRecord route;
    route.type = type;
    route.device_id = id;
    route.physical_fingerprint = fingerprint;
    route.driver_version = "1.0";
    route.runtime_version = "3.10.0";
    route.certified = certified;
    return route;
}

cyxwiz::TrainingBenchmarkResult Bench(const std::string& backend, int id, const std::string& fingerprint,
                                      double tokens_per_second, bool ok = true) {
    cyxwiz::TrainingBenchmarkResult result;
    result.ok = ok;
    result.backend = backend;
    result.device_id = id;
    result.physical_fingerprint = fingerprint;
    result.build = "1.2.3";
    result.tokens_per_second = tokens_per_second;
    return result;
}

}  // namespace

int main() {
    cyxwiz::RouteQualificationSnapshot snapshot;
    snapshot.matrix_id = "matrix";
    snapshot.compute_contract_id = "contract";
    snapshot.routes = {Route(cyxwiz::DeviceType::CUDA, 0, "uuid:gpu", true),
                       Route(cyxwiz::DeviceType::OPENCL, 0, "uuid:gpu", true),
                       Route(cyxwiz::DeviceType::OPENCL, 1, "uuid:igpu", false),
                       Route(cyxwiz::DeviceType::CPU, 0, "", true)};

    std::cout << "compute score\n";
    auto capability = cyxwiz::BuildMachineCapability(
        snapshot,
        {Bench("arrayfire_cuda", 0, "uuid:gpu", 20000.0), Bench("arrayfire_opencl", 0, "uuid:gpu", 15000.0),
         Bench("arrayfire_opencl", 1, "uuid:igpu", 90000.0), Bench("arrayfire_cpu", 0, "", 0.0, false)},
        "1.2.3");
    Check(capability.routes.size() == 4, "every verified route is listed");
    Check(capability.compute_score == 20000.0 && capability.compute_score_route == "arrayfire_cuda:0",
          "score is the fastest current benchmark on a certified route");
    Check(capability.routes[2].benchmark && capability.routes[2].benchmark_current,
          "an uncertified route keeps its benchmark but does not score");
    Check(capability.routes[3].benchmark && !capability.routes[3].benchmark->ok,
          "a failed benchmark is reported, not scored");

    std::cout << "staleness\n";
    auto other_build = cyxwiz::BuildMachineCapability(snapshot, {Bench("arrayfire_cuda", 0, "uuid:gpu", 20000.0)},
                                                      "1.2.4");
    Check(!other_build.routes[0].benchmark_current && other_build.compute_score == 0.0,
          "a benchmark from another build is not current");
    auto swapped = cyxwiz::BuildMachineCapability(snapshot, {Bench("arrayfire_cuda", 0, "uuid:other", 20000.0)},
                                                  "1.2.3");
    Check(!swapped.routes[0].benchmark_current && swapped.compute_score == 0.0,
          "a benchmark from another physical device is not current");
    auto old_benchmark = Bench("arrayfire_cuda", 0, "uuid:gpu", 20000.0);
    old_benchmark.benchmark_id = "cyxwiz-causal-lm-train-v0";
    Check(!cyxwiz::BuildMachineCapability(snapshot, {old_benchmark}, "1.2.3").routes[0].benchmark_current,
          "another benchmark version is not current");
    auto none = cyxwiz::BuildMachineCapability(snapshot, {}, "1.2.3");
    Check(!none.routes[0].benchmark && none.compute_score == 0.0, "no benchmark: score 0");

    std::cout << "environment fingerprint\n";
    Check(capability.environment.fingerprint.size() == 64, "fingerprint is SHA-256 hex");
    Check(none.environment.fingerprint == capability.environment.fingerprint,
          "benchmarks do not change the fingerprint");
    Check(other_build.environment.fingerprint != capability.environment.fingerprint,
          "another build changes the fingerprint");
    auto new_driver = snapshot;
    new_driver.routes[0].driver_version = "2.0";
    Check(cyxwiz::BuildMachineCapability(new_driver, {}, "1.2.3").environment.fingerprint !=
              capability.environment.fingerprint,
          "a driver update changes the fingerprint");
    Check(capability.environment.cyxwiz_build == "1.2.3" && capability.environment.route_matrix_id == "matrix" &&
              !capability.environment.os.empty(),
          "environment names build, OS and route matrix");

    std::cout << (g_failures == 0 ? "PASS" : "FAILED") << " (" << g_failures << " failures)\n";
    return g_failures == 0 ? 0 : 1;
}
