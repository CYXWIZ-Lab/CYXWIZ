#include "core/route_qualification_service.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace {

bool Require(bool condition, const std::string& message) {
    if (condition) return true;
    std::cerr << "Route qualification process contract failed: " << message
              << '\n';
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_route_qualification_process PROBE_EXE\n";
        return 2;
    }
    const std::filesystem::path probe = argv[1];
    cyxwiz::RouteProbeInvocation invocation;
    invocation.executable = probe;
    invocation.type = cyxwiz::DeviceType::CPU;
    invocation.device_id = 0;
    invocation.operation = "sum";
    invocation.timeout = std::chrono::seconds(20);

    const auto passed = cyxwiz::RunIsolatedRouteProbe(invocation, [] {
        return false;
    });
    if (!Require(passed.status == cyxwiz::RouteProbeStatus::Passed,
                 "ArrayFire CPU sum probe did not pass") ||
        !Require(passed.output.find("operation=sum") != std::string::npos,
                 "bounded child output lost the exact operation") ||
        !Require(!passed.last_probe_stage.empty(),
                 "bounded child output lost the final probe stage")) {
        std::cerr << passed.output << '\n' << passed.infrastructure_error << '\n';
        return 1;
    }

    invocation.operation = "dense_compute_benchmark";
    invocation.timeout = std::chrono::seconds(60);
    const auto benchmark = cyxwiz::RunIsolatedRouteProbe(invocation, [] {
        return false;
    });
    if (!Require(benchmark.status == cyxwiz::RouteProbeStatus::Passed,
                 "ArrayFire CPU dense benchmark did not pass") ||
        !Require(benchmark.output.find(
                     "benchmark_id=cyxwiz-dense-compute-v1") !=
                     std::string::npos,
                 "dense benchmark did not identify its fixed workload") ||
        !Require(benchmark.output.find("median_iteration_ms=") !=
                     std::string::npos,
                 "dense benchmark did not report its median")) {
        std::cerr << benchmark.output << '\n'
                  << benchmark.infrastructure_error << '\n';
        return 1;
    }

    invocation.operation = "sum";

    const auto cancelled = cyxwiz::RunIsolatedRouteProbe(invocation, [] {
        return true;
    });
    if (!Require(cancelled.status == cyxwiz::RouteProbeStatus::Cancelled,
                 "cancellation did not terminate the isolated child")) {
        return 1;
    }

    invocation.timeout = std::chrono::milliseconds(1);
    const auto timed_out = cyxwiz::RunIsolatedRouteProbe(invocation, [] {
        return false;
    });
    if (!Require(timed_out.status == cyxwiz::RouteProbeStatus::TimedOut,
                 "timeout did not terminate the isolated child")) {
        return 1;
    }

    invocation.timeout = std::chrono::seconds(20);
    invocation.output_limit_bytes = 1;
    const auto bounded = cyxwiz::RunIsolatedRouteProbe(invocation, [] {
        return false;
    });
    if (!Require(
            bounded.status == cyxwiz::RouteProbeStatus::InfrastructureFailure,
            "output-limit termination was not reported as infrastructure failure") ||
        !Require(bounded.infrastructure_error.find("output limit") !=
                     std::string::npos,
                 "output-limit failure did not retain its reason")) {
        return 1;
    }

    std::cout << "Route qualification process contracts passed\n";
    return 0;
}
