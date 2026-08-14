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
    if (argc != 3) {
        std::cerr <<
            "usage: test_route_qualification_process PROBE_EXE ENV_PROBE_EXE\n";
        return 2;
    }
    const std::filesystem::path probe = argv[1];
    const std::filesystem::path environment_probe =
        std::filesystem::absolute(argv[2]);
    cyxwiz::RouteProbeInvocation invocation;
    invocation.executable = probe;
    invocation.type = cyxwiz::DeviceType::CPU;
    invocation.device_id = 0;
    invocation.operation = "sum";
    invocation.timeout = std::chrono::seconds(20);

#ifdef _WIN32
    _putenv_s("AF_PATH", "inherited-arrayfire-path-must-not-reach-child");
    _putenv_s("PYTHONPATH", "inherited-python-path-must-not-reach-child");
    cyxwiz::RuntimeQualificationIdentity candidate_identity;
    candidate_identity.runtime_set_id = "set-v1";
    candidate_identity.generation = 9;
    candidate_identity.base_pack_id = "base-v1";
    candidate_identity.backend_packs = {
        {cyxwiz::DeviceType::OPENCL, "opencl-v2"}};
    invocation.executable = environment_probe;
    invocation.runtime_root = environment_probe.parent_path();
    invocation.working_directory = environment_probe.parent_path();
    invocation.runtime_dll_directories = {
        environment_probe.parent_path()};
    invocation.runtime_identity = candidate_identity;
    const auto candidate = cyxwiz::RunIsolatedRouteProbe(invocation, [] {
        return false;
    });
    _putenv_s("AF_PATH", "");
    _putenv_s("PYTHONPATH", "");
    if (!Require(candidate.status == cyxwiz::RouteProbeStatus::Passed,
                 "candidate-runtime environment probe did not pass") ||
        !Require(candidate.output.find("runtime_set=set-v1") !=
                     std::string::npos,
                 "candidate runtime-set identity did not reach the child") ||
        !Require(candidate.output.find("runtime_generation=9") !=
                     std::string::npos,
                 "candidate generation did not reach the child") ||
        !Require(candidate.output.find("base_pack=base-v1") !=
                     std::string::npos,
                 "candidate base identity did not reach the child") ||
        !Require(candidate.output.find("opencl_pack=opencl-v2") !=
                     std::string::npos,
                 "candidate backend-pack identity did not reach the child") ||
        !Require(candidate.output.find("af_path=<unset>") !=
                     std::string::npos,
                 "inherited ArrayFire override reached the candidate child") ||
        !Require(candidate.output.find("python_path=<unset>") !=
                     std::string::npos,
                 "inherited Python override reached the candidate child")) {
        std::cerr << candidate.output << '\n'
                  << candidate.infrastructure_error << '\n';
        return 1;
    }
    const auto isolated_root = std::filesystem::temp_directory_path() /
        "cyxwiz-candidate-probe-containment";
    std::error_code containment_error;
    std::filesystem::create_directories(isolated_root, containment_error);
    invocation.runtime_root = isolated_root;
    const auto escaped = cyxwiz::RunIsolatedRouteProbe(invocation, [] {
        return false;
    });
    std::filesystem::remove_all(isolated_root, containment_error);
    if (!Require(
            escaped.status ==
                cyxwiz::RouteProbeStatus::InfrastructureFailure,
            "candidate probe accepted paths outside its runtime root") ||
        !Require(
            escaped.infrastructure_error.find("runtime root") !=
                std::string::npos,
            "candidate probe path rejection lost its reason")) {
        return 1;
    }
    invocation.runtime_root.clear();
    invocation.working_directory.clear();
    invocation.runtime_dll_directories.clear();
    invocation.runtime_identity.reset();
#endif
    invocation.executable = probe;

    cyxwiz::RouteProbeInvocation discovery_invocation;
    discovery_invocation.executable = probe;
    discovery_invocation.type = cyxwiz::DeviceType::CPU;
    discovery_invocation.timeout = std::chrono::seconds(20);
    const auto discovered = cyxwiz::DiscoverIsolatedBackendRoutes(
        discovery_invocation, [] { return false; });
    if (!Require(discovered.status == cyxwiz::RouteProbeStatus::Passed,
                 "real CPU route discovery did not pass") ||
        !Require(!discovered.routes.empty(),
                 "real CPU route discovery returned no routes") ||
        !Require(discovered.routes.front().type == cyxwiz::DeviceType::CPU,
                 "real CPU route discovery relabeled its backend") ||
        !Require(discovered.routes.front().device_id == 0,
                 "real CPU route discovery lost the backend-local ordinal")) {
        std::cerr << discovered.message << '\n';
        return 1;
    }

    cyxwiz::RouteProbeInvocation fixture_discovery;
    fixture_discovery.executable = environment_probe;
    fixture_discovery.type = cyxwiz::DeviceType::OPENCL;
    fixture_discovery.timeout = std::chrono::seconds(5);
#ifdef _WIN32
    _putenv_s("CYXWIZ_TEST_ROUTE_INVENTORY_MODE", "valid");
#else
    setenv("CYXWIZ_TEST_ROUTE_INVENTORY_MODE", "valid", 1);
#endif
    const auto fixture_routes = cyxwiz::DiscoverIsolatedBackendRoutes(
        fixture_discovery, [] { return false; });
    if (!Require(fixture_routes.status == cyxwiz::RouteProbeStatus::Passed,
                 "valid strict route inventory fixture was rejected") ||
        !Require(fixture_routes.routes.size() == 1,
                 "valid strict route inventory fixture changed route count") ||
        !Require(fixture_routes.routes.front().provider_known &&
                     fixture_routes.routes.front().provider ==
                         "Fixture Provider",
                 "valid strict route inventory lost provider evidence") ||
        !Require(fixture_routes.routes.front().physical_fingerprint_known,
                 "valid strict route inventory lost stable identity")) {
        std::cerr << fixture_routes.message << '\n';
        return 1;
    }
#ifdef _WIN32
    _putenv_s("CYXWIZ_TEST_ROUTE_INVENTORY_MODE", "unknown_field");
#else
    setenv("CYXWIZ_TEST_ROUTE_INVENTORY_MODE", "unknown_field", 1);
#endif
    const auto unknown_field = cyxwiz::DiscoverIsolatedBackendRoutes(
        fixture_discovery, [] { return false; });
#ifdef _WIN32
    _putenv_s("CYXWIZ_TEST_ROUTE_INVENTORY_MODE", "duplicate_id");
#else
    setenv("CYXWIZ_TEST_ROUTE_INVENTORY_MODE", "duplicate_id", 1);
#endif
    const auto duplicate_id = cyxwiz::DiscoverIsolatedBackendRoutes(
        fixture_discovery, [] { return false; });
#ifdef _WIN32
    _putenv_s("CYXWIZ_TEST_ROUTE_INVENTORY_MODE", "");
#else
    unsetenv("CYXWIZ_TEST_ROUTE_INVENTORY_MODE");
#endif
    if (!Require(
            unknown_field.status ==
                cyxwiz::RouteProbeStatus::InfrastructureFailure,
            "route inventory accepted an unknown schema field") ||
        !Require(
            duplicate_id.status ==
                cyxwiz::RouteProbeStatus::InfrastructureFailure,
            "route inventory accepted duplicate backend-local ordinals")) {
        return 1;
    }

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
