#include "../src/core/installer_verification_summary.h"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

cyxwiz::RouteQualificationRecord PassedRoute(
    cyxwiz::DeviceType type, int device_id, std::string pack_id,
    std::string display_name, double median_ms) {
    cyxwiz::RouteQualificationRecord route;
    route.type = type;
    route.device_id = device_id;
    route.pack_id = std::move(pack_id);
    route.display_name = std::move(display_name);
    route.operation_count = cyxwiz::kRouteQualificationOperationCount;
    route.pass_count = route.operation_count;
    route.certified = true;
    route.benchmark_id = cyxwiz::kRoutePerformanceBenchmarkId;
    route.benchmark_sample_count = 5;
    route.benchmark_iterations_per_sample = 10;
    route.benchmark_median_iteration_ms = median_ms;
    return route;
}

void TestBestMeasuredAndProductionSafeFailureText() {
    cyxwiz::RouteQualificationSnapshot snapshot;
    snapshot.matrix_id =
        "ticket89-local-v2-post-intel-cpu-opencl:crash=0 timeout=1 failed=0";
    snapshot.runtime_set_id = "runtime-v1";
    snapshot.base_pack_id = "base-v1";
    snapshot.routes.push_back(PassedRoute(
        cyxwiz::DeviceType::CPU, 0, "base-v1", "System CPU", 4.0));
    snapshot.routes.push_back(PassedRoute(
        cyxwiz::DeviceType::CUDA, 0, "cuda-v1", "Discrete GPU", 2.0));

    cyxwiz::RouteQualificationRecord failed;
    failed.type = cyxwiz::DeviceType::ONEAPI;
    failed.device_id = 0;
    failed.pack_id = "oneapi-v1";
    failed.display_name = "Integrated GPU";
    failed.operation_count = cyxwiz::kRouteQualificationOperationCount;
    failed.pass_count = failed.operation_count - 1;
    failed.timeout_count = 1;
    failed.failure.stage = cyxwiz::RouteFailureStage::Operation;
    failed.failure.category = cyxwiz::RouteFailureCategory::Timeout;
    failed.failure.observed_fact = snapshot.matrix_id;
    failed.failure.evidence_id = snapshot.matrix_id;
    failed.benchmark_message = snapshot.matrix_id;
    snapshot.routes.push_back(std::move(failed));

    cyxwiz::RuntimeQualificationIdentity active;
    active.runtime_set_id = "runtime-v1";
    active.generation = 3;
    active.base_pack_id = "base-v1";
    active.backend_packs.push_back(
        {cyxwiz::DeviceType::CUDA, "cuda-v1"});

    const auto summary = cyxwiz::BuildInstallerVerificationSummary(
        snapshot, active);
    Check(summary.evidence_matches_runtime,
          "matching runtime evidence should be displayed");
    Check(summary.passed_count == 2 && summary.attention_count == 1,
          "summary should classify passed and failed routes");
    Check(summary.comparable_benchmark_count == 2,
          "two active verified routes should be comparable");
    Check(summary.routes[1].best_measured,
          "lowest comparable median should be the best measured route");
    Check(summary.routes[2].status ==
              cyxwiz::InstallerRouteVerificationStatus::TimedOut,
          "timeout should have a typed user-facing status");

    std::string visible = summary.headline + summary.performance_message;
    for (const auto& route : summary.routes) {
        visible += route.reason;
        visible += route.recommended_action;
    }
    Check(visible.find("ticket89") == std::string::npos,
          "internal ticket keys must not appear in production text");
    Check(visible.find("crash=0") == std::string::npos,
          "raw qualification counters must not be copied into production text");
}

void TestSingleBenchmarkDoesNotClaimFastest() {
    cyxwiz::RouteQualificationSnapshot snapshot;
    snapshot.runtime_set_id = "runtime-v1";
    snapshot.base_pack_id = "base-v1";
    snapshot.routes.push_back(PassedRoute(
        cyxwiz::DeviceType::CPU, 0, "base-v1", "System CPU", 4.0));
    cyxwiz::RuntimeQualificationIdentity active;
    active.runtime_set_id = "runtime-v1";
    active.base_pack_id = "base-v1";

    const auto summary = cyxwiz::BuildInstallerVerificationSummary(
        snapshot, active);
    Check(summary.comparable_benchmark_count == 1,
          "one benchmark should be retained as evidence");
    Check(!summary.routes.front().best_measured,
          "one benchmark must not be labeled best");
    Check(summary.performance_message.find("verify another route") !=
              std::string::npos,
          "single-route evidence should request a comparison");
}

void TestStaleRuntimeEvidenceIsNotPresentedAsCurrent() {
    cyxwiz::RouteQualificationSnapshot snapshot;
    snapshot.runtime_set_id = "old-runtime";
    snapshot.base_pack_id = "old-base";
    snapshot.routes.push_back(PassedRoute(
        cyxwiz::DeviceType::CPU, 0, "old-base", "Old CPU", 1.0));
    cyxwiz::RuntimeQualificationIdentity active;
    active.runtime_set_id = "runtime-v1";
    active.base_pack_id = "base-v1";

    const auto summary = cyxwiz::BuildInstallerVerificationSummary(
        snapshot, active);
    Check(summary.evidence_available && !summary.evidence_matches_runtime,
          "stale evidence should be identified explicitly");
    Check(summary.routes.empty(),
          "stale routes should not be presented as current results");
}

}  // namespace

int main() {
    TestBestMeasuredAndProductionSafeFailureText();
    TestSingleBenchmarkDoesNotClaimFastest();
    TestStaleRuntimeEvidenceIsNotPresentedAsCurrent();
    std::cout << "Installer verification summary tests passed\n";
    return 0;
}
