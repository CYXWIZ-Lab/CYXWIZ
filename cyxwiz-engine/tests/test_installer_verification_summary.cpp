#include "../src/core/backend_pack_decision_reconciliation.h"

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

cyxwiz::BackendPackManagerRecord ActivePack(std::string pack_id,
                                            std::string backend) {
    cyxwiz::BackendPackManagerRecord record;
    record.pack_id = std::move(pack_id);
    record.installed_pack_id = record.pack_id;
    record.backend = std::move(backend);
    record.catalog_support =
        cyxwiz::BackendPackCatalogSupport::Supported;
    record.installed = true;
    record.active = true;
    record.compatibility.emplace();
    record.compatibility->catalog_support =
        cyxwiz::runtime::BackendPackSupportStatus::Supported;
    record.compatibility->eligibility =
        cyxwiz::runtime::BackendPackEligibility::Compatible;
    record.compatibility->recommendation_target_eligible = true;
    record.compatibility->install_recommendation = cyxwiz::runtime::
        BackendPackInstallRecommendation::AvailableAfterVerification;
    return record;
}

void TestVerificationReconcilesWithoutChangingCompatibilityPolicy() {
    cyxwiz::RouteQualificationSnapshot snapshot;
    snapshot.runtime_set_id = "runtime-v1";
    snapshot.base_pack_id = "base-v1";
    snapshot.routes.push_back(PassedRoute(
        cyxwiz::DeviceType::CPU, 0, "base-v1", "System CPU", 4.0));
    snapshot.routes.push_back(PassedRoute(
        cyxwiz::DeviceType::CUDA, 0, "cuda-v1", "Discrete GPU", 2.0));

    cyxwiz::RouteQualificationRecord crashed;
    crashed.type = cyxwiz::DeviceType::ONEAPI;
    crashed.device_id = 0;
    crashed.pack_id = "oneapi-v1";
    crashed.operation_count = cyxwiz::kRouteQualificationOperationCount;
    crashed.pass_count = crashed.operation_count - 1;
    crashed.crash_count = 1;
    crashed.failure.category =
        cyxwiz::RouteFailureCategory::ChildProcessCrash;
    snapshot.routes.push_back(std::move(crashed));

    cyxwiz::RuntimeQualificationIdentity active;
    active.runtime_set_id = "runtime-v1";
    active.base_pack_id = "base-v1";
    active.backend_packs.push_back(
        {cyxwiz::DeviceType::CUDA, "cuda-v1"});
    active.backend_packs.push_back(
        {cyxwiz::DeviceType::ONEAPI, "oneapi-v1"});
    const auto summary = cyxwiz::BuildInstallerVerificationSummary(
        snapshot, active);

    std::vector records{
        ActivePack("base-v1", "cpu"), ActivePack("cuda-v1", "cuda"),
        ActivePack("oneapi-v1", "oneapi")};
    cyxwiz::ReconcileBackendPackDecisionEvidence(records, summary);

    const auto& cuda = *records[1].compatibility;
    Check(cuda.verification_status ==
              cyxwiz::runtime::BackendPackRouteVerificationStatus::Passed &&
              cuda.training_authorization == cyxwiz::runtime::
                  BackendPackTrainingAuthorizationStatus::Authorized &&
              cuda.performance_status == cyxwiz::runtime::
                  BackendPackPerformanceStatus::PreferredMeasured &&
              cuda.install_recommendation == cyxwiz::runtime::
                  BackendPackInstallRecommendation::Recommended &&
              records[1].training_authorized,
          "the best comparable verified active pack should be recommended");

    const auto& oneapi = *records[2].compatibility;
    Check(oneapi.verification_status ==
              cyxwiz::runtime::BackendPackRouteVerificationStatus::Crashed &&
              oneapi.training_authorization == cyxwiz::runtime::
                  BackendPackTrainingAuthorizationStatus::Rejected &&
              oneapi.install_recommendation == cyxwiz::runtime::
                  BackendPackInstallRecommendation::AvailableAfterVerification &&
              oneapi.eligibility ==
                  cyxwiz::runtime::BackendPackEligibility::Compatible &&
              !records[2].training_authorized,
          "a local crash must reject that route without globally blocking the "
          "signed oneAPI pack");

    records[1].compatibility->recommendation_target_eligible = false;
    cyxwiz::ReconcileBackendPackDecisionEvidence(records, summary);
    Check(records[1].compatibility->performance_status == cyxwiz::runtime::
              BackendPackPerformanceStatus::PreferredMeasured &&
              records[1].compatibility->install_recommendation ==
                  cyxwiz::runtime::BackendPackInstallRecommendation::
                      AvailableAfterVerification,
          "benchmark evidence must not override signed recommendation "
          "eligibility");

    auto stale_summary = summary;
    stale_summary.evidence_matches_runtime = false;
    records[1].compatibility->recommendation_target_eligible = true;
    cyxwiz::ReconcileBackendPackDecisionEvidence(records, stale_summary);
    Check(records[1].compatibility->verification_status == cyxwiz::runtime::
              BackendPackRouteVerificationStatus::Stale &&
              records[1].compatibility->training_authorization == cyxwiz::
                  runtime::BackendPackTrainingAuthorizationStatus::
                      NotEvaluated &&
              records[1].compatibility->performance_status == cyxwiz::runtime::
                  BackendPackPerformanceStatus::NotMeasured &&
              records[1].compatibility->install_recommendation ==
                  cyxwiz::runtime::BackendPackInstallRecommendation::
                      AvailableAfterVerification,
          "stale evidence must clear local authorization and recommendation");
}

}  // namespace

int main() {
    TestBestMeasuredAndProductionSafeFailureText();
    TestSingleBenchmarkDoesNotClaimFastest();
    TestStaleRuntimeEvidenceIsNotPresentedAsCurrent();
    TestVerificationReconcilesWithoutChangingCompatibilityPolicy();
    std::cout << "Installer verification summary tests passed\n";
    return 0;
}
