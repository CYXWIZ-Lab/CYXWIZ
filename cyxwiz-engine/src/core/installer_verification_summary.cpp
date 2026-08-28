#include "installer_verification_summary.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace cyxwiz {
namespace {

const char* BackendName(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "CPU";
        case DeviceType::CUDA: return "CUDA";
        case DeviceType::OPENCL: return "OpenCL";
        case DeviceType::ONEAPI: return "oneAPI";
        default: return "Unknown";
    }
}

std::string ActivePackId(
    const RuntimeQualificationIdentity& active_runtime,
    DeviceType type) {
    if (type == DeviceType::CPU) return active_runtime.base_pack_id;
    const auto pack = std::find_if(
        active_runtime.backend_packs.begin(),
        active_runtime.backend_packs.end(),
        [type](const BackendPackQualificationIdentity& candidate) {
            return candidate.type == type;
        });
    return pack == active_runtime.backend_packs.end()
        ? std::string{} : pack->pack_id;
}

InstallerRouteVerificationStatus StatusFor(
    const RouteQualificationRecord& route) {
    if (route.certified && route.pass_count == route.operation_count &&
        route.failure_count == 0 && route.unavailable_count == 0 &&
        route.timeout_count == 0 && route.crash_count == 0) {
        return InstallerRouteVerificationStatus::Passed;
    }
    if (route.crash_count > 0 ||
        route.failure.category == RouteFailureCategory::ChildProcessCrash) {
        return InstallerRouteVerificationStatus::Crashed;
    }
    if (route.timeout_count > 0 ||
        route.failure.category == RouteFailureCategory::Timeout) {
        return InstallerRouteVerificationStatus::TimedOut;
    }
    if (route.failure_count > 0 || route.unavailable_count > 0 ||
        route.failure.category != RouteFailureCategory::None) {
        return InstallerRouteVerificationStatus::Failed;
    }
    return InstallerRouteVerificationStatus::NotVerified;
}

std::string CountedReason(const RouteQualificationRecord& route,
                          InstallerRouteVerificationStatus status,
                          bool active) {
    std::ostringstream reason;
    switch (status) {
        case InstallerRouteVerificationStatus::Passed:
            reason << "Passed all " << route.operation_count
                   << " required operations";
            if (!active) reason << ", but this route is not active";
            reason << '.';
            break;
        case InstallerRouteVerificationStatus::Crashed:
            reason << route.crash_count
                   << " required operation(s) crashed in an isolated probe.";
            break;
        case InstallerRouteVerificationStatus::TimedOut:
            reason << route.timeout_count
                   << " required operation(s) exceeded the verification timeout.";
            break;
        case InstallerRouteVerificationStatus::Failed:
            if (route.failure_count > 0) {
                reason << route.failure_count
                       << " required operation(s) returned an error.";
            } else if (route.unavailable_count > 0) {
                reason << route.unavailable_count
                       << " required operation(s) were unavailable.";
            } else {
                reason << "The route did not satisfy the released operation contract.";
            }
            break;
        case InstallerRouteVerificationStatus::NotVerified:
            reason << "The retained evidence does not prove the complete operation contract.";
            break;
    }
    return reason.str();
}

std::string ActionFor(InstallerRouteVerificationStatus status) {
    switch (status) {
        case InstallerRouteVerificationStatus::Passed:
            return {};
        case InstallerRouteVerificationStatus::Crashed:
        case InstallerRouteVerificationStatus::TimedOut:
            return "Update the provider or driver and verify again, or use another verified route.";
        case InstallerRouteVerificationStatus::Failed:
            return "Check the provider requirements, update the driver or runtime, and verify again.";
        case InstallerRouteVerificationStatus::NotVerified:
            return "Run Verify All again with the current runtime.";
    }
    return "Run Verify All again.";
}

std::string RouteLabel(const InstallerRouteVerificationResult& route) {
    std::ostringstream label;
    label << route.backend;
    if (!route.display_name.empty()) label << " - " << route.display_name;
    label << " (device " << route.device_id << ')';
    return label.str();
}

}  // namespace

InstallerVerificationSummary BuildInstallerVerificationSummary(
    const std::optional<RouteQualificationSnapshot>& snapshot,
    const RuntimeQualificationIdentity& active_runtime) {
    InstallerVerificationSummary summary;
    if (!snapshot.has_value()) {
        summary.headline =
            "No local verification results are available. Run Verify All to evaluate this machine.";
        summary.performance_message =
            "No comparable performance benchmark is available yet.";
        return summary;
    }

    summary.evidence_available = true;
    summary.evidence_matches_runtime =
        !active_runtime.runtime_set_id.empty() &&
        snapshot->runtime_set_id == active_runtime.runtime_set_id &&
        snapshot->base_pack_id == active_runtime.base_pack_id;
    if (!summary.evidence_matches_runtime) {
        summary.headline =
            "Saved verification results belong to a different runtime. Run Verify All again.";
        summary.performance_message =
            "No current performance comparison is available.";
        return summary;
    }

    for (const auto& record : snapshot->routes) {
        InstallerRouteVerificationResult route;
        route.type = record.type;
        route.device_id = record.device_id;
        route.pack_id = record.pack_id;
        route.backend = BackendName(record.type);
        route.display_name = record.display_name;
        route.active = !record.pack_id.empty() &&
            record.pack_id == ActivePackId(active_runtime, record.type);
        route.status = StatusFor(record);
        route.reason = CountedReason(record, route.status, route.active);
        route.recommended_action = ActionFor(route.status);
        route.benchmark_available =
            route.active &&
            route.status == InstallerRouteVerificationStatus::Passed &&
            record.benchmark_id == kRoutePerformanceBenchmarkId &&
            record.benchmark_sample_count > 0 &&
            record.benchmark_iterations_per_sample > 0 &&
            std::isfinite(record.benchmark_median_iteration_ms) &&
            record.benchmark_median_iteration_ms > 0.0;
        route.benchmark_median_iteration_ms =
            route.benchmark_available
                ? record.benchmark_median_iteration_ms : 0.0;
        if (route.status == InstallerRouteVerificationStatus::Passed) {
            ++summary.passed_count;
        } else {
            ++summary.attention_count;
        }
        summary.routes.push_back(std::move(route));
    }

    summary.comparable_benchmark_count = static_cast<std::size_t>(
        std::count_if(
            summary.routes.begin(), summary.routes.end(),
            [](const InstallerRouteVerificationResult& route) {
                return route.benchmark_available;
            }));
    if (summary.comparable_benchmark_count >= 2) {
        const auto fastest = std::min_element(
            summary.routes.begin(), summary.routes.end(),
            [](const InstallerRouteVerificationResult& left,
               const InstallerRouteVerificationResult& right) {
                if (!left.benchmark_available) return false;
                if (!right.benchmark_available) return true;
                return left.benchmark_median_iteration_ms <
                    right.benchmark_median_iteration_ms;
            });
        fastest->best_measured = true;
        std::ostringstream message;
        message << "Best measured configuration: " << RouteLabel(*fastest)
                << " at " << fastest->benchmark_median_iteration_ms
                << " ms median per benchmark iteration (compared across "
                << summary.comparable_benchmark_count
                << " active verified routes).";
        summary.performance_message = message.str();
    } else if (summary.comparable_benchmark_count == 1) {
        summary.performance_message =
            "One active verified route has benchmark evidence; verify another route before comparing performance.";
    } else {
        summary.performance_message =
            "No comparable performance benchmark is available yet.";
    }

    std::ostringstream headline;
    headline << "Latest local verification: " << summary.passed_count
             << " passed, " << summary.attention_count
             << " need attention.";
    summary.headline = headline.str();
    return summary;
}

const char* InstallerRouteVerificationStatusName(
    InstallerRouteVerificationStatus status) {
    switch (status) {
        case InstallerRouteVerificationStatus::Passed: return "Passed";
        case InstallerRouteVerificationStatus::Failed: return "Failed";
        case InstallerRouteVerificationStatus::TimedOut: return "Timed out";
        case InstallerRouteVerificationStatus::Crashed: return "Crashed";
        case InstallerRouteVerificationStatus::NotVerified:
            return "Not verified";
    }
    return "Not verified";
}

}  // namespace cyxwiz
