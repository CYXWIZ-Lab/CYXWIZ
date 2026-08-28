#pragma once

#include "route_qualification_snapshot.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

enum class InstallerRouteVerificationStatus {
    Passed,
    Failed,
    TimedOut,
    Crashed,
    NotVerified
};

struct InstallerRouteVerificationResult {
    DeviceType type = DeviceType::CPU;
    int device_id = 0;
    std::string pack_id;
    std::string backend;
    std::string display_name;
    InstallerRouteVerificationStatus status =
        InstallerRouteVerificationStatus::NotVerified;
    std::string reason;
    std::string recommended_action;
    bool active = false;
    bool benchmark_available = false;
    double benchmark_median_iteration_ms = 0.0;
    bool best_measured = false;
};

struct InstallerVerificationSummary {
    bool evidence_available = false;
    bool evidence_matches_runtime = false;
    std::string headline;
    std::string performance_message;
    std::size_t passed_count = 0;
    std::size_t attention_count = 0;
    std::size_t comparable_benchmark_count = 0;
    std::vector<InstallerRouteVerificationResult> routes;
};

InstallerVerificationSummary BuildInstallerVerificationSummary(
    const std::optional<RouteQualificationSnapshot>& snapshot,
    const RuntimeQualificationIdentity& active_runtime);

const char* InstallerRouteVerificationStatusName(
    InstallerRouteVerificationStatus status);

}  // namespace cyxwiz
