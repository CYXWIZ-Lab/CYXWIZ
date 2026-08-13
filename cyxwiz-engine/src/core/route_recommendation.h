#pragma once

#include "route_qualification_snapshot.h"

#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

enum class RouteRecommendationClass {
    SamePhysicalDevice,
    ArrayFireCpuRecovery
};

struct RouteRecommendation {
    DeviceInfo route;
    RouteRecommendationClass remediation =
        RouteRecommendationClass::ArrayFireCpuRecovery;
    std::string reason;
};

struct RouteRecommendationRejection {
    DeviceInfo route;
    std::string reason;
};

struct RouteRecommendationResult {
    std::vector<RouteRecommendation> recommendations;
    std::vector<RouteRecommendationRejection> rejections;
};

struct RoutePerformanceRecommendation {
    DeviceType type = DeviceType::CPU;
    int device_id = 0;
    std::string display_name;
    std::string benchmark_id;
    int sample_count = 0;
    int iterations_per_sample = 0;
    double median_iteration_ms = 0.0;
};

RouteRecommendationResult RecommendExecutionRoutes(
    const DeviceInfo& failed_route,
    const std::vector<DeviceInfo>& inventory,
    const std::optional<RouteQualificationSnapshot>& qualification);

std::optional<RoutePerformanceRecommendation>
RecommendFastestVerifiedRoute(
    const std::optional<RouteQualificationSnapshot>& qualification);

}  // namespace cyxwiz
