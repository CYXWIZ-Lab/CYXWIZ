#include "route_recommendation.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {
namespace {

std::string Lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return value;
}

bool IsSameRoute(const DeviceInfo& left, const DeviceInfo& right) {
    return left.type == right.type && left.device_id == right.device_id;
}

bool HasStableSameDeviceIdentity(const DeviceInfo& left,
                                 const DeviceInfo& right) {
    return left.kind != DeviceKind::Unknown && left.kind == right.kind &&
           left.identity_confidence ==
               DeviceIdentityConfidence::StableHardware &&
           right.identity_confidence ==
               DeviceIdentityConfidence::StableHardware &&
           left.physical_fingerprint_known &&
           right.physical_fingerprint_known &&
           Lower(left.physical_fingerprint) ==
               Lower(right.physical_fingerprint);
}

std::optional<DeviceType> SameDeviceAlternativeBackend(
    const DeviceInfo& failed_route) {
    switch (failed_route.type) {
        case DeviceType::CUDA:
            return DeviceType::OPENCL;
        case DeviceType::ONEAPI:
            return DeviceType::OPENCL;
        case DeviceType::OPENCL:
            if (failed_route.kind == DeviceKind::GPU &&
                failed_route.hardware_vendor_id_known &&
                failed_route.hardware_vendor_id == 0x10de) {
                return DeviceType::CUDA;
            }
            if (failed_route.kind == DeviceKind::GPU ||
                failed_route.kind == DeviceKind::CPU) {
                return DeviceType::ONEAPI;
            }
            return std::nullopt;
        default:
            return std::nullopt;
    }
}

void Reject(RouteRecommendationResult& result, const DeviceInfo& route,
            std::string reason) {
    result.rejections.push_back({route, std::move(reason)});
}

}  // namespace

RouteRecommendationResult RecommendExecutionRoutes(
    const DeviceInfo& failed_route,
    const std::vector<DeviceInfo>& inventory,
    const std::optional<RouteQualificationSnapshot>& qualification) {
    RouteRecommendationResult result;
    const auto same_device_backend =
        SameDeviceAlternativeBackend(failed_route);

    for (const auto& candidate : inventory) {
        if (IsSameRoute(candidate, failed_route)) {
            Reject(result, candidate, "This is the route that failed");
            continue;
        }

        const bool same_device_candidate =
            same_device_backend.has_value() &&
            candidate.type == *same_device_backend;
        const bool cpu_recovery = candidate.type == DeviceType::CPU &&
                                  candidate.kind == DeviceKind::CPU;
        if (!same_device_candidate && !cpu_recovery) {
            Reject(result, candidate,
                   "Backend is not an eligible recovery for the failed route");
            continue;
        }

        if (same_device_candidate &&
            !HasStableSameDeviceIdentity(failed_route, candidate)) {
            Reject(result, candidate,
                   "Stable physical identity does not prove the same device");
            continue;
        }

        const auto decision =
            EvaluateRouteQualification(candidate, qualification);
        if (!decision.qualified) {
            Reject(result, candidate, decision.message);
            continue;
        }

        RouteRecommendation recommendation;
        recommendation.route = candidate;
        if (same_device_candidate) {
            recommendation.remediation =
                RouteRecommendationClass::SamePhysicalDevice;
            recommendation.reason =
                "Certified alternative backend on the same physical device";
        } else {
            recommendation.remediation =
                RouteRecommendationClass::ArrayFireCpuRecovery;
            recommendation.reason =
                "Certified ArrayFire CPU recovery on a different device";
        }
        result.recommendations.push_back(std::move(recommendation));
    }

    std::stable_sort(
        result.recommendations.begin(), result.recommendations.end(),
        [](const RouteRecommendation& left,
           const RouteRecommendation& right) {
            return left.remediation ==
                       RouteRecommendationClass::SamePhysicalDevice &&
                   right.remediation !=
                       RouteRecommendationClass::SamePhysicalDevice;
        });
    return result;
}

std::optional<RoutePerformanceRecommendation>
RecommendFastestVerifiedRoute(
    const std::optional<RouteQualificationSnapshot>& qualification) {
    if (!qualification.has_value()) return std::nullopt;

    const RouteQualificationRecord* fastest = nullptr;
    for (const auto& route : qualification->routes) {
        if (!route.certified ||
            route.benchmark_id != kRoutePerformanceBenchmarkId ||
            route.benchmark_sample_count <= 0 ||
            route.benchmark_iterations_per_sample <= 0 ||
            !(route.benchmark_median_iteration_ms > 0.0)) {
            continue;
        }
        if (!fastest || route.benchmark_median_iteration_ms <
                            fastest->benchmark_median_iteration_ms) {
            fastest = &route;
        }
    }
    if (!fastest) return std::nullopt;

    RoutePerformanceRecommendation recommendation;
    recommendation.type = fastest->type;
    recommendation.device_id = fastest->device_id;
    recommendation.display_name = fastest->display_name;
    recommendation.benchmark_id = fastest->benchmark_id;
    recommendation.sample_count = fastest->benchmark_sample_count;
    recommendation.iterations_per_sample =
        fastest->benchmark_iterations_per_sample;
    recommendation.median_iteration_ms =
        fastest->benchmark_median_iteration_ms;
    return recommendation;
}

}  // namespace cyxwiz
