#pragma once

#include "route_qualification_service.h"

#include "backend_pack_lifecycle_service.h"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>

namespace cyxwiz {

struct BackendPackQualificationAdapterOptions {
    std::filesystem::path runtime_root;
    std::filesystem::path probe_executable;
    std::filesystem::path cache_path;
    RuntimeQualificationFailurePolicy failure_policy =
        RuntimeQualificationFailurePolicy::KeepInstalledUnqualified;
    std::chrono::milliseconds operation_timeout{20000};
    std::chrono::milliseconds discovery_timeout{20000};
    std::size_t output_limit_bytes = 64 * 1024;
};

using BackendPackRouteDiscovery = std::function<
    IsolatedRouteDiscoveryResult(
        RouteProbeInvocation,
        const RouteQualificationCancelCheck&)>;

runtime::BackendPackQualificationHook CreateBackendPackQualificationHook(
    std::shared_ptr<RouteQualificationService> service,
    BackendPackQualificationAdapterOptions options,
    BackendPackRouteDiscovery discover = {});

}  // namespace cyxwiz
