#pragma once

#include "algorithms/arrayfire_backend_utils.h"

#include <cyxwiz/device.h>

#include <filesystem>
#include <optional>
#include <string>

namespace cyxwiz {

struct PreferredComputeRoute {
    DeviceType type = DeviceType::CPU;
    int last_device_id = 0;
    std::string physical_fingerprint;
};

struct ComputeRuntimeConfig {
    int schema = 1;
    std::optional<PreferredComputeRoute> preferred_route;
    ArrayFireFallbackPolicy default_fallback_policy =
        ArrayFireFallbackPolicy::AllowNativeCpuFallback;
};

struct ComputeRuntimeConfigLoadResult {
    bool loaded = false;
    bool file_exists = false;
    ComputeRuntimeConfig config;
    std::string message;
};

ComputeRuntimeConfigLoadResult LoadComputeRuntimeConfig(
    const std::filesystem::path& path);
bool SaveComputeRuntimeConfigAtomic(
    const std::filesystem::path& path,
    const ComputeRuntimeConfig& config,
    std::string& error);
bool UpdatePreferredComputeRouteAtomic(
    const std::filesystem::path& path,
    const PreferredComputeRoute& route,
    std::string& error);
bool UpdateDefaultFallbackPolicyAtomic(
    const std::filesystem::path& path,
    ArrayFireFallbackPolicy policy,
    std::string& error);

} // namespace cyxwiz
