#pragma once

#include "execution_policy.h"

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
    // 2: default_execution_policy (compatibility / strict), TOFIX140 F9.
    int schema = 2;
    std::optional<PreferredComputeRoute> preferred_route;
    ExecutionPolicy default_execution_policy =
        ExecutionPolicy::Compatibility;
};

struct ComputeRuntimeConfigLoadResult {
    bool loaded = false;
    bool file_exists = false;
    // The file is valid JSON written by an older Engine (a lower schema). Its
    // content is not read; the next saved choice replaces the file.
    bool older_schema = false;
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
bool UpdateDefaultExecutionPolicyAtomic(
    const std::filesystem::path& path,
    ExecutionPolicy policy,
    std::string& error);

} // namespace cyxwiz
