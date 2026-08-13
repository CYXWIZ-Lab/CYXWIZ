#include "compute_runtime_config.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <fstream>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace cyxwiz {
namespace {

std::optional<DeviceType> ParseBackend(const std::string& backend) {
    if (backend == "cpu") return DeviceType::CPU;
    if (backend == "cuda") return DeviceType::CUDA;
    if (backend == "opencl") return DeviceType::OPENCL;
    if (backend == "oneapi") return DeviceType::ONEAPI;
    return std::nullopt;
}

const char* BackendName(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "cpu";
        case DeviceType::CUDA: return "cuda";
        case DeviceType::OPENCL: return "opencl";
        case DeviceType::ONEAPI: return "oneapi";
        default: return "unsupported";
    }
}

const char* FallbackPolicyName(ArrayFireFallbackPolicy policy) {
    return policy == ArrayFireFallbackPolicy::ForbidNativeCpuFallback
        ? "forbid_native_cpu_fallback"
        : "allow_native_cpu_fallback";
}

std::optional<ArrayFireFallbackPolicy> ParseFallbackPolicy(
    const std::string& policy) {
    if (policy == "allow_native_cpu_fallback") {
        return ArrayFireFallbackPolicy::AllowNativeCpuFallback;
    }
    if (policy == "forbid_native_cpu_fallback") {
        return ArrayFireFallbackPolicy::ForbidNativeCpuFallback;
    }
    return std::nullopt;
}

bool PublishAtomic(const std::filesystem::path& temporary,
                   const std::filesystem::path& target,
                   std::string& error) {
#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(), target.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        error = "could not publish compute runtime config atomically: " +
                std::system_category().message(
                    static_cast<int>(GetLastError()));
        return false;
    }
    return true;
#else
    std::error_code ec;
    std::filesystem::rename(temporary, target, ec);
    if (ec) {
        error = "could not publish compute runtime config atomically: " +
                ec.message();
        return false;
    }
    return true;
#endif
}

} // namespace

ComputeRuntimeConfigLoadResult LoadComputeRuntimeConfig(
    const std::filesystem::path& path) {
    ComputeRuntimeConfigLoadResult result;
    std::error_code ec;
    result.file_exists = std::filesystem::exists(path, ec) && !ec;
    if (!result.file_exists) {
        result.message = "Compute runtime configuration not found: " +
                         path.string();
        return result;
    }

    try {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("could not open file");
        nlohmann::json document;
        input >> document;
        if (!document.is_object() || !document.contains("schema") ||
            !document.at("schema").is_number_integer() ||
            document.at("schema").get<int>() != 1) {
            throw std::runtime_error(
                "unsupported compute runtime configuration schema");
        }

        ComputeRuntimeConfig config;
        if (!document.contains("default_fallback_policy") ||
            !document.at("default_fallback_policy").is_string()) {
            throw std::runtime_error(
                "default_fallback_policy must be a string");
        }
        const auto policy = ParseFallbackPolicy(
            document.at("default_fallback_policy").get<std::string>());
        if (!policy.has_value()) {
            throw std::runtime_error("unknown default_fallback_policy");
        }
        config.default_fallback_policy = *policy;

        if (document.contains("preferred_route") &&
            !document.at("preferred_route").is_null()) {
            const auto& route = document.at("preferred_route");
            if (!route.is_object() || !route.contains("backend") ||
                !route.at("backend").is_string() ||
                !route.contains("last_device_id") ||
                !route.at("last_device_id").is_number_integer()) {
                throw std::runtime_error("preferred_route is malformed");
            }
            const auto type = ParseBackend(
                route.at("backend").get<std::string>());
            const int device_id = route.at("last_device_id").get<int>();
            if (!type.has_value() || device_id < 0) {
                throw std::runtime_error(
                    "preferred_route backend or device ID is invalid");
            }
            PreferredComputeRoute preferred;
            preferred.type = *type;
            preferred.last_device_id = device_id;
            if (route.contains("physical_fingerprint") &&
                !route.at("physical_fingerprint").is_null()) {
                if (!route.at("physical_fingerprint").is_string()) {
                    throw std::runtime_error(
                        "physical_fingerprint must be a string or null");
                }
                preferred.physical_fingerprint =
                    route.at("physical_fingerprint").get<std::string>();
            }
            config.preferred_route = std::move(preferred);
        }

        result.config = std::move(config);
        result.loaded = true;
        result.message = "Compute runtime configuration loaded";
    } catch (const std::exception& exception) {
        result.message = "Compute runtime configuration is invalid: " +
                         std::string(exception.what());
    }
    return result;
}

bool SaveComputeRuntimeConfigAtomic(
    const std::filesystem::path& path,
    const ComputeRuntimeConfig& config,
    std::string& error) {
    error.clear();
    if (config.schema != 1) {
        error = "unsupported compute runtime configuration schema";
        return false;
    }
    if (config.preferred_route.has_value() &&
        (config.preferred_route->last_device_id < 0 ||
         std::string(BackendName(config.preferred_route->type)) ==
             "unsupported")) {
        error = "preferred compute route is invalid";
        return false;
    }

    nlohmann::json document = {
        {"schema", 1},
        {"default_fallback_policy",
         FallbackPolicyName(config.default_fallback_policy)}};
    if (config.preferred_route.has_value()) {
        const auto& route = *config.preferred_route;
        document["preferred_route"] = {
            {"backend", BackendName(route.type)},
            {"last_device_id", route.last_device_id},
            {"physical_fingerprint",
             route.physical_fingerprint.empty()
                 ? nlohmann::json(nullptr)
                 : nlohmann::json(route.physical_fingerprint)}};
    } else {
        document["preferred_route"] = nullptr;
    }

    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
        error = "could not create compute runtime directory: " + ec.message();
        return false;
    }

    const auto nonce =
        std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path temporary = path;
    temporary += ".tmp." + std::to_string(nonce);
    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            error = "could not open temporary compute runtime config";
            return false;
        }
        output << document.dump(2) << '\n';
        output.flush();
        if (!output.good()) {
            error = "could not write temporary compute runtime config";
            output.close();
            std::filesystem::remove(temporary, ec);
            return false;
        }
        output.close();
        if (!PublishAtomic(temporary, path, error)) {
            std::filesystem::remove(temporary, ec);
            return false;
        }
        return true;
    } catch (const std::exception& exception) {
        error = "compute runtime configuration save failed: " +
                std::string(exception.what());
        std::filesystem::remove(temporary, ec);
        return false;
    }
}

bool UpdatePreferredComputeRouteAtomic(
    const std::filesystem::path& path,
    const PreferredComputeRoute& route,
    std::string& error) {
    const auto loaded = LoadComputeRuntimeConfig(path);
    if (loaded.file_exists && !loaded.loaded) {
        error = loaded.message;
        return false;
    }
    ComputeRuntimeConfig config = loaded.loaded
        ? loaded.config
        : ComputeRuntimeConfig{};
    config.preferred_route = route;
    return SaveComputeRuntimeConfigAtomic(path, config, error);
}

bool UpdateDefaultFallbackPolicyAtomic(
    const std::filesystem::path& path,
    ArrayFireFallbackPolicy policy,
    std::string& error) {
    const auto loaded = LoadComputeRuntimeConfig(path);
    if (loaded.file_exists && !loaded.loaded) {
        error = loaded.message;
        return false;
    }
    ComputeRuntimeConfig config = loaded.loaded
        ? loaded.config
        : ComputeRuntimeConfig{};
    config.default_fallback_policy = policy;
    return SaveComputeRuntimeConfigAtomic(path, config, error);
}

} // namespace cyxwiz
