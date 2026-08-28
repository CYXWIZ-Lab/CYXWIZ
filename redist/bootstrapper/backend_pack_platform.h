#pragma once

#include <string_view>

namespace cyxwiz::runtime {

constexpr std::string_view CurrentBackendPackPlatformId() {
#ifdef _WIN32
    return "win64";
#elif defined(__APPLE__)
    return "macos";
#elif defined(__linux__)
    return "linux64";
#else
    return {};
#endif
}

constexpr std::string_view CurrentBackendPackArchitectureId() {
#if defined(_M_ARM64) || defined(__aarch64__)
    return "arm64";
#elif defined(_M_X64) || defined(__x86_64__)
    return "x86_64";
#else
    return {};
#endif
}

constexpr std::string_view CurrentEngineExecutableName() {
#ifdef _WIN32
    return "cyxwiz-engine.exe";
#else
    return "cyxwiz-engine";
#endif
}

constexpr std::string_view CurrentRouteProbeExecutableName() {
#ifdef _WIN32
    return "cyxwiz-route-probe.exe";
#else
    return "cyxwiz-route-probe";
#endif
}

constexpr std::string_view CurrentBackendPackInstallerExecutableName() {
#ifdef _WIN32
    return "cyxwiz-backend-pack-installer.exe";
#else
    return "cyxwiz-backend-pack-installer";
#endif
}

constexpr std::string_view CurrentRuntimeBootstrapperExecutableName() {
#ifdef _WIN32
    return "cyxwiz-runtime-bootstrapper.exe";
#else
    return "cyxwiz-runtime-bootstrapper";
#endif
}

constexpr std::string_view CurrentProductRemovalFinalizerExecutableName() {
#ifdef _WIN32
    return "cyxwiz-product-removal-finalizer.exe";
#else
    return "cyxwiz-product-removal-finalizer";
#endif
}

constexpr std::string_view CurrentInstallerManagerExecutableName() {
#ifdef _WIN32
    return "cyxwiz-installer.exe";
#else
    return "cyxwiz-installer";
#endif
}

}  // namespace cyxwiz::runtime
