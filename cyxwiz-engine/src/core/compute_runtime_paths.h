#pragma once

#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <optional>
#include <utility>

namespace cyxwiz {

namespace compute_runtime_paths_detail {

inline std::mutex& RootMutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::optional<std::filesystem::path>& TestRootOverride() {
    static std::optional<std::filesystem::path> root;
    return root;
}

inline std::filesystem::path EnvironmentPath(const char* name) {
    const char* value = std::getenv(name);
    return value && *value ? std::filesystem::path(value)
                           : std::filesystem::path{};
}

inline std::filesystem::path DefaultRoot() {
#ifdef _WIN32
    auto base = EnvironmentPath("LOCALAPPDATA");
    if (base.empty()) base = EnvironmentPath("APPDATA");
    if (!base.empty()) return base / "CyxWiz" / "compute";
#elif defined(__APPLE__)
    const auto home = EnvironmentPath("HOME");
    if (!home.empty()) {
        return home / "Library" / "Application Support" / "CyxWiz" /
               "compute";
    }
#else
    auto base = EnvironmentPath("XDG_STATE_HOME");
    if (base.empty()) {
        const auto home = EnvironmentPath("HOME");
        if (!home.empty()) base = home / ".local" / "state";
    }
    if (!base.empty()) return base / "cyxwiz" / "compute";
#endif

    std::error_code ec;
    const auto temp = std::filesystem::temp_directory_path(ec);
    return (ec ? std::filesystem::path(".") : temp) / "cyxwiz" /
           "compute";
}

} // namespace compute_runtime_paths_detail

inline std::filesystem::path GetComputeRuntimeRoot() {
    std::lock_guard<std::mutex> lock(
        compute_runtime_paths_detail::RootMutex());
    const auto& override_root =
        compute_runtime_paths_detail::TestRootOverride();
    if (override_root.has_value()) return *override_root;
    static const std::filesystem::path root =
        compute_runtime_paths_detail::DefaultRoot().lexically_normal();
    return root;
}

inline std::filesystem::path GetComputeRuntimeConfigPath() {
    return GetComputeRuntimeRoot() / "runtime-config.json";
}

inline std::filesystem::path GetRouteQualificationCachePath() {
    return GetComputeRuntimeRoot() / "route-qualification.json";
}

class ScopedComputeRuntimeRootOverrideForTesting {
public:
    explicit ScopedComputeRuntimeRootOverrideForTesting(
        std::filesystem::path root) {
        std::error_code ec;
        auto absolute_root = std::filesystem::absolute(root, ec);
        if (ec) absolute_root = std::move(root);

        std::lock_guard<std::mutex> lock(
            compute_runtime_paths_detail::RootMutex());
        previous_ = compute_runtime_paths_detail::TestRootOverride();
        compute_runtime_paths_detail::TestRootOverride() =
            absolute_root.lexically_normal();
    }

    ~ScopedComputeRuntimeRootOverrideForTesting() {
        std::lock_guard<std::mutex> lock(
            compute_runtime_paths_detail::RootMutex());
        compute_runtime_paths_detail::TestRootOverride() =
            std::move(previous_);
    }

    ScopedComputeRuntimeRootOverrideForTesting(
        const ScopedComputeRuntimeRootOverrideForTesting&) = delete;
    ScopedComputeRuntimeRootOverrideForTesting& operator=(
        const ScopedComputeRuntimeRootOverrideForTesting&) = delete;

private:
    std::optional<std::filesystem::path> previous_;
};

} // namespace cyxwiz
