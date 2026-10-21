#pragma once

#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <optional>
#include <utility>

namespace cyxwiz {

namespace debug_run_paths_detail {

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
    if (base.empty()) {
        base = EnvironmentPath("APPDATA");
    }
    if (!base.empty()) {
        return base / "CyxWiz" / "Engine" / "debug_runs";
    }
#elif defined(__APPLE__)
    const auto home = EnvironmentPath("HOME");
    if (!home.empty()) {
        return home / "Library" / "Application Support" / "CyxWiz" /
               "Engine" / "debug_runs";
    }
#else
    auto base = EnvironmentPath("XDG_STATE_HOME");
    if (base.empty()) {
        const auto home = EnvironmentPath("HOME");
        if (!home.empty()) {
            base = home / ".local" / "state";
        }
    }
    if (!base.empty()) {
        return base / "cyxwiz" / "engine" / "debug_runs";
    }
#endif

    std::error_code ec;
    const auto temp = std::filesystem::temp_directory_path(ec);
    return (ec ? std::filesystem::path(".") : temp) /
           "cyxwiz" / "engine" / "debug_runs";
}

} // namespace debug_run_paths_detail

inline std::filesystem::path GetDebugRunRoot() {
    std::lock_guard<std::mutex> lock(debug_run_paths_detail::RootMutex());
    const auto& override_root = debug_run_paths_detail::TestRootOverride();
    if (override_root.has_value()) {
        return *override_root;
    }
    static const std::filesystem::path root =
        debug_run_paths_detail::DefaultRoot().lexically_normal();
    return root;
}

class ScopedDebugRunRootOverrideForTesting {
public:
    explicit ScopedDebugRunRootOverrideForTesting(
        std::filesystem::path root) {
        std::error_code ec;
        auto absolute_root = std::filesystem::absolute(root, ec);
        if (ec) {
            absolute_root = std::move(root);
        }

        std::lock_guard<std::mutex> lock(debug_run_paths_detail::RootMutex());
        previous_ = debug_run_paths_detail::TestRootOverride();
        debug_run_paths_detail::TestRootOverride() =
            absolute_root.lexically_normal();
    }

    ~ScopedDebugRunRootOverrideForTesting() {
        std::lock_guard<std::mutex> lock(debug_run_paths_detail::RootMutex());
        debug_run_paths_detail::TestRootOverride() = std::move(previous_);
    }

    ScopedDebugRunRootOverrideForTesting(
        const ScopedDebugRunRootOverrideForTesting&) = delete;
    ScopedDebugRunRootOverrideForTesting& operator=(
        const ScopedDebugRunRootOverrideForTesting&) = delete;

private:
    std::optional<std::filesystem::path> previous_;
};

} // namespace cyxwiz
