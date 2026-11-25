#pragma once

#include "user_folders.h"

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

inline std::filesystem::path DefaultRoot() {
    // A test run (or anyone) can point run records elsewhere, so tests never
    // write the user's last training trace or debug runs.
    if (const char* overridden = std::getenv("CYXWIZ_DEBUG_RUN_ROOT"); overridden && *overridden) {
        return std::filesystem::path(overridden);
    }
    // The Engine's state folder (user_folders.h); empty when unknown.
    return UserFile(EngineStateDir(), "debug_runs");
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

// The Studio Debugger's run store: one folder per run (session.json, layer
// captures).
inline std::filesystem::path GetDebugRunStoreRoot() {
    return UserFile(GetDebugRunRoot(), "studio");
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
