#pragma once

#include "user_folders.h"

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

// The machine-local state folder (user_folders.h); empty when unknown.
inline std::filesystem::path DefaultRoot() {
    return UserFile(UserStateDir(), "compute");
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
    return UserFile(GetComputeRuntimeRoot(), "runtime-config.json");
}

inline std::filesystem::path GetRouteQualificationCachePath() {
    return UserFile(GetComputeRuntimeRoot(), "route-qualification.json");
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
