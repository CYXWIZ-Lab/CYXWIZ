#pragma once

#include "compute_runtime_paths.h"

#include <cyxwiz/backend_placement_observation.h>

#include <spdlog/spdlog.h>

#include <filesystem>
#include <string>

namespace cyxwiz {

// Session lifecycle for the persistent backend placement observation cache
// (tofix67 slice 1). The compiler and runtime read/write the process-global
// observation store directly; these helpers only bridge that store to disk so
// runtime fallback and preflight-probe evidence survives across engine runs.

// Merge the persisted cache into the process-global store. Call once at
// engine startup, before the first graph compile. A missing file is a normal
// first run, not an error.
inline void LoadPlacementObservationCacheAtStartup() {
    const auto path = GetPlacementObservationCachePath();
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return;
    }
    std::string error;
    if (LoadBackendPlacementObservationCache(path.string(), &error)) {
        spdlog::info(
            "Placement observation cache loaded: {} ({} observations)",
            path.string(),
            SnapshotBackendPlacementObservations().size());
    } else {
        spdlog::warn("Placement observation cache load failed ({}): {}",
                     path.string(), error);
    }
}

// Persist the process-global store so the next engine run's compiles see this
// run's evidence. Call after a training run ends and at engine shutdown. An
// empty store is a no-op so a clean run never truncates evidence persisted by
// an earlier session.
inline void SavePlacementObservationCache() {
    if (SnapshotBackendPlacementObservations().empty()) {
        return;
    }
    const auto path = GetPlacementObservationCachePath();
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::string error;
    if (!SaveBackendPlacementObservationCache(path.string(), &error)) {
        spdlog::warn("Placement observation cache save failed ({}): {}",
                     path.string(), error);
    }
}

} // namespace cyxwiz
