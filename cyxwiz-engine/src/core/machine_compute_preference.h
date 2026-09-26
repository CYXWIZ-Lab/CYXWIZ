#pragma once

// This machine's saved compute preference (runtime-config.json, written by
// Preferences > Devices): the preferred verified route and the fallback
// policy. The Engine applies it at startup; headless hosts (Server Node
// daemon, training profiler) apply it the same way so they train on the route
// the machine verified instead of a default one (TOFIX118 P3/P8).

#include "compute_runtime_config.h"
#include "compute_runtime_paths.h"
#include "execution_device_preferences.h"

namespace cyxwiz {

// Applies the preference when the file loads; returns the load result (its
// message says why it was not applied). Never creates the file.
inline ComputeRuntimeConfigLoadResult ApplyMachineComputePreference() {
    auto result = LoadComputeRuntimeConfig(GetComputeRuntimeConfigPath());
    if (!result.loaded) return result;
    SetNextRunExecutionPolicy(result.config.default_fallback_policy);
    if (result.config.preferred_route.has_value()) {
        const auto& preferred = *result.config.preferred_route;
        CommitExecutionDeviceSelectionState(
            {preferred.type, preferred.last_device_id, preferred.physical_fingerprint});
    }
    return result;
}

}  // namespace cyxwiz
