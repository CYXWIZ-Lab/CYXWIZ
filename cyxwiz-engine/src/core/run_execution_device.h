#pragma once

// The device a training run uses, chosen the same way for every kind of run
// (supervised training and PPO; TOFIX206): the user's route from Preferences >
// Devices and the next-run execution policy (Strict refuses a route that fails
// its check, Compatibility falls back to the ArrayFire CPU route and says so),
// activated on the calling worker thread, and the bind recorded in the runtime
// log and the training trace with the run's id.

#include "execution_device_preferences.h"

#include <string>

namespace cyxwiz {

// The run's policy: the queued next-run policy when there is one, else
// Strict when `strict_by_default`, else Compatibility.
ExecutionPolicy ResolveRunExecutionPolicy(bool strict_by_default, const char* who);

// Activates the run's route on this thread (PrepareExecutionDeviceForRun).
// Throws with the reason when no route can run; a Compatibility fallback is
// returned with selection_fallback_applied and its reason in `error`.
ExecutionDeviceContext PrepareRunExecutionDevice(ExecutionPolicy policy, const char* who,
                                                 const RouteUnsupportedReason& route_unsupported = {});

// "ExecutionDeviceContext.Bind" in the runtime log (category "device") and the
// training trace, under `run_id`.
void RecordRunExecutionDeviceBind(const ExecutionDeviceContext& context, const std::string& run_id,
                                  const char* who);

}  // namespace cyxwiz
