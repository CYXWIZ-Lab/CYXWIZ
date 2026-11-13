#pragma once

// The compute route a node trains jobs on, for admission and for the job's
// own thread.
//
// ArrayFire's active backend is per thread. The daemon activates its route at
// startup on the main thread; a gRPC handler or job thread that never
// activated one reports ArrayFire's default (oneAPI on an Intel-only machine,
// found 2026-11-11 on the Dell: admission saw arrayfire_oneapi:0 while the
// node had activated and verified arrayfire_opencl:0). So the route comes
// from what the process chose, never from Device::GetCurrentDevice().

#include "core/execution_device_preferences.h"

#include <cyxwiz/device.h>

#include <optional>

namespace cyxwiz::servernode {

// The machine's saved compute preference, else the device the daemon
// activated at startup; empty when neither is known.
inline std::optional<ProcessDeviceSelection> NodeJobRoute() {
    if (const auto saved = GetSavedExecutionDeviceSelection()) {
        return ProcessDeviceSelection{saved->type, saved->device_id};
    }
    return Device::GetProcessDevice();
}

}  // namespace cyxwiz::servernode
