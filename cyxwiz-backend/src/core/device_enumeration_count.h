#pragma once

// Internal to the backend and its tests; not part of the plugin SDK.

#include "cyxwiz/api_export.h"

#include <cstdint>

namespace cyxwiz::detail {

// How many times this process has enumerated its devices
// (Device::GetAvailableDevices, which probes every ArrayFire backend). A test
// reads it around a path that must never enumerate (Compile's provider
// target), so the path cannot start probing again unnoticed.
CYXWIZ_API uint64_t DeviceEnumerationCount();

} // namespace cyxwiz::detail
