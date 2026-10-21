#pragma once

#include "cyxwiz/device.h"

namespace cyxwiz::detail {

// Promotes identity confidence only from provider-reported stable fields.
CYXWIZ_API void FinalizeDeviceIdentity(DeviceInfo& info);

// The caller must have already selected the requested ArrayFire route.
void EnrichSelectedDeviceIdentity(DeviceInfo& info);

} // namespace cyxwiz::detail
