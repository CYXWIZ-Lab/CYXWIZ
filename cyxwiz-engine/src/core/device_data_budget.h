#pragma once

// What one batcher may keep on the device for a whole run (TOFIX178 S0-b,
// 1.3b): a quarter of the active device's memory (the model, its gradients
// and the optimizer keep the rest); on ArrayFire's CPU backend a quarter of
// the data budget. Unknown device memory: nothing.

#include "data_budget.h"

#include <arrayfire.h>
#include <cyxwiz/device.h>

#include <cstdint>

namespace cyxwiz {

inline uint64_t ResidentDataBudgetBytes() {
    if (af::getActiveBackend() == AF_BACKEND_CPU) return DataBudgetBytes() / 4;
    const Device* device = Device::GetCurrentDevice();
    if (!device) return 0;
    const DeviceInfo info = device->GetInfo();
    return info.memory_total_known && info.memory_total > 0 ? info.memory_total / 4 : 0;
}

}  // namespace cyxwiz
