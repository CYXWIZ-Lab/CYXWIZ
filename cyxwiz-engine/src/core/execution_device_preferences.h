#pragma once

#include "algorithms/arrayfire_backend_utils.h"

#include <cyxwiz/cyxwiz.h>

#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

namespace cyxwiz {

struct PendingExecutionDeviceSelection {
    DeviceType type = DeviceType::CPU;
    int device_id = 0;
};

inline std::string ExecutionDeviceSelectionBackendName(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "arrayfire_cpu";
        case DeviceType::CUDA: return "arrayfire_cuda";
        case DeviceType::OPENCL: return "arrayfire_opencl";
        case DeviceType::ONEAPI: return "arrayfire_oneapi";
        case DeviceType::METAL: return "unsupported_metal";
        case DeviceType::VULKAN: return "unsupported_vulkan";
        default: return "arrayfire_unknown";
    }
}

inline bool IsArrayFireExecutionDeviceSelection(DeviceType type) {
    switch (type) {
        case DeviceType::CPU:
        case DeviceType::CUDA:
        case DeviceType::OPENCL:
        case DeviceType::ONEAPI:
            return true;
        case DeviceType::METAL:
        case DeviceType::VULKAN:
        default:
            return false;
    }
}

inline std::mutex& PendingExecutionDeviceSelectionMutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::optional<PendingExecutionDeviceSelection>&
PendingExecutionDeviceSelectionSlot() {
    static std::optional<PendingExecutionDeviceSelection> selection;
    return selection;
}

inline void SetPendingExecutionDeviceSelection(DeviceType type, int device_id) {
    if (!IsArrayFireExecutionDeviceSelection(type)) {
        throw std::invalid_argument(
            "Requested device type is not a supported ArrayFire execution backend");
    }

    std::lock_guard<std::mutex> lock(PendingExecutionDeviceSelectionMutex());
    PendingExecutionDeviceSelectionSlot() =
        PendingExecutionDeviceSelection{type, device_id};
}

inline std::optional<PendingExecutionDeviceSelection>
GetPendingExecutionDeviceSelection() {
    std::lock_guard<std::mutex> lock(PendingExecutionDeviceSelectionMutex());
    return PendingExecutionDeviceSelectionSlot();
}

inline void ClearPendingExecutionDeviceSelection() {
    std::lock_guard<std::mutex> lock(PendingExecutionDeviceSelectionMutex());
    PendingExecutionDeviceSelectionSlot().reset();
}

inline bool ApplyPendingExecutionDeviceSelection() {
    const auto pending = GetPendingExecutionDeviceSelection();
    if (!pending.has_value()) {
        return false;
    }

    Device device(pending->type, pending->device_id);
    device.SetActive();
    ClearPendingExecutionDeviceSelection();
    return true;
}

inline std::mutex& NextRunExecutionPolicyMutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::optional<ArrayFireFallbackPolicy>&
NextRunExecutionPolicySlot() {
    static std::optional<ArrayFireFallbackPolicy> policy;
    return policy;
}

inline void SetNextRunExecutionPolicy(ArrayFireFallbackPolicy policy) {
    std::lock_guard<std::mutex> lock(NextRunExecutionPolicyMutex());
    NextRunExecutionPolicySlot() = policy;
}

inline std::optional<ArrayFireFallbackPolicy> GetNextRunExecutionPolicy() {
    std::lock_guard<std::mutex> lock(NextRunExecutionPolicyMutex());
    return NextRunExecutionPolicySlot();
}

inline void ClearNextRunExecutionPolicy() {
    std::lock_guard<std::mutex> lock(NextRunExecutionPolicyMutex());
    NextRunExecutionPolicySlot().reset();
}

inline const char* ExecutionPolicyDisplayName(
    ArrayFireFallbackPolicy policy) {
    return policy == ArrayFireFallbackPolicy::ForbidNativeCpuFallback
        ? "Strict ArrayFire residency"
        : "Compatibility with recorded fallback";
}

} // namespace cyxwiz
