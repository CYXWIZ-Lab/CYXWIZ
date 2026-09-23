#pragma once

// Device family enum, kept dependency-free so contracts that only need to
// NAME a device family (e.g. neural_provider.h's device-keyed dispatch) can
// include it without pulling in ArrayFire/platform headers via device.h.

namespace cyxwiz {

enum class DeviceType {
    CPU = 0,
    CUDA = 1,
    OPENCL = 2,
    METAL = 3,
    VULKAN = 4,
    ONEAPI = 5
};

} // namespace cyxwiz
