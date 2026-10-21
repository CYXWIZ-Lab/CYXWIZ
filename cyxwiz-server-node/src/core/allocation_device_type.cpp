// allocation_device_type.cpp - the runtime behind a lent GPU.
#include "core/allocation_device_type.h"

namespace cyxwiz::servernode::core {

int AllocationTypeOf(cyxwiz::DeviceType type) {
    switch (type) {
        case cyxwiz::DeviceType::CPU: return kAllocCpu;
        case cyxwiz::DeviceType::CUDA: return kAllocCuda;
        case cyxwiz::DeviceType::OPENCL: return kAllocOpenCl;
        case cyxwiz::DeviceType::ONEAPI: return kAllocOpenCl;
        case cyxwiz::DeviceType::METAL: return kAllocMetal;
        case cyxwiz::DeviceType::VULKAN: return kAllocNone;
    }
    return kAllocNone;
}

int ResolveGpuAllocationType(const std::string& name, int gpu_position, const std::vector<KnownDevice>& known) {
    for (const auto& device : known) {
        if (device.type != cyxwiz::DeviceType::CPU && device.name == name) return AllocationTypeOf(device.type);
    }
    int position = 0;
    for (const auto& device : known) {
        if (device.type == cyxwiz::DeviceType::CPU) continue;
        if (position++ == gpu_position) return AllocationTypeOf(device.type);
    }
    return kAllocCuda;
}

}  // namespace cyxwiz::servernode::core
