// allocation_device_type.h - the runtime behind a lent GPU (TOFIX139 P8 finding).
// The GUI's allocation only says "GPU" or "CPU"; the backend knows which
// runtime drives each device (CUDA, OpenCL, oneAPI, Metal). The daemon
// resolves the allocation's type here before it registers with the central
// server, which routes jobs by that type.
#pragma once

#include <cyxwiz/device_type.h>

#include <string>
#include <vector>

namespace cyxwiz::servernode::core {

// Allocation device types, as DeviceAllocation::device_type carries them:
// 0 = CPU, 1 = CUDA, 2 = OpenCL, 3 = Metal; -1 = not shareable.
constexpr int kAllocCpu = 0;
constexpr int kAllocCuda = 1;
constexpr int kAllocOpenCl = 2;
constexpr int kAllocMetal = 3;
constexpr int kAllocNone = -1;

struct KnownDevice {
    cyxwiz::DeviceType type = cyxwiz::DeviceType::CPU;
    int device_id = 0;
    std::string name;
};

// The allocation type for a backend device type. oneAPI devices are
// registered as OpenCL until the protocol carries oneAPI (they run the
// OpenCL route too); Vulkan is not a training runtime here.
int AllocationTypeOf(cyxwiz::DeviceType type);

// The allocation type of the GPU the GUI named: the backend device with the
// same name, else the GPU at the same position among the backend's GPUs (the
// GUI lists them in the backend's order), else CUDA as before.
int ResolveGpuAllocationType(const std::string& name, int gpu_position, const std::vector<KnownDevice>& known);

}  // namespace cyxwiz::servernode::core
