#include "cyxwiz/device.h"
#include <spdlog/spdlog.h>

namespace cyxwiz {

static Device* g_current_device = nullptr;

Device::Device(DeviceType type, int device_id)
    : type_(type), device_id_(device_id) {
}

Device::~Device() {
}

DeviceInfo Device::GetInfo() const {
    DeviceInfo info;
    info.type = type_;
    info.device_id = device_id_;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (type_ == DeviceType::CUDA || type_ == DeviceType::OPENCL) {
        char name[256];
        char platform[256];
        char toolkit[256];
        char compute[256];
        af::deviceInfo(name, platform, toolkit, compute);

        info.name = std::string(name);
        // ArrayFire 3.9 API: use deviceMemInfo instead of getAvailableMemory
        size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
        af::deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
        info.memory_total = alloc_bytes + lock_bytes;
        info.memory_available = lock_bytes;
        // Note: ArrayFire doesn't expose compute units directly
        info.compute_units = 0;
        info.supports_fp64 = true; // Assume true for now
        info.supports_fp16 = false;

        spdlog::debug("Device info: {}, Platform: {}", name, platform);
    } else
#endif
    {
        info.name = "CPU";
        info.memory_total = 0;
        info.memory_available = 0;
        info.compute_units = 0;
        info.supports_fp64 = true;
        info.supports_fp16 = false;
    }

    return info;
}

void Device::SetActive() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (type_ == DeviceType::CUDA || type_ == DeviceType::OPENCL) {
        af::setDevice(device_id_);
    }
#endif
    g_current_device = this;
}

bool Device::IsActive() const {
    return g_current_device == this;
}

std::vector<DeviceInfo> Device::GetAvailableDevices() {
    std::vector<DeviceInfo> devices;

    // Always add CPU
    Device cpu_device(DeviceType::CPU, 0);
    devices.push_back(cpu_device.GetInfo());

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        int device_count = af::getDeviceCount();
        for (int i = 0; i < device_count; i++) {
            af::setDevice(i);
            Device gpu_device(DeviceType::CUDA, i);
            devices.push_back(gpu_device.GetInfo());
        }
    } catch (const af::exception& e) {
        spdlog::warn("Failed to enumerate GPU devices: {}", e.what());
    }
#endif

    return devices;
}

Device* Device::GetCurrentDevice() {
    return g_current_device;
}

} // namespace cyxwiz
