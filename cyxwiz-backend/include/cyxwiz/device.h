#pragma once

#include "api_export.h"
#include <string>
#include <vector>
#include <memory>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

enum class DeviceType {
    CPU = 0,
    CUDA = 1,
    OPENCL = 2,
    METAL = 3,
    VULKAN = 4
};

struct DeviceInfo {
    DeviceType type;
    int device_id;
    std::string name;
    size_t memory_total;
    size_t memory_available;
    int compute_units;
    bool supports_fp64;
    bool supports_fp16;
};

class CYXWIZ_API Device {
public:
    Device(DeviceType type, int device_id = 0);
    ~Device();

    DeviceType GetType() const { return type_; }
    int GetDeviceId() const { return device_id_; }
    DeviceInfo GetInfo() const;

    void SetActive();
    bool IsActive() const;

    static std::vector<DeviceInfo> GetAvailableDevices();
    static Device* GetCurrentDevice();

private:
    DeviceType type_;
    int device_id_;
};

} // namespace cyxwiz
