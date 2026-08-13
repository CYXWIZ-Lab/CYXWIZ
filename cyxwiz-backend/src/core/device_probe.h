#pragma once

#include "cyxwiz/device.h"

#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::detail {

enum class DeviceProbeStage {
    BackendSelection,
    Enumeration,
    DeviceSelection,
    Metadata
};

struct DeviceProbeStatus {
    bool success = false;
    int error_code = 0;
    std::string message;
};

struct DeviceCountProbeResult {
    DeviceProbeStatus status;
    int count = 0;
};

struct DeviceProbeFailure {
    DeviceType type = DeviceType::CPU;
    int device_id = -1;
    DeviceProbeStage stage = DeviceProbeStage::BackendSelection;
    int error_code = 0;
    std::string message;
};

struct DeviceInventoryProbeResult {
    std::vector<DeviceInfo> devices;
    std::vector<DeviceProbeFailure> failures;
};

class ArrayFireProbeAdapter {
public:
    virtual ~ArrayFireProbeAdapter() = default;

    virtual DeviceProbeStatus SelectBackend(DeviceType type) = 0;
    virtual DeviceCountProbeResult GetDeviceCount(DeviceType type) = 0;
    virtual DeviceProbeStatus SelectDevice(DeviceType type, int device_id) = 0;
    virtual DeviceInfo QuerySelectedDeviceMetadata(DeviceType type,
                                                   int device_id) = 0;
};

inline void RecordProbeFailure(DeviceInventoryProbeResult& result,
                               DeviceType type,
                               int device_id,
                               DeviceProbeStage stage,
                               const DeviceProbeStatus& status) {
    result.failures.push_back(
        {type, device_id, stage, status.error_code, status.message});
}

inline DeviceInventoryProbeResult ProbeAvailableArrayFireDevices(
    ArrayFireProbeAdapter& adapter,
    const std::vector<DeviceType>& backend_types) {
    DeviceInventoryProbeResult result;

    for (const auto type : backend_types) {
        const auto backend = adapter.SelectBackend(type);
        if (!backend.success) {
            RecordProbeFailure(result,
                               type,
                               -1,
                               DeviceProbeStage::BackendSelection,
                               backend);
            continue;
        }

        const auto enumeration = adapter.GetDeviceCount(type);
        if (!enumeration.status.success) {
            RecordProbeFailure(result,
                               type,
                               -1,
                               DeviceProbeStage::Enumeration,
                               enumeration.status);
            continue;
        }
        if (enumeration.count <= 0) {
            RecordProbeFailure(
                result,
                type,
                -1,
                DeviceProbeStage::Enumeration,
                {false, 0, "Backend loaded but exposed no compatible devices"});
            continue;
        }

        for (int device_id = 0; device_id < enumeration.count; ++device_id) {
            const auto selection = adapter.SelectDevice(type, device_id);
            if (!selection.success) {
                RecordProbeFailure(result,
                                   type,
                                   device_id,
                                   DeviceProbeStage::DeviceSelection,
                                   selection);
                continue;
            }

            auto info = adapter.QuerySelectedDeviceMetadata(type, device_id);
            info.type = type;
            info.device_id = device_id;
            info.backend_available = true;
            info.device_selectable = true;
            if (info.metadata_status == DeviceMetadataStatus::Unsupported ||
                info.metadata_status == DeviceMetadataStatus::Failed) {
                RecordProbeFailure(
                    result,
                    type,
                    device_id,
                    DeviceProbeStage::Metadata,
                    {false, info.metadata_error_code, info.metadata_message});
            }
            result.devices.push_back(std::move(info));
        }
    }

    return result;
}

} // namespace cyxwiz::detail
