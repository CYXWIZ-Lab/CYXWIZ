#pragma once

#include "api_export.h"
#include <cstdint>
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
    VULKAN = 4,
    ONEAPI = 5
};

enum class DeviceMetadataStatus {
    NotQueried = 0,
    Available = 1,
    Unsupported = 2,
    Failed = 3
};

enum class DeviceKind {
    Unknown = 0,
    CPU = 1,
    GPU = 2,
    Accelerator = 3
};

enum class DeviceIdentityConfidence {
    Unknown = 0,
    BackendLocal = 1,
    ProviderReported = 2,
    StableHardware = 3
};

enum class DeviceActivationStage {
    NotStarted = 0,
    BackendSelection = 1,
    DeviceSelection = 2,
    ExecutionValidation = 3,
    EffectiveStateQuery = 4,
    Complete = 5
};

struct DeviceActivationResult {
    DeviceType requested_type = DeviceType::CPU;
    int requested_device_id = 0;
    DeviceType effective_type = DeviceType::CPU;
    int effective_device_id = 0;
    DeviceActivationStage stage = DeviceActivationStage::NotStarted;
    bool success = false;
    bool execution_validated = false;
    int error_code = 0;
    std::string message;
};

struct DeviceInfo {
    DeviceType type = DeviceType::CPU;
    int device_id = 0;
    std::string name;
    size_t memory_total = 0;
    size_t memory_available = 0;
    int compute_units = 0;
    bool supports_fp64 = false;
    bool supports_fp16 = false;

    bool backend_available = false;
    bool device_selectable = false;
    bool execution_validated = false;
    DeviceKind kind = DeviceKind::Unknown;
    DeviceIdentityConfidence identity_confidence =
        DeviceIdentityConfidence::Unknown;
    std::string provider;
    std::string driver_version;
    uint32_t hardware_vendor_id = 0;
    uint32_t hardware_device_id = 0;
    int pci_domain = 0;
    int pci_bus = 0;
    int pci_device = 0;
    int pci_function = 0;
    std::string hardware_uuid;
    std::string hardware_luid;
    std::string physical_fingerprint;
    bool provider_known = false;
    bool driver_version_known = false;
    bool hardware_vendor_id_known = false;
    bool hardware_device_id_known = false;
    bool pci_location_known = false;
    bool hardware_uuid_known = false;
    bool hardware_luid_known = false;
    bool physical_fingerprint_known = false;
    DeviceMetadataStatus metadata_status = DeviceMetadataStatus::NotQueried;
    int metadata_error_code = 0;
    std::string metadata_message;
    bool name_known = false;
    bool name_is_fallback = false;
    bool memory_total_known = false;
    bool memory_available_known = false;
    bool compute_units_known = false;
    bool supports_fp64_known = false;
    bool supports_fp16_known = false;
};

enum class DeviceRouteResolutionStatus {
    Resolved,
    FingerprintMissing,
    NotFound,
    Ambiguous
};

struct DeviceRouteResolution {
    DeviceRouteResolutionStatus status =
        DeviceRouteResolutionStatus::FingerprintMissing;
    DeviceType type = DeviceType::CPU;
    int device_id = -1;
};

CYXWIZ_API const char* DeviceMetadataStatusName(DeviceMetadataStatus status);
CYXWIZ_API const char* DeviceKindName(DeviceKind kind);
CYXWIZ_API const char* DeviceIdentityConfidenceName(
    DeviceIdentityConfidence confidence);
CYXWIZ_API const char* DeviceActivationStageName(DeviceActivationStage stage);
CYXWIZ_API DeviceRouteResolution ResolvePhysicalDeviceRoute(
    const std::vector<DeviceInfo>& inventory,
    DeviceType type,
    const std::string& physical_fingerprint);
CYXWIZ_API bool IsUncertifiedOneAPITrainingEnabled();

class CYXWIZ_API Device {
public:
    Device(DeviceType type, int device_id = 0);
    ~Device();

    DeviceType GetType() const { return type_; }
    int GetDeviceId() const { return device_id_; }
    DeviceInfo GetInfo() const;

    DeviceActivationResult ActivateExact(bool validate_execution) const;
    void SetActive();
    bool IsActive() const;

    static std::vector<DeviceInfo> GetAvailableDevices();
    static Device* GetCurrentDevice();

private:
    DeviceType type_;
    int device_id_;
};

} // namespace cyxwiz
