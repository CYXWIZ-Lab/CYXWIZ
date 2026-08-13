#include "cyxwiz/device.h"
#include "device_identity.h"
#include "device_probe.h"
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <set>
#include <tuple>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

namespace {

const char* DeviceTypeDisplayName(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "CPU";
        case DeviceType::CUDA: return "CUDA";
        case DeviceType::OPENCL: return "OpenCL";
        case DeviceType::ONEAPI: return "oneAPI";
        case DeviceType::METAL: return "Metal";
        case DeviceType::VULKAN: return "Vulkan";
        default: return "Unknown";
    }
}

std::string FallbackDeviceName(DeviceType type, int device_id) {
    return std::string(DeviceTypeDisplayName(type)) + " device " +
           std::to_string(device_id);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
bool TryGetArrayFireBackend(DeviceType type, af::Backend& backend) {
    switch (type) {
        case DeviceType::CPU:
            backend = AF_BACKEND_CPU;
            return true;
        case DeviceType::CUDA:
            backend = AF_BACKEND_CUDA;
            return true;
        case DeviceType::OPENCL:
            backend = AF_BACKEND_OPENCL;
            return true;
        case DeviceType::ONEAPI:
            backend = AF_BACKEND_ONEAPI;
            return true;
        default:
            return false;
    }
}

bool IsArrayFireBackendBuilt(DeviceType type) {
    switch (type) {
        case DeviceType::CPU:
        case DeviceType::ONEAPI:
            return true;
        case DeviceType::CUDA:
#ifdef CYXWIZ_ENABLE_CUDA
            return true;
#else
            return false;
#endif
        case DeviceType::OPENCL:
#ifdef CYXWIZ_ENABLE_OPENCL
            return true;
#else
            return false;
#endif
        default:
            return false;
    }
}

DeviceType DeviceTypeFromArrayFireBackend(af::Backend backend) {
    switch (backend) {
        case AF_BACKEND_CUDA: return DeviceType::CUDA;
        case AF_BACKEND_OPENCL: return DeviceType::OPENCL;
        case AF_BACKEND_ONEAPI: return DeviceType::ONEAPI;
        case AF_BACKEND_CPU:
        default:
            return DeviceType::CPU;
    }
}

const char* ArrayFireErrorName(af_err error) {
    const char* name = af_err_to_string(error);
    return name != nullptr ? name : "unknown ArrayFire error";
}

class ScopedArrayFireDeviceState {
public:
    ScopedArrayFireDeviceState() {
        try {
            backend_ = af::getActiveBackend();
            backend_known_ = true;
            device_ = af::getDevice();
            device_known_ = true;
        } catch (const af::exception&) {
            backend_known_ = false;
            device_known_ = false;
        }
    }

    ~ScopedArrayFireDeviceState() {
        if (!restore_ || !backend_known_) return;
        try {
            af::setBackend(backend_);
            if (device_known_) af::setDevice(device_);
        } catch (af::exception& error) {
            spdlog::warn(
                "Temporary ArrayFire device query could not restore backend/device: error={} ({})",
                static_cast<int>(error.err()),
                ArrayFireErrorName(error.err()));
        }
    }

    ScopedArrayFireDeviceState(const ScopedArrayFireDeviceState&) = delete;
    ScopedArrayFireDeviceState& operator=(const ScopedArrayFireDeviceState&) = delete;

    void Dismiss() noexcept { restore_ = false; }

private:
    af::Backend backend_ = AF_BACKEND_DEFAULT;
    int device_ = 0;
    bool backend_known_ = false;
    bool device_known_ = false;
    bool restore_ = true;
};

void LogMetadataLimitationOnce(const DeviceInfo& info) {
    if (info.type != DeviceType::ONEAPI ||
        info.metadata_status == DeviceMetadataStatus::Available) {
        return;
    }
    static std::once_flag warning_once;
    std::call_once(warning_once, [&info]() {
        spdlog::warn(
            "oneAPI device {} discovered; detailed device properties are {} "
            "category=device backend=arrayfire_oneapi device_id={} "
            "probe_stage=metadata capability_status={} arrayfire_error={}",
            info.device_id,
            info.metadata_status == DeviceMetadataStatus::Unsupported
                ? "unsupported by the installed ArrayFire oneAPI plugin"
                : "unavailable",
            info.device_id,
            DeviceMetadataStatusName(info.metadata_status),
            info.metadata_error_code);
    });
}

DeviceInfo QuerySelectedArrayFireDeviceInfo(DeviceType type, int device_id) {
    DeviceInfo info{};
    info.type = type;
    info.device_id = device_id;
    info.name = FallbackDeviceName(type, device_id);
    info.name_is_fallback = true;
    info.backend_available = true;
    info.device_selectable = true;
    info.identity_confidence = DeviceIdentityConfidence::BackendLocal;
    if (type == DeviceType::CPU) {
        info.kind = DeviceKind::CPU;
    } else if (type == DeviceType::CUDA) {
        info.kind = DeviceKind::GPU;
    }

    char name[256] = {};
    char platform[256] = {};
    char toolkit[256] = {};
    char compute[256] = {};
    try {
        af::deviceInfo(name, platform, toolkit, compute);
        info.name = std::string(name);
        info.name_known = true;
        info.name_is_fallback = false;
        info.metadata_status = DeviceMetadataStatus::Available;
        if (platform[0] != '\0') {
            info.provider = platform;
            info.provider_known = true;
        }
        spdlog::debug("Device info: {}, Platform: {}", name, platform);
    } catch (af::exception& error) {
        info.metadata_error_code = static_cast<int>(error.err());
        info.metadata_status = error.err() == AF_ERR_NOT_SUPPORTED
            ? DeviceMetadataStatus::Unsupported
            : DeviceMetadataStatus::Failed;
        info.metadata_message = error.err() == AF_ERR_NOT_SUPPORTED
            ? "Detailed device properties are unsupported by the installed ArrayFire backend"
            : "Detailed device properties could not be queried";
        LogMetadataLimitationOnce(info);
    }

    detail::EnrichSelectedDeviceIdentity(info);

    return info;
}

class ProductionArrayFireProbeAdapter final
    : public detail::ArrayFireProbeAdapter {
public:
    detail::DeviceProbeStatus SelectBackend(DeviceType type) override {
        af::Backend backend = AF_BACKEND_DEFAULT;
        if (!TryGetArrayFireBackend(type, backend) ||
            !IsArrayFireBackendBuilt(type)) {
            return {false, 0, "ArrayFire backend is not enabled in this build"};
        }
        try {
            af::setBackend(backend);
            return {true, 0, {}};
        } catch (af::exception& error) {
            return Failure(error);
        }
    }

    detail::DeviceCountProbeResult GetDeviceCount(DeviceType) override {
        try {
            return {{true, 0, {}}, af::getDeviceCount()};
        } catch (af::exception& error) {
            return {Failure(error), 0};
        }
    }

    detail::DeviceProbeStatus SelectDevice(DeviceType, int device_id) override {
        try {
            af::setDevice(device_id);
            return {true, 0, {}};
        } catch (af::exception& error) {
            return Failure(error);
        }
    }

    DeviceInfo QuerySelectedDeviceMetadata(DeviceType type,
                                           int device_id) override {
        return QuerySelectedArrayFireDeviceInfo(type, device_id);
    }

private:
    static detail::DeviceProbeStatus Failure(af::exception& error) {
        return {false,
                static_cast<int>(error.err()),
                ArrayFireErrorName(error.err())};
    }
};

const char* DeviceProbeStageName(detail::DeviceProbeStage stage) {
    switch (stage) {
        case detail::DeviceProbeStage::BackendSelection:
            return "backend_selection";
        case detail::DeviceProbeStage::Enumeration:
            return "enumeration";
        case detail::DeviceProbeStage::DeviceSelection:
            return "device_selection";
        case detail::DeviceProbeStage::Metadata:
            return "metadata";
        default:
            return "unknown";
    }
}

const char* DeviceProbeFailureCategory(
    const detail::DeviceProbeFailure& failure) {
    switch (static_cast<af_err>(failure.error_code)) {
        case AF_ERR_LOAD_LIB:
        case AF_ERR_LOAD_SYM:
        case AF_ERR_NOT_CONFIGURED:
            return "backend_pack_missing_or_incompatible";
        case AF_ERR_DRIVER:
            return "driver_or_provider_failure";
        case AF_ERR_RUNTIME:
            return "runtime_or_provider_initialization_failure";
        default:
            break;
    }
    if (failure.stage == detail::DeviceProbeStage::Enumeration &&
        failure.error_code == 0) {
        return "no_compatible_device_or_provider";
    }
    return "backend_probe_failure";
}

const char* DeviceProbeFailureRemediation(
    const detail::DeviceProbeFailure& failure) {
    switch (static_cast<af_err>(failure.error_code)) {
        case AF_ERR_LOAD_LIB:
        case AF_ERR_LOAD_SYM:
        case AF_ERR_NOT_CONFIGURED:
            return "Install the matching ArrayFire backend pack and its exact transitive runtime dependencies";
        case AF_ERR_DRIVER:
            return "Install or update the hardware vendor driver/provider";
        case AF_ERR_RUNTIME:
            return "Verify the backend runtime, provider, and hardware driver versions";
        default:
            break;
    }
    if (failure.stage == detail::DeviceProbeStage::Enumeration &&
        failure.error_code == 0) {
        return "Install a compatible provider/driver or select another backend";
    }
    return "Inspect the backend pack, runtime provider, and driver configuration";
}

void LogDeviceProbeFailureOnce(const detail::DeviceProbeFailure& failure) {
    static std::mutex mutex;
    static std::set<std::tuple<int, int, int>> logged_failures;
    const auto key = std::make_tuple(static_cast<int>(failure.type),
                                     static_cast<int>(failure.stage),
                                     failure.error_code);
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!logged_failures.insert(key).second) {
            return;
        }
    }

    spdlog::warn(
        "{} backend unavailable: category={} stage={} device={} error={} ({}) remediation={}",
        DeviceTypeDisplayName(failure.type),
        DeviceProbeFailureCategory(failure),
        DeviceProbeStageName(failure.stage),
        failure.device_id,
        failure.error_code,
        failure.message,
        DeviceProbeFailureRemediation(failure));
}
#endif

} // namespace

const char* DeviceMetadataStatusName(DeviceMetadataStatus status) {
    switch (status) {
        case DeviceMetadataStatus::Available: return "available";
        case DeviceMetadataStatus::Unsupported: return "unsupported";
        case DeviceMetadataStatus::Failed: return "failed";
        case DeviceMetadataStatus::NotQueried:
        default:
            return "not_queried";
    }
}

const char* DeviceKindName(DeviceKind kind) {
    switch (kind) {
        case DeviceKind::CPU: return "cpu";
        case DeviceKind::GPU: return "gpu";
        case DeviceKind::Accelerator: return "accelerator";
        case DeviceKind::Unknown:
        default:
            return "unknown";
    }
}

const char* DeviceIdentityConfidenceName(
    DeviceIdentityConfidence confidence) {
    switch (confidence) {
        case DeviceIdentityConfidence::BackendLocal:
            return "backend_local";
        case DeviceIdentityConfidence::ProviderReported:
            return "provider_reported";
        case DeviceIdentityConfidence::StableHardware:
            return "stable_hardware";
        case DeviceIdentityConfidence::Unknown:
        default:
            return "unknown";
    }
}

const char* DeviceActivationStageName(DeviceActivationStage stage) {
    switch (stage) {
        case DeviceActivationStage::BackendSelection:
            return "backend_selection";
        case DeviceActivationStage::DeviceSelection:
            return "device_selection";
        case DeviceActivationStage::ExecutionValidation:
            return "execution_validation";
        case DeviceActivationStage::EffectiveStateQuery:
            return "effective_state_query";
        case DeviceActivationStage::Complete:
            return "complete";
        case DeviceActivationStage::NotStarted:
        default:
            return "not_started";
    }
}

bool IsUncertifiedOneAPITrainingEnabled() {
    const char* value =
        std::getenv("CYXWIZ_ENABLE_UNCERTIFIED_ONEAPI_TRAINING");
    return value != nullptr && std::string(value) == "1";
}

Device::Device(DeviceType type, int device_id)
    : type_(type), device_id_(device_id) {
}

Device::~Device() = default;

DeviceInfo Device::GetInfo() const {
    DeviceInfo info{};
    info.type = type_;
    info.device_id = device_id_;
    info.name = FallbackDeviceName(type_, device_id_);
    info.name_is_fallback = true;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    af::Backend backend = AF_BACKEND_DEFAULT;
    if (TryGetArrayFireBackend(type_, backend) &&
        IsArrayFireBackendBuilt(type_)) {
        ScopedArrayFireDeviceState restore_state;
        try {
            af::setBackend(backend);
            info.backend_available = true;
            const int count = af::getDeviceCount();
            if (device_id_ < 0 || device_id_ >= count) {
                return info;
            }
            af::setDevice(device_id_);
            info.device_selectable = true;
        } catch (const af::exception&) {
            return info;
        }

        info = QuerySelectedArrayFireDeviceInfo(type_, device_id_);
    } else
#endif
    {
        info.backend_available = type_ == DeviceType::CPU;
        info.device_selectable = type_ == DeviceType::CPU;
        if (info.device_selectable) {
            info.kind = DeviceKind::CPU;
            info.identity_confidence =
                DeviceIdentityConfidence::BackendLocal;
        }
    }

    return info;
}

DeviceActivationResult Device::ActivateExact(bool validate_execution) const {
    DeviceActivationResult result{};
    result.requested_type = type_;
    result.requested_device_id = device_id_;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    af::Backend backend = AF_BACKEND_DEFAULT;
    if (!TryGetArrayFireBackend(type_, backend) ||
        !IsArrayFireBackendBuilt(type_)) {
        result.stage = DeviceActivationStage::BackendSelection;
        result.message =
            "Requested ArrayFire backend is not enabled in this build";
        return result;
    }
    ScopedArrayFireDeviceState restore_on_failure;

    try {
        result.stage = DeviceActivationStage::BackendSelection;
        af::setBackend(backend);

        result.stage = DeviceActivationStage::DeviceSelection;
        const int count = af::getDeviceCount();
        if (device_id_ < 0 || device_id_ >= count) {
            result.message = "Requested ArrayFire device ID is out of range";
            return result;
        }
        af::setDevice(device_id_);

        if (validate_execution) {
            result.stage = DeviceActivationStage::ExecutionValidation;
            af::array input = af::constant(1.0f, af::dim4(4), f32);
            af::array output = input + 2.0f;
            output.eval();
            af::sync();
            result.execution_validated = true;
        }

        result.stage = DeviceActivationStage::EffectiveStateQuery;
        result.effective_type =
            DeviceTypeFromArrayFireBackend(af::getActiveBackend());
        result.effective_device_id = af::getDevice();
        if (result.effective_type != type_ ||
            result.effective_device_id != device_id_) {
            result.message =
                "ArrayFire effective backend/device differs from the request";
            return result;
        }

        result.stage = DeviceActivationStage::Complete;
        result.success = true;
        result.message = "Requested ArrayFire backend/device activated";
        restore_on_failure.Dismiss();
    } catch (af::exception& error) {
        result.error_code = static_cast<int>(error.err());
        result.message = ArrayFireErrorName(error.err());
        try {
            result.effective_type =
                DeviceTypeFromArrayFireBackend(af::getActiveBackend());
            result.effective_device_id = af::getDevice();
        } catch (const af::exception&) {
            // The failed stage and error remain authoritative.
        }
    }
#else
    result.stage = DeviceActivationStage::Complete;
    result.success = type_ == DeviceType::CPU && device_id_ == 0;
    result.execution_validated = result.success && validate_execution;
    result.effective_type = DeviceType::CPU;
    result.effective_device_id = 0;
    result.message = result.success
        ? "Native development CPU device selected"
        : "ArrayFire backend is unavailable in this reduced build";
#endif

    return result;
}

void Device::SetActive() {
    const auto activation = ActivateExact(false);
    if (activation.success) {
        spdlog::info("Switched to {} backend, device {}",
                     DeviceTypeDisplayName(type_), device_id_);
        return;
    }

    spdlog::warn(
        "Failed to switch to requested backend: type={} device={} stage={} error={} ({}); falling back to ArrayFire CPU",
        static_cast<int>(type_),
        device_id_,
        DeviceActivationStageName(activation.stage),
        activation.error_code,
        activation.message);
    const auto cpu_activation =
        Device(DeviceType::CPU, 0).ActivateExact(false);
    if (cpu_activation.success) {
        type_ = DeviceType::CPU;
        device_id_ = 0;
    } else {
        spdlog::error(
            "ArrayFire CPU activation failed: stage={} error={} ({})",
            DeviceActivationStageName(cpu_activation.stage),
            cpu_activation.error_code,
            cpu_activation.message);
    }
}

bool Device::IsActive() const {
    const auto* active = GetCurrentDevice();
    return active != nullptr &&
           active->GetType() == type_ &&
           active->GetDeviceId() == device_id_;
}

std::vector<DeviceInfo> Device::GetAvailableDevices() {
    std::vector<DeviceInfo> devices;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    ScopedArrayFireDeviceState restore_state;
    ProductionArrayFireProbeAdapter adapter;
    std::vector<DeviceType> backend_types = {DeviceType::CPU};
#ifdef CYXWIZ_ENABLE_CUDA
    backend_types.push_back(DeviceType::CUDA);
#endif
    backend_types.push_back(DeviceType::ONEAPI);
#ifdef CYXWIZ_ENABLE_OPENCL
    backend_types.push_back(DeviceType::OPENCL);
#endif
    auto probe = detail::ProbeAvailableArrayFireDevices(adapter, backend_types);
    devices = std::move(probe.devices);
    for (const auto& failure : probe.failures) {
        if (failure.stage == detail::DeviceProbeStage::Metadata) {
            continue;
        }
        LogDeviceProbeFailureOnce(failure);
    }
#else
    DeviceInfo cpu = Device(DeviceType::CPU, 0).GetInfo();
    cpu.backend_available = true;
    cpu.device_selectable = true;
    devices.push_back(std::move(cpu));
#endif

    spdlog::info("Total devices found: {}", devices.size());
    return devices;
}

Device* Device::GetCurrentDevice() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    // Query ArrayFire directly. The previous implementation retained a pointer
    // to the Device object passed to SetActive(); GUI callers use temporary
    // Device values, so that pointer became null/dangling as soon as the caller
    // returned even though ArrayFire remained on the selected backend.
    thread_local Device active_device(DeviceType::CPU, 0);
    try {
        switch (af::getActiveBackend()) {
            case AF_BACKEND_CUDA:
                active_device.type_ = DeviceType::CUDA;
                break;
            case AF_BACKEND_OPENCL:
                active_device.type_ = DeviceType::OPENCL;
                break;
            case AF_BACKEND_ONEAPI:
                active_device.type_ = DeviceType::ONEAPI;
                break;
            case AF_BACKEND_CPU:
            default:
                active_device.type_ = DeviceType::CPU;
                break;
        }
        active_device.device_id_ = af::getDevice();
        return &active_device;
    } catch (const af::exception&) {
        return nullptr;
    }
#else
    static Device cpu_device(DeviceType::CPU, 0);
    return &cpu_device;
#endif
}

} // namespace cyxwiz
