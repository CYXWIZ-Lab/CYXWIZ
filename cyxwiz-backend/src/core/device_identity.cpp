#include "device_identity.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

namespace cyxwiz::detail {
namespace {

std::string LowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](char value) {
        return static_cast<char>(
            std::tolower(static_cast<unsigned char>(value)));
    });
    return value;
}

std::string HexBytes(const unsigned char* bytes, size_t size) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (size_t index = 0; index < size; ++index) {
        out << std::setw(2) << static_cast<unsigned int>(bytes[index]);
    }
    return out.str();
}

std::string PciFingerprint(const DeviceInfo& info) {
    std::ostringstream out;
    out << "pci:" << std::hex << std::setfill('0')
        << std::setw(4) << info.hardware_vendor_id << ':'
        << std::setw(4) << info.pci_domain << ':'
        << std::setw(2) << info.pci_bus << ':'
        << std::setw(2) << info.pci_device << '.'
        << info.pci_function;
    return LowerAscii(out.str());
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
class DynamicLibrary {
public:
#ifdef _WIN32
    explicit DynamicLibrary(const wchar_t* name, bool already_loaded = false)
        : handle_(already_loaded ? GetModuleHandleW(name) : LoadLibraryW(name)),
          owns_(!already_loaded && handle_ != nullptr) {}

    ~DynamicLibrary() {
        if (owns_) FreeLibrary(handle_);
    }

    void* Find(const char* symbol) const {
        return handle_ != nullptr
            ? reinterpret_cast<void*>(GetProcAddress(handle_, symbol))
            : nullptr;
    }

private:
    HMODULE handle_ = nullptr;
    bool owns_ = false;
#else
    explicit DynamicLibrary(const char* name, bool already_loaded = false)
        : handle_(dlopen(name,
                         RTLD_LAZY |
                             (already_loaded ? RTLD_NOLOAD : RTLD_LOCAL))) {}

    ~DynamicLibrary() {
        if (handle_ != nullptr) dlclose(handle_);
    }

    void* Find(const char* symbol) const {
        return handle_ != nullptr ? dlsym(handle_, symbol) : nullptr;
    }

private:
    void* handle_ = nullptr;
#endif
};

template <typename Function>
Function FindFunction(const DynamicLibrary& library, const char* symbol) {
    return reinterpret_cast<Function>(library.Find(symbol));
}

#ifdef CYXWIZ_ENABLE_OPENCL
// OpenCL identity is resolved at runtime, like CUDA above: the portable backend
// must not import OpenCL or afopencl, so a machine without the OpenCL pack or
// an ICD loader still loads it. Values below are the Khronos cl.h/cl_ext.h ABI.
using ClDeviceId = void*;
using ClPlatformId = void*;
using ClInt = std::int32_t;
using ClUint = std::uint32_t;
using ClUlong = std::uint64_t;
using ClBool = ClUint;

constexpr ClInt kClSuccess = 0;
constexpr ClBool kClTrue = 1;
constexpr ClUint kClDeviceType = 0x1000;
constexpr ClUint kClDeviceVendorId = 0x1001;
constexpr ClUint kClDeviceGlobalMemSize = 0x101F;
constexpr ClUint kClDeviceVendor = 0x102C;
constexpr ClUint kClDriverVersion = 0x102D;
constexpr ClUint kClDeviceExtensions = 0x1030;
constexpr ClUint kClDevicePlatform = 0x1031;
constexpr ClUint kClPlatformVendor = 0x0903;
constexpr ClUlong kClDeviceTypeCpu = 1U << 1U;
constexpr ClUlong kClDeviceTypeGpu = 1U << 2U;
constexpr ClUlong kClDeviceTypeAccelerator = 1U << 3U;
constexpr ClUint kClDeviceUuidKhr = 0x106A;
constexpr ClUint kClDeviceLuidValidKhr = 0x106C;
constexpr ClUint kClDeviceLuidKhr = 0x106D;
constexpr std::size_t kClUuidSizeKhr = 16;
constexpr std::size_t kClLuidSizeKhr = 8;
constexpr ClUint kClDevicePciBusInfoKhr = 0x410F;
constexpr ClUint kClDevicePciBusIdNv = 0x4008;
constexpr ClUint kClDevicePciSlotIdNv = 0x4009;
constexpr ClUint kClDevicePciDomainIdNv = 0x400A;

struct ClPciBusInfoKhr {
    ClUint pci_domain;
    ClUint pci_bus;
    ClUint pci_device;
    ClUint pci_function;
};

struct OpenClIdentityApi {
    using GetDeviceInfo =
        ClInt (*)(ClDeviceId, ClUint, std::size_t, void*, std::size_t*);
    using GetPlatformInfo =
        ClInt (*)(ClPlatformId, ClUint, std::size_t, void*, std::size_t*);
    GetDeviceInfo get_device_info = nullptr;
    GetPlatformInfo get_platform_info = nullptr;

    template <typename Value>
    bool Query(ClDeviceId device, ClUint field, Value& value) const {
        return get_device_info(device, field, sizeof(Value), &value,
                               nullptr) == kClSuccess;
    }

    std::string QueryString(ClDeviceId device, ClUint field) const {
        std::size_t size = 0;
        if (get_device_info(device, field, 0, nullptr, &size) != kClSuccess ||
            size <= 1) {
            return {};
        }
        std::string value(size, '\0');
        if (get_device_info(device, field, size, value.data(), nullptr) !=
            kClSuccess) {
            return {};
        }
        while (!value.empty() && value.back() == '\0') value.pop_back();
        return value;
    }

    std::string QueryPlatformString(ClPlatformId platform, ClUint field) const {
        std::size_t size = 0;
        if (!get_platform_info ||
            get_platform_info(platform, field, 0, nullptr, &size) !=
                kClSuccess ||
            size <= 1) {
            return {};
        }
        std::string value(size, '\0');
        if (get_platform_info(platform, field, size, value.data(), nullptr) !=
            kClSuccess) {
            return {};
        }
        while (!value.empty() && value.back() == '\0') value.pop_back();
        return value;
    }
};

bool HasOpenClExtension(const std::string& extensions,
                        const std::string& extension) {
    size_t position = 0;
    while ((position = extensions.find(extension, position)) !=
           std::string::npos) {
        const bool left = position == 0 || extensions[position - 1] == ' ';
        const size_t end = position + extension.size();
        const bool right = end == extensions.size() || extensions[end] == ' ';
        if (left && right) return true;
        position = end;
    }
    return false;
}

void EnrichOpenClIdentity(DeviceInfo& info) {
#ifdef _WIN32
    DynamicLibrary arrayfire_opencl(L"afopencl.dll", true);
    DynamicLibrary opencl(L"OpenCL.dll");
#elif defined(__APPLE__)
    DynamicLibrary arrayfire_opencl("libafopencl.dylib", true);
    DynamicLibrary opencl("/System/Library/Frameworks/OpenCL.framework/OpenCL");
#else
    DynamicLibrary arrayfire_opencl("libafopencl.so", true);
    DynamicLibrary opencl("libOpenCL.so.1");
#endif

    using AfGetDeviceId = af_err (*)(ClDeviceId*);
    const auto af_get_device_id =
        FindFunction<AfGetDeviceId>(arrayfire_opencl, "afcl_get_device_id");
    OpenClIdentityApi api;
    api.get_device_info = FindFunction<OpenClIdentityApi::GetDeviceInfo>(
        opencl, "clGetDeviceInfo");
    api.get_platform_info = FindFunction<OpenClIdentityApi::GetPlatformInfo>(
        opencl, "clGetPlatformInfo");

    ClDeviceId device = nullptr;
    if (!af_get_device_id || !api.get_device_info ||
        af_get_device_id(&device) != AF_SUCCESS || device == nullptr) {
        return;
    }

    ClUlong type = 0;
    if (api.Query(device, kClDeviceType, type)) {
        if ((type & kClDeviceTypeCpu) != 0) {
            info.kind = DeviceKind::CPU;
        } else if ((type & kClDeviceTypeGpu) != 0) {
            info.kind = DeviceKind::GPU;
        } else if ((type & kClDeviceTypeAccelerator) != 0) {
            info.kind = DeviceKind::Accelerator;
        }
    }

    ClUlong total_memory = 0;
    if (api.Query(device, kClDeviceGlobalMemSize, total_memory) &&
        total_memory > 0) {
        info.memory_total = static_cast<size_t>(total_memory);
        info.memory_total_known = true;
    }

    info.provider = api.QueryString(device, kClDeviceVendor);
    info.provider_known = !info.provider.empty();
    info.driver_version = api.QueryString(device, kClDriverVersion);
    info.driver_version_known = !info.driver_version.empty();
    ClUint vendor_id = 0;
    info.hardware_vendor_id_known =
        api.Query(device, kClDeviceVendorId, vendor_id);
    if (info.hardware_vendor_id_known) info.hardware_vendor_id = vendor_id;

    ClPlatformId platform = nullptr;
    if (!info.provider_known &&
        api.Query(device, kClDevicePlatform, platform) && platform) {
        info.provider = api.QueryPlatformString(platform, kClPlatformVendor);
        info.provider_known = !info.provider.empty();
    }

    const std::string extensions =
        api.QueryString(device, kClDeviceExtensions);

    if (HasOpenClExtension(extensions, "cl_khr_device_uuid")) {
        std::array<unsigned char, kClUuidSizeKhr> uuid{};
        if (api.Query(device, kClDeviceUuidKhr, uuid)) {
            info.hardware_uuid = HexBytes(uuid.data(), uuid.size());
            info.hardware_uuid_known = !info.hardware_uuid.empty();
        }
        ClBool luid_valid = 0;
        std::array<unsigned char, kClLuidSizeKhr> luid{};
        if (api.Query(device, kClDeviceLuidValidKhr, luid_valid) &&
            luid_valid == kClTrue &&
            api.Query(device, kClDeviceLuidKhr, luid)) {
            info.hardware_luid = HexBytes(luid.data(), luid.size());
            info.hardware_luid_known = !info.hardware_luid.empty();
        }
    }

    if (HasOpenClExtension(extensions, "cl_khr_pci_bus_info")) {
        ClPciBusInfoKhr pci{};
        if (api.Query(device, kClDevicePciBusInfoKhr, pci)) {
            info.pci_domain = static_cast<int>(pci.pci_domain);
            info.pci_bus = static_cast<int>(pci.pci_bus);
            info.pci_device = static_cast<int>(pci.pci_device);
            info.pci_function = static_cast<int>(pci.pci_function);
            info.pci_location_known = true;
        }
    }

    if (!info.pci_location_known &&
        HasOpenClExtension(extensions, "cl_nv_device_attribute_query")) {
        ClUint bus = 0;
        ClUint slot = 0;
        ClUint domain = 0;
        if (api.Query(device, kClDevicePciBusIdNv, bus) &&
            api.Query(device, kClDevicePciSlotIdNv, slot)) {
            api.Query(device, kClDevicePciDomainIdNv, domain);
            info.pci_domain = static_cast<int>(domain);
            info.pci_bus = static_cast<int>(bus);
            info.pci_device = static_cast<int>(slot >> 3U);
            info.pci_function = static_cast<int>(slot & 0x7U);
            info.pci_location_known = true;
        }
    }
}
#endif

struct CudaUuid {
    char bytes[16];
};

void EnrichCudaIdentity(DeviceInfo& info) {
#ifdef _WIN32
    DynamicLibrary arrayfire_cuda(L"afcuda.dll", true);
    DynamicLibrary cuda_driver(L"nvcuda.dll");
#else
    DynamicLibrary arrayfire_cuda("libafcuda.so", true);
    DynamicLibrary cuda_driver("libcuda.so.1");
#endif

    using AfGetNativeId = af_err (*)(int*, int);
    using CuInit = int (*)(unsigned int);
    using CuDeviceGet = int (*)(int*, int);
    using CuDeviceGetUuid = int (*)(CudaUuid*, int);
    using CuDeviceGetAttribute = int (*)(int*, int, int);
    using CuDeviceTotalMem = int (*)(size_t*, int);

    const auto af_get_native_id =
        FindFunction<AfGetNativeId>(arrayfire_cuda, "afcu_get_native_id");
    const auto cu_init = FindFunction<CuInit>(cuda_driver, "cuInit");
    const auto cu_device_get =
        FindFunction<CuDeviceGet>(cuda_driver, "cuDeviceGet");
    auto cu_device_get_uuid = FindFunction<CuDeviceGetUuid>(
        cuda_driver, "cuDeviceGetUuid_v2");
    if (!cu_device_get_uuid) {
        cu_device_get_uuid = FindFunction<CuDeviceGetUuid>(
            cuda_driver, "cuDeviceGetUuid");
    }
    const auto cu_device_get_attribute =
        FindFunction<CuDeviceGetAttribute>(cuda_driver,
                                          "cuDeviceGetAttribute");
    const auto cu_device_total_mem =
        FindFunction<CuDeviceTotalMem>(cuda_driver, "cuDeviceTotalMem_v2");

    if (!af_get_native_id || !cu_init || !cu_device_get ||
        cu_init(0) != 0) {
        return;
    }

    int native_id = 0;
    int cuda_device = 0;
    if (af_get_native_id(&native_id, info.device_id) != AF_SUCCESS ||
        cu_device_get(&cuda_device, native_id) != 0) {
        return;
    }

    info.provider = "NVIDIA CUDA";
    info.provider_known = true;
    info.hardware_vendor_id = 0x10de;
    info.hardware_vendor_id_known = true;

    // Total device memory (admission compares job estimates against it).
    size_t total_memory = 0;
    if (cu_device_total_mem && cu_device_total_mem(&total_memory, cuda_device) == 0 &&
        total_memory > 0) {
        info.memory_total = total_memory;
        info.memory_total_known = true;
    }

    if (cu_device_get_uuid) {
        CudaUuid uuid{};
        if (cu_device_get_uuid(&uuid, cuda_device) == 0) {
            info.hardware_uuid = HexBytes(
                reinterpret_cast<const unsigned char*>(uuid.bytes),
                sizeof(uuid.bytes));
            info.hardware_uuid_known = !info.hardware_uuid.empty();
        }
    }

    if (cu_device_get_attribute) {
        constexpr int kPciBusId = 33;
        constexpr int kPciDeviceId = 34;
        constexpr int kPciDomainId = 50;
        int domain = 0;
        int bus = 0;
        int device = 0;
        if (cu_device_get_attribute(&domain, kPciDomainId, cuda_device) == 0 &&
            cu_device_get_attribute(&bus, kPciBusId, cuda_device) == 0 &&
            cu_device_get_attribute(&device, kPciDeviceId, cuda_device) == 0) {
            info.pci_domain = domain;
            info.pci_bus = bus;
            info.pci_device = device;
            info.pci_function = 0;
            info.pci_location_known = true;
        }
    }
}
#endif

} // namespace

void FinalizeDeviceIdentity(DeviceInfo& info) {
    info.physical_fingerprint.clear();
    info.physical_fingerprint_known = false;

    if (info.hardware_uuid_known && !info.hardware_uuid.empty()) {
        info.hardware_uuid = LowerAscii(info.hardware_uuid);
        info.physical_fingerprint = "uuid:" + info.hardware_uuid;
    } else if (info.hardware_luid_known && !info.hardware_luid.empty()) {
        info.hardware_luid = LowerAscii(info.hardware_luid);
        info.physical_fingerprint = "luid:" + info.hardware_luid;
    } else if (info.pci_location_known &&
               info.hardware_vendor_id_known) {
        info.physical_fingerprint = PciFingerprint(info);
    }

    info.physical_fingerprint_known = !info.physical_fingerprint.empty();
    if (info.physical_fingerprint_known) {
        info.identity_confidence = DeviceIdentityConfidence::StableHardware;
    } else if (info.provider_known || info.driver_version_known ||
               info.hardware_vendor_id_known ||
               info.hardware_device_id_known || info.pci_location_known ||
               info.hardware_uuid_known || info.hardware_luid_known) {
        info.identity_confidence =
            DeviceIdentityConfidence::ProviderReported;
    }
}

DeviceRouteResolution ResolvePhysicalDeviceRouteInternal(
    const std::vector<DeviceInfo>& inventory,
    DeviceType type,
    const std::string& physical_fingerprint) {
    DeviceRouteResolution result;
    result.type = type;
    if (physical_fingerprint.empty()) return result;

    const std::string normalized = LowerAscii(physical_fingerprint);
    for (const auto& device : inventory) {
        if (device.type != type || !device.physical_fingerprint_known ||
            LowerAscii(device.physical_fingerprint) != normalized) {
            continue;
        }
        if (result.status == DeviceRouteResolutionStatus::Resolved) {
            result.status = DeviceRouteResolutionStatus::Ambiguous;
            result.device_id = -1;
            return result;
        }
        result.status = DeviceRouteResolutionStatus::Resolved;
        result.device_id = device.device_id;
    }

    if (result.status != DeviceRouteResolutionStatus::Resolved) {
        result.status = DeviceRouteResolutionStatus::NotFound;
    }
    return result;
}

void EnrichSelectedDeviceIdentity(DeviceInfo& info) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        if (info.type == DeviceType::CUDA) {
            EnrichCudaIdentity(info);
        }
#ifdef CYXWIZ_ENABLE_OPENCL
        else if (info.type == DeviceType::OPENCL) {
            EnrichOpenClIdentity(info);
        }
#endif
    } catch (const std::exception&) {
        // Optional identity telemetry remains explicitly unknown.
    }
#endif
    FinalizeDeviceIdentity(info);
}

} // namespace cyxwiz::detail

namespace cyxwiz {

DeviceRouteResolution ResolvePhysicalDeviceRoute(
    const std::vector<DeviceInfo>& inventory,
    DeviceType type,
    const std::string& physical_fingerprint) {
    return detail::ResolvePhysicalDeviceRouteInternal(
        inventory, type, physical_fingerprint);
}

} // namespace cyxwiz
