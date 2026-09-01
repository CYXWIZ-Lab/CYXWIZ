#include "device_identity.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <string>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>

#ifdef CYXWIZ_ENABLE_OPENCL
#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <OpenCL/cl_ext.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif
#include <af/opencl.h>
#endif

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
#ifdef CYXWIZ_ENABLE_OPENCL
template <typename Value>
bool QueryOpenClValue(cl_device_id device,
                      cl_device_info field,
                      Value& value) {
    return clGetDeviceInfo(device,
                           field,
                           sizeof(Value),
                           &value,
                           nullptr) == CL_SUCCESS;
}

std::string QueryOpenClString(cl_device_id device, cl_device_info field) {
    size_t size = 0;
    if (clGetDeviceInfo(device, field, 0, nullptr, &size) != CL_SUCCESS ||
        size <= 1) {
        return {};
    }
    std::string value(size, '\0');
    if (clGetDeviceInfo(device, field, size, value.data(), nullptr) !=
        CL_SUCCESS) {
        return {};
    }
    while (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}

std::string QueryOpenClPlatformString(cl_platform_id platform,
                                     cl_platform_info field) {
    size_t size = 0;
    if (clGetPlatformInfo(platform, field, 0, nullptr, &size) != CL_SUCCESS ||
        size <= 1) {
        return {};
    }
    std::string value(size, '\0');
    if (clGetPlatformInfo(platform, field, size, value.data(), nullptr) !=
        CL_SUCCESS) {
        return {};
    }
    while (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}

#if (defined(CL_DEVICE_UUID_KHR) && defined(CL_UUID_SIZE_KHR)) || \
    (defined(CL_DEVICE_LUID_KHR) && defined(CL_LUID_SIZE_KHR) && \
     defined(CL_DEVICE_LUID_VALID_KHR)) || \
    defined(CL_DEVICE_PCI_BUS_INFO_KHR) || \
    (defined(CL_DEVICE_PCI_BUS_ID_NV) && defined(CL_DEVICE_PCI_SLOT_ID_NV))
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
#endif

void EnrichOpenClIdentity(DeviceInfo& info) {
    const cl_device_id device = afcl::getDeviceId();

    cl_device_type type = 0;
    if (QueryOpenClValue(device, CL_DEVICE_TYPE, type)) {
        if ((type & CL_DEVICE_TYPE_CPU) != 0) {
            info.kind = DeviceKind::CPU;
        } else if ((type & CL_DEVICE_TYPE_GPU) != 0) {
            info.kind = DeviceKind::GPU;
        } else if ((type & CL_DEVICE_TYPE_ACCELERATOR) != 0) {
            info.kind = DeviceKind::Accelerator;
        }
    }

    cl_ulong total_memory = 0;
    if (QueryOpenClValue(device, CL_DEVICE_GLOBAL_MEM_SIZE, total_memory) &&
        total_memory > 0) {
        info.memory_total = static_cast<size_t>(total_memory);
        info.memory_total_known = true;
    }

    info.provider = QueryOpenClString(device, CL_DEVICE_VENDOR);
    info.provider_known = !info.provider.empty();
    info.driver_version = QueryOpenClString(device, CL_DRIVER_VERSION);
    info.driver_version_known = !info.driver_version.empty();
    info.hardware_vendor_id_known = QueryOpenClValue(
        device, CL_DEVICE_VENDOR_ID, info.hardware_vendor_id);

    cl_platform_id platform = nullptr;
    if (!info.provider_known &&
        QueryOpenClValue(device, CL_DEVICE_PLATFORM, platform) && platform) {
        info.provider = QueryOpenClPlatformString(platform, CL_PLATFORM_VENDOR);
        info.provider_known = !info.provider.empty();
    }

    const std::string extensions =
        QueryOpenClString(device, CL_DEVICE_EXTENSIONS);

#if defined(CL_DEVICE_UUID_KHR) && defined(CL_UUID_SIZE_KHR)
    if (HasOpenClExtension(extensions, "cl_khr_device_uuid")) {
        std::array<unsigned char, CL_UUID_SIZE_KHR> uuid{};
        if (QueryOpenClValue(device, CL_DEVICE_UUID_KHR, uuid)) {
            info.hardware_uuid = HexBytes(uuid.data(), uuid.size());
            info.hardware_uuid_known = !info.hardware_uuid.empty();
        }
    }
#endif

#if defined(CL_DEVICE_LUID_KHR) && defined(CL_LUID_SIZE_KHR) && \
    defined(CL_DEVICE_LUID_VALID_KHR)
    if (HasOpenClExtension(extensions, "cl_khr_device_uuid")) {
        cl_bool luid_valid = CL_FALSE;
        std::array<unsigned char, CL_LUID_SIZE_KHR> luid{};
        if (QueryOpenClValue(
                device, CL_DEVICE_LUID_VALID_KHR, luid_valid) &&
            luid_valid == CL_TRUE &&
            QueryOpenClValue(device, CL_DEVICE_LUID_KHR, luid)) {
            info.hardware_luid = HexBytes(luid.data(), luid.size());
            info.hardware_luid_known = !info.hardware_luid.empty();
        }
    }
#endif

#if defined(CL_DEVICE_PCI_BUS_INFO_KHR)
    if (HasOpenClExtension(extensions, "cl_khr_pci_bus_info")) {
        cl_device_pci_bus_info_khr pci{};
        if (QueryOpenClValue(device, CL_DEVICE_PCI_BUS_INFO_KHR, pci)) {
            info.pci_domain = static_cast<int>(pci.pci_domain);
            info.pci_bus = static_cast<int>(pci.pci_bus);
            info.pci_device = static_cast<int>(pci.pci_device);
            info.pci_function = static_cast<int>(pci.pci_function);
            info.pci_location_known = true;
        }
    }
#endif

#if defined(CL_DEVICE_PCI_BUS_ID_NV) && defined(CL_DEVICE_PCI_SLOT_ID_NV)
    if (!info.pci_location_known &&
        HasOpenClExtension(extensions, "cl_nv_device_attribute_query")) {
        cl_uint bus = 0;
        cl_uint slot = 0;
        cl_uint domain = 0;
        if (QueryOpenClValue(device, CL_DEVICE_PCI_BUS_ID_NV, bus) &&
            QueryOpenClValue(device, CL_DEVICE_PCI_SLOT_ID_NV, slot)) {
#if defined(CL_DEVICE_PCI_DOMAIN_ID_NV)
            QueryOpenClValue(device, CL_DEVICE_PCI_DOMAIN_ID_NV, domain);
#endif
            info.pci_domain = static_cast<int>(domain);
            info.pci_bus = static_cast<int>(bus);
            info.pci_device = static_cast<int>(slot >> 3U);
            info.pci_function = static_cast<int>(slot & 0x7U);
            info.pci_location_known = true;
        }
    }
#endif
}
#endif

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
