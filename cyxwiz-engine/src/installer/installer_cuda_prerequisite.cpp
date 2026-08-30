#include "installer_cuda_prerequisite.h"

#include <algorithm>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace cyxwiz::installer {
namespace {

class CudaDriverLibrary {
public:
#ifdef _WIN32
    CudaDriverLibrary()
        : handle_(::LoadLibraryExW(
              L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32)) {}

    ~CudaDriverLibrary() {
        if (handle_) ::FreeLibrary(handle_);
    }

    void* Find(const char* name) const {
        return handle_
            ? reinterpret_cast<void*>(::GetProcAddress(handle_, name))
            : nullptr;
    }

private:
    HMODULE handle_ = nullptr;
#elif defined(__APPLE__)
    void* Find(const char*) const { return nullptr; }
#else
    CudaDriverLibrary()
        : handle_(::dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL)) {}

    ~CudaDriverLibrary() {
        if (handle_) ::dlclose(handle_);
    }

    void* Find(const char* name) const {
        return handle_ ? ::dlsym(handle_, name) : nullptr;
    }

private:
    void* handle_ = nullptr;
#endif

public:
    bool Loaded() const {
#if defined(__APPLE__)
        return false;
#else
        return handle_ != nullptr;
#endif
    }
};

template <typename Function>
Function FindFunction(
    const CudaDriverLibrary& library,
    const char* name) {
    return reinterpret_cast<Function>(library.Find(name));
}

std::string FormatCudaDriverApiVersion(int version) {
    if (version <= 0) return {};
    const int major = version / 1000;
    const int minor = (version % 1000) / 10;
    return std::to_string(major) + "." + std::to_string(minor);
}

}  // namespace

InstallerCudaPrerequisiteState EvaluateInstallerCudaDriverProbe(
    const InstallerCudaDriverProbe& probe) {
    InstallerCudaPrerequisiteState state;
    state.driver_detected = probe.driver_library_found;
    if (!probe.driver_library_found) {
        state.message =
            "No NVIDIA display driver was detected on this system.";
        return state;
    }
    if (!probe.required_api_found) {
        state.message =
            "The NVIDIA driver is present but its CUDA driver API is incomplete.";
        return state;
    }
    if (probe.initialization_status != 0) {
        state.message =
            "The NVIDIA CUDA driver is present but could not initialize a device.";
        return state;
    }
    state.device_count = std::max(0, probe.device_count);
    state.driver_api_version =
        FormatCudaDriverApiVersion(probe.driver_api_version);
    state.device_available = state.device_count > 0;
    if (!state.device_available) {
        state.message =
            "The NVIDIA CUDA driver is present, but no CUDA device is available.";
        return state;
    }
    state.message = "Detected " + std::to_string(state.device_count) +
        " NVIDIA CUDA device(s)";
    if (!state.driver_api_version.empty()) {
        state.message += " with driver API " + state.driver_api_version;
    }
    state.message +=
        ". The NVIDIA driver prerequisite is already satisfied; CyxWiz will "
        "install only its signed app-local CUDA backend pack.";
    return state;
}

InstallerCudaPrerequisiteState DetectInstallerCudaPrerequisite() {
    CudaDriverLibrary library;
    InstallerCudaDriverProbe probe;
    probe.driver_library_found = library.Loaded();
    if (!probe.driver_library_found) {
        return EvaluateInstallerCudaDriverProbe(probe);
    }

    using CuInit = int (*)(unsigned int);
    using CuDeviceGetCount = int (*)(int*);
    using CuDriverGetVersion = int (*)(int*);
    const auto cu_init = FindFunction<CuInit>(library, "cuInit");
    const auto cu_device_get_count =
        FindFunction<CuDeviceGetCount>(library, "cuDeviceGetCount");
    const auto cu_driver_get_version =
        FindFunction<CuDriverGetVersion>(library, "cuDriverGetVersion");
    probe.required_api_found =
        cu_init && cu_device_get_count && cu_driver_get_version;
    if (!probe.required_api_found) {
        return EvaluateInstallerCudaDriverProbe(probe);
    }

    probe.initialization_status = cu_init(0);
    if (probe.initialization_status == 0) {
        if (cu_device_get_count(&probe.device_count) != 0) {
            probe.device_count = 0;
        }
        if (cu_driver_get_version(&probe.driver_api_version) != 0) {
            probe.driver_api_version = 0;
        }
    }
    return EvaluateInstallerCudaDriverProbe(probe);
}

}  // namespace cyxwiz::installer
