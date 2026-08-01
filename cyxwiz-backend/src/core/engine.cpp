#include "cyxwiz/cyxwiz.h"
#include "cyxwiz/engine.h"
#include <spdlog/spdlog.h>
#include <atomic>
#include <mutex>
#include <string>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

static std::atomic<bool> g_initialized{false};

namespace {

std::mutex& LifecycleMutex() {
    static std::mutex mutex;
    return mutex;
}

} // namespace

#ifdef CYXWIZ_HAS_ARRAYFIRE
namespace {

const char* BackendToString(af::Backend backend) {
    switch (backend) {
        case AF_BACKEND_CPU: return "CPU";
        case AF_BACKEND_CUDA: return "CUDA";
        case AF_BACKEND_OPENCL: return "OpenCL";
        default: return "Unknown";
    }
}

#ifdef CYXWIZ_ENABLE_OPENCL
bool IsLikelyDiscreteGpu(const std::string& device_name) {
    return device_name.find("NVIDIA") != std::string::npos ||
           device_name.find("AMD") != std::string::npos ||
           device_name.find("Radeon") != std::string::npos ||
           device_name.find("GeForce") != std::string::npos;
}

bool TryActivateOpenCLBackend() {
    try {
        af::setBackend(AF_BACKEND_OPENCL);
        int num_devices = af::getDeviceCount();
        if (num_devices <= 0) {
            spdlog::warn("OpenCL backend detected but no OpenCL devices are available");
            return false;
        }

        int best_device = 0;
        bool found_discrete = false;

        for (int i = 0; i < num_devices; ++i) {
            af::setDevice(i);
            char d_name[256], d_platform[256], d_toolkit[256], d_compute[256];
            af::deviceInfo(d_name, d_platform, d_toolkit, d_compute);

            if (!found_discrete && IsLikelyDiscreteGpu(std::string(d_name))) {
                best_device = i;
                found_discrete = true;
                spdlog::info("OpenCL discrete GPU candidate found: {} (device {})", d_name, i);
                break;
            }
        }

        af::setDevice(best_device);
        char d_name[256], d_platform[256], d_toolkit[256], d_compute[256];
        af::deviceInfo(d_name, d_platform, d_toolkit, d_compute);
        spdlog::info("OpenCL backend active - Device {}: {}", best_device, d_name);
        return true;
    } catch (const af::exception& e) {
        spdlog::warn("Failed to activate OpenCL backend: {}", e.what());
        return false;
    }
}
#endif

#ifdef CYXWIZ_ENABLE_CUDA
bool TryActivateCUDABackend() {
    try {
        af::setBackend(AF_BACKEND_CUDA);
        int cuda_count = af::getDeviceCount();
        if (cuda_count <= 0) {
            spdlog::warn("CUDA backend detected but no CUDA devices are available");
            return false;
        }

        af::setDevice(0);
        char d_name[256], d_platform[256], d_toolkit[256], d_compute[256];
        af::deviceInfo(d_name, d_platform, d_toolkit, d_compute);
        spdlog::info("CUDA backend active - Device: {}", d_name);
        return true;
    } catch (const af::exception& e) {
        spdlog::warn("Failed to activate CUDA backend: {}", e.what());
        return false;
    }
}
#endif

bool TryActivateCpuBackend() {
    try {
        af::setBackend(AF_BACKEND_CPU);
        af::setDevice(0);
        spdlog::info("CPU backend active");
        return true;
    } catch (const af::exception& e) {
        spdlog::error("Failed to activate CPU backend: {}", e.what());
        return false;
    }
}

}  // namespace
#endif

bool Initialize() {
    std::lock_guard<std::mutex> lock(LifecycleMutex());

    if (g_initialized) {
        spdlog::warn("CyxWiz backend already initialized");
        return true;
    }

    spdlog::info("Initializing CyxWiz Backend v{}.{}.{}",
                 CYXWIZ_VERSION_MAJOR,
                 CYXWIZ_VERSION_MINOR,
                 CYXWIZ_VERSION_PATCH);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    bool backend_activated = false;

#ifdef CYXWIZ_ENABLE_CUDA
    backend_activated = TryActivateCUDABackend();
#endif

#ifdef CYXWIZ_ENABLE_OPENCL
    if (!backend_activated) {
        backend_activated = TryActivateOpenCLBackend();
    }
#endif

    if (!backend_activated) {
        backend_activated = TryActivateCpuBackend();
    }

    if (!backend_activated) {
        spdlog::error("ArrayFire initialization failed: no usable backend (OpenCL/CUDA/CPU)");
        return false;
    }

    try {
        af::Backend active_backend = af::getActiveBackend();
        spdlog::info("ArrayFire initialized successfully with {} backend", BackendToString(active_backend));
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire initialized but active backend query failed: {}", e.what());
    }
#else
    spdlog::warn("ArrayFire not available - using CPU-only mode");
#endif

#ifdef CYXWIZ_DEBUG
    spdlog::set_level(spdlog::level::debug);
    spdlog::info("Debug mode enabled");
#endif

    g_initialized = true;
    return true;
}

bool IsInitialized() {
    return g_initialized.load(std::memory_order_acquire);
}

void Shutdown() {
    std::lock_guard<std::mutex> lock(LifecycleMutex());

    if (!g_initialized) {
        return;
    }

    spdlog::info("Shutting down CyxWiz Backend");

#ifdef CYXWIZ_HAS_ARRAYFIRE
    af::deviceGC();
#endif

    g_initialized = false;
}

const char* GetVersionString() {
    static const std::string version =
        std::to_string(CYXWIZ_VERSION_MAJOR) + "." +
        std::to_string(CYXWIZ_VERSION_MINOR) + "." +
        std::to_string(CYXWIZ_VERSION_PATCH);
    return version.c_str();
}

} // namespace cyxwiz
