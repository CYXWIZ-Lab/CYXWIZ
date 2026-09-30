#include "cyxwiz/cyxwiz.h"
#include "cyxwiz/engine.h"
#include <spdlog/spdlog.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

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
        case AF_BACKEND_ONEAPI: return "oneAPI";
        default: return "Unknown";
    }
}

const char* ArrayFireErrorName(af_err error) {
    const char* name = af_err_to_string(error);
    return name != nullptr ? name : "unknown ArrayFire error";
}

void LogActiveDevice(const char* backend_name, DeviceType type, int device_id) {
    // Only successful startup activations reach here: this is the process
    // device until the owner selects another one.
    Device::RecordProcessDevice(type, device_id);
    const DeviceInfo info = Device(type, device_id).GetInfo();
    spdlog::info(
        "{} backend active - Device {}: {} (metadata={})",
        backend_name,
        device_id,
        info.name,
        DeviceMetadataStatusName(info.metadata_status));
}

void LogActivationFailure(const char* backend_name, af::exception& error) {
    spdlog::warn(
        "Failed to activate {} backend: error={} ({})",
        backend_name,
        static_cast<int>(error.err()),
        ArrayFireErrorName(error.err()));
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
            const DeviceInfo info = Device(DeviceType::OPENCL, i).GetInfo();
            if (!found_discrete && info.name_known && IsLikelyDiscreteGpu(info.name)) {
                best_device = i;
                found_discrete = true;
                spdlog::info(
                    "OpenCL discrete GPU candidate found: {} (device {})",
                    info.name,
                    i);
                break;
            }
        }

        af::setDevice(best_device);
        LogActiveDevice("OpenCL", DeviceType::OPENCL, best_device);
        return true;
    } catch (af::exception& error) {
        LogActivationFailure("OpenCL", error);
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
        LogActiveDevice("CUDA", DeviceType::CUDA, 0);
        return true;
    } catch (af::exception& error) {
        LogActivationFailure("CUDA", error);
        return false;
    }
}
#endif

bool TryActivateOneAPIBackend() {
    try {
        af::setBackend(AF_BACKEND_ONEAPI);
        int oneapi_count = af::getDeviceCount();
        if (oneapi_count <= 0) {
            spdlog::warn("oneAPI backend detected but no oneAPI devices are available");
            return false;
        }

        af::setDevice(0);
        LogActiveDevice("oneAPI", DeviceType::ONEAPI, 0);
        return true;
    } catch (af::exception& error) {
        LogActivationFailure("oneAPI", error);
        return false;
    }
}

bool TryActivateCpuBackend() {
    try {
        af::setBackend(AF_BACKEND_CPU);
        af::setDevice(0);
        Device::RecordProcessDevice(DeviceType::CPU, 0);
        spdlog::info("ArrayFire CPU backend active");
        return true;
    } catch (af::exception& error) {
        spdlog::error(
            "Failed to activate ArrayFire CPU backend: error={} ({})",
            static_cast<int>(error.err()),
            ArrayFireErrorName(error.err()));
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

    // Each activation step is timed, and a watchdog names a step that runs
    // longer than 30 s (first-run driver/kernel caches can take minutes on a
    // busy machine; without this a stall looks like a silent hang).
    std::mutex step_mutex;
    std::condition_variable step_cv;
    const char* current_step = "starting";
    auto step_started = std::chrono::steady_clock::now();
    bool init_done = false;
    std::thread watchdog([&]() {
        std::unique_lock<std::mutex> guard(step_mutex);
        while (!step_cv.wait_for(guard, std::chrono::seconds(30),
                                 [&]() { return init_done; })) {
            const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - step_started).count();
            spdlog::warn(
                "Backend initialization is still running: {} for {} s. First "
                "use of a GPU backend can take minutes while drivers build "
                "their caches, especially when the machine is busy.",
                current_step, seconds);
        }
    });
    const auto run_step = [&](const char* name, const auto& activate) {
        {
            std::lock_guard<std::mutex> guard(step_mutex);
            current_step = name;
            step_started = std::chrono::steady_clock::now();
        }
        spdlog::info("Backend initialization: trying {}", name);
        const auto started = std::chrono::steady_clock::now();
        const bool activated = activate();
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started).count();
        spdlog::info("Backend initialization: {} {} in {:.1f} s", name,
                     activated ? "activated" : "not activated",
                     static_cast<double>(elapsed_ms) / 1000.0);
        return activated;
    };
    const auto stop_watchdog = [&]() {
        {
            std::lock_guard<std::mutex> guard(step_mutex);
            init_done = true;
        }
        step_cv.notify_all();
        if (watchdog.joinable()) watchdog.join();
    };

#ifdef CYXWIZ_ENABLE_CUDA
    backend_activated = run_step("CUDA", [] { return TryActivateCUDABackend(); });
#endif

    if (!backend_activated && IsUncertifiedOneAPITrainingEnabled()) {
        backend_activated = run_step("oneAPI", [] { return TryActivateOneAPIBackend(); });
    } else if (!backend_activated) {
        spdlog::info(
            "Skipping automatic oneAPI activation because training support "
            "is not certified by this CyxWiz release; this is not a "
            "device-specific qualification result. Discovery remains "
            "available and "
            "CYXWIZ_ENABLE_UNCERTIFIED_ONEAPI_TRAINING=1 enables isolated "
            "diagnostics");
    }

#ifdef CYXWIZ_ENABLE_OPENCL
    if (!backend_activated) {
        backend_activated = run_step("OpenCL", [] { return TryActivateOpenCLBackend(); });
    }
#endif

    if (!backend_activated) {
        backend_activated = run_step("CPU", [] { return TryActivateCpuBackend(); });
    }
    stop_watchdog();

    if (!backend_activated) {
        spdlog::error("ArrayFire initialization failed: no usable backend (CUDA/oneAPI/OpenCL/CPU)");
        return false;
    }

    try {
        af::Backend active_backend = af::getActiveBackend();
        spdlog::info("ArrayFire initialized successfully with {} backend", BackendToString(active_backend));
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire initialized but active backend query failed: {}", e.what());
    }
#else
    spdlog::warn("ArrayFire not available - backend compiled in native CPU-only mode");
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
