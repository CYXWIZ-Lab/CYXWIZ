#include "cyxwiz/cyxwiz.h"
#include "cyxwiz/engine.h"
#include <spdlog/spdlog.h>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

static bool g_initialized = false;

bool Initialize() {
    if (g_initialized) {
        spdlog::warn("CyxWiz backend already initialized");
        return true;
    }

    spdlog::info("Initializing CyxWiz Backend v{}.{}.{}",
                 CYXWIZ_VERSION_MAJOR,
                 CYXWIZ_VERSION_MINOR,
                 CYXWIZ_VERSION_PATCH);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::info();
        spdlog::info("ArrayFire initialized successfully");

        // Prioritize CUDA backend over OpenCL when both are available
#ifdef CYXWIZ_ENABLE_CUDA
        try {
            af::setBackend(AF_BACKEND_CUDA);
            af::setDevice(0);
            // ArrayFire 3.9 API: deviceInfo requires output parameters
            char d_name[256], d_platform[256], d_toolkit[256], d_compute[256];
            af::deviceInfo(d_name, d_platform, d_toolkit, d_compute);
            spdlog::info("CUDA backend active - Device: {}", d_name);
        } catch (const af::exception& e) {
            spdlog::warn("Failed to set CUDA backend: {}", e.what());
#ifdef CYXWIZ_ENABLE_OPENCL
            // Fallback to OpenCL if CUDA fails
            af::setBackend(AF_BACKEND_OPENCL);
            spdlog::info("OpenCL backend available (fallback)");
#endif
        }
#elif defined(CYXWIZ_ENABLE_OPENCL)
        af::setBackend(AF_BACKEND_OPENCL);
        spdlog::info("OpenCL backend available");
#endif

    } catch (const af::exception& e) {
        spdlog::error("ArrayFire initialization failed: {}", e.what());
        return false;
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

void Shutdown() {
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
    static char version[32];
    snprintf(version, sizeof(version), "%d.%d.%d",
             CYXWIZ_VERSION_MAJOR,
             CYXWIZ_VERSION_MINOR,
             CYXWIZ_VERSION_PATCH);
    return version;
}

} // namespace cyxwiz
