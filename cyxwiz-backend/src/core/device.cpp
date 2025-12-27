#include "cyxwiz/device.h"
#include <spdlog/spdlog.h>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
// Include platform-specific headers for memory queries
#ifdef CYXWIZ_ENABLE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef CYXWIZ_ENABLE_OPENCL
#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <af/opencl.h>  // For afcl namespace
#endif
#endif

namespace cyxwiz {

static Device* g_current_device = nullptr;

Device::Device(DeviceType type, int device_id)
    : type_(type), device_id_(device_id) {
}

Device::~Device() {
}

DeviceInfo Device::GetInfo() const {
    DeviceInfo info;
    info.type = type_;
    info.device_id = device_id_;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (type_ == DeviceType::CUDA || type_ == DeviceType::OPENCL) {
        char name[256];
        char platform[256];
        char toolkit[256];
        char compute[256];
        af::deviceInfo(name, platform, toolkit, compute);

        info.name = std::string(name);

        // Get actual device memory using platform-specific APIs
        af::Backend backend = af::getActiveBackend();

        if (backend == AF_BACKEND_CUDA) {
#ifdef CYXWIZ_ENABLE_CUDA
            // Use CUDA API to get actual device memory
            spdlog::info("CUDA backend detected, querying memory for device {}", device_id_);

            // First, ensure we're querying the correct CUDA device
            cudaError_t set_err = cudaSetDevice(device_id_);
            if (set_err != cudaSuccess) {
                spdlog::error("Failed to set CUDA device {}: {}", device_id_, cudaGetErrorString(set_err));
                info.memory_total = 0;
                info.memory_available = 0;
            } else {
                spdlog::info("Successfully set CUDA device {}", device_id_);
                size_t free_bytes = 0, total_bytes = 0;
                cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
                if (err == cudaSuccess) {
                    info.memory_total = total_bytes;
                    info.memory_available = free_bytes;
                    spdlog::info("CUDA device {}: {} GB total, {} GB free (raw: {} / {})",
                        device_id_,
                        total_bytes / (1024.0 * 1024.0 * 1024.0),
                        free_bytes / (1024.0 * 1024.0 * 1024.0),
                        total_bytes,
                        free_bytes);
                } else {
                    spdlog::error("Failed to query CUDA memory: {}", cudaGetErrorString(err));
                    info.memory_total = 0;
                    info.memory_available = 0;
                }
            }
#else
            // CUDA not available, fall back to zero
            // This is expected in GUI mode - GPU metrics come from daemon
            spdlog::debug("CYXWIZ_ENABLE_CUDA not defined, GPU memory info unavailable locally");
            info.memory_total = 0;
            info.memory_available = 0;
#endif
        } else if (backend == AF_BACKEND_OPENCL) {
            // For OpenCL, try to get memory info or estimate from device name
            bool got_memory = false;

#ifdef CYXWIZ_ENABLE_OPENCL
            // Use OpenCL API to get device memory
            try {
                cl_device_id cl_device = afcl::getDeviceId();
                if (cl_device != nullptr) {
                    cl_ulong total_mem = 0;
                    cl_int err = clGetDeviceInfo(cl_device, CL_DEVICE_GLOBAL_MEM_SIZE,
                        sizeof(cl_ulong), &total_mem, nullptr);

                    if (err == CL_SUCCESS && total_mem > 0) {
                        size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
                        af::deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

                        info.memory_total = total_mem;
                        info.memory_available = total_mem - (alloc_bytes + lock_bytes);
                        got_memory = true;

                        spdlog::info("OpenCL device {} memory: {:.2f} GB total, {:.2f} GB available",
                            device_id_,
                            total_mem / (1024.0 * 1024.0 * 1024.0),
                            info.memory_available / (1024.0 * 1024.0 * 1024.0));
                    } else {
                        spdlog::warn("OpenCL clGetDeviceInfo failed with error: {}", err);
                    }
                } else {
                    spdlog::warn("OpenCL afcl::getDeviceId() returned null");
                }
            } catch (const std::exception& e) {
                spdlog::warn("OpenCL memory query exception: {}", e.what());
            }
#endif
            // Fallback: estimate memory from device name
            if (!got_memory) {
                std::string device_name = info.name;

                // Intel integrated graphics
                if (device_name.find("UHD") != std::string::npos ||
                    device_name.find("Iris") != std::string::npos ||
                    device_name.find("Intel(R) Graphics") != std::string::npos ||
                    (device_name.find("Intel") != std::string::npos && device_name.find("Graphics") != std::string::npos)) {
                    // Intel integrated GPUs share system RAM, typically allocate 2-8 GB
                    info.memory_total = 4LL * 1024 * 1024 * 1024; // 4 GB shared estimate
                    info.memory_available = static_cast<size_t>(info.memory_total * 0.8);
                    spdlog::info("Estimated memory for Intel GPU '{}': {} GB shared", device_name,
                        info.memory_total / (1024.0 * 1024.0 * 1024.0));
                }
                // AMD integrated
                else if (device_name.find("Radeon") != std::string::npos &&
                         device_name.find("Vega") != std::string::npos) {
                    info.memory_total = 2LL * 1024 * 1024 * 1024; // 2 GB shared
                    info.memory_available = static_cast<size_t>(info.memory_total * 0.8);
                }
                // NVIDIA via OpenCL (shouldn't happen if CUDA is available)
                else if (device_name.find("NVIDIA") != std::string::npos ||
                         device_name.find("GeForce") != std::string::npos) {
                    // Try to match common GPU models
                    if (device_name.find("1050") != std::string::npos) {
                        info.memory_total = 4LL * 1024 * 1024 * 1024;
                    } else if (device_name.find("1060") != std::string::npos) {
                        info.memory_total = 6LL * 1024 * 1024 * 1024;
                    } else if (device_name.find("1070") != std::string::npos ||
                               device_name.find("1080") != std::string::npos) {
                        info.memory_total = 8LL * 1024 * 1024 * 1024;
                    } else {
                        info.memory_total = 4LL * 1024 * 1024 * 1024; // Default 4GB
                    }
                    info.memory_available = static_cast<size_t>(info.memory_total * 0.9);
                }
                else {
                    // Unknown device, provide a reasonable estimate
                    info.memory_total = 2LL * 1024 * 1024 * 1024;
                    info.memory_available = static_cast<size_t>(info.memory_total * 0.8);
                    spdlog::debug("Unknown OpenCL device '{}', using 2GB estimate", device_name);
                }
            }

        } else {
            // Unknown backend, use ArrayFire's memory info as fallback
            size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
            af::deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
            info.memory_total = alloc_bytes + lock_bytes;
            info.memory_available = lock_bytes;
        }

        // Note: ArrayFire doesn't expose compute units directly
        info.compute_units = 0;
        info.supports_fp64 = true; // Assume true for now
        info.supports_fp16 = false;

        spdlog::debug("Device info: {}, Platform: {}", name, platform);
    } else
#endif
    {
        info.name = "CPU";
        info.memory_total = 0;
        info.memory_available = 0;
        info.compute_units = 0;
        info.supports_fp64 = true;
        info.supports_fp16 = false;
    }

    return info;
}

void Device::SetActive() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        // Switch to the correct backend first
        if (type_ == DeviceType::CUDA) {
            af::setBackend(AF_BACKEND_CUDA);
            af::setDevice(device_id_);
            spdlog::info("Set active device: CUDA device {}", device_id_);
        } else if (type_ == DeviceType::OPENCL) {
            af::setBackend(AF_BACKEND_OPENCL);
            af::setDevice(device_id_);
            spdlog::info("Set active device: OpenCL device {}", device_id_);
        } else if (type_ == DeviceType::CPU) {
            af::setBackend(AF_BACKEND_CPU);
            spdlog::info("Set active device: CPU");
        }
    } catch (const af::exception& e) {
        spdlog::error("Failed to set active device: {}", e.what());
        return;
    }
#endif
    g_current_device = this;
}

bool Device::IsActive() const {
    return g_current_device == this;
}

std::vector<DeviceInfo> Device::GetAvailableDevices() {
    std::vector<DeviceInfo> devices;

    // Always add CPU as first option
    Device cpu_device(DeviceType::CPU, 0);
    devices.push_back(cpu_device.GetInfo());

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // Save current backend to restore later
    af::Backend original_backend = af::getActiveBackend();
    int original_device = af::getDevice();

    // Get available backends bitmask
    int available_backends = af::getAvailableBackends();
    spdlog::debug("Available backends bitmask: {}", available_backends);

    // Enumerate CUDA devices first (preferred)
    if (available_backends & AF_BACKEND_CUDA) {
        try {
            af::setBackend(AF_BACKEND_CUDA);
            int cuda_count = af::getDeviceCount();
            spdlog::debug("Found {} CUDA device(s)", cuda_count);

            for (int i = 0; i < cuda_count; i++) {
                af::setDevice(i);
                Device cuda_device(DeviceType::CUDA, i);
                devices.push_back(cuda_device.GetInfo());
            }
        } catch (const af::exception& e) {
            spdlog::warn("Failed to enumerate CUDA devices: {}", e.what());
        }
    }

    // Enumerate OpenCL devices (includes Intel GPU, AMD GPU)
    if (available_backends & AF_BACKEND_OPENCL) {
        try {
            af::setBackend(AF_BACKEND_OPENCL);
            int opencl_count = af::getDeviceCount();
            spdlog::debug("Found {} OpenCL device(s)", opencl_count);

            for (int i = 0; i < opencl_count; i++) {
                af::setDevice(i);
                Device opencl_device(DeviceType::OPENCL, i);
                DeviceInfo info = opencl_device.GetInfo();

                // Skip if this is the same as a CUDA device (avoid duplicates)
                bool is_duplicate = false;
                for (const auto& existing : devices) {
                    if (existing.type == DeviceType::CUDA &&
                        info.name.find("NVIDIA") != std::string::npos) {
                        is_duplicate = true;
                        break;
                    }
                }

                if (!is_duplicate) {
                    devices.push_back(info);
                }
            }
        } catch (const af::exception& e) {
            spdlog::warn("Failed to enumerate OpenCL devices: {}", e.what());
        }
    }

    // Restore original backend and device
    try {
        af::setBackend(original_backend);
        if (af::getDeviceCount() > original_device) {
            af::setDevice(original_device);
        }
    } catch (const af::exception& e) {
        spdlog::warn("Failed to restore original backend: {}", e.what());
    }
#endif

    return devices;
}

Device* Device::GetCurrentDevice() {
    return g_current_device;
}

} // namespace cyxwiz
