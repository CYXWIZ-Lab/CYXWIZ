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
#include <CL/cl.h>
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
#ifdef CYXWIZ_ENABLE_OPENCL
            // Use OpenCL API to get device memory
            // ArrayFire stores device info that we can query directly
            try {
                // Get the OpenCL device ID from ArrayFire's internal context
                cl_device_id cl_device = afcl::getDeviceId();

                cl_ulong total_mem = 0;
                cl_int err = clGetDeviceInfo(cl_device, CL_DEVICE_GLOBAL_MEM_SIZE,
                    sizeof(cl_ulong), &total_mem, nullptr);

                if (err == CL_SUCCESS && total_mem > 0) {
                    // OpenCL doesn't have a standard way to get available memory
                    // We'll estimate it based on ArrayFire's memory manager
                    size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
                    af::deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

                    info.memory_total = total_mem;
                    // Estimate available as total minus what ArrayFire has allocated
                    info.memory_available = total_mem - (alloc_bytes + lock_bytes);

                    spdlog::debug("OpenCL device {}: {} GB total, ~{} GB available (AF allocated: {} MB)",
                        device_id_,
                        total_mem / (1024.0 * 1024.0 * 1024.0),
                        info.memory_available / (1024.0 * 1024.0 * 1024.0),
                        (alloc_bytes + lock_bytes) / (1024.0 * 1024.0));
                } else {
                    spdlog::warn("Failed to query OpenCL memory: clGetDeviceInfo returned error {}", err);
                    info.memory_total = 0;
                    info.memory_available = 0;
                }
            } catch (const std::exception& e) {
                spdlog::warn("Failed to query OpenCL memory: {}", e.what());
                // Fallback: try to estimate from device name
                // Many GPUs have standard memory sizes we can guess from
                std::string device_name = info.name;
                info.memory_total = 0;
                info.memory_available = 0;

                // Try to extract memory size from device name (e.g., "GTX 1050 Ti 4GB")
                if (device_name.find("1050") != std::string::npos) {
                    info.memory_total = 4LL * 1024 * 1024 * 1024; // 4 GB
                    info.memory_available = static_cast<size_t>(info.memory_total * 0.9); // Estimate 90% available
                    spdlog::info("Using estimated memory for {}: {} GB", device_name,
                        info.memory_total / (1024.0 * 1024.0 * 1024.0));
                } else if (device_name.find("UHD") != std::string::npos ||
                           device_name.find("Intel") != std::string::npos) {
                    // Integrated Intel graphics typically share system RAM
                    info.memory_total = 2LL * 1024 * 1024 * 1024; // 2 GB shared
                    info.memory_available = static_cast<size_t>(info.memory_total * 0.8);
                    spdlog::info("Using estimated memory for {}: {} GB shared", device_name,
                        info.memory_total / (1024.0 * 1024.0 * 1024.0));
                }
            }
#else
            // OpenCL not available
            info.memory_total = 0;
            info.memory_available = 0;
#endif
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
    if (type_ == DeviceType::CUDA || type_ == DeviceType::OPENCL) {
        af::setDevice(device_id_);
    }
#endif
    g_current_device = this;
}

bool Device::IsActive() const {
    return g_current_device == this;
}

std::vector<DeviceInfo> Device::GetAvailableDevices() {
    std::vector<DeviceInfo> devices;

    // Always add CPU
    Device cpu_device(DeviceType::CPU, 0);
    devices.push_back(cpu_device.GetInfo());

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        // Detect which backend ArrayFire is using
        af::Backend backend = af::getActiveBackend();
        DeviceType device_type = DeviceType::CPU;

        switch (backend) {
            case AF_BACKEND_CUDA:
                device_type = DeviceType::CUDA;
                spdlog::debug("ArrayFire backend: CUDA");
                break;
            case AF_BACKEND_OPENCL:
                device_type = DeviceType::OPENCL;
                spdlog::debug("ArrayFire backend: OpenCL");
                break;
            case AF_BACKEND_CPU:
                device_type = DeviceType::CPU;
                spdlog::debug("ArrayFire backend: CPU");
                break;
            default:
                spdlog::warn("Unknown ArrayFire backend: {}", static_cast<int>(backend));
                device_type = DeviceType::CPU;
                break;
        }

        int device_count = af::getDeviceCount();
        spdlog::debug("Enumerating {} device(s) with backend type: {}",
            device_count, static_cast<int>(device_type));

        for (int i = 0; i < device_count; i++) {
            af::setDevice(i);
            Device gpu_device(device_type, i);
            devices.push_back(gpu_device.GetInfo());
        }
    } catch (const af::exception& e) {
        spdlog::warn("Failed to enumerate GPU devices: {}", e.what());
    }
#endif

    return devices;
}

Device* Device::GetCurrentDevice() {
    return g_current_device;
}

} // namespace cyxwiz
