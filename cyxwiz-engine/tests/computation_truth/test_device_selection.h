#pragma once

// Device selection for the computation-truth tools, following the existing
// test conventions: CYXWIZ_TEST_ARRAYFIRE_BACKEND=opencl|cuda|cpu picks the
// ArrayFire backend, CYXWIZ_OPENCL_TEST_DEVICE=N picks the OpenCL GPU (0 =
// the NVIDIA card, 1 = the Intel UHD 630 on the dev box). Unset = default.

#include <cyxwiz/device.h>
#include <cyxwiz/neural_provider.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace cyxwiz::test {

// Returns false when a requested device could not be activated.
inline bool SelectTestDeviceFromEnvironment() {
    const char* backend = std::getenv("CYXWIZ_TEST_ARRAYFIRE_BACKEND");
    if (backend == nullptr || backend[0] == '\0') return true;
    const std::string name(backend);
    DeviceType type = DeviceType::CPU;
    int id = 0;
    if (name == "cuda") {
        type = DeviceType::CUDA;
    } else if (name == "opencl") {
        type = DeviceType::OPENCL;
        if (const char* index = std::getenv("CYXWIZ_OPENCL_TEST_DEVICE")) id = std::atoi(index);
    } else if (name != "cpu") {
        std::fprintf(stderr, "unknown CYXWIZ_TEST_ARRAYFIRE_BACKEND '%s'\n", backend);
        return false;
    }
    const auto result = Device(type, id).ActivateExact(true);
    if (!result.success) {
        std::fprintf(stderr, "could not activate %s device %d\n", backend, id);
        return false;
    }
    std::printf("device: %s %d\n", NeuralDevicePlatformName(CaptureCurrentNeuralDeviceTarget().platform), id);
    return true;
}

}  // namespace cyxwiz::test
