#pragma once

// Device selection for the computation-truth tools, following the existing
// test conventions: CYXWIZ_TEST_ARRAYFIRE_BACKEND=opencl|cuda|oneapi|cpu picks the
// ArrayFire backend, CYXWIZ_OPENCL_TEST_DEVICE=N picks the OpenCL GPU (0 =
// the NVIDIA card, 1 = the Intel UHD 630 on the dev box). Unset = default.
//
// The analysis algorithms compute in float64 where the device has it and in
// float32 otherwise (cyxwiz-backend arrayfire_host_values.h), so a check names
// both tolerances: DeviceTolerance(float64, float32) (TOFIX145 item 7).

#include <cyxwiz/device.h>
#include <cyxwiz/neural_provider.h>

#include <arrayfire.h>
// <windows.h> (through ArrayFire) defines LoadImage as LoadImageA, which would
// rename ImageUtils::LoadImage in the tests that include this (as
// core/materialization_memory_guard.h does), and `small` as char
// (rpcndr.h), which breaks a variable of that name.
#ifdef LoadImage
#undef LoadImage
#endif
#ifdef small
#undef small
#endif

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>

namespace cyxwiz::test {

// The device CYXWIZ_TEST_ARRAYFIRE_BACKEND asks for (nullopt: none asked).
struct RequestedTestDevice {
    DeviceType type = DeviceType::CPU;
    int id = 0;
    std::string name;  // the variable's value: cpu, cuda, opencl, oneapi
};

// False (with the reason on stderr) when the variable names no backend.
inline bool ReadRequestedTestDevice(std::optional<RequestedTestDevice>& out) {
    out.reset();
    const char* backend = std::getenv("CYXWIZ_TEST_ARRAYFIRE_BACKEND");
    if (backend == nullptr || backend[0] == '\0') return true;
    RequestedTestDevice request;
    request.name = backend;
    if (request.name == "cuda") {
        request.type = DeviceType::CUDA;
    } else if (request.name == "opencl") {
        request.type = DeviceType::OPENCL;
        if (const char* index = std::getenv("CYXWIZ_OPENCL_TEST_DEVICE")) request.id = std::atoi(index);
    } else if (request.name == "oneapi") {
        request.type = DeviceType::ONEAPI;
    } else if (request.name != "cpu") {
        std::fprintf(stderr, "unknown CYXWIZ_TEST_ARRAYFIRE_BACKEND '%s'\n", backend);
        return false;
    }
    out = request;
    return true;
}

// Whether the active device computes in float64. Read once, after the device is
// selected.
inline bool DeviceComputesInDouble() {
    static const bool has_double = af::isDoubleAvailable(af::getDevice());
    return has_double;
}

// A check's tolerance on the active device.
inline double DeviceTolerance(double float64_tolerance, double float32_tolerance) {
    return DeviceComputesInDouble() ? float64_tolerance : float32_tolerance;
}

// Activates exactly `device` (ActivateExact fails when ArrayFire makes another
// backend or device active) and prints the device that runs.
inline bool ActivateTestDevice(const RequestedTestDevice& device) {
    const auto result = Device(device.type, device.id).ActivateExact(true);
    if (!result.success) {
        std::fprintf(stderr, "could not activate %s device %d: %s\n", device.name.c_str(), device.id,
                     result.message.c_str());
        return false;
    }
    std::printf("device: %s %d (%s)\n", NeuralDevicePlatformName(CaptureCurrentNeuralDeviceTarget().platform),
                result.effective_device_id, af::isDoubleAvailable(af::getDevice()) ? "float64" : "float32");
    std::fflush(stdout);
    return true;
}

// The requested device, activated exactly. Nothing asked: the default device,
// said so. Returns false when the request is unknown or could not be activated.
inline bool SelectTestDeviceFromEnvironment() {
    std::optional<RequestedTestDevice> request;
    if (!ReadRequestedTestDevice(request)) return false;
    if (!request) {
        std::printf("device: the default (CYXWIZ_TEST_ARRAYFIRE_BACKEND not set)\n");
        std::fflush(stdout);
        return true;
    }
    return ActivateTestDevice(*request);
}

}  // namespace cyxwiz::test
