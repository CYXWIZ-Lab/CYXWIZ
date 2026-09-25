// The process-wide selected device vs ArrayFire's thread-local active backend.
//
// Regression for the Dell run (2026-09-25): the training executor resolved
// "the current process device" on its freshly created worker thread, where
// ArrayFire reports its default backend (oneAPI on a machine without CUDA),
// not the OpenCL route the Engine had activated on the main thread. The run
// then asked for an unqualified oneAPI route and failed preflight. The
// process device is now recorded by activations and read across threads.
#include <catch2/catch_test_macros.hpp>

#include <cyxwiz/device.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <arrayfire.h>

#include <optional>
#include <thread>

namespace {

std::optional<cyxwiz::ProcessDeviceSelection> ReadOnWorkerThread() {
    std::optional<cyxwiz::ProcessDeviceSelection> seen;
    std::thread worker([&] { seen = cyxwiz::Device::GetProcessDevice(); });
    worker.join();
    return seen;
}

struct RestoreCpu {
    ~RestoreCpu() {
        try {
            cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).SetActive();
        } catch (...) {
        }
    }
};

}  // namespace

TEST_CASE("SetActive records the process device for other threads", "[device][process_device]") {
    RestoreCpu restore;
    cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).SetActive();
    const auto main_view = cyxwiz::Device::GetProcessDevice();
    REQUIRE(main_view.has_value());
    CHECK(main_view->type == cyxwiz::DeviceType::CPU);
    CHECK(main_view->device_id == 0);

    const auto worker_view = ReadOnWorkerThread();
    REQUIRE(worker_view.has_value());
    CHECK(worker_view->type == cyxwiz::DeviceType::CPU);
    CHECK(worker_view->device_id == 0);
}

TEST_CASE("A worker thread's ArrayFire device is not the process device", "[device][process_device]") {
    // Pick a non-CPU route when one exists so the two notions can differ.
    std::optional<cyxwiz::DeviceInfo> accelerator;
    for (const auto& info : cyxwiz::Device::GetAvailableDevices()) {
        if (info.device_selectable &&
            (info.type == cyxwiz::DeviceType::OPENCL || info.type == cyxwiz::DeviceType::CUDA)) {
            accelerator = info;
            break;
        }
    }
    if (!accelerator) {
        WARN("no CUDA/OpenCL route on this machine; cross-thread check limited to CPU");
        return;
    }
    RestoreCpu restore;
    cyxwiz::Device selected(accelerator->type, accelerator->device_id);
    const auto activation = selected.ActivateExact(true);
    if (!activation.success) {
        WARN("route could not be activated here: " << activation.message);
        return;
    }
    cyxwiz::Device::RecordProcessDevice(accelerator->type, accelerator->device_id);

    // On a fresh thread the thread-local runtime device is ArrayFire's
    // default, which need not be the selected route; the process record is.
    const auto worker_view = ReadOnWorkerThread();
    REQUIRE(worker_view.has_value());
    CHECK(worker_view->type == accelerator->type);
    CHECK(worker_view->device_id == accelerator->device_id);
}
