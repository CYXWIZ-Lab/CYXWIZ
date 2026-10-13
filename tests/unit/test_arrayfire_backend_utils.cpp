#include <catch2/catch_test_macros.hpp>

#include "algorithms/arrayfire_backend_utils.h"

#include <cyxwiz/device.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_host_sync_observer_count = 0;
cyxwiz::ArrayFireHostSyncEvent g_last_host_sync_event;

void CaptureHostSyncEvent(const cyxwiz::ArrayFireHostSyncEvent& event) {
    ++g_host_sync_observer_count;
    g_last_host_sync_event = event;
}

} // namespace

TEST_CASE("ArrayFire GPU decision follows active backend across device switches",
          "[arrayfire][device_switch]") {
    const cyxwiz::Device* current = cyxwiz::Device::GetCurrentDevice();
    REQUIRE(current != nullptr);
    const cyxwiz::DeviceType original_type = current->GetType();
    const int original_id = current->GetDeviceId();

    struct RestoreDevice {
        cyxwiz::DeviceType type;
        int id;
        ~RestoreDevice() {
            cyxwiz::Device(type, id).SetActive();
        }
    } restore{original_type, original_id};

    cyxwiz::Device cpu(cyxwiz::DeviceType::CPU, 0);
    cpu.SetActive();
    REQUIRE(cpu.IsActive());
    REQUIRE(cyxwiz::IsCurrentArrayFireBackendAvailable());

    const auto devices = cyxwiz::Device::GetAvailableDevices();
    for (const auto& info : devices) {
        if (info.type != cyxwiz::DeviceType::CUDA &&
            info.type != cyxwiz::DeviceType::OPENCL) {
            continue;
        }

        cyxwiz::Device accelerator(info.type, info.device_id);
        accelerator.SetActive();
        REQUIRE(accelerator.IsActive());
        REQUIRE(cyxwiz::IsCurrentArrayFireBackendAvailable());

        cpu.SetActive();
        REQUIRE(cpu.IsActive());
    }
}

TEST_CASE("ArrayFire fallback reasons classify backend failures", "[arrayfire][fallback]") {
    const char* overflow =
        "NVRTC_ERROR_COMPILATION: Formal parameter space overflowed "
        "(4097 bytes required, max 4096 bytes allowed)";
    REQUIRE(cyxwiz::ClassifyArrayFireBackendFallbackReason(overflow) ==
            cyxwiz::BackendFallbackReason::CudaJitParamOverflow);
    REQUIRE(std::string(cyxwiz::BackendFallbackReasonName(
                cyxwiz::BackendFallbackReason::CudaJitParamOverflow)) ==
            "cuda_jit_param_overflow");

    REQUIRE(cyxwiz::ClassifyArrayFireBackendFallbackReason(
                "ArrayFire JIT compile failed") ==
            cyxwiz::BackendFallbackReason::ArrayFireJitCompileFailure);
    REQUIRE(cyxwiz::ClassifyArrayFireBackendFallbackReason(
                "device lost") ==
            cyxwiz::BackendFallbackReason::GpuBackendException);
    REQUIRE(std::string(cyxwiz::BackendFallbackReasonName(
                cyxwiz::BackendFallbackReason::UnsupportedOperation)) ==
            "unsupported_operation");
    REQUIRE(std::string(cyxwiz::BackendFallbackReasonName(
                cyxwiz::BackendFallbackReason::BackendUnavailable)) ==
            "backend_unavailable");
    REQUIRE(cyxwiz::ClassifyArrayFireBackendFallbackReason(
                "ArrayFire backend unavailable") ==
            cyxwiz::BackendFallbackReason::BackendUnavailable);
}

TEST_CASE("Scoped ArrayFire host sync attribution reaches observer and restores",
          "[arrayfire][host_sync]") {
    const auto original_attribution =
        cyxwiz::GetArrayFireHostSyncAttribution();
    const auto original_observer = cyxwiz::GetArrayFireHostSyncObserver();
    g_host_sync_observer_count = 0;
    g_last_host_sync_event = {};

    {
        const cyxwiz::ScopedArrayFireHostSyncObserver observer(
            &CaptureHostSyncEvent);
        const cyxwiz::ScopedArrayFireHostSyncAttribution attribution(
            cyxwiz::ArrayFireHostSyncCategory::LossScalarReadback,
            "UnitTest::ComputeLoss");
        cyxwiz::ArrayFireHostSyncEvent event;
        event.operation_name = "Tensor::EnsureHostCurrent";
        event.reason_code = "tensor_host_materialization";
        event.tensor_shape = {1};
        event.tensor_dtype = "float32";
        event.tensor_layout = "arrayfire_native";
        event.bytes = sizeof(float);
        cyxwiz::NotifyArrayFireHostSync(event);

        REQUIRE(g_host_sync_observer_count == 1);
        REQUIRE(g_last_host_sync_event.attribution_category ==
                "loss_scalar_readback");
        REQUIRE(g_last_host_sync_event.attribution_operation ==
                "UnitTest::ComputeLoss");
        REQUIRE(g_last_host_sync_event.tensor_shape ==
                std::vector<size_t>{1});
        REQUIRE(g_last_host_sync_event.selected_backend ==
                cyxwiz::CurrentArrayFireBackendName());
    }

    const auto restored = cyxwiz::GetArrayFireHostSyncAttribution();
    REQUIRE(restored.category == original_attribution.category);
    REQUIRE(restored.operation_name == original_attribution.operation_name);
    REQUIRE(cyxwiz::GetArrayFireHostSyncObserver() == original_observer);
}
