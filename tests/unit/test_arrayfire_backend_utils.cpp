#include <catch2/catch_test_macros.hpp>

#include "algorithms/arrayfire_backend_utils.h"

#include <cyxwiz/device.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char* kForceFallbackEnv =
    "CYXWIZ_TEST_FORCE_ARRAYFIRE_FALLBACK";

int g_fallback_observer_count = 0;
cyxwiz::ArrayFireNativeCpuFallbackEvent g_last_fallback_event;
int g_host_sync_observer_count = 0;
cyxwiz::ArrayFireHostSyncEvent g_last_host_sync_event;

void CaptureFallbackEvent(
    const cyxwiz::ArrayFireNativeCpuFallbackEvent& event) {
    ++g_fallback_observer_count;
    g_last_fallback_event = event;
}

void CaptureHostSyncEvent(const cyxwiz::ArrayFireHostSyncEvent& event) {
    ++g_host_sync_observer_count;
    g_last_host_sync_event = event;
}

void SetEnvVar(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void ClearEnvVar(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value)
        : name_(name) {
        const char* previous = std::getenv(name);
        if (previous != nullptr) {
            had_previous_ = true;
            previous_ = previous;
        }
        if (value == nullptr) {
            ClearEnvVar(name_);
        } else {
            SetEnvVar(name_, value);
        }
    }

    ~ScopedEnvVar() {
        if (had_previous_) {
            SetEnvVar(name_, previous_.c_str());
        } else {
            ClearEnvVar(name_);
        }
    }

private:
    const char* name_;
    bool had_previous_ = false;
    std::string previous_;
};

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
    REQUIRE_FALSE(cyxwiz::IsCurrentArrayFireBackendGpu());

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
        REQUIRE(cyxwiz::IsCurrentArrayFireBackendGpu());

        cpu.SetActive();
        REQUIRE(cpu.IsActive());
        REQUIRE_FALSE(cyxwiz::IsCurrentArrayFireBackendGpu());
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
}

TEST_CASE("ArrayFire fallback log gate is once per operation reason and context",
          "[arrayfire][fallback]") {
    const std::string context_a =
        cyxwiz::BuildArrayFireBackendFallbackContext(
            cyxwiz::BuildTensorShapeContext(
                "input", std::vector<size_t>{2, 3}),
            "test-backend");
    const std::string context_b =
        cyxwiz::BuildArrayFireBackendFallbackContext(
            cyxwiz::BuildTensorShapeContext(
                "input", std::vector<size_t>{4, 3}),
            "test-backend");

    REQUIRE(cyxwiz::ShouldLogArrayFireBackendFallbackOnce(
        "UnitTestArrayFireFallback::Forward",
        cyxwiz::BackendFallbackReason::GpuBackendException,
        context_a));
    REQUIRE_FALSE(cyxwiz::ShouldLogArrayFireBackendFallbackOnce(
        "UnitTestArrayFireFallback::Forward",
        cyxwiz::BackendFallbackReason::GpuBackendException,
        context_a));
    REQUIRE(cyxwiz::ShouldLogArrayFireBackendFallbackOnce(
        "UnitTestArrayFireFallback::Forward",
        cyxwiz::BackendFallbackReason::GpuBackendException,
        context_b));
}

TEST_CASE("ArrayFire fallback messages can suppress compiler dumps",
          "[arrayfire][fallback]") {
    const std::string context =
        cyxwiz::BuildArrayFireBackendFallbackContext("batch=8", "cuda");
    const std::string message = cyxwiz::BuildArrayFireBackendFallbackMessage(
        "UnitTestArrayFireFallback::Backward",
        cyxwiz::BackendFallbackReason::CudaJitParamOverflow,
        false,
        "very long NVRTC compiler output",
        context);

    REQUIRE(message.find("reason=cuda_jit_param_overflow") != std::string::npos);
    REQUIRE(message.find("Training continues") != std::string::npos);
    REQUIRE(message.find("native CPU") != std::string::npos);
    REQUIRE(message.find("batch=8") != std::string::npos);
    REQUIRE(message.find("very long NVRTC compiler output") == std::string::npos);
}

TEST_CASE("ArrayFire fallback policy defaults to allowing native CPU fallback",
          "[arrayfire][fallback][policy]") {
    const cyxwiz::ArrayFireFallbackPolicy original =
        cyxwiz::GetArrayFireFallbackPolicy();

    cyxwiz::SetArrayFireFallbackPolicy(
        cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
    REQUIRE_FALSE(cyxwiz::IsArrayFireNativeCpuFallbackForbidden());
    REQUIRE_NOTHROW(cyxwiz::ThrowIfArrayFireNativeCpuFallbackForbidden(
        "UnitTestArrayFireFallback::Forward",
        cyxwiz::BackendFallbackReason::BackendInternalError,
        "synthetic failure",
        "backend=cpu; input=[2x3]"));

    cyxwiz::SetArrayFireFallbackPolicy(original);
}

TEST_CASE("ArrayFire native CPU fallback observer sees allowed and strict attempts",
          "[arrayfire][fallback][policy]") {
    const cyxwiz::ArrayFireFallbackPolicy original_policy =
        cyxwiz::GetArrayFireFallbackPolicy();
    const auto original_observer =
        cyxwiz::GetArrayFireNativeCpuFallbackObserver();

    g_fallback_observer_count = 0;
    g_last_fallback_event = {};
    cyxwiz::SetArrayFireFallbackPolicy(
        cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);

    {
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver observer(
            &CaptureFallbackEvent);
        REQUIRE_NOTHROW(cyxwiz::ThrowIfArrayFireNativeCpuFallbackForbidden(
            "UnitTestArrayFireFallback::Forward",
            cyxwiz::BackendFallbackReason::BackendInternalError,
            "synthetic failure",
            "backend=cpu; input=[2x3]"));
        REQUIRE(g_fallback_observer_count == 1);
        REQUIRE(g_last_fallback_event.operation_name ==
                "UnitTestArrayFireFallback::Forward");
        REQUIRE(g_last_fallback_event.reason_code ==
                "backend_internal_error");
        REQUIRE_FALSE(g_last_fallback_event.fallback_forbidden);

        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        REQUIRE_THROWS_AS(cyxwiz::ThrowIfArrayFireNativeCpuFallbackForbidden(
                              "UnitTestArrayFireFallback::Backward",
                              cyxwiz::BackendFallbackReason::UnsupportedShape,
                              "synthetic shape failure",
                              "backend=cuda; input=[4x5]"),
                          std::runtime_error);
        REQUIRE(g_fallback_observer_count == 2);
        REQUIRE(g_last_fallback_event.operation_name ==
                "UnitTestArrayFireFallback::Backward");
        REQUIRE(g_last_fallback_event.reason_code == "unsupported_shape");
        REQUIRE(g_last_fallback_event.fallback_forbidden);
    }

    REQUIRE(cyxwiz::GetArrayFireNativeCpuFallbackObserver() ==
            original_observer);

    cyxwiz::SetArrayFireFallbackPolicy(original_policy);
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

TEST_CASE("Scoped ArrayFire fallback policy forbids and restores fallback",
          "[arrayfire][fallback][policy]") {
    const cyxwiz::ArrayFireFallbackPolicy original =
        cyxwiz::GetArrayFireFallbackPolicy();

    cyxwiz::SetArrayFireFallbackPolicy(
        cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        REQUIRE(cyxwiz::IsArrayFireNativeCpuFallbackForbidden());

        bool threw = false;
        try {
            cyxwiz::ThrowIfArrayFireNativeCpuFallbackForbidden(
                "UnitTestArrayFireFallback::Backward",
                cyxwiz::BackendFallbackReason::UnsupportedShape,
                "synthetic shape failure",
                "backend=cuda; input=[4x5]");
        } catch (const std::runtime_error& e) {
            threw = true;
            const std::string message = e.what();
            REQUIRE(message.find("native CPU fallback is forbidden") !=
                    std::string::npos);
            REQUIRE(message.find("reason=unsupported_shape") !=
                    std::string::npos);
            REQUIRE(message.find("backend=cuda; input=[4x5]") !=
                    std::string::npos);
        }
        REQUIRE(threw);
    }
    REQUIRE_FALSE(cyxwiz::IsArrayFireNativeCpuFallbackForbidden());

    cyxwiz::SetArrayFireFallbackPolicy(original);
}

TEST_CASE("ArrayFire forced fallback test hook parses requested operations",
          "[arrayfire][fallback]") {
    {
        ScopedEnvVar env(kForceFallbackEnv, nullptr);
        REQUIRE_FALSE(cyxwiz::ShouldForceArrayFireBackendFallbackForTesting(
            "DenseLayer::Forward"));
    }

#ifdef NDEBUG
    {
        ScopedEnvVar env(kForceFallbackEnv, "*");
        REQUIRE_FALSE(cyxwiz::ShouldForceArrayFireBackendFallbackForTesting(
            "DenseLayer::Forward"));
    }
#else
    {
        ScopedEnvVar env(kForceFallbackEnv, "*");
        REQUIRE(cyxwiz::ShouldForceArrayFireBackendFallbackForTesting(
            "DenseLayer::Forward"));
    }
    {
        ScopedEnvVar env(
            kForceFallbackEnv,
            " DenseLayer::Backward ; AttentionLayer::Forward ");
        REQUIRE(cyxwiz::ShouldForceArrayFireBackendFallbackForTesting(
            "DenseLayer::Backward"));
        REQUIRE_FALSE(cyxwiz::ShouldForceArrayFireBackendFallbackForTesting(
            "DenseLayer::Forward"));
    }
    {
        ScopedEnvVar env(kForceFallbackEnv, ", ; DenseLayer::Forward");
        REQUIRE(cyxwiz::ShouldForceArrayFireBackendFallbackForTesting(
            "DenseLayer::Forward"));
        REQUIRE_FALSE(cyxwiz::ShouldForceArrayFireBackendFallbackForTesting(
            "DenseLayer::Backward"));
    }
#endif
}
