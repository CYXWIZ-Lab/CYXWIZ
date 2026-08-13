#pragma once

#include "algorithms/arrayfire_backend_utils.h"
#include "route_qualification_snapshot.h"

#include <cyxwiz/device.h>

#include <atomic>
#include <cstdint>
#include <sstream>
#include <string>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

struct ExecutionRouteQualification {
    bool evidence_available = false;
    bool qualified = false;
    std::string matrix_id;
    std::string message;
};

struct ExecutionDeviceContext {
    std::string platform = "arrayfire";
    std::string requested_backend;
    int requested_device_id = 0;
    std::string effective_backend;
    int effective_device_id = 0;
    std::string device_name;
    std::string route_identity;
    std::string physical_fingerprint;
    DeviceIdentityConfidence identity_confidence =
        DeviceIdentityConfidence::Unknown;
    ExecutionRouteQualification requested_qualification;
    ExecutionRouteQualification effective_qualification;
    uint64_t capability_generation = 0;
    ArrayFireFallbackPolicy fallback_policy =
        ArrayFireFallbackPolicy::AllowNativeCpuFallback;
    bool valid = false;
    bool activation_succeeded = false;
    bool execution_validated = false;
    bool selection_fallback_applied = false;
    std::string preflight_stage = "not_started";
    int preflight_error_code = 0;
    std::string error;

    std::string FallbackPolicyName() const {
        return fallback_policy ==
                ArrayFireFallbackPolicy::ForbidNativeCpuFallback
            ? "forbid_native_cpu_fallback"
            : "allow_native_cpu_fallback";
    }

    std::string Describe() const {
        std::ostringstream out;
        out << "platform=" << platform
            << " requested_backend=" << requested_backend
            << " requested_device=" << requested_device_id
            << " effective_backend=" << effective_backend
            << " effective_device=" << effective_device_id
            << " device_name='" << device_name << "'"
            << " route_identity='" << route_identity << "'"
            << " physical_identity='"
            << (physical_fingerprint.empty() ? "unknown"
                                             : physical_fingerprint)
            << "' identity_confidence="
            << DeviceIdentityConfidenceName(identity_confidence)
            << " requested_qualification="
            << (requested_qualification.qualified ? "certified" : "rejected")
            << " requested_verification="
            << RouteQualificationEvidenceLabel(
                   requested_qualification.matrix_id)
            << " effective_qualification="
            << (effective_qualification.qualified ? "certified" : "rejected")
            << " effective_verification="
            << RouteQualificationEvidenceLabel(
                   effective_qualification.matrix_id)
            << " generation=" << capability_generation
            << " fallback_policy=" << FallbackPolicyName()
            << " activation=" << (activation_succeeded ? "ok" : "failed")
            << " execution="
            << (execution_validated ? "validated" : "not_validated")
            << " preflight_stage=" << preflight_stage
            << " selection_fallback="
            << (selection_fallback_applied ? "true" : "false");
        if (preflight_error_code != 0) {
            out << " preflight_error=" << preflight_error_code;
        }
        if (!error.empty()) {
            out << " error='" << error << "'";
        }
        return out.str();
    }
};

inline uint64_t NextExecutionDeviceCapabilityGeneration() {
    static std::atomic<uint64_t> generation{0};
    return generation.fetch_add(1, std::memory_order_relaxed) + 1;
}

inline std::string ArrayFireBackendNameForContext(int backend) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    switch (backend) {
        case AF_BACKEND_CPU: return "arrayfire_cpu";
        case AF_BACKEND_CUDA: return "arrayfire_cuda";
        case AF_BACKEND_OPENCL: return "arrayfire_opencl";
        case AF_BACKEND_ONEAPI: return "arrayfire_oneapi";
        default: return "arrayfire_unknown";
    }
#else
    (void)backend;
    return "native_cpu";
#endif
}

inline DeviceType DeviceTypeForContextBackend(int backend) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    switch (backend) {
        case AF_BACKEND_CUDA: return DeviceType::CUDA;
        case AF_BACKEND_OPENCL: return DeviceType::OPENCL;
        case AF_BACKEND_ONEAPI: return DeviceType::ONEAPI;
        case AF_BACKEND_CPU:
        default:
            return DeviceType::CPU;
    }
#else
    (void)backend;
    return DeviceType::CPU;
#endif
}

inline ExecutionDeviceContext CaptureCurrentExecutionDeviceContext(
    ArrayFireFallbackPolicy fallback_policy) {
    ExecutionDeviceContext context;
    context.fallback_policy = fallback_policy;
    context.capability_generation = NextExecutionDeviceCapabilityGeneration();

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::Backend backend = af::getActiveBackend();
        const int device_id = af::getDevice();
        context.platform = "arrayfire";
        context.requested_backend =
            ArrayFireBackendNameForContext(static_cast<int>(backend));
        context.requested_device_id = device_id;
        context.effective_backend = context.requested_backend;
        context.effective_device_id = device_id;

        const DeviceInfo info =
            Device(DeviceTypeForContextBackend(static_cast<int>(backend)),
                   device_id)
                .GetInfo();
        context.device_name = info.name;
        context.route_identity =
            context.platform + ":" + context.effective_backend + ":" +
            std::to_string(context.effective_device_id);
        context.physical_fingerprint = info.physical_fingerprint;
        context.identity_confidence = info.identity_confidence;
        context.activation_succeeded = true;
        context.preflight_stage = "capture";
        context.valid = true;
    } catch (af::exception& e) {
        context.valid = false;
        context.platform = "arrayfire";
        context.requested_backend = "query_failed";
        context.effective_backend = "query_failed";
        context.route_identity = "arrayfire:query_failed";
        context.preflight_stage = "effective_state_query";
        context.preflight_error_code = static_cast<int>(e.err());
        const char* error_name = af_err_to_string(e.err());
        context.error = error_name != nullptr ? error_name
                                              : "unknown ArrayFire error";
    }
#else
    context.platform = "native";
    context.requested_backend = "native_cpu";
    context.effective_backend = "native_cpu";
    context.device_name = "native CPU";
    context.route_identity = "native:native_cpu:0";
    context.identity_confidence = DeviceIdentityConfidence::BackendLocal;
    context.activation_succeeded = true;
    context.execution_validated = true;
    context.preflight_stage = "complete";
    context.valid = true;
#endif

    return context;
}

inline const ExecutionDeviceContext*& CurrentExecutionDeviceContextSlot() {
    thread_local const ExecutionDeviceContext* context = nullptr;
    return context;
}

inline const ExecutionDeviceContext* CurrentExecutionDeviceContext() {
    return CurrentExecutionDeviceContextSlot();
}

inline std::atomic<int>& ActiveExecutionDeviceContextCount() {
    static std::atomic<int> active_count{0};
    return active_count;
}

inline bool HasActiveExecutionDeviceContext() {
    return ActiveExecutionDeviceContextCount().load(std::memory_order_acquire) >
           0;
}

class ScopedActiveExecutionDeviceContext {
public:
    ScopedActiveExecutionDeviceContext() {
        ActiveExecutionDeviceContextCount().fetch_add(
            1, std::memory_order_acq_rel);
    }

    ScopedActiveExecutionDeviceContext(
        const ScopedActiveExecutionDeviceContext&) = delete;
    ScopedActiveExecutionDeviceContext& operator=(
        const ScopedActiveExecutionDeviceContext&) = delete;

    ~ScopedActiveExecutionDeviceContext() {
        ActiveExecutionDeviceContextCount().fetch_sub(
            1, std::memory_order_acq_rel);
    }
};

class ScopedExecutionDeviceContext {
public:
    explicit ScopedExecutionDeviceContext(const ExecutionDeviceContext& context)
        : previous_(CurrentExecutionDeviceContextSlot()) {
        CurrentExecutionDeviceContextSlot() = &context;
    }

    ScopedExecutionDeviceContext(const ScopedExecutionDeviceContext&) = delete;
    ScopedExecutionDeviceContext& operator=(
        const ScopedExecutionDeviceContext&) = delete;

    ~ScopedExecutionDeviceContext() {
        CurrentExecutionDeviceContextSlot() = previous_;
    }

private:
    const ExecutionDeviceContext* previous_ = nullptr;
};

} // namespace cyxwiz
