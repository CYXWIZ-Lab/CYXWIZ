#pragma once

#include "algorithms/arrayfire_backend_utils.h"

#include <atomic>
#include <cstdint>
#include <sstream>
#include <string>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

struct ExecutionDeviceContext {
    std::string platform = "arrayfire";
    std::string requested_backend;
    int requested_device_id = 0;
    std::string effective_backend;
    int effective_device_id = 0;
    std::string device_name;
    std::string stable_identity;
    uint64_t capability_generation = 0;
    ArrayFireFallbackPolicy fallback_policy =
        ArrayFireFallbackPolicy::AllowNativeCpuFallback;
    bool valid = false;
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
            << " identity='" << stable_identity << "'"
            << " generation=" << capability_generation
            << " fallback_policy=" << FallbackPolicyName();
        if (!valid && !error.empty()) {
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

        char name[256] = {};
        char platform[256] = {};
        char toolkit[256] = {};
        char compute[256] = {};
        af::deviceInfo(name, platform, toolkit, compute);
        context.device_name = name[0] == '\0' ? context.effective_backend
                                              : std::string(name);
        context.stable_identity =
            context.platform + ":" + context.effective_backend + ":" +
            std::to_string(context.effective_device_id) + ":" +
            context.device_name;
        context.valid = true;
    } catch (const af::exception& e) {
        context.valid = false;
        context.platform = "arrayfire";
        context.requested_backend = "query_failed";
        context.effective_backend = "query_failed";
        context.stable_identity = "arrayfire:query_failed";
        context.error = e.what();
    }
#else
    context.platform = "native";
    context.requested_backend = "native_cpu";
    context.effective_backend = "native_cpu";
    context.device_name = "native CPU";
    context.stable_identity = "native:native_cpu:0:native CPU";
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
