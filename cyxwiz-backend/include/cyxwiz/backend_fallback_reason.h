#pragma once

namespace cyxwiz {

// Single authority for the GPU/backend failure taxonomy (tofix67 slice 2).
// The typed enum is the source of truth; every string form of a reason code
// (placement observation records, logs, cache JSON, UI text) must be derived
// through BackendFallbackReasonName. Do not define these strings anywhere
// else — a source-scan test enforces that this header stays the only
// definition site.
enum class BackendFallbackReason {
    BackendUnavailable,
    GpuBackendException,
    ArrayFireJitCompileFailure,
    CudaJitParamOverflow,
    GpuOutOfMemory,
    UnsupportedDtype,
    UnsupportedShape,
    UnsupportedOperation,
    BackendCompileTimeout,
    BackendInternalError,
    // Native neural-network provider reasons (tofix68). Provider failures
    // must never be described as generic GPU-memory exhaustion unless the
    // workspace failure proves it.
    NvidiaProviderUnavailable,
    NvidiaProviderUnsupportedContract,
    NvidiaProviderWorkspaceExhausted,
    NvidiaProviderExecutionFailed,
    // OpenCL provider (device-keyed dispatch tenant #2), same four shapes.
    OpenclProviderUnavailable,
    OpenclProviderUnsupportedContract,
    OpenclProviderWorkspaceExhausted,
    OpenclProviderExecutionFailed,
};

constexpr const char* BackendFallbackReasonName(BackendFallbackReason reason) {
    switch (reason) {
    case BackendFallbackReason::BackendUnavailable:
        return "backend_unavailable";
    case BackendFallbackReason::CudaJitParamOverflow:
        return "cuda_jit_param_overflow";
    case BackendFallbackReason::ArrayFireJitCompileFailure:
        return "arrayfire_jit_compile_failure";
    case BackendFallbackReason::GpuBackendException:
        return "gpu_backend_exception";
    case BackendFallbackReason::GpuOutOfMemory:
        return "gpu_out_of_memory";
    case BackendFallbackReason::UnsupportedDtype:
        return "unsupported_dtype";
    case BackendFallbackReason::UnsupportedShape:
        return "unsupported_shape";
    case BackendFallbackReason::UnsupportedOperation:
        return "unsupported_operation";
    case BackendFallbackReason::BackendCompileTimeout:
        return "backend_compile_timeout";
    case BackendFallbackReason::BackendInternalError:
        return "backend_internal_error";
    case BackendFallbackReason::NvidiaProviderUnavailable:
        return "nvidia_provider_unavailable";
    case BackendFallbackReason::NvidiaProviderUnsupportedContract:
        return "nvidia_provider_unsupported_contract";
    case BackendFallbackReason::NvidiaProviderWorkspaceExhausted:
        return "nvidia_provider_workspace_exhausted";
    case BackendFallbackReason::NvidiaProviderExecutionFailed:
        return "nvidia_provider_execution_failed";
    case BackendFallbackReason::OpenclProviderUnavailable:
        return "opencl_provider_unavailable";
    case BackendFallbackReason::OpenclProviderUnsupportedContract:
        return "opencl_provider_unsupported_contract";
    case BackendFallbackReason::OpenclProviderWorkspaceExhausted:
        return "opencl_provider_workspace_exhausted";
    case BackendFallbackReason::OpenclProviderExecutionFailed:
        return "opencl_provider_execution_failed";
    }
    return "backend_internal_error";
}

} // namespace cyxwiz
