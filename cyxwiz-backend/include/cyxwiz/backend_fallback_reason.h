// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
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
    // Owner ruling 2026-09-23: the OpenCL tenant declines tuples it measured
    // slower than native CPU on (hidden below the retention floor). Not a
    // contract failure, so it gets its own persisted name.
    OpenclProviderBelowRetentionFloor,
    // oneAPI (SYCL) provider, device-resident attention (tofix112 phase 5b).
    OneapiProviderUnavailable,
    OneapiProviderUnsupportedContract,
    OneapiProviderExecutionFailed,
    // Why no provider serves a device family at all (the compute path
    // standard). Before these, an absent provider had no reason: the
    // operation ran on ArrayFire and nothing said why.
    //  - NotBuilt: this build made no provider for the family (its compile
    //    define is off) and none is installed beside it.
    //  - NotInstalled: the build has the provider, its module or kernel
    //    library is absent at run time.
    //  - AbiMismatch: the module or kernel library was found and belongs to
    //    another version of the provider interface.
    //  - Disabled: providers are turned off for the process
    //    (CYXWIZ_DISABLE_NEURAL_PROVIDERS, or a test).
    //  - NotOffered: no provider implements this operation for this device
    //    family, in any build. ArrayFire is the designed path; nothing to fix.
    //  - TargetMismatch: the request's device, or the thread's ArrayFire
    //    device, is not the device the run bound (or ArrayFire's active
    //    device could not be read): no provider is asked, ArrayFire runs it
    //    where its data is. An Engine fault, never a user's choice.
    ProviderNotBuilt,
    ProviderNotInstalled,
    ProviderAbiMismatch,
    ProviderDisabled,
    ProviderNotOffered,
    TargetMismatch,
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
    case BackendFallbackReason::OpenclProviderBelowRetentionFloor:
        return "opencl_provider_below_retention_floor";
    case BackendFallbackReason::OneapiProviderUnavailable:
        return "oneapi_provider_unavailable";
    case BackendFallbackReason::OneapiProviderUnsupportedContract:
        return "oneapi_provider_unsupported_contract";
    case BackendFallbackReason::OneapiProviderExecutionFailed:
        return "oneapi_provider_execution_failed";
    case BackendFallbackReason::ProviderNotBuilt:
        return "provider_not_built";
    case BackendFallbackReason::ProviderNotInstalled:
        return "provider_not_installed";
    case BackendFallbackReason::ProviderAbiMismatch:
        return "provider_abi_mismatch";
    case BackendFallbackReason::ProviderDisabled:
        return "provider_disabled";
    case BackendFallbackReason::ProviderNotOffered:
        return "provider_not_offered";
    case BackendFallbackReason::TargetMismatch:
        return "target_mismatch";
    }
    return "backend_internal_error";
}

} // namespace cyxwiz
