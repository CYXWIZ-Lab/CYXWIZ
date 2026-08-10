#pragma once

#include "cyxwiz/api_export.h"

#include <string>
#include <vector>

namespace cyxwiz {

enum class BackendFallbackReason {
    GpuBackendException,
    ArrayFireJitCompileFailure,
    CudaJitParamOverflow,
    GpuOutOfMemory,
    UnsupportedDtype,
    UnsupportedShape,
    BackendCompileTimeout,
    BackendInternalError,
};

enum class ArrayFireFallbackPolicy {
    AllowNativeCpuFallback,
    ForbidNativeCpuFallback,
};

struct ArrayFireNativeCpuFallbackEvent {
    std::string operation_name;
    std::string reason_code;
    std::string selected_backend;
    std::string context;
    std::string error_message;
    bool fallback_forbidden = false;
};

struct ArrayFireHostSyncEvent {
    std::string operation_name;
    std::string selected_backend;
    std::string reason_code;
    std::string context;
    uint64_t bytes = 0;
};

using ArrayFireNativeCpuFallbackObserver =
    void (*)(const ArrayFireNativeCpuFallbackEvent& event);
using ArrayFireHostSyncObserver =
    void (*)(const ArrayFireHostSyncEvent& event);

class CYXWIZ_API ScopedArrayFireFallbackPolicy {
public:
    explicit ScopedArrayFireFallbackPolicy(ArrayFireFallbackPolicy policy);
    ~ScopedArrayFireFallbackPolicy();

    ScopedArrayFireFallbackPolicy(const ScopedArrayFireFallbackPolicy&) = delete;
    ScopedArrayFireFallbackPolicy& operator=(
        const ScopedArrayFireFallbackPolicy&) = delete;

private:
    ArrayFireFallbackPolicy previous_;
};

class CYXWIZ_API ScopedArrayFireNativeCpuFallbackObserver {
public:
    explicit ScopedArrayFireNativeCpuFallbackObserver(
        ArrayFireNativeCpuFallbackObserver observer);
    ~ScopedArrayFireNativeCpuFallbackObserver();

    ScopedArrayFireNativeCpuFallbackObserver(
        const ScopedArrayFireNativeCpuFallbackObserver&) = delete;
    ScopedArrayFireNativeCpuFallbackObserver& operator=(
        const ScopedArrayFireNativeCpuFallbackObserver&) = delete;

private:
    ArrayFireNativeCpuFallbackObserver previous_;
};

class CYXWIZ_API ScopedArrayFireHostSyncObserver {
public:
    explicit ScopedArrayFireHostSyncObserver(
        ArrayFireHostSyncObserver observer);
    ~ScopedArrayFireHostSyncObserver();

    ScopedArrayFireHostSyncObserver(
        const ScopedArrayFireHostSyncObserver&) = delete;
    ScopedArrayFireHostSyncObserver& operator=(
        const ScopedArrayFireHostSyncObserver&) = delete;

private:
    ArrayFireHostSyncObserver previous_;
};

CYXWIZ_API ArrayFireFallbackPolicy GetArrayFireFallbackPolicy();
CYXWIZ_API void SetArrayFireFallbackPolicy(ArrayFireFallbackPolicy policy);
CYXWIZ_API bool IsArrayFireNativeCpuFallbackForbidden();
CYXWIZ_API ArrayFireNativeCpuFallbackObserver
GetArrayFireNativeCpuFallbackObserver();
CYXWIZ_API void SetArrayFireNativeCpuFallbackObserver(
    ArrayFireNativeCpuFallbackObserver observer);
CYXWIZ_API ArrayFireHostSyncObserver GetArrayFireHostSyncObserver();
CYXWIZ_API void SetArrayFireHostSyncObserver(
    ArrayFireHostSyncObserver observer);
CYXWIZ_API const char* BackendFallbackReasonName(BackendFallbackReason reason);
CYXWIZ_API bool IsCudaJitFormalParameterOverflow(const char* message);
CYXWIZ_API BackendFallbackReason ClassifyArrayFireBackendFallbackReason(
    const char* message);

CYXWIZ_API std::string BuildTensorShapeContext(
    const char* tensor_name,
    const std::vector<size_t>& shape);
CYXWIZ_API std::string CurrentArrayFireBackendName();
CYXWIZ_API bool IsCurrentArrayFireBackendAvailable();
CYXWIZ_API bool IsCurrentArrayFireBackendGpu();
CYXWIZ_API std::string BuildArrayFireBackendFallbackContext(
    const std::string& shape_or_node_context,
    const std::string& backend_name = CurrentArrayFireBackendName());
CYXWIZ_API std::string BuildArrayFireBackendFallbackMessage(
    const char* operation_name,
    BackendFallbackReason reason,
    bool include_error_text,
    const char* error_message,
    const std::string& context = {});
CYXWIZ_API void ThrowIfArrayFireNativeCpuFallbackForbidden(
    const char* operation_name,
    BackendFallbackReason reason,
    const char* error_message,
    const std::string& context = {});
CYXWIZ_API bool ShouldLogArrayFireBackendFallbackOnce(
    const char* operation_name,
    BackendFallbackReason reason,
    const std::string& context = {});
CYXWIZ_API bool ShouldForceArrayFireBackendFallbackForTesting(
    const char* operation_name);

} // namespace cyxwiz
