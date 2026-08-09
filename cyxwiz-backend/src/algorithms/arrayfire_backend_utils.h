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

CYXWIZ_API const char* BackendFallbackReasonName(BackendFallbackReason reason);
CYXWIZ_API bool IsCudaJitFormalParameterOverflow(const char* message);
CYXWIZ_API BackendFallbackReason ClassifyArrayFireBackendFallbackReason(
    const char* message);

CYXWIZ_API std::string BuildTensorShapeContext(
    const char* tensor_name,
    const std::vector<size_t>& shape);
CYXWIZ_API std::string CurrentArrayFireBackendName();
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
CYXWIZ_API bool ShouldLogArrayFireBackendFallbackOnce(
    const char* operation_name,
    BackendFallbackReason reason,
    const std::string& context = {});
CYXWIZ_API bool ShouldForceArrayFireBackendFallbackForTesting(
    const char* operation_name);

} // namespace cyxwiz
