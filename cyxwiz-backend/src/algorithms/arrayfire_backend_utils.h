#pragma once

#include <string>
#include <vector>

namespace cyxwiz {

enum class BackendFallbackReason {
    GpuBackendException,
    ArrayFireJitCompileFailure,
    CudaJitParamOverflow,
};

const char* BackendFallbackReasonName(BackendFallbackReason reason);
bool IsCudaJitFormalParameterOverflow(const char* message);
BackendFallbackReason ClassifyArrayFireBackendFallbackReason(const char* message);

std::string BuildTensorShapeContext(
    const char* tensor_name,
    const std::vector<size_t>& shape);
std::string CurrentArrayFireBackendName();
std::string BuildArrayFireBackendFallbackContext(
    const std::string& shape_or_node_context,
    const std::string& backend_name = CurrentArrayFireBackendName());
std::string BuildArrayFireBackendFallbackMessage(
    const char* operation_name,
    BackendFallbackReason reason,
    bool include_error_text,
    const char* error_message,
    const std::string& context = {});
bool ShouldLogArrayFireBackendFallbackOnce(
    const char* operation_name,
    BackendFallbackReason reason,
    const std::string& context = {});
bool ShouldForceArrayFireBackendFallbackForTesting(
    const char* operation_name);

} // namespace cyxwiz
