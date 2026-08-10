#include "optimizer_utils.h"

#include "../arrayfire_backend_utils.h"
#include "cyxwiz/tensor.h"

#include <spdlog/spdlog.h>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {
namespace optimizer_detail {

bool OptimizerArrayFireAvailable() {
    return IsCurrentArrayFireBackendAvailable();
}

void LogOptimizerFallbackOnce(
    const char* operation_name,
    const std::string& parameter_name,
    const Tensor& parameter,
    const char* error_message) {
    const BackendFallbackReason reason =
        ClassifyArrayFireBackendFallbackReason(error_message);
    const std::string tensor_name =
        parameter_name.empty() ? "parameter" : parameter_name;
    const std::string context = BuildArrayFireBackendFallbackContext(
        BuildTensorShapeContext(tensor_name.c_str(), parameter.Shape()));
    ThrowIfArrayFireNativeCpuFallbackForbidden(
        operation_name,
        reason,
        error_message,
        context);
    if (ShouldLogArrayFireBackendFallbackOnce(operation_name, reason, context)) {
        spdlog::warn("{}",
            BuildArrayFireBackendFallbackMessage(
                operation_name,
                reason,
                reason != BackendFallbackReason::CudaJitParamOverflow,
                error_message,
                context));
    }
}

} // namespace optimizer_detail
} // namespace cyxwiz
