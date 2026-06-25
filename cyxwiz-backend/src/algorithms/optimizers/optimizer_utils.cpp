#include "optimizer_utils.h"

#include "../arrayfire_backend_utils.h"
#include "cyxwiz/tensor.h"

#include <spdlog/spdlog.h>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {
namespace optimizer_detail {

namespace {

bool s_use_gpu = false;
bool s_gpu_checked = false;

} // namespace

bool OptimizerGpuAvailable() {
    if (s_gpu_checked) {
        return s_use_gpu;
    }
    s_gpu_checked = true;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::Backend backend = af::getActiveBackend();
        s_use_gpu = (backend == AF_BACKEND_CUDA || backend == AF_BACKEND_OPENCL);
    } catch (const af::exception&) {
        s_use_gpu = false;
    }
#endif

    return s_use_gpu;
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
