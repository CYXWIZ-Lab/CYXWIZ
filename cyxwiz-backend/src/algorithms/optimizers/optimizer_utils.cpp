#include "optimizer_utils.h"

#include "../arrayfire_backend_utils.h"
#include "cyxwiz/tensor.h"

#include <spdlog/spdlog.h>
#include <stdexcept>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {
namespace optimizer_detail {

bool OptimizerArrayFireAvailable() {
    return IsCurrentArrayFireBackendAvailable();
}

bool PrepareOptimizerNativeCpuFallback(
    const char* operation_name,
    const std::string& parameter_name,
    const Tensor& parameter,
    bool arrayfire_available) {
    if (ShouldForceArrayFireBackendFallbackForTesting(operation_name)) {
        LogOptimizerFallbackOnce(
            operation_name,
            parameter_name,
            parameter,
            BackendFallbackReason::GpuBackendException,
            "forced ArrayFire backend fallback test hook");
        return true;
    }
    if (!arrayfire_available) {
        LogOptimizerFallbackOnce(
            operation_name,
            parameter_name,
            parameter,
            BackendFallbackReason::BackendUnavailable,
            "ArrayFire backend unavailable");
        return true;
    }
    return false;
}

void LogOptimizerFallbackOnce(
    const char* operation_name,
    const std::string& parameter_name,
    const Tensor& parameter,
    const char* error_message) {
    const BackendFallbackReason reason =
        ClassifyArrayFireBackendFallbackReason(error_message);
    LogOptimizerFallbackOnce(
        operation_name,
        parameter_name,
        parameter,
        reason,
        error_message);
}

void LogOptimizerFallbackOnce(
    const char* operation_name,
    const std::string& parameter_name,
    const Tensor& parameter,
    BackendFallbackReason reason,
    const char* error_message) {
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

void ValidateOptimizerStepTensors(
    const char* operation_name,
    const std::string& parameter_name,
    const Tensor& parameter,
    const Tensor& gradient) {
    const std::string operation =
        operation_name == nullptr ? "Optimizer::Step" : operation_name;
    const std::string name =
        parameter_name.empty() ? "parameter" : parameter_name;
    if (parameter.GetDataType() != DataType::Float32 ||
        gradient.GetDataType() != DataType::Float32) {
        throw std::invalid_argument(
            operation + " requires Float32 parameter and gradient tensors for '" +
            name + "'.");
    }
    if (parameter.Shape() != gradient.Shape()) {
        throw std::invalid_argument(
            operation + " gradient shape does not match parameter '" + name +
            "'.");
    }
}

} // namespace optimizer_detail
} // namespace cyxwiz
