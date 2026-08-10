// Prevent Windows.h from defining min/max macros that conflict with ArrayFire
#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/activations/relu.h"
#include "cyxwiz/backend_placement_observation.h"
#include "../arrayfire_backend_utils.h"
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <spdlog/spdlog.h>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {
namespace {

#ifdef CYXWIZ_HAS_ARRAYFIRE
void LogReluFallbackOnce(
    const char* operation_name,
    const char* error_message,
    const Tensor& tensor,
    const char* tensor_name) {
    const BackendFallbackReason reason =
        ClassifyArrayFireBackendFallbackReason(error_message);
    const std::string context = BuildArrayFireBackendFallbackContext(
        BuildTensorShapeContext(tensor_name, tensor.Shape()));
    const std::string message = BuildArrayFireBackendFallbackMessage(
        operation_name,
        reason,
        reason != BackendFallbackReason::CudaJitParamOverflow,
        error_message,
        context);
    RecordBackendPlacementObservationForActiveDevice(
        "ReLU",
        CurrentArrayFireBackendName(),
        "float32",
        BuildActivationPlacementShapeSignature(tensor.Shape(), "float32"),
        BackendFallbackReasonName(reason),
        BackendPlacementObservationSource::RuntimeFallback,
        message);
    ThrowIfArrayFireNativeCpuFallbackForbidden(
        operation_name,
        reason,
        error_message,
        context);
    if (ShouldLogArrayFireBackendFallbackOnce(
            operation_name, reason, context)) {
        spdlog::warn("{}", message);
    }
}
#endif

} // namespace

Tensor ReLU::Forward(const Tensor& input) {
    if (input.GetDataType() != DataType::Float32) {
        throw std::runtime_error("ReLU only supports Float32 tensors");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (IsCurrentArrayFireBackendAvailable()) {
        try {
            af::array input_gpu = input.GetSemanticArray();

            // ReLU: max(0, x)
            af::array output_gpu = af::max(input_gpu, 0.0f);
            output_gpu.eval();

            return Tensor::FromSemanticArray(output_gpu, input.Shape());
        } catch (const af::exception& e) {
            LogReluFallbackOnce("ReLU::Forward", e.what(), input, "input");
        }
    }
#endif

    // CPU fallback
    Tensor output(input.Shape(), input.GetDataType());
    size_t num_elements = input.NumElements();
    const float* input_data = static_cast<const float*>(input.Data());
    float* output_data = static_cast<float*>(output.Data());

    for (size_t i = 0; i < num_elements; i++) {
        output_data[i] = std::max(0.0f, input_data[i]);
    }

    return output;
}

Tensor ReLU::Backward(const Tensor& grad_output, const Tensor& input) {
    if (grad_output.Shape() != input.Shape()) {
        throw std::runtime_error("ReLU::Backward: gradient and input shapes must match");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (IsCurrentArrayFireBackendAvailable()) {
        try {
            af::array grad_gpu = grad_output.GetSemanticArray();
            af::array input_gpu = input.GetSemanticArray();

            // Gradient: grad * (input > 0)
            af::array mask = input_gpu > 0.0f;
            af::array grad_input_gpu = grad_gpu * mask.as(f32);
            grad_input_gpu.eval();

            return Tensor::FromSemanticArray(
                grad_input_gpu, grad_output.Shape());
        } catch (const af::exception& e) {
            LogReluFallbackOnce(
                "ReLU::Backward", e.what(), grad_output, "grad_output");
        }
    }
#endif

    // CPU fallback
    Tensor grad_input(input.Shape(), input.GetDataType());
    size_t num_elements = input.NumElements();
    const float* grad_out_data = static_cast<const float*>(grad_output.Data());
    const float* input_data = static_cast<const float*>(input.Data());
    float* grad_in_data = static_cast<float*>(grad_input.Data());

    for (size_t i = 0; i < num_elements; i++) {
        grad_in_data[i] = input_data[i] > 0.0f ? grad_out_data[i] : 0.0f;
    }

    return grad_input;
}

} // namespace cyxwiz
