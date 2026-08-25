#include "cyxwiz/activations/tanh.h"
#include "cyxwiz/backend_placement_observation.h"
#include "../arrayfire_backend_utils.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <spdlog/spdlog.h>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {
namespace {

#ifdef CYXWIZ_HAS_ARRAYFIRE
void LogTanhFallbackOnce(
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
        "Tanh",
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

Tensor Tanh::Forward(const Tensor& input) {
    if (input.GetDataType() != DataType::Float32) {
        throw std::runtime_error("Tanh only supports Float32 tensors");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (IsCurrentArrayFireBackendAvailable()) {
        try {
            af::array input_gpu = input.GetSemanticArray();

            // Tanh
            af::array output_gpu = af::tanh(input_gpu);
            output_gpu.eval();

            return Tensor::FromSemanticArray(output_gpu, input.Shape());
        } catch (const af::exception& e) {
            LogTanhFallbackOnce("Tanh::Forward", e.what(), input, "input");
        }
    }
#endif

    // CPU fallback
    Tensor output(input.Shape(), input.GetDataType());
    size_t num_elements = input.NumElements();
    const float* input_data = input.ReadData<float>();
    float* output_data = output.MutableData<float>();

    for (size_t i = 0; i < num_elements; i++) {
        output_data[i] = std::tanh(input_data[i]);
    }

    return output;
}

Tensor Tanh::Backward(const Tensor& grad_output, const Tensor& input) {
    if (grad_output.Shape() != input.Shape()) {
        throw std::runtime_error("Tanh::Backward: gradient and input shapes must match");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (IsCurrentArrayFireBackendAvailable()) {
        try {
            af::array grad_gpu = grad_output.GetSemanticArray();
            af::array input_gpu = input.GetSemanticArray();

            // Gradient: grad * (1 - tanh(x)^2)
            af::array tanh_val = af::tanh(input_gpu);
            af::array grad_input_gpu = grad_gpu * (1.0f - tanh_val * tanh_val);
            grad_input_gpu.eval();

            return Tensor::FromSemanticArray(
                grad_input_gpu, grad_output.Shape());
        } catch (const af::exception& e) {
            LogTanhFallbackOnce(
                "Tanh::Backward", e.what(), grad_output, "grad_output");
        }
    }
#endif

    // CPU fallback
    Tensor grad_input(input.Shape(), input.GetDataType());
    size_t num_elements = input.NumElements();
    const float* grad_out_data = grad_output.ReadData<float>();
    const float* input_data = input.ReadData<float>();
    float* grad_in_data = grad_input.MutableData<float>();

    for (size_t i = 0; i < num_elements; i++) {
        float tanh_val = std::tanh(input_data[i]);
        grad_in_data[i] = grad_out_data[i] * (1.0f - tanh_val * tanh_val);
    }

    return grad_input;
}

} // namespace cyxwiz
