#include "cyxwiz/activations/sigmoid.h"
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
void LogSigmoidFallbackOnce(
    const char* operation_name,
    const char* error_message,
    const Tensor& tensor,
    const char* tensor_name) {
    const BackendFallbackReason reason =
        ClassifyArrayFireBackendFallbackReason(error_message);
    const std::string context = BuildArrayFireBackendFallbackContext(
        BuildTensorShapeContext(tensor_name, tensor.Shape()));
    if (ShouldLogArrayFireBackendFallbackOnce(
            operation_name, reason, context)) {
        spdlog::warn("{}",
                     BuildArrayFireBackendFallbackMessage(
                         operation_name,
                         reason,
                         reason != BackendFallbackReason::CudaJitParamOverflow,
                         error_message,
                         context));
    }
}
#endif

} // namespace

static bool s_use_gpu = false;
static bool s_gpu_checked = false;

static bool UseGPU() {
    if (!s_gpu_checked) {
        s_gpu_checked = true;
#ifdef CYXWIZ_HAS_ARRAYFIRE
        try {
            af::Backend backend = af::getActiveBackend();
            s_use_gpu = (backend == AF_BACKEND_CUDA || backend == AF_BACKEND_OPENCL);
        } catch (...) {
            s_use_gpu = false;
        }
#endif
    }
    return s_use_gpu;
}

Tensor Sigmoid::Forward(const Tensor& input) {
    if (input.GetDataType() != DataType::Float32) {
        throw std::runtime_error("Sigmoid only supports Float32 tensors");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (UseGPU()) {
        try {
            af::array input_gpu = input.GetArray();

            // Sigmoid: 1 / (1 + exp(-x))
            af::array output_gpu = af::sigmoid(input_gpu);
            output_gpu.eval();

            return Tensor(output_gpu);
        } catch (const af::exception& e) {
            LogSigmoidFallbackOnce(
                "Sigmoid::Forward", e.what(), input, "input");
        }
    }
#endif

    // CPU fallback
    Tensor output(input.Shape(), input.GetDataType());
    size_t num_elements = input.NumElements();
    const float* input_data = static_cast<const float*>(input.Data());
    float* output_data = static_cast<float*>(output.Data());

    for (size_t i = 0; i < num_elements; i++) {
        output_data[i] = 1.0f / (1.0f + std::exp(-input_data[i]));
    }

    return output;
}

Tensor Sigmoid::Backward(const Tensor& grad_output, const Tensor& input) {
    if (grad_output.Shape() != input.Shape()) {
        throw std::runtime_error("Sigmoid::Backward: gradient and input shapes must match");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (UseGPU()) {
        try {
            af::array grad_gpu = grad_output.GetArray();
            af::array input_gpu = input.GetArray();

            // Gradient: grad * sigmoid(x) * (1 - sigmoid(x))
            af::array sigmoid_val = af::sigmoid(input_gpu);
            af::array grad_input_gpu = grad_gpu * sigmoid_val * (1.0f - sigmoid_val);
            grad_input_gpu.eval();

            return Tensor(grad_input_gpu);
        } catch (const af::exception& e) {
            LogSigmoidFallbackOnce(
                "Sigmoid::Backward", e.what(), grad_output, "grad_output");
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
        float sigmoid_val = 1.0f / (1.0f + std::exp(-input_data[i]));
        grad_in_data[i] = grad_out_data[i] * sigmoid_val * (1.0f - sigmoid_val);
    }

    return grad_input;
}

} // namespace cyxwiz
