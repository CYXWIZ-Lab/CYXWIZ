#include "cyxwiz/layers/dropout.h"
#include "layer_arrayfire_utils.h"
#include "../arrayfire_backend_utils.h"

#include <random>
#include <stdexcept>
#include <string>

#include <spdlog/spdlog.h>

namespace cyxwiz {

#ifdef CYXWIZ_HAS_ARRAYFIRE
static void HandleDropoutArrayFireFallback(
    const char* operation_name,
    const Tensor& tensor,
    const char* error_message)
{
    const BackendFallbackReason reason = ClassifyArrayFireBackendFallbackReason(error_message);
    const std::string context = BuildArrayFireBackendFallbackContext(
        BuildTensorShapeContext("input", tensor.Shape()));
    ThrowIfArrayFireNativeCpuFallbackForbidden(
        operation_name, reason, error_message, context);
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
#endif

DropoutLayer::DropoutLayer(float p) : p_(p) {
    if (p < 0.0f || p > 1.0f) {
        throw std::invalid_argument("Dropout probability must be in [0, 1]");
    }
}

Tensor DropoutLayer::Forward(const Tensor& input) {
    has_forward_ = false;
    forward_used_dropout_ = training_ && p_ > 0.0f;
    output_shape_ = input.Shape();
    output_dtype_ = input.GetDataType();

    if (!forward_used_dropout_) {
        has_forward_ = true;
        return input;
    }
    if (input.GetDataType() != DataType::Float32) {
        throw std::runtime_error(
            "Dropout training requires Float32 input");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    bool use_native_cpu = false;
    if (ShouldForceArrayFireBackendFallbackForTesting(
            "DropoutLayer::Forward")) {
        HandleDropoutArrayFireFallback(
            "DropoutLayer::Forward", input,
            "forced ArrayFire backend fallback test hook");
        use_native_cpu = true;
    }
    if (!use_native_cpu) {
        try {
            af::array x = TensorToAf(input);
            af::array mask;
            af::array output;

            if (p_ == 1.0f) {
                mask = af::constant(0.0f, x.dims(), af::dtype::f32);
                output = mask;
            } else {
                af::array random = af::randu(x.dims(), af::dtype::f32);
                mask = (random > p_).as(af::dtype::f32);
                output = x * mask * (1.0f / (1.0f - p_));
            }
            mask.eval();
            output.eval();

            mask_ = Tensor::FromSemanticArray(mask, input.Shape());
            Tensor result = Tensor::FromSemanticArray(output, input.Shape());
            has_forward_ = true;
            return result;
        } catch (const af::exception& error) {
            HandleDropoutArrayFireFallback(
                "DropoutLayer::Forward", input, error.what());
        }
    }
#endif

    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath,
        "DropoutLayer::Forward");
    Tensor output(input.Shape(), DataType::Float32);
    mask_ = Tensor(input.Shape(), DataType::Float32);
    float* output_data = output.MutableData<float>();
    float* mask_data = mask_.MutableData<float>();
    if (p_ == 1.0f) {
        for (size_t index = 0; index < input.NumElements(); ++index) {
            output_data[index] = 0.0f;
            mask_data[index] = 0.0f;
        }
        has_forward_ = true;
        return output;
    }

    const float* input_data = input.ReadData<float>();
    const float scale = 1.0f / (1.0f - p_);
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t index = 0; index < input.NumElements(); ++index) {
        mask_data[index] = dist(rng) > p_ ? 1.0f : 0.0f;
        output_data[index] = input_data[index] * mask_data[index] * scale;
    }

    has_forward_ = true;
    return output;
}

Tensor DropoutLayer::Backward(const Tensor& grad_output) {
    if (!has_forward_) {
        throw std::logic_error(
            "DropoutLayer::Backward requires a successful Forward call");
    }
    if (grad_output.Shape() != output_shape_) {
        throw std::runtime_error(
            "Dropout backward gradient shape does not match Forward output");
    }
    if (grad_output.GetDataType() != output_dtype_) {
        throw std::runtime_error(
            "Dropout backward gradient dtype does not match Forward output");
    }
    if (!forward_used_dropout_) {
        return grad_output;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    bool use_native_cpu = false;
    if (ShouldForceArrayFireBackendFallbackForTesting(
            "DropoutLayer::Backward")) {
        HandleDropoutArrayFireFallback(
            "DropoutLayer::Backward", grad_output,
            "forced ArrayFire backend fallback test hook");
        use_native_cpu = true;
    }
    if (!use_native_cpu) {
        try {
            af::array grad_out = TensorToAf(grad_output);
            af::array mask = TensorToAf(mask_);
            const float scale = p_ == 1.0f ? 0.0f : 1.0f / (1.0f - p_);
            af::array dx = grad_out * mask * scale;
            dx.eval();
            return Tensor::FromSemanticArray(dx, grad_output.Shape());
        } catch (const af::exception& error) {
            HandleDropoutArrayFireFallback(
                "DropoutLayer::Backward", grad_output, error.what());
        }
    }
#endif

    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath,
        "DropoutLayer::Backward");
    Tensor grad_input(grad_output.Shape(), DataType::Float32);
    const float* grad_data = grad_output.ReadData<float>();
    const float* mask_data = mask_.ReadData<float>();
    float* grad_input_data = grad_input.MutableData<float>();
    const float scale = p_ == 1.0f ? 0.0f : 1.0f / (1.0f - p_);
    for (size_t index = 0; index < grad_output.NumElements(); ++index) {
        grad_input_data[index] = grad_data[index] * mask_data[index] * scale;
    }

    return grad_input;
}

} // namespace cyxwiz
