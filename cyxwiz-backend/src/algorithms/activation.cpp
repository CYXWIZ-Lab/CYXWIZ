#include "cyxwiz/activation.h"
#include "cyxwiz/tensor.h"
#include "arrayfire_backend_utils.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

// Undefine Windows macros that conflict with ArrayFire functions
// Must be AFTER all includes (Windows headers define these)
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz {

namespace {

void ValidateFloat32UnaryActivation(const Tensor& input, const char* name) {
    if (input.GetDataType() != DataType::Float32) {
        throw std::runtime_error(std::string(name) + " only supports Float32 tensors");
    }
}

void ValidateFloat32ActivationBackward(const Tensor& grad_output,
                                       const Tensor& input,
                                       const char* name) {
    ValidateFloat32UnaryActivation(input, name);
    if (grad_output.GetDataType() != DataType::Float32) {
        throw std::runtime_error(std::string(name) + " gradient only supports Float32 tensors");
    }
    if (grad_output.Shape() != input.Shape()) {
        throw std::runtime_error(std::string(name) + " backward requires matching gradient and input shapes");
    }
}

Tensor CpuReLUForward(const Tensor& input) {
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath, "ReLU::Forward");
    ValidateFloat32UnaryActivation(input, "ReLU");
    Tensor output(input.Shape(), input.GetDataType());
    const float* in = input.ReadData<float>();
    float* out = output.MutableData<float>();
    for (size_t i = 0; i < input.NumElements(); ++i) {
        out[i] = std::max(0.0f, in[i]);
    }
    return output;
}

Tensor CpuReLUBackward(const Tensor& grad_output, const Tensor& input) {
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath, "ReLU::Backward");
    ValidateFloat32ActivationBackward(grad_output, input, "ReLU");
    Tensor grad_input(input.Shape(), input.GetDataType());
    const float* grad = grad_output.ReadData<float>();
    const float* in = input.ReadData<float>();
    float* out = grad_input.MutableData<float>();
    for (size_t i = 0; i < input.NumElements(); ++i) {
        out[i] = in[i] > 0.0f ? grad[i] : 0.0f;
    }
    return grad_input;
}

Tensor CpuSigmoidForward(const Tensor& input) {
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath, "Sigmoid::Forward");
    ValidateFloat32UnaryActivation(input, "Sigmoid");
    Tensor output(input.Shape(), input.GetDataType());
    const float* in = input.ReadData<float>();
    float* out = output.MutableData<float>();
    for (size_t i = 0; i < input.NumElements(); ++i) {
        out[i] = 1.0f / (1.0f + std::exp(-in[i]));
    }
    return output;
}

Tensor CpuSigmoidBackward(const Tensor& grad_output, const Tensor& input) {
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath, "Sigmoid::Backward");
    ValidateFloat32ActivationBackward(grad_output, input, "Sigmoid");
    Tensor grad_input(input.Shape(), input.GetDataType());
    const float* grad = grad_output.ReadData<float>();
    const float* in = input.ReadData<float>();
    float* out = grad_input.MutableData<float>();
    for (size_t i = 0; i < input.NumElements(); ++i) {
        const float value = 1.0f / (1.0f + std::exp(-in[i]));
        out[i] = grad[i] * value * (1.0f - value);
    }
    return grad_input;
}

Tensor CpuTanhForward(const Tensor& input) {
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath, "Tanh::Forward");
    ValidateFloat32UnaryActivation(input, "Tanh");
    Tensor output(input.Shape(), input.GetDataType());
    const float* in = input.ReadData<float>();
    float* out = output.MutableData<float>();
    for (size_t i = 0; i < input.NumElements(); ++i) {
        out[i] = std::tanh(in[i]);
    }
    return output;
}

Tensor CpuTanhBackward(const Tensor& grad_output, const Tensor& input) {
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath, "Tanh::Backward");
    ValidateFloat32ActivationBackward(grad_output, input, "Tanh");
    Tensor grad_input(input.Shape(), input.GetDataType());
    const float* grad = grad_output.ReadData<float>();
    const float* in = input.ReadData<float>();
    float* out = grad_input.MutableData<float>();
    for (size_t i = 0; i < input.NumElements(); ++i) {
        const float value = std::tanh(in[i]);
        out[i] = grad[i] * (1.0f - value * value);
    }
    return grad_input;
}

int NormalizeActivationAxis(int axis, int rank, const char* name) {
    if (rank <= 0) {
        throw std::runtime_error(std::string(name) + " requires at least one tensor dimension");
    }
    int normalized = axis < 0 ? axis + rank : axis;
    if (normalized < 0 || normalized >= rank) {
        throw std::runtime_error(std::string(name) + " axis is out of range");
    }
    return normalized;
}

std::vector<size_t> RowMajorStrides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return strides;
}

Tensor CpuSoftmaxForward(const Tensor& input, int axis, Tensor* cached_output) {
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath, "Softmax::Forward");
    ValidateFloat32UnaryActivation(input, "Softmax");
    const std::vector<size_t>& shape = input.Shape();
    const int actual_axis = NormalizeActivationAxis(axis, static_cast<int>(shape.size()), "Softmax");
    const std::vector<size_t> strides = RowMajorStrides(shape);
    const size_t axis_size = shape[static_cast<size_t>(actual_axis)];
    const size_t axis_stride = strides[static_cast<size_t>(actual_axis)];
    const size_t outer_count = input.NumElements() / axis_size;

    Tensor output(shape, input.GetDataType());
    const float* in = input.ReadData<float>();
    float* out = output.MutableData<float>();

    for (size_t outer = 0; outer < outer_count; ++outer) {
        const size_t before_axis = outer / axis_stride;
        const size_t after_axis = outer % axis_stride;
        const size_t base = before_axis * axis_size * axis_stride + after_axis;

        float max_value = in[base];
        for (size_t i = 1; i < axis_size; ++i) {
            max_value = std::max(max_value, in[base + i * axis_stride]);
        }

        float sum_exp = 0.0f;
        for (size_t i = 0; i < axis_size; ++i) {
            const float value = std::exp(in[base + i * axis_stride] - max_value);
            out[base + i * axis_stride] = value;
            sum_exp += value;
        }

        for (size_t i = 0; i < axis_size; ++i) {
            out[base + i * axis_stride] /= sum_exp;
        }
    }

    if (cached_output) {
        *cached_output = output;
    }
    return output;
}

Tensor CpuSoftmaxBackward(const Tensor& grad_output, const Tensor& input, int axis, const Tensor& cached_output) {
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath, "Softmax::Backward");
    ValidateFloat32ActivationBackward(grad_output, input, "Softmax");
    // Backward receives the source input, so derive the Jacobian from that
    // input rather than trusting a same-shaped cache from an earlier call.
    (void)cached_output;
    Tensor softmax_out = CpuSoftmaxForward(input, axis, nullptr);

    const std::vector<size_t>& shape = input.Shape();
    const int actual_axis = NormalizeActivationAxis(axis, static_cast<int>(shape.size()), "Softmax");
    const std::vector<size_t> strides = RowMajorStrides(shape);
    const size_t axis_size = shape[static_cast<size_t>(actual_axis)];
    const size_t axis_stride = strides[static_cast<size_t>(actual_axis)];
    const size_t outer_count = input.NumElements() / axis_size;

    Tensor grad_input(input.Shape(), input.GetDataType());
    const float* grad = grad_output.ReadData<float>();
    const float* softmax = softmax_out.ReadData<float>();
    float* out = grad_input.MutableData<float>();

    for (size_t outer = 0; outer < outer_count; ++outer) {
        const size_t before_axis = outer / axis_stride;
        const size_t after_axis = outer % axis_stride;
        const size_t base = before_axis * axis_size * axis_stride + after_axis;

        float dot = 0.0f;
        for (size_t i = 0; i < axis_size; ++i) {
            const size_t index = base + i * axis_stride;
            dot += grad[index] * softmax[index];
        }

        for (size_t i = 0; i < axis_size; ++i) {
            const size_t index = base + i * axis_stride;
            out[index] = softmax[index] * (grad[index] - dot);
        }
    }

    return grad_input;
}

template <typename Fn>
Tensor CpuElementwiseActivationForward(const Tensor& input, const char* name, Fn&& fn) {
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath,
        std::string(name) + "::Forward");
    ValidateFloat32UnaryActivation(input, name);
    Tensor output(input.Shape(), input.GetDataType());
    const float* in = input.ReadData<float>();
    float* out = output.MutableData<float>();
    for (size_t i = 0; i < input.NumElements(); ++i) {
        out[i] = fn(in[i]);
    }
    return output;
}

template <typename Fn>
Tensor CpuElementwiseActivationBackward(const Tensor& grad_output,
                                        const Tensor& input,
                                        const char* name,
                                        Fn&& derivative) {
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath,
        std::string(name) + "::Backward");
    ValidateFloat32ActivationBackward(grad_output, input, name);
    Tensor grad_input(input.Shape(), input.GetDataType());
    const float* grad = grad_output.ReadData<float>();
    const float* in = input.ReadData<float>();
    float* out = grad_input.MutableData<float>();
    for (size_t i = 0; i < input.NumElements(); ++i) {
        out[i] = grad[i] * derivative(in[i]);
    }
    return grad_input;
}

float CpuSigmoidValue(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    const float exp_x = std::exp(x);
    return exp_x / (1.0f + exp_x);
}

float CpuSoftplus(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return std::exp(x);
    }
    return std::log1p(std::exp(x));
}

float CpuGELU(float x) {
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    constexpr float gelu_const = 0.044715f;
    const float inner = sqrt_2_over_pi * (x + gelu_const * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

float CpuGELUDerivative(float x) {
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    constexpr float gelu_const = 0.044715f;
    const float x2 = x * x;
    const float inner = sqrt_2_over_pi * (x + gelu_const * x * x2);
    const float tanh_inner = std::tanh(inner);
    const float sech2_inner = 1.0f - tanh_inner * tanh_inner;
    const float d_inner = sqrt_2_over_pi * (1.0f + 3.0f * gelu_const * x2);
    return 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2_inner * d_inner;
}

float CpuMish(float x) {
    return x * std::tanh(CpuSoftplus(x));
}

float CpuMishDerivative(float x) {
    const float softplus = CpuSoftplus(x);
    const float tanh_sp = std::tanh(softplus);
    const float sech2_sp = 1.0f - tanh_sp * tanh_sp;
    return tanh_sp + x * sech2_sp * CpuSigmoidValue(x);
}

} // namespace

// ============================================================================
// Helper Functions for ArrayFire Integration
// ============================================================================

#ifdef CYXWIZ_HAS_ARRAYFIRE

// Helper: Create ArrayFire array from Tensor
static af::array TensorToAf(const Tensor& t) {
    return t.GetArray();
}

// Helper: Create Tensor from ArrayFire array
static Tensor AfToTensor(const af::array& arr) {
    af::array materialized = arr;
    materialized.eval();
    return Tensor(materialized);
}

static void LogActivationFallbackOnce(
    const char* operation_name,
    const char* error_message,
    const Tensor& tensor,
    const char* tensor_name) {
    const BackendFallbackReason reason =
        ClassifyArrayFireBackendFallbackReason(error_message);
    const std::string context = BuildArrayFireBackendFallbackContext(
        BuildTensorShapeContext(tensor_name, tensor.Shape()));
    ThrowIfArrayFireNativeCpuFallbackForbidden(
        operation_name,
        reason,
        error_message,
        context);
    const bool log_fallback =
        ShouldLogArrayFireBackendFallbackOnce(
            operation_name, reason, context);
    if (log_fallback) {
        spdlog::warn("{}",
                     BuildArrayFireBackendFallbackMessage(
                         operation_name, reason,
                         reason != BackendFallbackReason::CudaJitParamOverflow,
                         error_message, context));
    }
}

// Constants for GELU approximation
static const float SQRT_2_OVER_PI = 0.7978845608028654f;  // sqrt(2/pi)
static const float GELU_CONST = 0.044715f;

#endif // CYXWIZ_HAS_ARRAYFIRE

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<Activation> CreateActivation(ActivationType type, float alpha) {
    switch (type) {
        case ActivationType::ReLU:
            return std::make_unique<ReLUActivation>();
        case ActivationType::Sigmoid:
            return std::make_unique<SigmoidActivation>();
        case ActivationType::Tanh:
            return std::make_unique<TanhActivation>();
        case ActivationType::Softmax:
            return std::make_unique<SoftmaxActivation>();
        case ActivationType::LeakyReLU:
            return std::make_unique<LeakyReLUActivation>(alpha);
        case ActivationType::ELU:
            return std::make_unique<ELUActivation>(alpha);
        case ActivationType::GELU:
            return std::make_unique<GELUActivation>();
        case ActivationType::Swish:
        case ActivationType::SiLU:
            return std::make_unique<SwishActivation>();
        case ActivationType::Mish:
            return std::make_unique<MishActivation>();
        case ActivationType::Hardswish:
            return std::make_unique<HardswishActivation>();
        case ActivationType::SELU:
            return std::make_unique<SELUActivation>();
        case ActivationType::PReLU:
            return std::make_unique<PReLUActivation>(1, alpha);
        default:
            throw std::runtime_error("Unknown activation type");
    }
}

// ============================================================================
// ReLU Implementation
// ============================================================================

Tensor ReLUActivation::Forward(const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);
        // ReLU: max(0, x)
        af::array output = af::max(x, 0.0f);
        return AfToTensor(output);
    } catch (const af::exception& e) {
        LogActivationFallbackOnce("ReLU::Forward", e.what(), input, "input");
    }
#endif
    return CpuReLUForward(input);
}

Tensor ReLUActivation::Backward(const Tensor& grad_output, const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array x = TensorToAf(input);
        // Gradient: grad * (x > 0)
        af::array dx = grad_out * (x > 0).as(af::dtype::f32);
        return AfToTensor(dx);
    } catch (const af::exception& e) {
        LogActivationFallbackOnce("ReLU::Backward", e.what(), grad_output, "grad_output");
    }
#endif
    return CpuReLUBackward(grad_output, input);
}

// ============================================================================
// LeakyReLU Implementation
// ============================================================================

Tensor LeakyReLUActivation::Forward(const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);
        // LeakyReLU: max(alpha*x, x) = x if x > 0 else alpha*x
        af::array positive = af::max(x, 0.0f);
        af::array negative = af::min(x, 0.0f) * alpha_;
        af::array output = positive + negative;
        return AfToTensor(output);
    } catch (const af::exception& e) {
        LogActivationFallbackOnce("LeakyReLU::Forward", e.what(), input, "input");
    }
#endif
    return CpuElementwiseActivationForward(input, "LeakyReLU", [this](float x) {
        return x > 0.0f ? x : alpha_ * x;
    });
}

Tensor LeakyReLUActivation::Backward(const Tensor& grad_output, const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array x = TensorToAf(input);
        // Use array expression instead of af::select with two scalars
        af::array mask = (x > 0).as(af::dtype::f32);
        af::array dx = grad_out * (mask + (1.0f - mask) * alpha_);
        return AfToTensor(dx);
    } catch (const af::exception& e) {
        LogActivationFallbackOnce("LeakyReLU::Backward", e.what(), grad_output, "grad_output");
    }
#endif
    return CpuElementwiseActivationBackward(grad_output, input, "LeakyReLU", [this](float x) {
        return x > 0.0f ? 1.0f : alpha_;
    });
}

// ============================================================================
// ELU Implementation
// ============================================================================

Tensor ELUActivation::Forward(const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);
        // ELU: x if x > 0 else alpha * (exp(x) - 1)
        af::array positive = af::max(x, 0.0f);
        af::array negative = alpha_ * (af::exp(af::min(x, 0.0f)) - 1.0f);
        // Only apply negative part where x <= 0
        af::array output = af::select(x > 0, x, negative);
        return AfToTensor(output);
    } catch (const af::exception& e) {
        LogActivationFallbackOnce("ELU::Forward", e.what(), input, "input");
    }
#endif
    return CpuElementwiseActivationForward(input, "ELU", [this](float x) {
        return x > 0.0f ? x : alpha_ * (std::exp(x) - 1.0f);
    });
}

Tensor ELUActivation::Backward(const Tensor& grad_output, const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array x = TensorToAf(input);
        // Gradient: grad * (1 if x > 0 else alpha * exp(x))
        af::array dx = grad_out * af::select(x > 0, 1.0f, alpha_ * af::exp(x));
        return AfToTensor(dx);
    } catch (const af::exception& e) {
        LogActivationFallbackOnce("ELU::Backward", e.what(), grad_output, "grad_output");
    }
#endif
    return CpuElementwiseActivationBackward(grad_output, input, "ELU", [this](float x) {
        return x > 0.0f ? 1.0f : alpha_ * std::exp(x);
    });
}

// ============================================================================
// GELU Implementation (Gaussian Error Linear Unit)
// ============================================================================

Tensor GELUActivation::Forward(const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        af::array inner = SQRT_2_OVER_PI * (x + GELU_CONST * af::pow(x, 3));
        af::array output = 0.5f * x * (1.0f + af::tanh(inner));
        return AfToTensor(output);
    } catch (const af::exception& e) {
        LogActivationFallbackOnce("GELU::Forward", e.what(), input, "input");
    }
#endif
    return CpuElementwiseActivationForward(input, "GELU", CpuGELU);
}

Tensor GELUActivation::Backward(const Tensor& grad_output, const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array x = TensorToAf(input);

        // GELU derivative (using approximation)
        // d/dx [0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))]
        af::array x3 = af::pow(x, 3);
        af::array inner = SQRT_2_OVER_PI * (x + GELU_CONST * x3);
        af::array tanh_inner = af::tanh(inner);
        af::array sech2_inner = 1.0f - tanh_inner * tanh_inner;

        // Derivative of tanh(inner) w.r.t. x
        af::array d_inner = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_CONST * x * x);

        // Full derivative
        af::array dx = grad_out * (0.5f * (1.0f + tanh_inner) +
                                    0.5f * x * sech2_inner * d_inner);

        return AfToTensor(dx);
    } catch (const af::exception& e) {
        LogActivationFallbackOnce("GELU::Backward", e.what(), grad_output, "grad_output");
    }
#endif
    return CpuElementwiseActivationBackward(grad_output, input, "GELU", CpuGELUDerivative);
}

// ============================================================================
// Swish / SiLU Implementation
// ============================================================================

Tensor SwishActivation::Forward(const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);
        // Swish: x * sigmoid(x)
        af::array sigmoid_x = af::sigmoid(x);
        af::array output = x * sigmoid_x;
        return AfToTensor(output);
    } catch (const af::exception& e) {
        LogActivationFallbackOnce("Swish::Forward", e.what(), input, "input");
    }
#endif
    return CpuElementwiseActivationForward(input, "Swish", [](float x) {
        return x * CpuSigmoidValue(x);
    });
}

Tensor SwishActivation::Backward(const Tensor& grad_output, const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array x = TensorToAf(input);

        // Swish derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        //                 = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        af::array sigmoid_x = af::sigmoid(x);
        af::array dx = grad_out * sigmoid_x * (1.0f + x * (1.0f - sigmoid_x));

        return AfToTensor(dx);
    } catch (const af::exception& e) {
        LogActivationFallbackOnce("Swish::Backward", e.what(), grad_output, "grad_output");
    }
#endif
    return CpuElementwiseActivationBackward(grad_output, input, "Swish", [](float x) {
        const float sigmoid = CpuSigmoidValue(x);
        return sigmoid * (1.0f + x * (1.0f - sigmoid));
    });
}

// ============================================================================
// Sigmoid Implementation
// ============================================================================

Tensor SigmoidActivation::Forward(const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);
        // Sigmoid: 1 / (1 + exp(-x))
        af::array output = af::sigmoid(x);
        return AfToTensor(output);
    } catch (const af::exception& e) {
        LogActivationFallbackOnce("Sigmoid::Forward", e.what(), input, "input");
    }
#endif
    return CpuSigmoidForward(input);
}

Tensor SigmoidActivation::Backward(const Tensor& grad_output, const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array x = TensorToAf(input);

        // Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
        af::array sigmoid_x = af::sigmoid(x);
        af::array dx = grad_out * sigmoid_x * (1.0f - sigmoid_x);

        return AfToTensor(dx);
    } catch (const af::exception& e) {
        LogActivationFallbackOnce("Sigmoid::Backward", e.what(), grad_output, "grad_output");
    }
#endif
    return CpuSigmoidBackward(grad_output, input);
}

// ============================================================================
// Tanh Implementation
// ============================================================================

Tensor TanhActivation::Forward(const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);
        af::array output = af::tanh(x);
        return AfToTensor(output);
    } catch (const af::exception& e) {
        LogActivationFallbackOnce("Tanh::Forward", e.what(), input, "input");
    }
#endif
    return CpuTanhForward(input);
}

Tensor TanhActivation::Backward(const Tensor& grad_output, const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array x = TensorToAf(input);

        // Tanh derivative: 1 - tanh(x)^2
        af::array tanh_x = af::tanh(x);
        af::array dx = grad_out * (1.0f - tanh_x * tanh_x);

        return AfToTensor(dx);
    } catch (const af::exception& e) {
        LogActivationFallbackOnce("Tanh::Backward", e.what(), grad_output, "grad_output");
    }
#endif
    return CpuTanhBackward(grad_output, input);
}

// ============================================================================
// Softmax Implementation
// ============================================================================

Tensor SoftmaxActivation::Forward(const Tensor& input) {
    ValidateFloat32UnaryActivation(input, "Softmax");
    const int actual_axis = NormalizeActivationAxis(
        axis_, static_cast<int>(input.Shape().size()), "Softmax");
    if (input.NumElements() == 0) {
        cached_output_ = input.Clone();
        return cached_output_;
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = input.GetSemanticArray();

        // For numerical stability, subtract max before exp
        af::array max_vals = af::max(x, actual_axis);
        max_vals.eval();

        // Tile max_vals to match x dimensions for subtraction
        af::dim4 tile_dims(1, 1, 1, 1);
        tile_dims[actual_axis] = x.dims(actual_axis);
        af::array x_stable = x - af::tile(max_vals, tile_dims);
        x_stable.eval();

        // Compute softmax: exp(x - max) / sum(exp(x - max))
        af::array exp_x = af::exp(x_stable);
        exp_x.eval();
        af::array sum_exp = af::sum(exp_x, actual_axis);
        sum_exp.eval();
        af::array output = exp_x / af::tile(sum_exp, tile_dims);
        output.eval();

        cached_output_ =
            Tensor::FromSemanticArray(output, input.Shape());
        return cached_output_;
    } catch (const af::exception& e) {
        LogActivationFallbackOnce(
            "Softmax::Forward", e.what(), input, "input");
    }
#endif
    return CpuSoftmaxForward(input, axis_, &cached_output_);
}

Tensor SoftmaxActivation::Backward(const Tensor& grad_output, const Tensor& input) {
    ValidateFloat32ActivationBackward(grad_output, input, "Softmax");
    const int actual_axis = NormalizeActivationAxis(
        axis_, static_cast<int>(input.Shape().size()), "Softmax");
    if (input.NumElements() == 0) {
        return grad_output.Clone();
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = grad_output.GetSemanticArray();
        af::array x = input.GetSemanticArray();

        af::array max_vals = af::max(x, actual_axis);
        af::dim4 tile_dims(1, 1, 1, 1);
        tile_dims[actual_axis] = x.dims(actual_axis);
        af::array exp_x = af::exp(x - af::tile(max_vals, tile_dims));
        af::array softmax_out =
            exp_x / af::tile(af::sum(exp_x, actual_axis), tile_dims);

        // Softmax backward: softmax * (grad - sum(grad * softmax))
        af::array grad_softmax = grad_out * softmax_out;
        grad_softmax.eval();
        af::array sum_grad_softmax = af::sum(grad_softmax, actual_axis);
        sum_grad_softmax.eval();

        af::array dx = softmax_out * (grad_out - af::tile(sum_grad_softmax, tile_dims));
        dx.eval();

        return Tensor::FromSemanticArray(dx, input.Shape());
    } catch (const af::exception& e) {
        LogActivationFallbackOnce(
            "Softmax::Backward", e.what(), grad_output, "grad_output");
    }
#endif
    return CpuSoftmaxBackward(grad_output, input, axis_, cached_output_);
}

// ============================================================================
// Mish Implementation
// ============================================================================

Tensor MishActivation::Forward(const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);
        // Stable softplus avoids exp overflow for large finite inputs.
        af::array softplus_x =
            af::max(x, 0.0f) + af::log(1.0f + af::exp(-af::abs(x)));
        af::array output = x * af::tanh(softplus_x);
        return AfToTensor(output);
    } catch (const af::exception& e) {
        LogActivationFallbackOnce("Mish::Forward", e.what(), input, "input");
    }
#endif
    return CpuElementwiseActivationForward(input, "Mish", CpuMish);
}

Tensor MishActivation::Backward(const Tensor& grad_output, const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array x = TensorToAf(input);

        // Mish derivative is complex:
        // d/dx [x * tanh(softplus(x))]
        // = tanh(softplus(x)) + x * sech^2(softplus(x)) * sigmoid(x)
        af::array softplus_x =
            af::max(x, 0.0f) + af::log(1.0f + af::exp(-af::abs(x)));
        af::array tanh_sp = af::tanh(softplus_x);
        af::array sech2_sp = 1.0f - tanh_sp * tanh_sp;
        af::array sigmoid_x = af::sigmoid(x);

        af::array dx = grad_out * (tanh_sp + x * sech2_sp * sigmoid_x);

        return AfToTensor(dx);
    } catch (const af::exception& e) {
        LogActivationFallbackOnce("Mish::Backward", e.what(), grad_output, "grad_output");
    }
#endif
    return CpuElementwiseActivationBackward(grad_output, input, "Mish", CpuMishDerivative);
}

// ============================================================================
// Hardswish Implementation
// ============================================================================

Tensor HardswishActivation::Forward(const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);
        // Hardswish: x * ReLU6(x + 3) / 6
        // = 0                   if x <= -3
        // = x                   if x >= 3
        // = x * (x + 3) / 6     otherwise

        // Use mask-based approach instead of nested af::select with scalars
        af::array mask_low = (x <= -3.0f).as(af::dtype::f32);
        af::array mask_high = (x >= 3.0f).as(af::dtype::f32);
        af::array mask_mid = (1.0f - mask_low) * (1.0f - mask_high);
        af::array output = mask_high * x + mask_mid * (x * (x + 3.0f) / 6.0f);
        return AfToTensor(output);
    } catch (const af::exception& e) {
        LogActivationFallbackOnce("Hardswish::Forward", e.what(), input, "input");
    }
#endif
    return CpuElementwiseActivationForward(input, "Hardswish", [](float x) {
        if (x <= -3.0f) {
            return 0.0f;
        }
        if (x >= 3.0f) {
            return x;
        }
        return x * (x + 3.0f) / 6.0f;
    });
}

Tensor HardswishActivation::Backward(const Tensor& grad_output, const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array x = TensorToAf(input);

        // Hardswish derivative:
        // = 0                   if x <= -3
        // = 1                   if x >= 3
        // = (2x + 3) / 6        otherwise

        // Use mask-based approach instead of nested af::select with scalars
        af::array mask_low = (x <= -3.0f).as(af::dtype::f32);
        af::array mask_high = (x >= 3.0f).as(af::dtype::f32);
        af::array mask_mid = (1.0f - mask_low) * (1.0f - mask_high);
        af::array dx = grad_out * (mask_high + mask_mid * ((2.0f * x + 3.0f) / 6.0f));
        return AfToTensor(dx);
    } catch (const af::exception& e) {
        LogActivationFallbackOnce("Hardswish::Backward", e.what(), grad_output, "grad_output");
    }
#endif
    return CpuElementwiseActivationBackward(grad_output, input, "Hardswish", [](float x) {
        if (x <= -3.0f) {
            return 0.0f;
        }
        if (x >= 3.0f) {
            return 1.0f;
        }
        return (2.0f * x + 3.0f) / 6.0f;
    });
}


// ============================================================================
// SELU Implementation - Scaled Exponential Linear Unit
// ============================================================================

Tensor SELUActivation::Forward(const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);
        // SELU: scale * (max(0, x) + alpha * min(0, exp(x) - 1))
        af::array positive = af::max(x, 0.0f);
        af::array negative = af::min(x, 0.0f);
        af::array output = SCALE * (positive + ALPHA * (af::exp(negative) - 1.0f));
        return AfToTensor(output);
    } catch (const af::exception& e) {
        LogActivationFallbackOnce("SELU::Forward", e.what(), input, "input");
    }
#endif
    return CpuElementwiseActivationForward(input, "SELU", [](float x) {
        return SCALE * (x > 0.0f ? x : ALPHA * (std::exp(x) - 1.0f));
    });
}

Tensor SELUActivation::Backward(const Tensor& grad_output, const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad = TensorToAf(grad_output);
        af::array x = TensorToAf(input);
        // d(SELU)/dx = scale for x > 0, scale * alpha * exp(x) for x <= 0
        af::array positive_mask = (x > 0).as(af::dtype::f32);
        af::array negative_mask = (x <= 0).as(af::dtype::f32);
        af::array grad_input = grad * SCALE *
            (positive_mask + ALPHA * af::exp(af::min(x, 0.0f)) * negative_mask);
        return AfToTensor(grad_input);
    } catch (const af::exception& e) {
        LogActivationFallbackOnce("SELU::Backward", e.what(), grad_output, "grad_output");
    }
#endif
    return CpuElementwiseActivationBackward(grad_output, input, "SELU", [](float x) {
        return SCALE * (x > 0.0f ? 1.0f : ALPHA * std::exp(x));
    });
}

// ============================================================================
// PReLU Implementation - Parametric ReLU
// ============================================================================

PReLUActivation::PReLUActivation(int num_parameters, float init)
    : num_parameters_(num_parameters) {
    if (num_parameters_ <= 0 || !std::isfinite(init)) {
        throw std::invalid_argument(
            "PReLU requires a positive parameter count and finite initial value");
    }
    // Initialize alpha with the init value
    alpha_ = Tensor({static_cast<size_t>(num_parameters)}, DataType::Float32);
    float* alpha_data = alpha_.MutableData<float>();
    for (int i = 0; i < num_parameters; ++i) {
        alpha_data[i] = init;
    }
    grad_alpha_ = Tensor::Zeros({static_cast<size_t>(num_parameters)});
}

void PReLUActivation::SetAlpha(const Tensor& alpha) {
    if (alpha.GetDataType() != DataType::Float32 ||
        alpha.Shape() != std::vector<size_t>{
                             static_cast<size_t>(num_parameters_)}) {
        throw std::invalid_argument(
            "PReLU alpha must be Float32 with shape [num_parameters]");
    }
    alpha_ = alpha;
}

Tensor PReLUActivation::Forward(const Tensor& input) {
    ValidateFloat32UnaryActivation(input, "PReLU");
    if (num_parameters_ != 1 &&
        (input.Shape().size() < 2 ||
         input.Shape()[1] != static_cast<size_t>(num_parameters_))) {
        throw std::invalid_argument(
            "PReLU channel parameters must match input dimension 1");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = input.GetSemanticArray();
        af::array alpha = alpha_.GetSemanticArray();

        // PReLU: max(0, x) + alpha * min(0, x)
        af::array positive = af::max(x, 0.0f);
        af::array negative = af::min(x, 0.0f);

        af::array output;
        if (num_parameters_ == 1) {
            // Shared alpha across all channels
            output = positive + alpha(0) * negative;
        } else {
            // Per-channel alpha (for CNNs)
            // Input shape assumed: [batch, channels, ...] or similar
            // Reshape alpha for broadcasting
            af::dim4 alpha_shape(1, 1, 1, 1);
            // Assume channels are in dim 1 (after batch in dim 0 for AF column-major)
            alpha_shape[1] = static_cast<dim_t>(num_parameters_);
            af::array alpha_bc = af::moddims(alpha, alpha_shape);

            // Tile to match input dimensions
            af::dim4 tile_dims = x.dims();
            tile_dims[1] = 1;  // Don't tile along channel dimension
            alpha_bc = af::tile(alpha_bc, tile_dims);

            output = positive + alpha_bc * negative;
        }

        return Tensor::FromSemanticArray(output, input.Shape());
    } catch (const af::exception& e) {
        LogActivationFallbackOnce("PReLU::Forward", e.what(), input, "input");
    }
#endif
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath, "PReLU::Forward");
    const auto& shape = input.Shape();
    size_t channel_span = 1;
    for (size_t dimension = 2; dimension < shape.size(); ++dimension) {
        channel_span *= shape[dimension];
    }

    const float* alpha = alpha_.ReadData<float>();
    const float* in = input.ReadData<float>();
    Tensor output(shape, input.GetDataType());
    float* out = output.MutableData<float>();
    for (size_t i = 0; i < input.NumElements(); ++i) {
        const size_t channel = num_parameters_ == 1
            ? 0
            : (i / channel_span) % shape[1];
        out[i] = in[i] > 0.0f ? in[i] : alpha[channel] * in[i];
    }
    return output;
}

Tensor PReLUActivation::Backward(const Tensor& grad_output, const Tensor& input) {
    ValidateFloat32ActivationBackward(grad_output, input, "PReLU");
    if (num_parameters_ != 1 &&
        (input.Shape().size() < 2 ||
         input.Shape()[1] != static_cast<size_t>(num_parameters_))) {
        throw std::invalid_argument(
            "PReLU channel parameters must match input dimension 1");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad = grad_output.GetSemanticArray();
        af::array x = input.GetSemanticArray();
        af::array alpha = alpha_.GetSemanticArray();

        // Gradient w.r.t input: 1 for x > 0, alpha for x <= 0
        af::array positive_mask = (x > 0).as(af::dtype::f32);
        af::array negative_mask = (x <= 0).as(af::dtype::f32);

        af::array grad_input;
        if (num_parameters_ == 1) {
            grad_input = grad * (positive_mask + alpha(0) * negative_mask);

            // Gradient w.r.t alpha: sum of grad * min(0, x)
            af::array grad_alpha_arr =
                af::sum(af::flat(grad * af::min(x, 0.0f)));
            grad_alpha_ = Tensor::FromSemanticArray(
                af::moddims(grad_alpha_arr, af::dim4(1)), {1});
        } else {
            // Per-channel
            af::dim4 alpha_shape(1, 1, 1, 1);
            alpha_shape[1] = static_cast<dim_t>(num_parameters_);
            af::array alpha_bc = af::moddims(alpha, alpha_shape);
            af::dim4 tile_dims = x.dims();
            tile_dims[1] = 1;
            alpha_bc = af::tile(alpha_bc, tile_dims);

            grad_input = grad * (positive_mask + alpha_bc * negative_mask);

            // Gradient w.r.t alpha per channel
            af::array negative = af::min(x, 0.0f);
            af::array grad_times_neg = grad * negative;
            // Sum over every semantic dimension except channel dimension 1.
            af::array grad_alpha_arr = grad_times_neg;
            for (int dimension = 3; dimension >= 0; --dimension) {
                if (dimension != 1) {
                    grad_alpha_arr = af::sum(grad_alpha_arr, dimension);
                }
            }
            grad_alpha_ = Tensor::FromSemanticArray(
                af::moddims(grad_alpha_arr, af::dim4(num_parameters_)),
                {static_cast<size_t>(num_parameters_)});
        }

        return Tensor::FromSemanticArray(grad_input, input.Shape());
    } catch (const af::exception& e) {
        LogActivationFallbackOnce("PReLU::Backward", e.what(), grad_output, "grad_output");
    }
#endif
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath, "PReLU::Backward");
    const auto& shape = input.Shape();
    size_t channel_span = 1;
    for (size_t dimension = 2; dimension < shape.size(); ++dimension) {
        channel_span *= shape[dimension];
    }

    const float* alpha = alpha_.ReadData<float>();
    Tensor grad_input(input.Shape(), input.GetDataType());
    grad_alpha_ = Tensor::Zeros({static_cast<size_t>(num_parameters_)});
    float* grad_alpha = grad_alpha_.MutableData<float>();
    const float* grad = grad_output.ReadData<float>();
    const float* in = input.ReadData<float>();
    float* out = grad_input.MutableData<float>();
    for (size_t i = 0; i < input.NumElements(); ++i) {
        const size_t channel = num_parameters_ == 1
            ? 0
            : (i / channel_span) % shape[1];
        out[i] = grad[i] * (in[i] > 0.0f ? 1.0f : alpha[channel]);
        if (in[i] < 0.0f) {
            grad_alpha[channel] += grad[i] * in[i];
        }
    }
    return grad_input;
}

} // namespace cyxwiz
