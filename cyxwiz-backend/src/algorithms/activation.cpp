#include "cyxwiz/activation.h"
#include "cyxwiz/tensor.h"
#include <stdexcept>
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

// ============================================================================
// Helper Functions for ArrayFire Integration
// ============================================================================

#ifdef CYXWIZ_HAS_ARRAYFIRE

// Helper: Convert CyxWiz DataType to ArrayFire dtype
static af::dtype ToAfType(DataType dtype) {
    switch (dtype) {
        case DataType::Float32: return af::dtype::f32;
        case DataType::Float64: return af::dtype::f64;
        case DataType::Int32: return af::dtype::s32;
        case DataType::Int64: return af::dtype::s64;
        case DataType::UInt8: return af::dtype::u8;
        default: throw std::runtime_error("Unsupported DataType for ArrayFire");
    }
}

// Helper: Create ArrayFire array from Tensor
static af::array TensorToAf(const Tensor& t) {
    const auto& shape = t.Shape();
    af::dim4 dims(1, 1, 1, 1);
    for (size_t i = 0; i < shape.size() && i < 4; i++) {
        dims[static_cast<unsigned int>(i)] = static_cast<dim_t>(shape[i]);
    }

    af::array arr(dims, ToAfType(t.GetDataType()));
    arr.write(t.Data(), arr.bytes(), afHost);
    return arr;
}

// Helper: Create Tensor from ArrayFire array
static Tensor AfToTensor(const af::array& arr) {
    std::vector<size_t> shape;
    for (unsigned int i = 0; i < 4; i++) {
        if (arr.dims(i) > 1 || i == 0) {
            shape.push_back(static_cast<size_t>(arr.dims(i)));
        } else if (i > 0 && arr.dims(i) == 1) {
            bool all_ones = true;
            for (unsigned int j = i; j < 4; j++) {
                if (arr.dims(j) != 1) {
                    all_ones = false;
                    break;
                }
            }
            if (all_ones) break;
            shape.push_back(static_cast<size_t>(arr.dims(i)));
        }
    }

    DataType dtype = DataType::Float32;
    switch (arr.type()) {
        case af::dtype::f32: dtype = DataType::Float32; break;
        case af::dtype::f64: dtype = DataType::Float64; break;
        case af::dtype::s32: dtype = DataType::Int32; break;
        case af::dtype::s64: dtype = DataType::Int64; break;
        case af::dtype::u8: dtype = DataType::UInt8; break;
        default: dtype = DataType::Float32;
    }

    Tensor result(shape, dtype);
    arr.host(result.Data());
    return result;
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
        spdlog::warn("ArrayFire ReLU::Forward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("ReLU forward requires ArrayFire");
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
        spdlog::warn("ArrayFire ReLU::Backward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("ReLU backward requires ArrayFire");
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
        spdlog::warn("ArrayFire LeakyReLU::Forward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("LeakyReLU forward requires ArrayFire");
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
        spdlog::warn("ArrayFire LeakyReLU::Backward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("LeakyReLU backward requires ArrayFire");
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
        spdlog::warn("ArrayFire ELU::Forward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("ELU forward requires ArrayFire");
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
        spdlog::warn("ArrayFire ELU::Backward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("ELU backward requires ArrayFire");
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
        spdlog::warn("ArrayFire GELU::Forward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("GELU forward requires ArrayFire");
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
        spdlog::warn("ArrayFire GELU::Backward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("GELU backward requires ArrayFire");
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
        spdlog::warn("ArrayFire Swish::Forward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("Swish forward requires ArrayFire");
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
        spdlog::warn("ArrayFire Swish::Backward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("Swish backward requires ArrayFire");
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
        spdlog::warn("ArrayFire Sigmoid::Forward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("Sigmoid forward requires ArrayFire");
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
        spdlog::warn("ArrayFire Sigmoid::Backward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("Sigmoid backward requires ArrayFire");
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
        spdlog::warn("ArrayFire Tanh::Forward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("Tanh forward requires ArrayFire");
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
        spdlog::warn("ArrayFire Tanh::Backward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("Tanh backward requires ArrayFire");
}

// ============================================================================
// Softmax Implementation
// ============================================================================

Tensor SoftmaxActivation::Forward(const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);

        // Determine the axis for softmax (default is last dimension)
        int actual_axis = axis_;
        if (actual_axis < 0) {
            actual_axis = static_cast<int>(x.numdims()) - 1;
        }

        // For numerical stability, subtract max before exp
        af::array max_vals = af::max(x, actual_axis);

        // Tile max_vals to match x dimensions for subtraction
        af::dim4 tile_dims(1, 1, 1, 1);
        tile_dims[actual_axis] = x.dims(actual_axis);
        af::array x_stable = x - af::tile(max_vals, tile_dims);

        // Compute softmax: exp(x - max) / sum(exp(x - max))
        af::array exp_x = af::exp(x_stable);
        af::array sum_exp = af::sum(exp_x, actual_axis);
        af::array output = exp_x / af::tile(sum_exp, tile_dims);

        cached_output_ = AfToTensor(output);
        return cached_output_;
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire Softmax::Forward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("Softmax forward requires ArrayFire");
}

Tensor SoftmaxActivation::Backward(const Tensor& grad_output, const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array softmax_out = TensorToAf(cached_output_);

        int actual_axis = axis_;
        if (actual_axis < 0) {
            actual_axis = static_cast<int>(softmax_out.numdims()) - 1;
        }

        // Softmax backward: softmax * (grad - sum(grad * softmax))
        af::array sum_grad_softmax = af::sum(grad_out * softmax_out, actual_axis);

        af::dim4 tile_dims(1, 1, 1, 1);
        tile_dims[actual_axis] = softmax_out.dims(actual_axis);

        af::array dx = softmax_out * (grad_out - af::tile(sum_grad_softmax, tile_dims));

        return AfToTensor(dx);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire Softmax::Backward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("Softmax backward requires ArrayFire");
}

// ============================================================================
// Mish Implementation
// ============================================================================

Tensor MishActivation::Forward(const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);
        // Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        af::array softplus_x = af::log(1.0f + af::exp(x));
        af::array output = x * af::tanh(softplus_x);
        return AfToTensor(output);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire Mish::Forward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("Mish forward requires ArrayFire");
}

Tensor MishActivation::Backward(const Tensor& grad_output, const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array x = TensorToAf(input);

        // Mish derivative is complex:
        // d/dx [x * tanh(softplus(x))]
        // = tanh(softplus(x)) + x * sech^2(softplus(x)) * sigmoid(x)
        af::array exp_x = af::exp(x);
        af::array softplus_x = af::log(1.0f + exp_x);
        af::array tanh_sp = af::tanh(softplus_x);
        af::array sech2_sp = 1.0f - tanh_sp * tanh_sp;
        af::array sigmoid_x = exp_x / (1.0f + exp_x);

        af::array dx = grad_out * (tanh_sp + x * sech2_sp * sigmoid_x);

        return AfToTensor(dx);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire Mish::Backward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("Mish backward requires ArrayFire");
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
        spdlog::warn("ArrayFire Hardswish::Forward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("Hardswish forward requires ArrayFire");
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
        spdlog::warn("ArrayFire Hardswish::Backward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("Hardswish backward requires ArrayFire");
}

} // namespace cyxwiz
