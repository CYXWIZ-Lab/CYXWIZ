#include "cyxwiz/layers/dropout.h"
#include "layer_arrayfire_utils.h"

#include <cmath>
#include <stdexcept>
#include <string>


namespace cyxwiz {

// One ArrayFire path (the CPU is ArrayFire's CPU backend): a device error is
// reported, not hidden behind host loops; a build without ArrayFire refuses.
#ifdef CYXWIZ_HAS_ARRAYFIRE
[[noreturn]] static void ThrowDropoutDeviceError(const char* operation, const af::exception& error) {
    throw std::runtime_error(std::string(operation) + " failed on the ArrayFire device: " + error.what());
}
#else
[[noreturn]] static void ThrowDropoutNeedsArrayFire() {
    throw std::runtime_error("Dropout runs on ArrayFire, and this build has no ArrayFire");
}
#endif

DropoutLayer::DropoutLayer(float p) : p_(p) {
    if (!std::isfinite(p) || p < 0.0f || p > 1.0f) {
        throw std::invalid_argument(
            "Dropout probability must be finite and in [0, 1]");
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
        ThrowDropoutDeviceError("DropoutLayer::Forward", error);
    }
#else
    ThrowDropoutNeedsArrayFire();
#endif
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
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array mask = TensorToAf(mask_);
        const float scale = p_ == 1.0f ? 0.0f : 1.0f / (1.0f - p_);
        af::array dx = grad_out * mask * scale;
        dx.eval();
        return Tensor::FromSemanticArray(dx, grad_output.Shape());
    } catch (const af::exception& error) {
        ThrowDropoutDeviceError("DropoutLayer::Backward", error);
    }
#else
    ThrowDropoutNeedsArrayFire();
#endif
}

} // namespace cyxwiz
