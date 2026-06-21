#include "cyxwiz/layers/dropout.h"
#include "layer_arrayfire_utils.h"

#include <random>
#include <stdexcept>

#include <spdlog/spdlog.h>

namespace cyxwiz {

DropoutLayer::DropoutLayer(float p) : p_(p) {
    if (p < 0.0f || p >= 1.0f) {
        throw std::invalid_argument("Dropout probability must be in [0, 1)");
    }
}

Tensor DropoutLayer::Forward(const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);

        if (training_ && p_ > 0.0f) {
            // Generate random mask
            af::array rand_mask = af::randu(x.dims(), af::dtype::f32);
            af::array mask = (rand_mask > p_).as(af::dtype::f32);

            // Scale by 1/(1-p) to maintain expected value
            float scale = 1.0f / (1.0f - p_);
            af::array output = x * mask * scale;

            mask_ = AfToTensor(mask);
            return AfToTensor(output);
        } else {
            // During inference, just pass through
            return input;
        }
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire DropoutLayer::Forward failed: {}", e.what());
    }
#endif

    if (!training_ || p_ <= 0.0f) {
        return input;
    }
    if (input.GetDataType() != DataType::Float32) {
        throw std::runtime_error("Dropout forward CPU fallback requires Float32 input");
    }

    Tensor output(input.Shape(), DataType::Float32);
    mask_ = Tensor(input.Shape(), DataType::Float32);
    const float* input_data = input.Data<float>();
    float* output_data = output.Data<float>();
    float* mask_data = mask_.Data<float>();
    const float scale = 1.0f / (1.0f - p_);

    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < input.NumElements(); ++i) {
        mask_data[i] = dist(rng) > p_ ? 1.0f : 0.0f;
        output_data[i] = input_data[i] * mask_data[i] * scale;
    }

    return output;
}

Tensor DropoutLayer::Backward(const Tensor& grad_output) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        if (training_ && p_ > 0.0f) {
            af::array grad_out = TensorToAf(grad_output);
            af::array mask = TensorToAf(mask_);

            // Apply same mask and scaling
            float scale = 1.0f / (1.0f - p_);
            af::array dx = grad_out * mask * scale;

            return AfToTensor(dx);
        } else {
            return grad_output;
        }
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire DropoutLayer::Backward failed: {}", e.what());
    }
#endif

    if (!training_ || p_ <= 0.0f) {
        return grad_output;
    }
    if (grad_output.GetDataType() != DataType::Float32 || mask_.GetDataType() != DataType::Float32) {
        throw std::runtime_error("Dropout backward CPU fallback requires Float32 tensors");
    }
    if (mask_.Shape() != grad_output.Shape()) {
        throw std::runtime_error("Dropout backward requires a forward mask matching grad_output");
    }

    Tensor grad_input(grad_output.Shape(), DataType::Float32);
    const float* grad_data = grad_output.Data<float>();
    const float* mask_data = mask_.Data<float>();
    float* grad_input_data = grad_input.Data<float>();
    const float scale = 1.0f / (1.0f - p_);
    for (size_t i = 0; i < grad_output.NumElements(); ++i) {
        grad_input_data[i] = grad_data[i] * mask_data[i] * scale;
    }

    return grad_input;
}

} // namespace cyxwiz
