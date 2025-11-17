#include "cyxwiz/activations/sigmoid.h"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace cyxwiz {

Tensor Sigmoid::Forward(const Tensor& input) {
    Tensor output(input.Shape(), input.GetDataType());

    size_t num_elements = input.NumElements();

    if (input.GetDataType() != DataType::Float32) {
        throw std::runtime_error("Sigmoid only supports Float32 tensors");
    }

    const float* input_data = static_cast<const float*>(input.Data());
    float* output_data = static_cast<float*>(output.Data());

    // Apply Sigmoid: f(x) = 1 / (1 + exp(-x))
    for (size_t i = 0; i < num_elements; i++) {
        output_data[i] = 1.0f / (1.0f + std::exp(-input_data[i]));
    }

    return output;
}

Tensor Sigmoid::Backward(const Tensor& grad_output, const Tensor& input) {
    if (grad_output.Shape() != input.Shape()) {
        throw std::runtime_error("Sigmoid::Backward: gradient and input shapes must match");
    }

    Tensor grad_input(input.Shape(), input.GetDataType());

    size_t num_elements = input.NumElements();
    const float* grad_out_data = static_cast<const float*>(grad_output.Data());
    const float* input_data = static_cast<const float*>(input.Data());
    float* grad_in_data = static_cast<float*>(grad_input.Data());

    // Gradient: f'(x) = sigmoid(x) * (1 - sigmoid(x))
    for (size_t i = 0; i < num_elements; i++) {
        float sigmoid_val = 1.0f / (1.0f + std::exp(-input_data[i]));
        grad_in_data[i] = grad_out_data[i] * sigmoid_val * (1.0f - sigmoid_val);
    }

    return grad_input;
}

} // namespace cyxwiz
