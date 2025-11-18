#include "cyxwiz/activations/tanh.h"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace cyxwiz {

Tensor Tanh::Forward(const Tensor& input) {
    Tensor output(input.Shape(), input.GetDataType());

    size_t num_elements = input.NumElements();

    if (input.GetDataType() != DataType::Float32) {
        throw std::runtime_error("Tanh only supports Float32 tensors");
    }

    const float* input_data = static_cast<const float*>(input.Data());
    float* output_data = static_cast<float*>(output.Data());

    // Apply Tanh: f(x) = tanh(x)
    for (size_t i = 0; i < num_elements; i++) {
        output_data[i] = std::tanh(input_data[i]);
    }

    return output;
}

Tensor Tanh::Backward(const Tensor& grad_output, const Tensor& input) {
    if (grad_output.Shape() != input.Shape()) {
        throw std::runtime_error("Tanh::Backward: gradient and input shapes must match");
    }

    Tensor grad_input(input.Shape(), input.GetDataType());

    size_t num_elements = input.NumElements();
    const float* grad_out_data = static_cast<const float*>(grad_output.Data());
    const float* input_data = static_cast<const float*>(input.Data());
    float* grad_in_data = static_cast<float*>(grad_input.Data());

    // Gradient: f'(x) = 1 - tanh(x)^2
    for (size_t i = 0; i < num_elements; i++) {
        float tanh_val = std::tanh(input_data[i]);
        grad_in_data[i] = grad_out_data[i] * (1.0f - tanh_val * tanh_val);
    }

    return grad_input;
}

} // namespace cyxwiz
