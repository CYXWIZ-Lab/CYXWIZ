#include "cyxwiz/activations/relu.h"
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace cyxwiz {

Tensor ReLU::Forward(const Tensor& input) {
    Tensor output(input.Shape(), input.GetDataType());

    const auto& shape = input.Shape();
    size_t num_elements = input.NumElements();

    // Only support Float32 for now
    if (input.GetDataType() != DataType::Float32) {
        throw std::runtime_error("ReLU only supports Float32 tensors");
    }

    const float* input_data = static_cast<const float*>(input.Data());
    float* output_data = static_cast<float*>(output.Data());

    // Apply ReLU: f(x) = max(0, x)
    for (size_t i = 0; i < num_elements; i++) {
        output_data[i] = std::max(0.0f, input_data[i]);
    }

    return output;
}

Tensor ReLU::Backward(const Tensor& grad_output, const Tensor& input) {
    if (grad_output.Shape() != input.Shape()) {
        throw std::runtime_error("ReLU::Backward: gradient and input shapes must match");
    }

    Tensor grad_input(input.Shape(), input.GetDataType());

    size_t num_elements = input.NumElements();
    const float* grad_out_data = static_cast<const float*>(grad_output.Data());
    const float* input_data = static_cast<const float*>(input.Data());
    float* grad_in_data = static_cast<float*>(grad_input.Data());

    // Gradient: f'(x) = 1 if x > 0, else 0
    for (size_t i = 0; i < num_elements; i++) {
        grad_in_data[i] = input_data[i] > 0.0f ? grad_out_data[i] : 0.0f;
    }

    return grad_input;
}

} // namespace cyxwiz
