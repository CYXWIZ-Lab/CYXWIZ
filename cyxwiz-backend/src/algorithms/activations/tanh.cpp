#include "cyxwiz/activations/tanh.h"
#include <stdexcept>
#include <string>

#include <arrayfire.h>

namespace cyxwiz {
Tensor Tanh::Forward(const Tensor& input) {
    if (input.GetDataType() != DataType::Float32) {
        throw std::runtime_error("Tanh only supports Float32 tensors");
    }

    try {
        af::array input_gpu = input.GetSemanticArray();

        // Tanh
        af::array output_gpu = af::tanh(input_gpu);
        output_gpu.eval();

        return Tensor::FromSemanticArray(output_gpu, input.Shape());
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("Tanh::Forward failed on the ArrayFire device: ") + e.what());
    }
}

Tensor Tanh::Backward(const Tensor& grad_output, const Tensor& input) {
    if (grad_output.Shape() != input.Shape()) {
        throw std::runtime_error("Tanh::Backward: gradient and input shapes must match");
    }

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
        throw std::runtime_error(std::string("Tanh::Backward failed on the ArrayFire device: ") + e.what());
    }
}

} // namespace cyxwiz
