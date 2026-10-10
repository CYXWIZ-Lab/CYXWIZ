#include "cyxwiz/activations/sigmoid.h"
#include <stdexcept>
#include <string>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {
Tensor Sigmoid::Forward(const Tensor& input) {
    if (input.GetDataType() != DataType::Float32) {
        throw std::runtime_error("Sigmoid only supports Float32 tensors");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array input_gpu = input.GetSemanticArray();

        // Sigmoid: 1 / (1 + exp(-x))
        af::array output_gpu = af::sigmoid(input_gpu);
        output_gpu.eval();

        return Tensor::FromSemanticArray(output_gpu, input.Shape());
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("Sigmoid::Forward failed on the ArrayFire device: ") + e.what());
    }
#else
    throw std::runtime_error("Sigmoid runs on ArrayFire, and this build has no ArrayFire");
#endif
}

Tensor Sigmoid::Backward(const Tensor& grad_output, const Tensor& input) {
    if (grad_output.Shape() != input.Shape()) {
        throw std::runtime_error("Sigmoid::Backward: gradient and input shapes must match");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_gpu = grad_output.GetSemanticArray();
        af::array input_gpu = input.GetSemanticArray();

        // Gradient: grad * sigmoid(x) * (1 - sigmoid(x))
        af::array sigmoid_val = af::sigmoid(input_gpu);
        af::array grad_input_gpu = grad_gpu * sigmoid_val * (1.0f - sigmoid_val);
        grad_input_gpu.eval();

        return Tensor::FromSemanticArray(
            grad_input_gpu, grad_output.Shape());
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("Sigmoid::Backward failed on the ArrayFire device: ") + e.what());
    }
#else
    throw std::runtime_error("Sigmoid runs on ArrayFire, and this build has no ArrayFire");
#endif
}

} // namespace cyxwiz
