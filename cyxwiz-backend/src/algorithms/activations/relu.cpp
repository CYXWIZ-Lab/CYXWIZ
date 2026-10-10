// Prevent Windows.h from defining min/max macros that conflict with ArrayFire
#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/activations/relu.h"
#include <stdexcept>
#include <string>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {
Tensor ReLU::Forward(const Tensor& input) {
    if (input.GetDataType() != DataType::Float32) {
        throw std::runtime_error("ReLU only supports Float32 tensors");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array input_gpu = input.GetSemanticArray();

        // ReLU: max(0, x)
        af::array output_gpu = af::max(input_gpu, 0.0f);
        output_gpu.eval();

        return Tensor::FromSemanticArray(output_gpu, input.Shape());
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("ReLU::Forward failed on the ArrayFire device: ") + e.what());
    }
#else
    throw std::runtime_error("ReLU runs on ArrayFire, and this build has no ArrayFire");
#endif
}

Tensor ReLU::Backward(const Tensor& grad_output, const Tensor& input) {
    if (grad_output.Shape() != input.Shape()) {
        throw std::runtime_error("ReLU::Backward: gradient and input shapes must match");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_gpu = grad_output.GetSemanticArray();
        af::array input_gpu = input.GetSemanticArray();

        // Gradient: grad * (input > 0)
        af::array mask = input_gpu > 0.0f;
        af::array grad_input_gpu = grad_gpu * mask.as(f32);
        grad_input_gpu.eval();

        return Tensor::FromSemanticArray(
            grad_input_gpu, grad_output.Shape());
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("ReLU::Backward failed on the ArrayFire device: ") + e.what());
    }
#else
    throw std::runtime_error("ReLU runs on ArrayFire, and this build has no ArrayFire");
#endif
}

} // namespace cyxwiz
