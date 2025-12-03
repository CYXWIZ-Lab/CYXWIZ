// Prevent Windows.h from defining min/max macros that conflict with ArrayFire
#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/activations/relu.h"
#include <cstring>
#include <algorithm>
#include <stdexcept>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

// GPU availability flag (shared across the library)
extern bool CheckGPUAvailable();

static bool s_use_gpu = false;
static bool s_gpu_checked = false;

static bool UseGPU() {
    if (!s_gpu_checked) {
        s_gpu_checked = true;
#ifdef CYXWIZ_HAS_ARRAYFIRE
        try {
            af::Backend backend = af::getActiveBackend();
            s_use_gpu = (backend == AF_BACKEND_CUDA || backend == AF_BACKEND_OPENCL);
        } catch (...) {
            s_use_gpu = false;
        }
#endif
    }
    return s_use_gpu;
}

Tensor ReLU::Forward(const Tensor& input) {
    Tensor output(input.Shape(), input.GetDataType());

    size_t num_elements = input.NumElements();

    if (input.GetDataType() != DataType::Float32) {
        throw std::runtime_error("ReLU only supports Float32 tensors");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (UseGPU()) {
        try {
            af::array input_gpu(static_cast<dim_t>(num_elements),
                                static_cast<const float*>(input.Data()));

            // ReLU: max(0, x)
            af::array output_gpu = af::max(input_gpu, 0.0f);

            output_gpu.host(output.Data<float>());
            return output;
        } catch (const af::exception& e) {
            // Fall through to CPU
        }
    }
#endif

    // CPU fallback
    const float* input_data = static_cast<const float*>(input.Data());
    float* output_data = static_cast<float*>(output.Data());

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

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (UseGPU()) {
        try {
            af::array grad_gpu(static_cast<dim_t>(num_elements),
                               static_cast<const float*>(grad_output.Data()));
            af::array input_gpu(static_cast<dim_t>(num_elements),
                                static_cast<const float*>(input.Data()));

            // Gradient: grad * (input > 0)
            af::array mask = input_gpu > 0.0f;
            af::array grad_input_gpu = grad_gpu * mask.as(f32);

            grad_input_gpu.host(grad_input.Data<float>());
            return grad_input;
        } catch (const af::exception& e) {
            // Fall through to CPU
        }
    }
#endif

    // CPU fallback
    const float* grad_out_data = static_cast<const float*>(grad_output.Data());
    const float* input_data = static_cast<const float*>(input.Data());
    float* grad_in_data = static_cast<float*>(grad_input.Data());

    for (size_t i = 0; i < num_elements; i++) {
        grad_in_data[i] = input_data[i] > 0.0f ? grad_out_data[i] : 0.0f;
    }

    return grad_input;
}

} // namespace cyxwiz
