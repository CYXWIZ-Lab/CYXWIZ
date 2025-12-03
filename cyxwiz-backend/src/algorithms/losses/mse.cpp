#include "cyxwiz/losses/mse.h"
#include <cmath>
#include <cstring>
#include <stdexcept>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

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

Tensor MSELoss::Forward(const Tensor& predictions, const Tensor& targets) {
    if (predictions.Shape() != targets.Shape()) {
        throw std::runtime_error("MSELoss::Forward: predictions and targets must have the same shape");
    }

    if (predictions.GetDataType() != DataType::Float32) {
        throw std::runtime_error("MSELoss only supports Float32 tensors");
    }

    size_t num_elements = predictions.NumElements();

    Tensor result({1}, DataType::Float32);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (UseGPU()) {
        try {
            af::array pred_gpu(static_cast<dim_t>(num_elements),
                               static_cast<const float*>(predictions.Data()));
            af::array target_gpu(static_cast<dim_t>(num_elements),
                                 static_cast<const float*>(targets.Data()));

            // MSE: mean((pred - target)^2)
            af::array diff = pred_gpu - target_gpu;
            af::array loss = af::mean(diff * diff);

            float loss_val;
            loss.host(&loss_val);
            result.Data<float>()[0] = loss_val;
            return result;
        } catch (const af::exception& e) {
            // Fall through to CPU
        }
    }
#endif

    // CPU fallback
    const float* pred_data = static_cast<const float*>(predictions.Data());
    const float* target_data = static_cast<const float*>(targets.Data());

    float sum = 0.0f;
    for (size_t i = 0; i < num_elements; i++) {
        float diff = pred_data[i] - target_data[i];
        sum += diff * diff;
    }

    result.Data<float>()[0] = sum / static_cast<float>(num_elements);
    return result;
}

Tensor MSELoss::Backward(const Tensor& predictions, const Tensor& targets) {
    if (predictions.Shape() != targets.Shape()) {
        throw std::runtime_error("MSELoss::Backward: predictions and targets must have the same shape");
    }

    Tensor grad_input(predictions.Shape(), predictions.GetDataType());

    size_t num_elements = predictions.NumElements();

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (UseGPU()) {
        try {
            af::array pred_gpu(static_cast<dim_t>(num_elements),
                               static_cast<const float*>(predictions.Data()));
            af::array target_gpu(static_cast<dim_t>(num_elements),
                                 static_cast<const float*>(targets.Data()));

            // Gradient: 2 * (pred - target) / N
            float scale = 2.0f / static_cast<float>(num_elements);
            af::array grad_gpu = scale * (pred_gpu - target_gpu);

            grad_gpu.host(grad_input.Data<float>());
            return grad_input;
        } catch (const af::exception& e) {
            // Fall through to CPU
        }
    }
#endif

    // CPU fallback
    const float* pred_data = static_cast<const float*>(predictions.Data());
    const float* target_data = static_cast<const float*>(targets.Data());
    float* grad_data = static_cast<float*>(grad_input.Data());

    float scale = 2.0f / static_cast<float>(num_elements);
    for (size_t i = 0; i < num_elements; i++) {
        grad_data[i] = scale * (pred_data[i] - target_data[i]);
    }

    return grad_input;
}

} // namespace cyxwiz
