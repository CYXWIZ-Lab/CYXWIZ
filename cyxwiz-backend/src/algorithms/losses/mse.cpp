#include "cyxwiz/losses/mse.h"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace cyxwiz {

Tensor MSELoss::Forward(const Tensor& predictions, const Tensor& targets) {
    if (predictions.Shape() != targets.Shape()) {
        throw std::runtime_error("MSELoss::Forward: predictions and targets must have the same shape");
    }

    if (predictions.GetDataType() != DataType::Float32) {
        throw std::runtime_error("MSELoss only supports Float32 tensors");
    }

    size_t num_elements = predictions.NumElements();
    const float* pred_data = static_cast<const float*>(predictions.Data());
    const float* target_data = static_cast<const float*>(targets.Data());

    // Compute MSE: mean((predictions - targets)^2)
    float sum = 0.0f;
    for (size_t i = 0; i < num_elements; i++) {
        float diff = pred_data[i] - target_data[i];
        sum += diff * diff;
    }

    float loss = sum / static_cast<float>(num_elements);

    // Return scalar tensor
    Tensor result({1}, DataType::Float32);
    float* result_data = static_cast<float*>(result.Data());
    result_data[0] = loss;

    return result;
}

Tensor MSELoss::Backward(const Tensor& predictions, const Tensor& targets) {
    if (predictions.Shape() != targets.Shape()) {
        throw std::runtime_error("MSELoss::Backward: predictions and targets must have the same shape");
    }

    Tensor grad_input(predictions.Shape(), predictions.GetDataType());

    size_t num_elements = predictions.NumElements();
    const float* pred_data = static_cast<const float*>(predictions.Data());
    const float* target_data = static_cast<const float*>(targets.Data());
    float* grad_data = static_cast<float*>(grad_input.Data());

    // Gradient: dL/dy = 2 * (predictions - targets) / N
    float scale = 2.0f / static_cast<float>(num_elements);
    for (size_t i = 0; i < num_elements; i++) {
        grad_data[i] = scale * (pred_data[i] - target_data[i]);
    }

    return grad_input;
}

} // namespace cyxwiz
