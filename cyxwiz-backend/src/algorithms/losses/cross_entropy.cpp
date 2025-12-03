// Prevent Windows.h from defining min/max macros that conflict with ArrayFire
#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/losses/cross_entropy.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <algorithm>

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

Tensor CrossEntropyLoss::Softmax(const Tensor& logits) const {
    auto shape = logits.Shape();
    if (shape.size() != 2) {
        throw std::runtime_error("Softmax expects 2D input [batch, num_classes]");
    }

    size_t batch_size = shape[0];
    size_t num_classes = shape[1];

    Tensor output(shape, logits.GetDataType());

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (UseGPU()) {
        try {
            // Row-major [batch, num_classes] -> AF column-major [num_classes, batch]
            af::array logits_gpu(static_cast<dim_t>(num_classes),
                                 static_cast<dim_t>(batch_size),
                                 static_cast<const float*>(logits.Data()));

            // Softmax: exp(x - max) / sum(exp(x - max))
            // Max along classes (dim 0 in AF)
            af::array max_val = af::tile(af::max(logits_gpu, 0), static_cast<dim_t>(num_classes), 1);
            af::array exp_val = af::exp(logits_gpu - max_val);
            af::array sum_exp = af::tile(af::sum(exp_val, 0), static_cast<dim_t>(num_classes), 1);
            af::array output_gpu = exp_val / sum_exp;

            output_gpu.host(output.Data<float>());
            return output;
        } catch (const af::exception& e) {
            // Fall through to CPU
        }
    }
#endif

    // CPU fallback
    const float* logits_data = static_cast<const float*>(logits.Data());
    float* output_data = static_cast<float*>(output.Data());

    for (size_t b = 0; b < batch_size; b++) {
        const float* logits_row = logits_data + b * num_classes;
        float* output_row = output_data + b * num_classes;

        // Find max for numerical stability
        float max_logit = logits_row[0];
        for (size_t c = 1; c < num_classes; c++) {
            max_logit = std::max(max_logit, logits_row[c]);
        }

        // Compute exp(x - max) and sum
        float sum_exp = 0.0f;
        for (size_t c = 0; c < num_classes; c++) {
            output_row[c] = std::exp(logits_row[c] - max_logit);
            sum_exp += output_row[c];
        }

        // Normalize
        for (size_t c = 0; c < num_classes; c++) {
            output_row[c] /= sum_exp;
        }
    }

    return output;
}

Tensor CrossEntropyLoss::Forward(const Tensor& predictions, const Tensor& targets) {
    auto pred_shape = predictions.Shape();
    auto target_shape = targets.Shape();

    if (pred_shape != target_shape) {
        throw std::runtime_error("CrossEntropyLoss::Forward: predictions and targets must have the same shape");
    }

    if (pred_shape.size() != 2) {
        throw std::runtime_error("CrossEntropyLoss expects 2D input [batch, num_classes]");
    }

    if (predictions.GetDataType() != DataType::Float32) {
        throw std::runtime_error("CrossEntropyLoss only supports Float32 tensors");
    }

    size_t batch_size = pred_shape[0];
    size_t num_classes = pred_shape[1];

    // Compute softmax
    Tensor probs = Softmax(predictions);

    Tensor result({1}, DataType::Float32);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (UseGPU()) {
        try {
            // Row-major [batch, num_classes] -> AF [num_classes, batch]
            af::array probs_gpu(static_cast<dim_t>(num_classes),
                                static_cast<dim_t>(batch_size),
                                static_cast<const float*>(probs.Data()));
            af::array target_gpu(static_cast<dim_t>(num_classes),
                                 static_cast<dim_t>(batch_size),
                                 static_cast<const float*>(targets.Data()));

            const float epsilon = 1e-7f;

            // Cross entropy: -mean(sum(targets * log(probs + eps)))
            af::array log_probs = af::log(probs_gpu + epsilon);
            af::array ce = -target_gpu * log_probs;
            af::array loss = af::mean(af::sum(ce, 0));

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
    const float* probs_data = static_cast<const float*>(probs.Data());
    const float* target_data = static_cast<const float*>(targets.Data());

    float total_loss = 0.0f;
    const float epsilon = 1e-7f;

    for (size_t b = 0; b < batch_size; b++) {
        float sample_loss = 0.0f;
        for (size_t c = 0; c < num_classes; c++) {
            size_t idx = b * num_classes + c;
            if (target_data[idx] > 0.0f) {
                float prob = std::max(probs_data[idx], epsilon);
                sample_loss += target_data[idx] * std::log(prob);
            }
        }
        total_loss -= sample_loss;
    }

    result.Data<float>()[0] = total_loss / static_cast<float>(batch_size);
    return result;
}

Tensor CrossEntropyLoss::Backward(const Tensor& predictions, const Tensor& targets) {
    auto pred_shape = predictions.Shape();
    auto target_shape = targets.Shape();

    if (pred_shape != target_shape) {
        throw std::runtime_error("CrossEntropyLoss::Backward: predictions and targets must have the same shape");
    }

    if (pred_shape.size() != 2) {
        throw std::runtime_error("CrossEntropyLoss expects 2D input [batch, num_classes]");
    }

    size_t batch_size = pred_shape[0];
    size_t num_classes = pred_shape[1];

    // Gradient: (softmax(predictions) - targets) / batch_size
    Tensor probs = Softmax(predictions);
    Tensor grad_input(pred_shape, predictions.GetDataType());

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (UseGPU()) {
        try {
            af::array probs_gpu(static_cast<dim_t>(num_classes),
                                static_cast<dim_t>(batch_size),
                                static_cast<const float*>(probs.Data()));
            af::array target_gpu(static_cast<dim_t>(num_classes),
                                 static_cast<dim_t>(batch_size),
                                 static_cast<const float*>(targets.Data()));

            float scale = 1.0f / static_cast<float>(batch_size);
            af::array grad_gpu = scale * (probs_gpu - target_gpu);

            grad_gpu.host(grad_input.Data<float>());
            return grad_input;
        } catch (const af::exception& e) {
            // Fall through to CPU
        }
    }
#endif

    // CPU fallback
    const float* probs_data = static_cast<const float*>(probs.Data());
    const float* target_data = static_cast<const float*>(targets.Data());
    float* grad_data = static_cast<float*>(grad_input.Data());

    float scale = 1.0f / static_cast<float>(batch_size);

    for (size_t i = 0; i < batch_size * num_classes; i++) {
        grad_data[i] = scale * (probs_data[i] - target_data[i]);
    }

    return grad_input;
}

} // namespace cyxwiz
