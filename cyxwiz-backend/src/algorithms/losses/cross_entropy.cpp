#include "cyxwiz/losses/cross_entropy.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace cyxwiz {

Tensor CrossEntropyLoss::Softmax(const Tensor& logits) const {
    auto shape = logits.Shape();
    if (shape.size() != 2) {
        throw std::runtime_error("Softmax expects 2D input [batch, num_classes]");
    }

    size_t batch_size = shape[0];
    size_t num_classes = shape[1];

    Tensor output(shape, logits.GetDataType());
    const float* logits_data = static_cast<const float*>(logits.Data());
    float* output_data = static_cast<float*>(output.Data());

    // Compute softmax for each sample in batch
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
    const float* probs_data = static_cast<const float*>(probs.Data());
    const float* target_data = static_cast<const float*>(targets.Data());

    // Compute cross entropy: -mean(sum(targets * log(probs)))
    float total_loss = 0.0f;
    const float epsilon = 1e-7f; // Small constant to avoid log(0)

    for (size_t b = 0; b < batch_size; b++) {
        float sample_loss = 0.0f;
        for (size_t c = 0; c < num_classes; c++) {
            size_t idx = b * num_classes + c;
            // Only accumulate loss where target is non-zero (one-hot encoding)
            if (target_data[idx] > 0.0f) {
                float prob = std::max(probs_data[idx], epsilon);
                sample_loss += target_data[idx] * std::log(prob);
            }
        }
        total_loss -= sample_loss; // Negative log likelihood
    }

    float loss = total_loss / static_cast<float>(batch_size);

    // Return scalar tensor
    Tensor result({1}, DataType::Float32);
    float* result_data = static_cast<float*>(result.Data());
    result_data[0] = loss;

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
