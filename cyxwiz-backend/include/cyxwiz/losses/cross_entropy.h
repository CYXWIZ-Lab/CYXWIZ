#pragma once

#include "../loss.h"
#include "../tensor.h"
#include "../api_export.h"

namespace cyxwiz {

/**
 * @brief Cross Entropy loss (with softmax)
 *
 * Forward: L = -mean(sum(targets * log(softmax(predictions))))
 * Backward: dL/dy = (softmax(predictions) - targets) / N
 *
 * Used for multi-class classification
 * Assumes predictions are logits (pre-softmax)
 * Assumes targets are one-hot encoded or class probabilities
 */
class CYXWIZ_API CrossEntropyLoss : public Loss {
public:
    CrossEntropyLoss() = default;
    ~CrossEntropyLoss() override = default;

    /**
     * @brief Forward pass: compute cross entropy loss
     * @param predictions Logits (pre-softmax), shape: [batch, num_classes]
     * @param targets One-hot targets, shape: [batch, num_classes]
     * @return Scalar loss value (averaged over batch)
     */
    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;

    /**
     * @brief Backward pass: compute gradient w.r.t predictions
     * @param predictions Logits (pre-softmax)
     * @param targets One-hot targets
     * @return Gradient tensor (same shape as predictions)
     */
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;

private:
    /**
     * @brief Compute softmax: exp(x_i) / sum(exp(x_j))
     * @param logits Input logits, shape: [batch, num_classes]
     * @return Softmax probabilities, same shape
     */
    Tensor Softmax(const Tensor& logits) const;
};

} // namespace cyxwiz
