#pragma once

#include "../loss.h"
#include "../tensor.h"
#include "../api_export.h"

namespace cyxwiz {

/**
 * @brief Mean Squared Error (MSE) loss
 *
 * Forward: L = mean((predictions - targets)^2)
 * Backward: dL/dy = 2 * (predictions - targets) / N
 *
 * Used for regression tasks
 */
class CYXWIZ_API MSELoss : public Loss {
public:
    MSELoss() = default;
    ~MSELoss() override = default;

    /**
     * @brief Forward pass: compute MSE loss
     * @param predictions Model predictions
     * @param targets Ground truth targets
     * @return Scalar loss value (averaged over all elements)
     */
    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;

    /**
     * @brief Backward pass: compute gradient w.r.t predictions
     * @param predictions Model predictions
     * @param targets Ground truth targets
     * @return Gradient tensor (same shape as predictions)
     */
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
};

} // namespace cyxwiz
