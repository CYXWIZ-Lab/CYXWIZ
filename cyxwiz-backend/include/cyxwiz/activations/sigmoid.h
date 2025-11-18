#pragma once

#include "../activation.h"
#include "../tensor.h"
#include "../api_export.h"

namespace cyxwiz {

/**
 * @brief Sigmoid activation
 *
 * Forward: f(x) = 1 / (1 + exp(-x))
 * Backward: f'(x) = f(x) * (1 - f(x))
 */
class CYXWIZ_API Sigmoid : public Activation {
public:
    Sigmoid() = default;
    ~Sigmoid() override = default;

    /**
     * @brief Forward pass: f(x) = 1 / (1 + exp(-x))
     * @param input Input tensor
     * @return Output tensor with Sigmoid applied element-wise
     */
    Tensor Forward(const Tensor& input) override;

    /**
     * @brief Backward pass: gradient computation
     * @param grad_output Gradient from next layer
     * @param input Original input from forward pass
     * @return Gradient w.r.t input
     */
    Tensor Backward(const Tensor& grad_output, const Tensor& input) override;
};

} // namespace cyxwiz
