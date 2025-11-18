#pragma once

#include "../activation.h"
#include "../tensor.h"
#include "../api_export.h"

namespace cyxwiz {

/**
 * @brief Tanh (Hyperbolic Tangent) activation
 *
 * Forward: f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 * Backward: f'(x) = 1 - f(x)^2
 */
class CYXWIZ_API Tanh : public Activation {
public:
    Tanh() = default;
    ~Tanh() override = default;

    /**
     * @brief Forward pass: f(x) = tanh(x)
     * @param input Input tensor
     * @return Output tensor with Tanh applied element-wise
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
