#pragma once

#include "../activation.h"
#include "../tensor.h"
#include "../api_export.h"

namespace cyxwiz {

/**
 * @brief ReLU (Rectified Linear Unit) activation
 *
 * Forward: f(x) = max(0, x)
 * Backward: f'(x) = 1 if x > 0, else 0
 */
class CYXWIZ_API ReLU : public Activation {
public:
    ReLU() = default;
    ~ReLU() override = default;

    /**
     * @brief Forward pass: f(x) = max(0, x)
     * @param input Input tensor
     * @return Output tensor with ReLU applied element-wise
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
