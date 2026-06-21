#pragma once

#include "cyxwiz/api_export.h"
#include "cyxwiz/layers/layer_base.h"
#include "cyxwiz/tensor.h"

#include <map>
#include <string>

namespace cyxwiz {

// ============================================================================
// Dropout Layer - Regularization
// ============================================================================

class CYXWIZ_API DropoutLayer : public Layer {
public:
    /**
     * Create a dropout layer
     * @param p Probability of dropping (default: 0.5)
     */
    explicit DropoutLayer(float p = 0.5f);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override { return {}; }
    void SetParameters(const std::map<std::string, Tensor>&) override {}
    std::string GetName() const override { return "Dropout"; }

private:
    float p_;
    Tensor mask_;  // Dropout mask for backward pass
};

} // namespace cyxwiz
