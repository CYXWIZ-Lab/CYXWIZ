#pragma once

#include "cyxwiz/api_export.h"
#include "cyxwiz/layers/layer_base.h"
#include "cyxwiz/tensor.h"

#include <map>
#include <string>
#include <vector>

namespace cyxwiz {

// ============================================================================
// Flatten Layer - Flatten spatial dimensions
// ============================================================================

class CYXWIZ_API FlattenLayer : public Layer {
public:
    FlattenLayer() = default;

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override { return {}; }
    void SetParameters(const std::map<std::string, Tensor>&) override {}
    std::string GetName() const override { return "Flatten"; }

private:
    std::vector<size_t> input_shape_;  // Original shape for backward
};

} // namespace cyxwiz
