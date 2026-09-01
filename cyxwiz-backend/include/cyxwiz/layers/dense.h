#pragma once

#include "layer_base.h"

#include <map>
#include <string>

namespace cyxwiz {

// ============================================================================
// Dense (Fully Connected) Layer
// ============================================================================

class CYXWIZ_API DenseLayer : public Layer {
public:
    DenseLayer(int in_features, int out_features, bool use_bias = true);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    // Legacy Layer serialization includes grad_* entries in this map.
    // New optimizer-facing code should use GetGradients() for an exact
    // parameter-name -> gradient-name mapping.
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::map<std::string, Tensor> GetGradients() const;
    std::string GetName() const override { return "Dense"; }

private:
    int in_features_;
    int out_features_;
    bool use_bias_;

    Tensor weights_;      // [out_features, in_features]
    Tensor bias_;         // [out_features]
    Tensor grad_weights_; // Gradient accumulator
    Tensor grad_bias_;    // Gradient accumulator
    bool has_cached_input_ = false;
};

} // namespace cyxwiz
