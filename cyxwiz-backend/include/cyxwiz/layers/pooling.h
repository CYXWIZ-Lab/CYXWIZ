#pragma once

#include "cyxwiz/api_export.h"
#include "cyxwiz/layers/layer_base.h"
#include "cyxwiz/tensor.h"

#include <map>
#include <string>

namespace cyxwiz {

// ============================================================================
// MaxPool2D Layer - 2D Max Pooling using ArrayFire
// ============================================================================

class CYXWIZ_API MaxPool2DLayer : public Layer {
public:
    /**
     * Create a 2D max pooling layer
     * @param pool_size Size of the pooling window (assumes square)
     * @param stride Stride of the pooling (default: same as pool_size)
     * @param padding Padding added to input (default: 0)
     */
    MaxPool2DLayer(int pool_size, int stride = -1, int padding = 0);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override { return {}; }
    void SetParameters(const std::map<std::string, Tensor>&) override {}
    std::string GetName() const override { return "MaxPool2D"; }

private:
    int pool_size_;
    int stride_;
    int padding_;

    Tensor max_indices_;  // Store indices for backward pass
};

// ============================================================================
// AvgPool2D Layer - 2D Average Pooling using ArrayFire
// ============================================================================

class CYXWIZ_API AvgPool2DLayer : public Layer {
public:
    /**
     * Create a 2D average pooling layer
     * @param pool_size Size of the pooling window (assumes square)
     * @param stride Stride of the pooling (default: same as pool_size)
     * @param padding Padding added to input (default: 0)
     */
    AvgPool2DLayer(int pool_size, int stride = -1, int padding = 0);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override { return {}; }
    void SetParameters(const std::map<std::string, Tensor>&) override {}
    std::string GetName() const override { return "AvgPool2D"; }

private:
    int pool_size_;
    int stride_;
    int padding_;
};

// ============================================================================
// GlobalAvgPool2D Layer - Global Average Pooling
// ============================================================================

class CYXWIZ_API GlobalAvgPool2DLayer : public Layer {
public:
    GlobalAvgPool2DLayer() = default;

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override { return {}; }
    void SetParameters(const std::map<std::string, Tensor>&) override {}
    std::string GetName() const override { return "GlobalAvgPool2D"; }
};

} // namespace cyxwiz
