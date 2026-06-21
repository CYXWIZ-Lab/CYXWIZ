#pragma once

#include "cyxwiz/api_export.h"
#include "cyxwiz/layers/layer_base.h"
#include "cyxwiz/tensor.h"

#include <map>
#include <string>
#include <vector>

namespace cyxwiz {

class CYXWIZ_API BatchNorm2DLayer : public Layer {
public:
    BatchNorm2DLayer(int num_features, float eps = 1e-5f, float momentum = 0.1f);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "BatchNorm2D"; }

private:
    int num_features_;
    float eps_;
    float momentum_;

    Tensor gamma_;
    Tensor beta_;
    Tensor running_mean_;
    Tensor running_var_;
    Tensor normalized_;
    Tensor std_inv_;
    Tensor grad_gamma_;
    Tensor grad_beta_;
};

class CYXWIZ_API LayerNormLayer : public Layer {
public:
    LayerNormLayer(const std::vector<int>& normalized_shape,
                   float eps = 1e-5f, bool elementwise_affine = true);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "LayerNorm"; }

private:
    std::vector<int> normalized_shape_;
    float eps_;
    bool elementwise_affine_;

    Tensor gamma_;
    Tensor beta_;
    Tensor grad_gamma_;
    Tensor grad_beta_;
    Tensor normalized_;
    Tensor std_inv_;
};

class CYXWIZ_API InstanceNorm2DLayer : public Layer {
public:
    InstanceNorm2DLayer(int num_features, float eps = 1e-5f, bool affine = false);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "InstanceNorm2D"; }

private:
    int num_features_;
    float eps_;
    bool affine_;

    Tensor gamma_;
    Tensor beta_;
    Tensor grad_gamma_;
    Tensor grad_beta_;
    Tensor normalized_;
    Tensor std_inv_;
};

class CYXWIZ_API GroupNormLayer : public Layer {
public:
    GroupNormLayer(int num_groups, int num_channels,
                   float eps = 1e-5f, bool affine = true);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "GroupNorm"; }

private:
    int num_groups_;
    int num_channels_;
    float eps_;
    bool affine_;

    Tensor gamma_;
    Tensor beta_;
    Tensor grad_gamma_;
    Tensor grad_beta_;
    Tensor normalized_;
    Tensor std_inv_;
};

} // namespace cyxwiz
