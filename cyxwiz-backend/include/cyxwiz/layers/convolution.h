#pragma once

#include "cyxwiz/api_export.h"
#include "cyxwiz/layers/layer_base.h"
#include "cyxwiz/tensor.h"

#include <map>
#include <string>

namespace cyxwiz {

// ============================================================================
// Conv1D Layer - 1D Convolution using ArrayFire
// ============================================================================

class CYXWIZ_API Conv1DLayer : public Layer {
public:
    Conv1DLayer(int in_channels, int out_channels, int kernel_size,
                int stride = 1, int padding = 0, int dilation = 1,
                bool use_bias = true);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "Conv1D"; }

    int GetInChannels() const { return in_channels_; }
    int GetOutChannels() const { return out_channels_; }
    int GetKernelSize() const { return kernel_size_; }
    int GetStride() const { return stride_; }
    int GetPadding() const { return padding_; }
    int GetDilation() const { return dilation_; }

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    int dilation_;
    bool use_bias_;

    Tensor weights_;
    Tensor bias_;
    Tensor grad_weights_;
    Tensor grad_bias_;
    bool has_forward_ = false;
};

// ============================================================================
// Conv2D Layer - 2D Convolution using ArrayFire
// ============================================================================

class CYXWIZ_API Conv2DLayer : public Layer {
public:
    Conv2DLayer(int in_channels, int out_channels, int kernel_size,
                int stride = 1, int padding = 0, bool use_bias = true);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "Conv2D"; }

    int GetInChannels() const { return in_channels_; }
    int GetOutChannels() const { return out_channels_; }
    int GetKernelSize() const { return kernel_size_; }
    int GetStride() const { return stride_; }
    int GetPadding() const { return padding_; }

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    bool use_bias_;

    Tensor weights_;
    Tensor bias_;
    Tensor grad_weights_;
    Tensor grad_bias_;
    bool has_forward_ = false;
};

// ============================================================================
// ConvTranspose2D Layer - 2D Transposed Convolution
// ============================================================================

class CYXWIZ_API ConvTranspose2DLayer : public Layer {
public:
    ConvTranspose2DLayer(int in_channels, int out_channels, int kernel_size,
                         int stride = 1, int padding = 0, int output_padding = 0,
                         bool use_bias = true);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "ConvTranspose2D"; }

    int GetInChannels() const { return in_channels_; }
    int GetOutChannels() const { return out_channels_; }
    int GetKernelSize() const { return kernel_size_; }
    int GetStride() const { return stride_; }
    int GetPadding() const { return padding_; }
    int GetOutputPadding() const { return output_padding_; }

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    int output_padding_;
    bool use_bias_;

    Tensor weights_;
    Tensor bias_;
    Tensor grad_weights_;
    Tensor grad_bias_;
};

} // namespace cyxwiz
