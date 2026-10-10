#pragma once

#include "cyxwiz/api_export.h"
#include "cyxwiz/layers/layer_base.h"
#include "cyxwiz/tensor.h"

#include <map>
#include <memory>
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

    // The sparse window gather for one input length, built on first use.
    struct DeviceGather;
    std::shared_ptr<DeviceGather> gather_;

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
    // Device-resident provider path (TOFIX140 A1b): tried before ArrayFire;
    // a failed provider is not retried for this layer.
    bool provider_disabled_ = false;
    bool provider_logged_ = false;
    bool TryProviderForward(const Tensor& input, const std::vector<size_t>& output_shape, Tensor& output);
    bool TryProviderBackward(const Tensor& grad_output, Tensor& grad_input);
};

// ============================================================================
// DepthwiseConv2D Layer - one set of kernels per input channel
// ============================================================================

// [H,W,C,N] -> [out_h,out_w,C*M,N], torch Conv2d(C, C*M, k, groups=C): output
// channel c*M + m convolves input channel c alone. Weights [k,k,1,C*M] (the
// Conv2D layout with one input channel per group), bias [C*M].
class CYXWIZ_API DepthwiseConv2DLayer : public Layer {
public:
    DepthwiseConv2DLayer(int channels, int depth_multiplier, int kernel_size,
                         int stride = 1, int padding = 0, bool use_bias = true);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "DepthwiseConv2D"; }

    int GetChannels() const { return channels_; }
    int GetDepthMultiplier() const { return multiplier_; }

private:
    int channels_;
    int multiplier_;
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
    bool has_forward_ = false;
};

} // namespace cyxwiz
