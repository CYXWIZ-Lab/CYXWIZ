#pragma once

#include "api_export.h"
#include "tensor.h"
#include <string>
#include <map>
#include <memory>
#include <vector>

namespace cyxwiz {

// ============================================================================
// Base Layer Class
// ============================================================================

class CYXWIZ_API Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor Forward(const Tensor& input) = 0;
    virtual Tensor Backward(const Tensor& grad_output) = 0;
    virtual std::map<std::string, Tensor> GetParameters() = 0;
    virtual void SetParameters(const std::map<std::string, Tensor>& params) = 0;

    // Training mode (affects BatchNorm, Dropout, etc.)
    virtual void SetTraining(bool training) { training_ = training; }
    bool IsTraining() const { return training_; }

    // Layer name for debugging/serialization
    virtual std::string GetName() const { return "Layer"; }

protected:
    bool training_ = true;
    Tensor cached_input_;  // For backward pass
};

// ============================================================================
// Dense (Fully Connected) Layer
// ============================================================================

class CYXWIZ_API DenseLayer : public Layer {
public:
    DenseLayer(int in_features, int out_features, bool use_bias = true);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "Dense"; }

private:
    int in_features_;
    int out_features_;
    bool use_bias_;

    Tensor weights_;      // [out_features, in_features]
    Tensor bias_;         // [out_features]
    Tensor grad_weights_; // Gradient accumulator
    Tensor grad_bias_;    // Gradient accumulator
};

// ============================================================================
// Conv2D Layer - 2D Convolution using ArrayFire
// ============================================================================

class CYXWIZ_API Conv2DLayer : public Layer {
public:
    /**
     * Create a 2D convolutional layer
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels (filters)
     * @param kernel_size Size of the convolution kernel (assumes square)
     * @param stride Stride of the convolution (default: 1)
     * @param padding Padding added to input (default: 0)
     * @param use_bias Whether to include bias (default: true)
     */
    Conv2DLayer(int in_channels, int out_channels, int kernel_size,
                int stride = 1, int padding = 0, bool use_bias = true);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "Conv2D"; }

    // Accessors
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

    Tensor weights_;      // [out_channels, in_channels, kernel_size, kernel_size]
    Tensor bias_;         // [out_channels]
    Tensor grad_weights_;
    Tensor grad_bias_;

    // Helper for im2col/col2im operations
    Tensor Im2Col(const Tensor& input, int kernel_h, int kernel_w,
                  int stride_h, int stride_w, int pad_h, int pad_w);
    Tensor Col2Im(const Tensor& col, int height, int width, int channels,
                  int kernel_h, int kernel_w, int stride_h, int stride_w,
                  int pad_h, int pad_w);
};

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

// ============================================================================
// BatchNorm2D Layer - Batch Normalization using ArrayFire
// ============================================================================

class CYXWIZ_API BatchNorm2DLayer : public Layer {
public:
    /**
     * Create a 2D batch normalization layer
     * @param num_features Number of features/channels
     * @param eps Small value for numerical stability (default: 1e-5)
     * @param momentum Momentum for running statistics (default: 0.1)
     */
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

    // Learnable parameters
    Tensor gamma_;    // Scale [num_features]
    Tensor beta_;     // Shift [num_features]

    // Running statistics (for inference)
    Tensor running_mean_;
    Tensor running_var_;

    // Cached for backward pass
    Tensor normalized_;
    Tensor std_inv_;

    // Gradients
    Tensor grad_gamma_;
    Tensor grad_beta_;
};

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
