#pragma once

#include "cyxwiz/api_export.h"
#include "cyxwiz/layers/layer_base.h"
#include "cyxwiz/tensor.h"

#include <map>
#include <string>

namespace cyxwiz {

// ============================================================================
// Upsample2D Layer - Spatial Upsampling (Nearest / Bilinear)
// ============================================================================

enum class UpsampleMode { Nearest, Bilinear };

class CYXWIZ_API Upsample2DLayer : public Layer {
public:
    /**
     * Create a 2D upsampling layer (no learnable parameters)
     * @param scale_factor Upsampling factor (default: 2)
     * @param mode Interpolation mode (default: Nearest)
     */
    explicit Upsample2DLayer(int scale_factor = 2, UpsampleMode mode = UpsampleMode::Nearest);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override { return {}; }
    void SetParameters(const std::map<std::string, Tensor>&) override {}
    std::string GetName() const override { return "Upsample2D"; }

    int GetScaleFactor() const { return scale_factor_; }
    UpsampleMode GetMode() const { return mode_; }

private:
    int scale_factor_;
    UpsampleMode mode_;
};

// ============================================================================
// PixelShuffle Layer - Sub-Pixel Convolution (Depth to Space)
// ============================================================================

class CYXWIZ_API PixelShuffleLayer : public Layer {
public:
    /**
     * Rearranges elements: (H, W, C*r^2) -> (H*r, W*r, C)
     * @param upscale_factor Upscaling factor r
     */
    explicit PixelShuffleLayer(int upscale_factor);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override { return {}; }
    void SetParameters(const std::map<std::string, Tensor>&) override {}
    std::string GetName() const override { return "PixelShuffle"; }

    int GetUpscaleFactor() const { return upscale_factor_; }

private:
    int upscale_factor_;
    int cached_in_channels_ = 0;
};

} // namespace cyxwiz
