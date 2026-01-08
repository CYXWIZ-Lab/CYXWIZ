#pragma once

#include "preprocessing_config.h"
#include <cyxwiz/tensor.h>
#include <vector>

namespace cyxwiz {

/**
 * Image preprocessing transform
 * Handles resizing, format conversion, etc.
 */
class ImageTransform {
public:
    /**
     * Create image transform with config
     * @param config Image preprocessing configuration
     */
    explicit ImageTransform(const ImagePreprocessingConfig& config);

    /**
     * Apply image preprocessing to input tensor
     * @param input Input tensor (assumes [batch, H, W, C] or [H, W, C] format)
     * @return Transformed tensor
     */
    Tensor Apply(const Tensor& input);

private:
    ImagePreprocessingConfig config_;

    /**
     * Resize single image
     * @param input Input image data [H, W, C]
     * @param in_height Input height
     * @param in_width Input width
     * @param in_channels Input channels
     * @param out_height Output height
     * @param out_width Output width
     * @return Resized image data
     */
    std::vector<float> ResizeImage(
        const std::vector<float>& input,
        int in_height, int in_width, int in_channels,
        int out_height, int out_width
    );

    /**
     * Convert RGB to grayscale
     * @param input Input image data [H, W, 3]
     * @param height Image height
     * @param width Image width
     * @return Grayscale image data [H, W, 1]
     */
    std::vector<float> RGBToGrayscale(
        const std::vector<float>& input,
        int height, int width
    );

    /**
     * Convert grayscale to RGB (replicate channel)
     * @param input Input image data [H, W, 1]
     * @param height Image height
     * @param width Image width
     * @return RGB image data [H, W, 3]
     */
    std::vector<float> GrayscaleToRGB(
        const std::vector<float>& input,
        int height, int width
    );

    /**
     * Bilinear interpolation helper
     */
    float BilinearInterpolate(
        const std::vector<float>& input,
        float x, float y, int channel,
        int width, int height, int channels
    );
};

} // namespace cyxwiz
