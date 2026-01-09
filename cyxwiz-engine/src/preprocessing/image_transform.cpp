#include "image_transform.h"
#include "../core/image_utils.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

namespace cyxwiz {

ImageTransform::ImageTransform(const ImagePreprocessingConfig& config)
    : config_(config) {
}

Tensor ImageTransform::Apply(const Tensor& input) {
    const auto& shape = input.Shape();
    if (shape.empty()) {
        return input;
    }

    // Get input data
    size_t num_elements = input.NumElements();
    const float* input_data = input.Data<float>();
    if (!input_data) {
        spdlog::error("ImageTransform: Failed to get input tensor data");
        return input;
    }
    std::vector<float> data(input_data, input_data + num_elements);

    // Determine if batch or single image
    bool is_batch = (shape.size() == 4);  // [batch, H, W, C]
    int batch_size = is_batch ? static_cast<int>(shape[0]) : 1;
    int height = is_batch ? static_cast<int>(shape[1]) : static_cast<int>(shape[0]);
    int width = is_batch ? static_cast<int>(shape[2]) : static_cast<int>(shape[1]);
    int channels = is_batch ? static_cast<int>(shape[3]) : (shape.size() >= 3 ? static_cast<int>(shape[2]) : 1);

    // Apply transformations
    std::vector<float> transformed_data = data;
    int out_height = height;
    int out_width = width;
    int out_channels = channels;

    // 1. Format conversion (Grayscale/RGB)
    if (config_.convert_to_grayscale && channels == 3) {
        // RGB -> Grayscale
        std::vector<float> grayscale_data;
        grayscale_data.reserve(batch_size * height * width);

        for (int b = 0; b < batch_size; ++b) {
            size_t offset = b * height * width * channels;
            std::vector<float> image_data(
                transformed_data.begin() + offset,
                transformed_data.begin() + offset + height * width * channels
            );
            auto gray = RGBToGrayscale(image_data, height, width);
            grayscale_data.insert(grayscale_data.end(), gray.begin(), gray.end());
        }

        transformed_data = grayscale_data;
        out_channels = 1;
        spdlog::debug("ImageTransform: Converted RGB to Grayscale");
    } else if (config_.convert_to_rgb && channels == 1) {
        // Grayscale -> RGB
        std::vector<float> rgb_data;
        rgb_data.reserve(batch_size * height * width * 3);

        for (int b = 0; b < batch_size; ++b) {
            size_t offset = b * height * width;
            std::vector<float> image_data(
                transformed_data.begin() + offset,
                transformed_data.begin() + offset + height * width
            );
            auto rgb = GrayscaleToRGB(image_data, height, width);
            rgb_data.insert(rgb_data.end(), rgb.begin(), rgb.end());
        }

        transformed_data = rgb_data;
        out_channels = 3;
        spdlog::debug("ImageTransform: Converted Grayscale to RGB");
    }

    // 2. Resizing
    if (config_.resize_mode != ResizeMode::None &&
        config_.target_width > 0 && config_.target_height > 0) {

        std::vector<float> resized_data;
        int target_w = config_.target_width;
        int target_h = config_.target_height;

        resized_data.reserve(batch_size * target_h * target_w * out_channels);

        for (int b = 0; b < batch_size; ++b) {
            size_t offset = b * out_height * out_width * out_channels;
            std::vector<float> image_data(
                transformed_data.begin() + offset,
                transformed_data.begin() + offset + out_height * out_width * out_channels
            );

            auto resized = ResizeImage(image_data, out_height, out_width, out_channels,
                                      target_h, target_w);
            resized_data.insert(resized_data.end(), resized.begin(), resized.end());
        }

        transformed_data = resized_data;
        out_height = target_h;
        out_width = target_w;
        spdlog::debug("ImageTransform: Resized to {}x{}", target_w, target_h);
    }

    // Create output tensor with new shape
    std::vector<size_t> out_shape;
    if (is_batch) {
        out_shape = {static_cast<size_t>(batch_size),
                     static_cast<size_t>(out_height),
                     static_cast<size_t>(out_width),
                     static_cast<size_t>(out_channels)};
    } else {
        out_shape = {static_cast<size_t>(out_height),
                     static_cast<size_t>(out_width),
                     static_cast<size_t>(out_channels)};
    }

    return Tensor(out_shape, transformed_data.data(), DataType::Float32);
}

std::vector<float> ImageTransform::ResizeImage(
    const std::vector<float>& input,
    int in_height, int in_width, int in_channels,
    int out_height, int out_width
) {
    // Use ImageUtils for high-quality OpenCV-based resize
    std::vector<float> output = input;  // Copy input data

    // Choose resize method based on upscale/downscale
    ImageUtils::ResizeMethod method = (out_width < in_width || out_height < in_height)
        ? ImageUtils::ResizeMethod::Area      // Best for downscaling
        : ImageUtils::ResizeMethod::Lanczos;  // Best for upscaling

    if (!ImageUtils::ResizeImage(output, in_width, in_height, in_channels,
                                  out_width, out_height, method)) {
        spdlog::error("ImageTransform: OpenCV resize failed, returning zeros");
        return std::vector<float>(out_height * out_width * in_channels, 0.0f);
    }

    return output;
}

std::vector<float> ImageTransform::RGBToGrayscale(
    const std::vector<float>& input,
    int height, int width
) {
    std::vector<float> output(height * width);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            int rgb_idx = idx * 3;

            // Luminosity method: 0.299*R + 0.587*G + 0.114*B
            float r = input[rgb_idx + 0];
            float g = input[rgb_idx + 1];
            float b = input[rgb_idx + 2];
            output[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }

    return output;
}

std::vector<float> ImageTransform::GrayscaleToRGB(
    const std::vector<float>& input,
    int height, int width
) {
    std::vector<float> output(height * width * 3);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int gray_idx = y * width + x;
            int rgb_idx = gray_idx * 3;
            float value = input[gray_idx];

            // Replicate gray value to all channels
            output[rgb_idx + 0] = value;
            output[rgb_idx + 1] = value;
            output[rgb_idx + 2] = value;
        }
    }

    return output;
}

// BilinearInterpolate method removed - now using OpenCV via ImageUtils for higher quality resize

} // namespace cyxwiz
