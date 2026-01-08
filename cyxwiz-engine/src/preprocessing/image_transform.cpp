#include "image_transform.h"
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
    std::vector<float> output(out_height * out_width * in_channels);

    float scale_x = static_cast<float>(in_width) / out_width;
    float scale_y = static_cast<float>(in_height) / out_height;

    for (int y = 0; y < out_height; ++y) {
        for (int x = 0; x < out_width; ++x) {
            float src_x = (x + 0.5f) * scale_x - 0.5f;
            float src_y = (y + 0.5f) * scale_y - 0.5f;

            for (int c = 0; c < in_channels; ++c) {
                float value = BilinearInterpolate(input, src_x, src_y, c,
                                                  in_width, in_height, in_channels);
                output[(y * out_width + x) * in_channels + c] = value;
            }
        }
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

float ImageTransform::BilinearInterpolate(
    const std::vector<float>& input,
    float x, float y, int channel,
    int width, int height, int channels
) {
    // Clamp coordinates
    x = std::clamp(x, 0.0f, width - 1.0f);
    y = std::clamp(y, 0.0f, height - 1.0f);

    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = std::min(x0 + 1, width - 1);
    int y1 = std::min(y0 + 1, height - 1);

    float dx = x - x0;
    float dy = y - y0;

    auto get_pixel = [&](int px, int py, int c) -> float {
        int idx = (py * width + px) * channels + c;
        return (idx >= 0 && idx < static_cast<int>(input.size())) ? input[idx] : 0.0f;
    };

    float v00 = get_pixel(x0, y0, channel);
    float v10 = get_pixel(x1, y0, channel);
    float v01 = get_pixel(x0, y1, channel);
    float v11 = get_pixel(x1, y1, channel);

    float v0 = v00 * (1 - dx) + v10 * dx;
    float v1 = v01 * (1 - dx) + v11 * dx;

    return v0 * (1 - dy) + v1 * dy;
}

} // namespace cyxwiz
