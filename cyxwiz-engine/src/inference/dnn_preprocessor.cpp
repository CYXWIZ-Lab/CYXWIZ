#include "dnn_preprocessor.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cyxwiz {

DNNBlob DNNPreprocessor::Preprocess(
    const std::vector<float>& image_data,
    int width, int height, int channels,
    const DNNPreprocessConfig& config)
{
    DNNBlob blob;
    blob.original_width = width;
    blob.original_height = height;
    blob.batch_size = 1;
    blob.channels = channels;
    blob.width = config.input_width;
    blob.height = config.input_height;

    // Validate input
    if (image_data.size() != static_cast<size_t>(width * height * channels)) {
        throw std::invalid_argument("Image data size mismatch");
    }

    std::vector<float> resized;

    // Step 1: Resize
    if (config.keep_aspect_ratio) {
        ResizeLetterbox(image_data, width, height, channels,
                       resized, config.input_width, config.input_height,
                       blob.scale, blob.pad_left, blob.pad_top,
                       config.letterbox_color);
    } else if (config.crop_center) {
        ResizeCenterCrop(image_data, width, height, channels,
                        resized, config.input_width, config.input_height);
        blob.scale = 1.0f;
        blob.pad_left = 0;
        blob.pad_top = 0;
    } else {
        ResizeBilinear(image_data, width, height, channels,
                      resized, config.input_width, config.input_height);
        blob.scale = 1.0f;
        blob.pad_left = 0;
        blob.pad_top = 0;
    }

    // Step 2: Swap R and B if needed
    if (config.swap_rb && channels == 3) {
        SwapRB(resized, config.input_width, config.input_height);
    }

    // Step 3: Normalize
    Normalize(resized, config.input_width, config.input_height, channels,
             config.scale_factor, config.mean, config.std_dev);

    // Step 4: Convert to CHW if needed
    if (config.channels_first) {
        HWCToCHW(resized, config.input_width, config.input_height, channels, blob.data);
    } else {
        blob.data = std::move(resized);
    }

    return blob;
}

DNNBlob DNNPreprocessor::PreprocessBatch(
    const std::vector<std::vector<float>>& images,
    const std::vector<int>& widths,
    const std::vector<int>& heights,
    int channels,
    const DNNPreprocessConfig& config)
{
    if (images.empty()) {
        throw std::invalid_argument("Empty image batch");
    }

    size_t batch_size = images.size();
    if (widths.size() != batch_size || heights.size() != batch_size) {
        throw std::invalid_argument("Dimension arrays size mismatch");
    }

    DNNBlob blob;
    blob.batch_size = static_cast<int>(batch_size);
    blob.channels = channels;
    blob.width = config.input_width;
    blob.height = config.input_height;
    blob.original_width = widths[0];  // Use first image dimensions
    blob.original_height = heights[0];

    size_t single_size = static_cast<size_t>(config.input_width * config.input_height * channels);
    blob.data.resize(batch_size * single_size);

    for (size_t i = 0; i < batch_size; ++i) {
        DNNBlob single = Preprocess(images[i], widths[i], heights[i], channels, config);

        // Copy to batch
        std::copy(single.data.begin(), single.data.end(),
                 blob.data.begin() + i * single_size);

        // Store scale/padding from first image
        if (i == 0) {
            blob.scale = single.scale;
            blob.pad_left = single.pad_left;
            blob.pad_top = single.pad_top;
        }
    }

    return blob;
}

std::pair<float, float> DNNPreprocessor::MapToOriginal(
    float x, float y,
    const DNNBlob& blob)
{
    // Convert from model space (0-1) to padded image pixels
    float px = x * blob.width;
    float py = y * blob.height;

    // Remove padding
    px -= blob.pad_left;
    py -= blob.pad_top;

    // Scale to original size
    float orig_x = px / blob.scale / blob.original_width;
    float orig_y = py / blob.scale / blob.original_height;

    // Clamp to valid range
    orig_x = std::max(0.0f, std::min(1.0f, orig_x));
    orig_y = std::max(0.0f, std::min(1.0f, orig_y));

    return {orig_x, orig_y};
}

std::array<float, 4> DNNPreprocessor::MapBoxToOriginal(
    float x, float y, float w, float h,
    const DNNBlob& blob)
{
    // Map center point
    auto [cx, cy] = MapToOriginal(x, y, blob);

    // Scale dimensions
    float scaled_w = w * blob.width / blob.scale / blob.original_width;
    float scaled_h = h * blob.height / blob.scale / blob.original_height;

    return {cx, cy, scaled_w, scaled_h};
}

// Preset configurations

DNNPreprocessConfig DNNPreprocessor::ImageNetConfig(int size) {
    DNNPreprocessConfig config;
    config.input_width = size;
    config.input_height = size;
    config.scale_factor = 1.0f / 255.0f;
    config.mean = {0.485f, 0.456f, 0.406f};  // ImageNet mean (RGB)
    config.std_dev = {0.229f, 0.224f, 0.225f};  // ImageNet std (RGB)
    config.swap_rb = true;  // RGB to BGR
    config.crop_center = true;
    config.keep_aspect_ratio = false;
    return config;
}

DNNPreprocessConfig DNNPreprocessor::YOLOConfig(int size) {
    DNNPreprocessConfig config;
    config.input_width = size;
    config.input_height = size;
    config.scale_factor = 1.0f / 255.0f;
    config.mean = {0.0f, 0.0f, 0.0f};
    config.std_dev = {1.0f, 1.0f, 1.0f};
    config.swap_rb = true;  // RGB to BGR
    config.keep_aspect_ratio = true;  // Letterbox
    config.letterbox_color = 0.5f;  // Gray padding
    return config;
}

DNNPreprocessConfig DNNPreprocessor::SSDConfig(int size, bool mobilenet) {
    DNNPreprocessConfig config;
    config.input_width = size;
    config.input_height = size;

    if (mobilenet) {
        // MobileNet-SSD uses simple scaling
        config.scale_factor = 0.007843f;  // 2/255
        config.mean = {127.5f, 127.5f, 127.5f};  // Will be multiplied by scale
    } else {
        // VGG-based SSD
        config.scale_factor = 1.0f;
        config.mean = {104.0f, 117.0f, 123.0f};  // BGR mean
    }

    config.std_dev = {1.0f, 1.0f, 1.0f};
    config.swap_rb = true;
    config.keep_aspect_ratio = false;
    return config;
}

DNNPreprocessConfig DNNPreprocessor::FaceDetectorConfig() {
    DNNPreprocessConfig config;
    config.input_width = 300;
    config.input_height = 300;
    config.scale_factor = 1.0f;
    config.mean = {104.0f, 177.0f, 123.0f};  // Face detector mean
    config.std_dev = {1.0f, 1.0f, 1.0f};
    config.swap_rb = false;  // Already BGR
    config.keep_aspect_ratio = false;
    return config;
}

DNNPreprocessConfig DNNPreprocessor::OpenPoseConfig(int height) {
    DNNPreprocessConfig config;
    // OpenPose typically uses 368 or 656 height, width varies with aspect ratio
    config.input_height = height;
    config.input_width = height;  // Will be adjusted for aspect ratio
    config.scale_factor = 1.0f / 255.0f;
    config.mean = {0.0f, 0.0f, 0.0f};
    config.std_dev = {1.0f, 1.0f, 1.0f};
    config.swap_rb = false;
    config.keep_aspect_ratio = true;
    return config;
}

DNNPreprocessConfig DNNPreprocessor::ClassificationConfig(int size) {
    // Generic classification config similar to ImageNet
    return ImageNetConfig(size);
}

// Private helper implementations

void DNNPreprocessor::ResizeLetterbox(
    const std::vector<float>& src, int src_w, int src_h, int channels,
    std::vector<float>& dst, int dst_w, int dst_h,
    float& scale, int& pad_left, int& pad_top,
    float fill_color)
{
    // Calculate scale to fit within destination
    float scale_w = static_cast<float>(dst_w) / src_w;
    float scale_h = static_cast<float>(dst_h) / src_h;
    scale = std::min(scale_w, scale_h);

    // Calculate new size
    int new_w = static_cast<int>(src_w * scale);
    int new_h = static_cast<int>(src_h * scale);

    // Calculate padding
    pad_left = (dst_w - new_w) / 2;
    pad_top = (dst_h - new_h) / 2;

    // Initialize output with fill color
    dst.resize(dst_w * dst_h * channels);
    std::fill(dst.begin(), dst.end(), fill_color);

    // Resize source to intermediate size
    std::vector<float> resized;
    ResizeBilinear(src, src_w, src_h, channels, resized, new_w, new_h);

    // Copy resized image to center of output
    for (int y = 0; y < new_h; ++y) {
        for (int x = 0; x < new_w; ++x) {
            int dst_idx = ((y + pad_top) * dst_w + (x + pad_left)) * channels;
            int src_idx = (y * new_w + x) * channels;
            for (int c = 0; c < channels; ++c) {
                dst[dst_idx + c] = resized[src_idx + c];
            }
        }
    }
}

void DNNPreprocessor::ResizeCenterCrop(
    const std::vector<float>& src, int src_w, int src_h, int channels,
    std::vector<float>& dst, int dst_w, int dst_h)
{
    // First resize to make smaller dimension equal to target
    float scale = std::max(
        static_cast<float>(dst_w) / src_w,
        static_cast<float>(dst_h) / src_h
    );

    int new_w = static_cast<int>(src_w * scale);
    int new_h = static_cast<int>(src_h * scale);

    std::vector<float> resized;
    ResizeBilinear(src, src_w, src_h, channels, resized, new_w, new_h);

    // Center crop
    int crop_x = (new_w - dst_w) / 2;
    int crop_y = (new_h - dst_h) / 2;

    dst.resize(dst_w * dst_h * channels);
    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            int dst_idx = (y * dst_w + x) * channels;
            int src_idx = ((y + crop_y) * new_w + (x + crop_x)) * channels;
            for (int c = 0; c < channels; ++c) {
                dst[dst_idx + c] = resized[src_idx + c];
            }
        }
    }
}

void DNNPreprocessor::ResizeBilinear(
    const std::vector<float>& src, int src_w, int src_h, int channels,
    std::vector<float>& dst, int dst_w, int dst_h)
{
    dst.resize(dst_w * dst_h * channels);

    float scale_x = static_cast<float>(src_w) / dst_w;
    float scale_y = static_cast<float>(src_h) / dst_h;

    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            float src_x = (x + 0.5f) * scale_x - 0.5f;
            float src_y = (y + 0.5f) * scale_y - 0.5f;

            int x0 = static_cast<int>(std::floor(src_x));
            int y0 = static_cast<int>(std::floor(src_y));
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            // Clamp to valid range
            x0 = std::max(0, std::min(x0, src_w - 1));
            x1 = std::max(0, std::min(x1, src_w - 1));
            y0 = std::max(0, std::min(y0, src_h - 1));
            y1 = std::max(0, std::min(y1, src_h - 1));

            float fx = src_x - std::floor(src_x);
            float fy = src_y - std::floor(src_y);

            for (int c = 0; c < channels; ++c) {
                float v00 = src[(y0 * src_w + x0) * channels + c];
                float v01 = src[(y0 * src_w + x1) * channels + c];
                float v10 = src[(y1 * src_w + x0) * channels + c];
                float v11 = src[(y1 * src_w + x1) * channels + c];

                float v0 = v00 * (1 - fx) + v01 * fx;
                float v1 = v10 * (1 - fx) + v11 * fx;
                float v = v0 * (1 - fy) + v1 * fy;

                dst[(y * dst_w + x) * channels + c] = v;
            }
        }
    }
}

void DNNPreprocessor::HWCToCHW(
    const std::vector<float>& hwc, int w, int h, int c,
    std::vector<float>& chw)
{
    chw.resize(hwc.size());

    for (int ch = 0; ch < c; ++ch) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int hwc_idx = (y * w + x) * c + ch;
                int chw_idx = ch * h * w + y * w + x;
                chw[chw_idx] = hwc[hwc_idx];
            }
        }
    }
}

void DNNPreprocessor::Normalize(
    std::vector<float>& data, int w, int h, int c,
    float scale,
    const std::array<float, 3>& mean,
    const std::array<float, 3>& std_dev)
{
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int ch = 0; ch < c; ++ch) {
                int idx = (y * w + x) * c + ch;

                // Scale from 0-1 to 0-255 if needed, then apply normalization
                float val = data[idx] * 255.0f;  // Convert back to 0-255

                // Apply scale, mean subtraction, and std normalization
                val = (val * scale - mean[ch % 3]) / std_dev[ch % 3];

                data[idx] = val;
            }
        }
    }
}

void DNNPreprocessor::SwapRB(std::vector<float>& data, int w, int h) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = (y * w + x) * 3;
            std::swap(data[idx], data[idx + 2]);  // Swap R and B
        }
    }
}

} // namespace cyxwiz
