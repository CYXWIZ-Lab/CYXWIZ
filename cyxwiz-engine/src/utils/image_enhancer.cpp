#include "image_enhancer.h"
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>

namespace cyxwiz {

bool ImageEnhancer::ValidateInput(const std::vector<float>& image_data,
                                   int width, int height, int channels) {
    if (width <= 0 || height <= 0 || channels <= 0) {
        spdlog::error("ImageEnhancer: Invalid dimensions ({}x{}, {} channels)",
                      width, height, channels);
        return false;
    }

    size_t expected_size = static_cast<size_t>(width) * height * channels;
    if (image_data.size() != expected_size) {
        spdlog::error("ImageEnhancer: Data size mismatch. Expected {}, got {}",
                      expected_size, image_data.size());
        return false;
    }

    return true;
}

bool ImageEnhancer::ApplyCLAHE(
    std::vector<float>& image_data,
    int width, int height, int channels,
    float clip_limit, int tile_size
) {
    if (!ValidateInput(image_data, width, height, channels)) {
        return false;
    }

    try {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
            clip_limit,
            cv::Size(tile_size, tile_size)
        );

        if (channels == 1) {
            // Grayscale - apply CLAHE directly
            cv::Mat gray(height, width, CV_32FC1, image_data.data());

            // Convert to 8-bit for CLAHE
            cv::Mat uint8_img;
            gray.convertTo(uint8_img, CV_8UC1, 255.0);

            // Apply CLAHE
            clahe->apply(uint8_img, uint8_img);

            // Convert back to float [0, 1]
            uint8_img.convertTo(gray, CV_32FC1, 1.0 / 255.0);
            std::memcpy(image_data.data(), gray.data,
                        image_data.size() * sizeof(float));
        } else if (channels == 3) {
            // RGB - convert to Lab, apply CLAHE to L channel
            cv::Mat rgb(height, width, CV_32FC3, image_data.data());

            // Convert to 8-bit
            cv::Mat uint8_rgb;
            rgb.convertTo(uint8_rgb, CV_8UC3, 255.0);

            // Convert RGB to Lab color space
            cv::Mat lab;
            cv::cvtColor(uint8_rgb, lab, cv::COLOR_RGB2Lab);

            // Split into L, a, b channels
            std::vector<cv::Mat> lab_channels;
            cv::split(lab, lab_channels);

            // Apply CLAHE to L channel only
            clahe->apply(lab_channels[0], lab_channels[0]);

            // Merge channels back
            cv::merge(lab_channels, lab);

            // Convert Lab back to RGB
            cv::cvtColor(lab, uint8_rgb, cv::COLOR_Lab2RGB);

            // Convert back to float [0, 1]
            uint8_rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
            std::memcpy(image_data.data(), rgb.data,
                        image_data.size() * sizeof(float));
        } else {
            spdlog::error("ImageEnhancer: CLAHE only supports 1 or 3 channels, got {}",
                          channels);
            return false;
        }

        spdlog::debug("ImageEnhancer: Applied CLAHE (clip_limit={}, tile_size={})",
                      clip_limit, tile_size);
        return true;

    } catch (const cv::Exception& e) {
        spdlog::error("ImageEnhancer: OpenCV exception in ApplyCLAHE: {}", e.what());
        return false;
    }
}

bool ImageEnhancer::ApplyDenoise(
    std::vector<float>& image_data,
    int width, int height, int channels,
    float strength, float color_strength,
    int template_size, int search_size
) {
    if (!ValidateInput(image_data, width, height, channels)) {
        return false;
    }

    try {
        if (channels == 1) {
            // Grayscale denoising
            cv::Mat gray(height, width, CV_32FC1, image_data.data());

            // Convert to 8-bit
            cv::Mat uint8_img;
            gray.convertTo(uint8_img, CV_8UC1, 255.0);

            // Apply denoising
            cv::Mat denoised;
            cv::fastNlMeansDenoising(uint8_img, denoised,
                                     strength, template_size, search_size);

            // Convert back to float
            denoised.convertTo(gray, CV_32FC1, 1.0 / 255.0);
            std::memcpy(image_data.data(), gray.data,
                        image_data.size() * sizeof(float));

        } else if (channels == 3) {
            // Color denoising
            cv::Mat rgb(height, width, CV_32FC3, image_data.data());

            // Convert to 8-bit
            cv::Mat uint8_rgb;
            rgb.convertTo(uint8_rgb, CV_8UC3, 255.0);

            // Apply color denoising
            cv::Mat denoised;
            cv::fastNlMeansDenoisingColored(uint8_rgb, denoised,
                                            strength, color_strength,
                                            template_size, search_size);

            // Convert back to float
            denoised.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
            std::memcpy(image_data.data(), rgb.data,
                        image_data.size() * sizeof(float));
        } else {
            spdlog::error("ImageEnhancer: Denoise only supports 1 or 3 channels, got {}",
                          channels);
            return false;
        }

        spdlog::debug("ImageEnhancer: Applied denoise (strength={})", strength);
        return true;

    } catch (const cv::Exception& e) {
        spdlog::error("ImageEnhancer: OpenCV exception in ApplyDenoise: {}", e.what());
        return false;
    }
}

bool ImageEnhancer::ApplySharpen(
    std::vector<float>& image_data,
    int width, int height, int channels,
    float amount, float radius, float threshold
) {
    if (!ValidateInput(image_data, width, height, channels)) {
        return false;
    }

    try {
        // Create cv::Mat wrapper
        int cv_type = (channels == 1) ? CV_32FC1 : CV_32FC3;
        cv::Mat img(height, width, cv_type, image_data.data());

        // Apply Gaussian blur
        cv::Mat blurred;
        int kernel_size = std::max(3, static_cast<int>(radius * 6 + 1));
        if (kernel_size % 2 == 0) kernel_size++; // Must be odd
        cv::GaussianBlur(img, blurred, cv::Size(kernel_size, kernel_size), radius);

        // Unsharp mask: original + amount * (original - blurred)
        cv::Mat sharpened = img + amount * (img - blurred);

        // Apply threshold if specified
        if (threshold > 0.0f) {
            cv::Mat diff = cv::abs(img - blurred);
            cv::Mat mask = diff > threshold;
            if (channels == 3) {
                // Convert single-channel mask to 3-channel
                std::vector<cv::Mat> mask_channels = {mask, mask, mask};
                cv::merge(mask_channels, mask);
            }
            mask.convertTo(mask, CV_32FC(channels), 1.0 / 255.0);
            sharpened = img.mul(1.0 - mask) + sharpened.mul(mask);
        }

        // Clamp to [0, 1] range
        sharpened = cv::max(0.0, cv::min(1.0, sharpened));

        // Copy back to input vector
        std::memcpy(image_data.data(), sharpened.data,
                    image_data.size() * sizeof(float));

        spdlog::debug("ImageEnhancer: Applied sharpen (amount={}, radius={})",
                      amount, radius);
        return true;

    } catch (const cv::Exception& e) {
        spdlog::error("ImageEnhancer: OpenCV exception in ApplySharpen: {}", e.what());
        return false;
    }
}

bool ImageEnhancer::ApplyPreset(
    std::vector<float>& image_data,
    int width, int height, int channels,
    EnhancementPreset preset
) {
    switch (preset) {
        case EnhancementPreset::None:
            // No enhancement
            return true;

        case EnhancementPreset::Light:
            // CLAHE(1.5) + light sharpen
            if (!ApplyCLAHE(image_data, width, height, channels, 1.5f, 8)) {
                return false;
            }
            return ApplySharpen(image_data, width, height, channels, 0.5f, 1.0f);

        case EnhancementPreset::Standard:
            // CLAHE(2.0) + denoise + sharpen
            if (!ApplyCLAHE(image_data, width, height, channels, 2.0f, 8)) {
                return false;
            }
            if (!ApplyDenoise(image_data, width, height, channels, 10.0f, 10.0f)) {
                return false;
            }
            return ApplySharpen(image_data, width, height, channels, 1.0f, 1.0f);

        case EnhancementPreset::Strong:
            // CLAHE(3.0) + strong denoise + sharpen
            if (!ApplyCLAHE(image_data, width, height, channels, 3.0f, 8)) {
                return false;
            }
            if (!ApplyDenoise(image_data, width, height, channels, 15.0f, 15.0f)) {
                return false;
            }
            return ApplySharpen(image_data, width, height, channels, 1.5f, 1.0f);

        case EnhancementPreset::Medical:
            // CLAHE(2.5) only (preserve detail, no artifacts from denoise/sharpen)
            return ApplyCLAHE(image_data, width, height, channels, 2.5f, 8);

        case EnhancementPreset::Custom:
            // User should apply individual methods
            spdlog::info("ImageEnhancer: Custom preset selected, apply methods individually");
            return true;

        default:
            spdlog::error("ImageEnhancer: Unknown preset {}", static_cast<int>(preset));
            return false;
    }
}

} // namespace cyxwiz
