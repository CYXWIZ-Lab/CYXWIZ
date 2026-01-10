#include "blur_transform.h"
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <cstring>
#include <algorithm>

namespace cyxwiz {

// ============================================================================
// Enum String Conversions
// ============================================================================

std::string BlurTypeToString(BlurType type) {
    switch (type) {
        case BlurType::Box:       return "box";
        case BlurType::Gaussian:  return "gaussian";
        case BlurType::Median:    return "median";
        case BlurType::Bilateral: return "bilateral";
        default:                  return "gaussian";
    }
}

BlurType StringToBlurType(const std::string& str) {
    if (str == "box")       return BlurType::Box;
    if (str == "gaussian")  return BlurType::Gaussian;
    if (str == "median")    return BlurType::Median;
    if (str == "bilateral") return BlurType::Bilateral;
    return BlurType::Gaussian;
}

// ============================================================================
// BlurConfig Implementation
// ============================================================================

nlohmann::json BlurConfig::ToJson() const {
    return {
        {"enabled", enabled},
        {"type", BlurTypeToString(type)},
        {"kernel_size", kernel_size},
        {"sigma", sigma},
        {"sigma_color", sigma_color},
        {"sigma_space", sigma_space}
    };
}

BlurConfig BlurConfig::FromJson(const nlohmann::json& j) {
    BlurConfig config;
    if (j.contains("enabled")) config.enabled = j["enabled"].get<bool>();
    if (j.contains("type")) config.type = StringToBlurType(j["type"].get<std::string>());
    if (j.contains("kernel_size")) config.kernel_size = j["kernel_size"].get<int>();
    if (j.contains("sigma")) config.sigma = j["sigma"].get<float>();
    if (j.contains("sigma_color")) config.sigma_color = j["sigma_color"].get<float>();
    if (j.contains("sigma_space")) config.sigma_space = j["sigma_space"].get<float>();
    return config;
}

bool BlurConfig::operator==(const BlurConfig& other) const {
    return enabled == other.enabled &&
           type == other.type &&
           kernel_size == other.kernel_size &&
           std::abs(sigma - other.sigma) < 0.001f &&
           std::abs(sigma_color - other.sigma_color) < 0.001f &&
           std::abs(sigma_space - other.sigma_space) < 0.001f;
}

// ============================================================================
// BlurTransform Implementation
// ============================================================================

bool BlurTransform::ValidateInput(const std::vector<float>& image_data,
                                   int width, int height, int channels) {
    if (width <= 0 || height <= 0 || channels <= 0) {
        spdlog::error("BlurTransform: Invalid dimensions ({}x{}, {} channels)",
                      width, height, channels);
        return false;
    }

    size_t expected_size = static_cast<size_t>(width) * height * channels;
    if (image_data.size() != expected_size) {
        spdlog::error("BlurTransform: Data size mismatch. Expected {}, got {}",
                      expected_size, image_data.size());
        return false;
    }

    if (channels != 1 && channels != 3) {
        spdlog::error("BlurTransform: Only 1 or 3 channels supported, got {}", channels);
        return false;
    }

    return true;
}

int BlurTransform::ValidateKernelSize(int kernel_size, BlurType type) {
    // Kernel size must be odd and positive
    if (kernel_size < 1) {
        spdlog::warn("BlurTransform: Kernel size {} too small, using 3", kernel_size);
        return 3;
    }

    // Make odd if even
    if (kernel_size % 2 == 0) {
        kernel_size += 1;
        spdlog::debug("BlurTransform: Adjusted kernel size to {} (must be odd)", kernel_size);
    }

    // Median blur has stricter limits in OpenCV
    if (type == BlurType::Median) {
        if (kernel_size > 255) {
            spdlog::warn("BlurTransform: Median kernel size {} too large, using 255", kernel_size);
            return 255;
        }
    } else {
        if (kernel_size > 31) {
            spdlog::warn("BlurTransform: Kernel size {} too large, using 31", kernel_size);
            return 31;
        }
    }

    return kernel_size;
}

bool BlurTransform::Apply(
    std::vector<float>& image_data,
    int width, int height, int channels,
    const BlurConfig& config
) {
    if (!config.enabled) {
        return true;  // Nothing to do
    }

    switch (config.type) {
        case BlurType::Box:
            return BoxBlur(image_data, width, height, channels, config.kernel_size);

        case BlurType::Gaussian:
            return GaussianBlur(image_data, width, height, channels,
                               config.kernel_size, config.sigma);

        case BlurType::Median:
            return MedianBlur(image_data, width, height, channels, config.kernel_size);

        case BlurType::Bilateral:
            return BilateralFilter(image_data, width, height, channels,
                                   config.kernel_size, config.sigma_color, config.sigma_space);

        default:
            spdlog::error("BlurTransform: Unknown blur type");
            return false;
    }
}

bool BlurTransform::BoxBlur(
    std::vector<float>& image_data,
    int width, int height, int channels,
    int kernel_size
) {
    if (!ValidateInput(image_data, width, height, channels)) {
        return false;
    }

    kernel_size = ValidateKernelSize(kernel_size, BlurType::Box);

    try {
        if (channels == 1) {
            cv::Mat img(height, width, CV_32FC1, image_data.data());
            cv::Mat result;
            cv::blur(img, result, cv::Size(kernel_size, kernel_size));
            std::memcpy(image_data.data(), result.data, image_data.size() * sizeof(float));
        } else {
            cv::Mat img(height, width, CV_32FC3, image_data.data());
            cv::Mat result;
            cv::blur(img, result, cv::Size(kernel_size, kernel_size));
            std::memcpy(image_data.data(), result.data, image_data.size() * sizeof(float));
        }

        spdlog::debug("BlurTransform: Applied box blur (kernel={})", kernel_size);
        return true;

    } catch (const cv::Exception& e) {
        spdlog::error("BlurTransform: OpenCV exception in BoxBlur: {}", e.what());
        return false;
    }
}

bool BlurTransform::GaussianBlur(
    std::vector<float>& image_data,
    int width, int height, int channels,
    int kernel_size,
    float sigma
) {
    if (!ValidateInput(image_data, width, height, channels)) {
        return false;
    }

    kernel_size = ValidateKernelSize(kernel_size, BlurType::Gaussian);

    // If sigma is 0, OpenCV will calculate it from kernel size
    double cv_sigma = sigma > 0.0f ? static_cast<double>(sigma) : 0.0;

    try {
        if (channels == 1) {
            cv::Mat img(height, width, CV_32FC1, image_data.data());
            cv::Mat result;
            cv::GaussianBlur(img, result, cv::Size(kernel_size, kernel_size), cv_sigma);
            std::memcpy(image_data.data(), result.data, image_data.size() * sizeof(float));
        } else {
            cv::Mat img(height, width, CV_32FC3, image_data.data());
            cv::Mat result;
            cv::GaussianBlur(img, result, cv::Size(kernel_size, kernel_size), cv_sigma);
            std::memcpy(image_data.data(), result.data, image_data.size() * sizeof(float));
        }

        spdlog::debug("BlurTransform: Applied Gaussian blur (kernel={}, sigma={})",
                      kernel_size, sigma);
        return true;

    } catch (const cv::Exception& e) {
        spdlog::error("BlurTransform: OpenCV exception in GaussianBlur: {}", e.what());
        return false;
    }
}

bool BlurTransform::MedianBlur(
    std::vector<float>& image_data,
    int width, int height, int channels,
    int kernel_size
) {
    if (!ValidateInput(image_data, width, height, channels)) {
        return false;
    }

    kernel_size = ValidateKernelSize(kernel_size, BlurType::Median);

    try {
        // MedianBlur requires 8-bit images
        if (channels == 1) {
            cv::Mat img(height, width, CV_32FC1, image_data.data());

            // Convert to 8-bit
            cv::Mat uint8_img;
            img.convertTo(uint8_img, CV_8UC1, 255.0);

            // Apply median blur
            cv::Mat result;
            cv::medianBlur(uint8_img, result, kernel_size);

            // Convert back to float
            result.convertTo(img, CV_32FC1, 1.0 / 255.0);
            std::memcpy(image_data.data(), img.data, image_data.size() * sizeof(float));

        } else {
            cv::Mat img(height, width, CV_32FC3, image_data.data());

            // Convert to 8-bit
            cv::Mat uint8_img;
            img.convertTo(uint8_img, CV_8UC3, 255.0);

            // Apply median blur
            cv::Mat result;
            cv::medianBlur(uint8_img, result, kernel_size);

            // Convert back to float
            result.convertTo(img, CV_32FC3, 1.0 / 255.0);
            std::memcpy(image_data.data(), img.data, image_data.size() * sizeof(float));
        }

        spdlog::debug("BlurTransform: Applied median blur (kernel={})", kernel_size);
        return true;

    } catch (const cv::Exception& e) {
        spdlog::error("BlurTransform: OpenCV exception in MedianBlur: {}", e.what());
        return false;
    }
}

bool BlurTransform::BilateralFilter(
    std::vector<float>& image_data,
    int width, int height, int channels,
    int diameter,
    float sigma_color,
    float sigma_space
) {
    if (!ValidateInput(image_data, width, height, channels)) {
        return false;
    }

    // Diameter validation
    if (diameter < -1) diameter = -1;
    if (diameter > 15) {
        spdlog::warn("BlurTransform: Bilateral diameter {} is large, may be slow", diameter);
    }

    try {
        // Bilateral filter works best with 8-bit images
        if (channels == 1) {
            cv::Mat img(height, width, CV_32FC1, image_data.data());

            // Convert to 8-bit
            cv::Mat uint8_img;
            img.convertTo(uint8_img, CV_8UC1, 255.0);

            // Apply bilateral filter
            cv::Mat result;
            cv::bilateralFilter(uint8_img, result, diameter,
                               static_cast<double>(sigma_color),
                               static_cast<double>(sigma_space));

            // Convert back to float
            result.convertTo(img, CV_32FC1, 1.0 / 255.0);
            std::memcpy(image_data.data(), img.data, image_data.size() * sizeof(float));

        } else {
            cv::Mat img(height, width, CV_32FC3, image_data.data());

            // Convert to 8-bit
            cv::Mat uint8_img;
            img.convertTo(uint8_img, CV_8UC3, 255.0);

            // Apply bilateral filter
            cv::Mat result;
            cv::bilateralFilter(uint8_img, result, diameter,
                               static_cast<double>(sigma_color),
                               static_cast<double>(sigma_space));

            // Convert back to float
            result.convertTo(img, CV_32FC3, 1.0 / 255.0);
            std::memcpy(image_data.data(), img.data, image_data.size() * sizeof(float));
        }

        spdlog::debug("BlurTransform: Applied bilateral filter (d={}, sigma_color={}, sigma_space={})",
                      diameter, sigma_color, sigma_space);
        return true;

    } catch (const cv::Exception& e) {
        spdlog::error("BlurTransform: OpenCV exception in BilateralFilter: {}", e.what());
        return false;
    }
}

} // namespace cyxwiz
