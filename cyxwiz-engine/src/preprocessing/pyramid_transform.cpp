#include "pyramid_transform.h"
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <cstring>
#include <algorithm>

namespace cyxwiz {

// ============================================================================
// Enum String Conversions
// ============================================================================

std::string PyramidOperationToString(PyramidOperation op) {
    switch (op) {
        case PyramidOperation::PyrDown:        return "pyr_down";
        case PyramidOperation::PyrUp:          return "pyr_up";
        case PyramidOperation::BuildGaussian:  return "build_gaussian";
        case PyramidOperation::BuildLaplacian: return "build_laplacian";
        default:                               return "pyr_down";
    }
}

PyramidOperation StringToPyramidOperation(const std::string& str) {
    if (str == "pyr_down")        return PyramidOperation::PyrDown;
    if (str == "pyr_up")          return PyramidOperation::PyrUp;
    if (str == "build_gaussian")  return PyramidOperation::BuildGaussian;
    if (str == "build_laplacian") return PyramidOperation::BuildLaplacian;
    return PyramidOperation::PyrDown;
}

// ============================================================================
// PyramidConfig Implementation
// ============================================================================

nlohmann::json PyramidConfig::ToJson() const {
    return {
        {"enabled", enabled},
        {"operation", PyramidOperationToString(operation)},
        {"levels", levels}
    };
}

PyramidConfig PyramidConfig::FromJson(const nlohmann::json& j) {
    PyramidConfig config;
    if (j.contains("enabled")) config.enabled = j["enabled"].get<bool>();
    if (j.contains("operation")) config.operation = StringToPyramidOperation(j["operation"].get<std::string>());
    if (j.contains("levels")) config.levels = j["levels"].get<int>();
    return config;
}

bool PyramidConfig::operator==(const PyramidConfig& other) const {
    return enabled == other.enabled &&
           operation == other.operation &&
           levels == other.levels;
}

// ============================================================================
// PyramidTransform Implementation
// ============================================================================

bool PyramidTransform::ValidateInput(const std::vector<float>& image_data,
                                      int width, int height, int channels) {
    if (width <= 0 || height <= 0 || channels <= 0) {
        spdlog::error("PyramidTransform: Invalid dimensions ({}x{}, {} channels)",
                      width, height, channels);
        return false;
    }

    size_t expected_size = static_cast<size_t>(width) * height * channels;
    if (image_data.size() != expected_size) {
        spdlog::error("PyramidTransform: Data size mismatch. Expected {}, got {}",
                      expected_size, image_data.size());
        return false;
    }

    if (channels != 1 && channels != 3) {
        spdlog::error("PyramidTransform: Only 1 or 3 channels supported, got {}", channels);
        return false;
    }

    return true;
}

bool PyramidTransform::Apply(
    std::vector<float>& image_data,
    int& width, int& height, int channels,
    const PyramidConfig& config
) {
    if (!config.enabled) {
        return true;
    }

    int levels = std::max(1, config.levels);

    switch (config.operation) {
        case PyramidOperation::PyrDown:
            return PyrDownMultiple(image_data, width, height, channels, levels);

        case PyramidOperation::PyrUp:
            return PyrUpMultiple(image_data, width, height, channels, levels);

        case PyramidOperation::BuildGaussian:
            return BuildGaussianLevel(image_data, width, height, channels, levels);

        case PyramidOperation::BuildLaplacian:
            return BuildLaplacianLevel(image_data, width, height, channels, levels);

        default:
            spdlog::error("PyramidTransform: Unknown operation");
            return false;
    }
}

bool PyramidTransform::PyrDown(
    std::vector<float>& image_data,
    int& width, int& height, int channels
) {
    if (!ValidateInput(image_data, width, height, channels)) {
        return false;
    }

    // Check minimum size
    if (width < 2 || height < 2) {
        spdlog::warn("PyramidTransform: Image too small for PyrDown ({}x{})", width, height);
        return true;  // Nothing to do
    }

    try {
        // Calculate new dimensions (OpenCV pyrDown rounds up for odd dimensions)
        int new_width = (width + 1) / 2;
        int new_height = (height + 1) / 2;

        if (channels == 1) {
            cv::Mat img(height, width, CV_32FC1, image_data.data());
            cv::Mat result;
            cv::pyrDown(img, result, cv::Size(new_width, new_height));

            image_data.resize(static_cast<size_t>(new_width) * new_height);
            std::memcpy(image_data.data(), result.data, image_data.size() * sizeof(float));
        } else {
            cv::Mat img(height, width, CV_32FC3, image_data.data());
            cv::Mat result;
            cv::pyrDown(img, result, cv::Size(new_width, new_height));

            image_data.resize(static_cast<size_t>(new_width) * new_height * channels);
            std::memcpy(image_data.data(), result.data, image_data.size() * sizeof(float));
        }

        width = new_width;
        height = new_height;

        spdlog::debug("PyramidTransform: PyrDown to {}x{}", new_width, new_height);
        return true;

    } catch (const cv::Exception& e) {
        spdlog::error("PyramidTransform: OpenCV exception in PyrDown: {}", e.what());
        return false;
    }
}

bool PyramidTransform::PyrUp(
    std::vector<float>& image_data,
    int& width, int& height, int channels
) {
    if (!ValidateInput(image_data, width, height, channels)) {
        return false;
    }

    try {
        int new_width = width * 2;
        int new_height = height * 2;

        if (channels == 1) {
            cv::Mat img(height, width, CV_32FC1, image_data.data());
            cv::Mat result;
            cv::pyrUp(img, result, cv::Size(new_width, new_height));

            image_data.resize(static_cast<size_t>(new_width) * new_height);
            std::memcpy(image_data.data(), result.data, image_data.size() * sizeof(float));
        } else {
            cv::Mat img(height, width, CV_32FC3, image_data.data());
            cv::Mat result;
            cv::pyrUp(img, result, cv::Size(new_width, new_height));

            image_data.resize(static_cast<size_t>(new_width) * new_height * channels);
            std::memcpy(image_data.data(), result.data, image_data.size() * sizeof(float));
        }

        width = new_width;
        height = new_height;

        spdlog::debug("PyramidTransform: PyrUp to {}x{}", new_width, new_height);
        return true;

    } catch (const cv::Exception& e) {
        spdlog::error("PyramidTransform: OpenCV exception in PyrUp: {}", e.what());
        return false;
    }
}

bool PyramidTransform::BuildGaussianLevel(
    std::vector<float>& image_data,
    int& width, int& height, int channels,
    int level
) {
    if (!ValidateInput(image_data, width, height, channels)) {
        return false;
    }

    if (level < 0) {
        spdlog::warn("PyramidTransform: Invalid level {}, using 0", level);
        level = 0;
    }

    if (level == 0) {
        return true;  // Original image, nothing to do
    }

    // Apply pyrDown 'level' times
    return PyrDownMultiple(image_data, width, height, channels, level);
}

bool PyramidTransform::BuildLaplacianLevel(
    std::vector<float>& image_data,
    int& width, int& height, int channels,
    int level
) {
    if (!ValidateInput(image_data, width, height, channels)) {
        return false;
    }

    if (level < 0) {
        spdlog::warn("PyramidTransform: Invalid level {}, using 0", level);
        level = 0;
    }

    try {
        // Build Gaussian pyramid up to level+1
        std::vector<cv::Mat> gaussian_pyramid;

        if (channels == 1) {
            cv::Mat current(height, width, CV_32FC1, image_data.data());
            gaussian_pyramid.push_back(current.clone());

            for (int i = 0; i <= level; ++i) {
                cv::Mat next;
                cv::pyrDown(gaussian_pyramid.back(), next);
                gaussian_pyramid.push_back(next);

                // Check minimum size
                if (next.cols < 2 || next.rows < 2) {
                    spdlog::warn("PyramidTransform: Reached minimum pyramid size at level {}", i);
                    break;
                }
            }

            // Laplacian[level] = Gaussian[level] - PyrUp(Gaussian[level+1])
            int target_level = std::min(level, static_cast<int>(gaussian_pyramid.size()) - 2);

            cv::Mat upsampled;
            cv::pyrUp(gaussian_pyramid[target_level + 1], upsampled,
                     gaussian_pyramid[target_level].size());

            cv::Mat laplacian = gaussian_pyramid[target_level] - upsampled;

            // Normalize Laplacian to 0-1 range (it can have negative values)
            double min_val, max_val;
            cv::minMaxLoc(laplacian, &min_val, &max_val);
            if (max_val > min_val) {
                laplacian = (laplacian - min_val) / (max_val - min_val);
            }

            // Update output
            width = laplacian.cols;
            height = laplacian.rows;
            image_data.resize(static_cast<size_t>(width) * height);
            std::memcpy(image_data.data(), laplacian.data, image_data.size() * sizeof(float));

        } else {
            cv::Mat current(height, width, CV_32FC3, image_data.data());
            gaussian_pyramid.push_back(current.clone());

            for (int i = 0; i <= level; ++i) {
                cv::Mat next;
                cv::pyrDown(gaussian_pyramid.back(), next);
                gaussian_pyramid.push_back(next);

                if (next.cols < 2 || next.rows < 2) {
                    spdlog::warn("PyramidTransform: Reached minimum pyramid size at level {}", i);
                    break;
                }
            }

            int target_level = std::min(level, static_cast<int>(gaussian_pyramid.size()) - 2);

            cv::Mat upsampled;
            cv::pyrUp(gaussian_pyramid[target_level + 1], upsampled,
                     gaussian_pyramid[target_level].size());

            cv::Mat laplacian = gaussian_pyramid[target_level] - upsampled;

            // Normalize per channel
            std::vector<cv::Mat> channels_vec;
            cv::split(laplacian, channels_vec);
            for (auto& ch : channels_vec) {
                double min_val, max_val;
                cv::minMaxLoc(ch, &min_val, &max_val);
                if (max_val > min_val) {
                    ch = (ch - min_val) / (max_val - min_val);
                }
            }
            cv::merge(channels_vec, laplacian);

            width = laplacian.cols;
            height = laplacian.rows;
            image_data.resize(static_cast<size_t>(width) * height * channels);
            std::memcpy(image_data.data(), laplacian.data, image_data.size() * sizeof(float));
        }

        spdlog::debug("PyramidTransform: Built Laplacian level {} ({}x{})",
                      level, width, height);
        return true;

    } catch (const cv::Exception& e) {
        spdlog::error("PyramidTransform: OpenCV exception in BuildLaplacianLevel: {}", e.what());
        return false;
    }
}

bool PyramidTransform::PyrDownMultiple(
    std::vector<float>& image_data,
    int& width, int& height, int channels,
    int levels
) {
    if (!ValidateInput(image_data, width, height, channels)) {
        return false;
    }

    levels = std::max(0, levels);

    for (int i = 0; i < levels; ++i) {
        if (width < 2 || height < 2) {
            spdlog::warn("PyramidTransform: Stopped at level {} (image too small)", i);
            break;
        }
        if (!PyrDown(image_data, width, height, channels)) {
            return false;
        }
    }

    return true;
}

bool PyramidTransform::PyrUpMultiple(
    std::vector<float>& image_data,
    int& width, int& height, int channels,
    int levels
) {
    if (!ValidateInput(image_data, width, height, channels)) {
        return false;
    }

    levels = std::max(0, levels);

    for (int i = 0; i < levels; ++i) {
        // Check for reasonable size limit
        if (width > 16384 || height > 16384) {
            spdlog::warn("PyramidTransform: Stopped at level {} (image too large)", i);
            break;
        }
        if (!PyrUp(image_data, width, height, channels)) {
            return false;
        }
    }

    return true;
}

} // namespace cyxwiz
