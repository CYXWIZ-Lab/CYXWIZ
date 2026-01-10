#include "morphology_transform.h"
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <cstring>

namespace cyxwiz {

// ============================================================================
// Enum String Conversions
// ============================================================================

std::string MorphOperationToString(MorphOperation op) {
    switch (op) {
        case MorphOperation::Erode:    return "erode";
        case MorphOperation::Dilate:   return "dilate";
        case MorphOperation::Open:     return "open";
        case MorphOperation::Close:    return "close";
        case MorphOperation::Gradient: return "gradient";
        case MorphOperation::TopHat:   return "tophat";
        case MorphOperation::BlackHat: return "blackhat";
        default:                       return "erode";
    }
}

MorphOperation StringToMorphOperation(const std::string& str) {
    if (str == "erode")    return MorphOperation::Erode;
    if (str == "dilate")   return MorphOperation::Dilate;
    if (str == "open")     return MorphOperation::Open;
    if (str == "close")    return MorphOperation::Close;
    if (str == "gradient") return MorphOperation::Gradient;
    if (str == "tophat")   return MorphOperation::TopHat;
    if (str == "blackhat") return MorphOperation::BlackHat;
    return MorphOperation::Erode;
}

std::string MorphShapeToString(MorphShape shape) {
    switch (shape) {
        case MorphShape::Rect:    return "rect";
        case MorphShape::Cross:   return "cross";
        case MorphShape::Ellipse: return "ellipse";
        default:                  return "rect";
    }
}

MorphShape StringToMorphShape(const std::string& str) {
    if (str == "rect")    return MorphShape::Rect;
    if (str == "cross")   return MorphShape::Cross;
    if (str == "ellipse") return MorphShape::Ellipse;
    return MorphShape::Rect;
}

// ============================================================================
// MorphologyConfig Implementation
// ============================================================================

nlohmann::json MorphologyConfig::ToJson() const {
    return {
        {"enabled", enabled},
        {"operation", MorphOperationToString(operation)},
        {"shape", MorphShapeToString(shape)},
        {"kernel_size", kernel_size},
        {"iterations", iterations}
    };
}

MorphologyConfig MorphologyConfig::FromJson(const nlohmann::json& j) {
    MorphologyConfig config;
    if (j.contains("enabled")) config.enabled = j["enabled"].get<bool>();
    if (j.contains("operation")) config.operation = StringToMorphOperation(j["operation"].get<std::string>());
    if (j.contains("shape")) config.shape = StringToMorphShape(j["shape"].get<std::string>());
    if (j.contains("kernel_size")) config.kernel_size = j["kernel_size"].get<int>();
    if (j.contains("iterations")) config.iterations = j["iterations"].get<int>();
    return config;
}

bool MorphologyConfig::operator==(const MorphologyConfig& other) const {
    return enabled == other.enabled &&
           operation == other.operation &&
           shape == other.shape &&
           kernel_size == other.kernel_size &&
           iterations == other.iterations;
}

// ============================================================================
// MorphologyTransform Implementation
// ============================================================================

bool MorphologyTransform::ValidateInput(const std::vector<float>& image_data,
                                         int width, int height, int channels) {
    if (width <= 0 || height <= 0 || channels <= 0) {
        spdlog::error("MorphologyTransform: Invalid dimensions ({}x{}, {} channels)",
                      width, height, channels);
        return false;
    }

    size_t expected_size = static_cast<size_t>(width) * height * channels;
    if (image_data.size() != expected_size) {
        spdlog::error("MorphologyTransform: Data size mismatch. Expected {}, got {}",
                      expected_size, image_data.size());
        return false;
    }

    if (channels != 1 && channels != 3) {
        spdlog::error("MorphologyTransform: Only 1 or 3 channels supported, got {}", channels);
        return false;
    }

    return true;
}

int MorphologyTransform::ValidateKernelSize(int kernel_size) {
    // Kernel size must be odd and positive
    if (kernel_size < 1) {
        spdlog::warn("MorphologyTransform: Kernel size {} too small, using 3", kernel_size);
        return 3;
    }
    if (kernel_size % 2 == 0) {
        spdlog::warn("MorphologyTransform: Kernel size {} is even, using {}",
                     kernel_size, kernel_size + 1);
        return kernel_size + 1;
    }
    if (kernel_size > 31) {
        spdlog::warn("MorphologyTransform: Kernel size {} too large, using 31", kernel_size);
        return 31;
    }
    return kernel_size;
}

bool MorphologyTransform::Apply(
    std::vector<float>& image_data,
    int width, int height, int channels,
    const MorphologyConfig& config
) {
    if (!config.enabled) {
        return true;  // Nothing to do
    }

    return ApplyOperation(image_data, width, height, channels,
                          config.operation, config.shape,
                          config.kernel_size, config.iterations);
}

bool MorphologyTransform::ApplyOperation(
    std::vector<float>& image_data,
    int width, int height, int channels,
    MorphOperation operation,
    MorphShape shape,
    int kernel_size,
    int iterations
) {
    if (!ValidateInput(image_data, width, height, channels)) {
        return false;
    }

    kernel_size = ValidateKernelSize(kernel_size);
    iterations = std::max(1, iterations);

    try {
        // Convert MorphShape to OpenCV shape
        int cv_shape;
        switch (shape) {
            case MorphShape::Rect:    cv_shape = cv::MORPH_RECT; break;
            case MorphShape::Cross:   cv_shape = cv::MORPH_CROSS; break;
            case MorphShape::Ellipse: cv_shape = cv::MORPH_ELLIPSE; break;
            default:                  cv_shape = cv::MORPH_RECT; break;
        }

        // Create structuring element
        cv::Mat kernel = cv::getStructuringElement(
            cv_shape,
            cv::Size(kernel_size, kernel_size)
        );

        // Convert MorphOperation to OpenCV operation
        int cv_operation;
        switch (operation) {
            case MorphOperation::Erode:    cv_operation = cv::MORPH_ERODE; break;
            case MorphOperation::Dilate:   cv_operation = cv::MORPH_DILATE; break;
            case MorphOperation::Open:     cv_operation = cv::MORPH_OPEN; break;
            case MorphOperation::Close:    cv_operation = cv::MORPH_CLOSE; break;
            case MorphOperation::Gradient: cv_operation = cv::MORPH_GRADIENT; break;
            case MorphOperation::TopHat:   cv_operation = cv::MORPH_TOPHAT; break;
            case MorphOperation::BlackHat: cv_operation = cv::MORPH_BLACKHAT; break;
            default:                       cv_operation = cv::MORPH_ERODE; break;
        }

        if (channels == 1) {
            // Grayscale processing
            cv::Mat img(height, width, CV_32FC1, image_data.data());

            // Convert to 8-bit for morphological operations
            cv::Mat uint8_img;
            img.convertTo(uint8_img, CV_8UC1, 255.0);

            // Apply morphological operation
            cv::Mat result;
            cv::morphologyEx(uint8_img, result, cv_operation, kernel,
                             cv::Point(-1, -1), iterations);

            // Convert back to float [0, 1]
            result.convertTo(img, CV_32FC1, 1.0 / 255.0);
            std::memcpy(image_data.data(), img.data, image_data.size() * sizeof(float));

        } else {  // channels == 3
            // RGB processing - apply to each channel
            cv::Mat img(height, width, CV_32FC3, image_data.data());

            // Convert to 8-bit
            cv::Mat uint8_img;
            img.convertTo(uint8_img, CV_8UC3, 255.0);

            // Apply morphological operation
            cv::Mat result;
            cv::morphologyEx(uint8_img, result, cv_operation, kernel,
                             cv::Point(-1, -1), iterations);

            // Convert back to float [0, 1]
            result.convertTo(img, CV_32FC3, 1.0 / 255.0);
            std::memcpy(image_data.data(), img.data, image_data.size() * sizeof(float));
        }

        spdlog::debug("MorphologyTransform: Applied {} (kernel={}, shape={}, iterations={})",
                      MorphOperationToString(operation), kernel_size,
                      MorphShapeToString(shape), iterations);
        return true;

    } catch (const cv::Exception& e) {
        spdlog::error("MorphologyTransform: OpenCV exception: {}", e.what());
        return false;
    } catch (const std::exception& e) {
        spdlog::error("MorphologyTransform: Exception: {}", e.what());
        return false;
    }
}

// ============================================================================
// Convenience Methods
// ============================================================================

bool MorphologyTransform::Erode(
    std::vector<float>& image_data,
    int width, int height, int channels,
    int kernel_size, int iterations, MorphShape shape
) {
    return ApplyOperation(image_data, width, height, channels,
                          MorphOperation::Erode, shape, kernel_size, iterations);
}

bool MorphologyTransform::Dilate(
    std::vector<float>& image_data,
    int width, int height, int channels,
    int kernel_size, int iterations, MorphShape shape
) {
    return ApplyOperation(image_data, width, height, channels,
                          MorphOperation::Dilate, shape, kernel_size, iterations);
}

bool MorphologyTransform::Open(
    std::vector<float>& image_data,
    int width, int height, int channels,
    int kernel_size, MorphShape shape
) {
    return ApplyOperation(image_data, width, height, channels,
                          MorphOperation::Open, shape, kernel_size, 1);
}

bool MorphologyTransform::Close(
    std::vector<float>& image_data,
    int width, int height, int channels,
    int kernel_size, MorphShape shape
) {
    return ApplyOperation(image_data, width, height, channels,
                          MorphOperation::Close, shape, kernel_size, 1);
}

bool MorphologyTransform::Gradient(
    std::vector<float>& image_data,
    int width, int height, int channels,
    int kernel_size, MorphShape shape
) {
    return ApplyOperation(image_data, width, height, channels,
                          MorphOperation::Gradient, shape, kernel_size, 1);
}

bool MorphologyTransform::TopHat(
    std::vector<float>& image_data,
    int width, int height, int channels,
    int kernel_size, MorphShape shape
) {
    return ApplyOperation(image_data, width, height, channels,
                          MorphOperation::TopHat, shape, kernel_size, 1);
}

bool MorphologyTransform::BlackHat(
    std::vector<float>& image_data,
    int width, int height, int channels,
    int kernel_size, MorphShape shape
) {
    return ApplyOperation(image_data, width, height, channels,
                          MorphOperation::BlackHat, shape, kernel_size, 1);
}

} // namespace cyxwiz
