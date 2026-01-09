#include "image_utils.h"
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>

namespace cyxwiz {

// =============================================================================
// Image Loading
// =============================================================================

bool ImageUtils::LoadImage(const std::string& path,
                            std::vector<float>& data,
                            int& width, int& height, int& channels) {
    // Load image using OpenCV (BGR format)
    cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);

    if (img.empty()) {
        spdlog::error("ImageUtils: Failed to load image: {}", path);
        return false;
    }

    width = img.cols;
    height = img.rows;
    channels = img.channels();

    // Convert BGR to RGB if color image
    if (channels == 3 || channels == 4) {
        if (channels == 4) {
            cv::cvtColor(img, img, cv::COLOR_BGRA2RGB);
            channels = 3;  // Drop alpha channel
        } else {
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        }
    }

    // Convert to float [0, 1]
    cv::Mat float_img;
    img.convertTo(float_img, CV_32FC(channels), 1.0 / 255.0);

    // Copy to output vector
    size_t num_pixels = width * height * channels;
    data.resize(num_pixels);
    std::memcpy(data.data(), float_img.data, num_pixels * sizeof(float));

    return true;
}

// =============================================================================
// Image Resizing
// =============================================================================

bool ImageUtils::ResizeImage(std::vector<float>& data,
                              int src_width, int src_height, int channels,
                              int dst_width, int dst_height,
                              ResizeMethod method) {
    if (src_width <= 0 || src_height <= 0 || channels <= 0) {
        spdlog::error("ImageUtils: Invalid source dimensions");
        return false;
    }

    if (dst_width <= 0 || dst_height <= 0) {
        spdlog::error("ImageUtils: Invalid target dimensions");
        return false;
    }

    // If already the correct size, nothing to do
    if (src_width == dst_width && src_height == dst_height) {
        return true;
    }

    // Create source Mat from float data
    cv::Mat src(src_height, src_width, CV_32FC(channels), data.data());

    // Map ResizeMethod to OpenCV interpolation flags
    int interp_flag;
    switch (method) {
        case ResizeMethod::Nearest:
            interp_flag = cv::INTER_NEAREST;
            break;
        case ResizeMethod::Bilinear:
            interp_flag = cv::INTER_LINEAR;
            break;
        case ResizeMethod::Bicubic:
            interp_flag = cv::INTER_CUBIC;
            break;
        case ResizeMethod::Lanczos:
            interp_flag = cv::INTER_LANCZOS4;
            break;
        case ResizeMethod::Area:
            interp_flag = cv::INTER_AREA;
            break;
        default:
            interp_flag = cv::INTER_CUBIC;
            break;
    }

    // Resize using OpenCV
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(dst_width, dst_height), 0, 0, interp_flag);

    // Copy back to output vector
    size_t num_pixels = dst_width * dst_height * channels;
    data.resize(num_pixels);
    std::memcpy(data.data(), dst.data, num_pixels * sizeof(float));

    return true;
}

// =============================================================================
// Color Space Conversion
// =============================================================================

bool ImageUtils::ConvertColorSpace(std::vector<float>& data,
                                    int width, int height,
                                    ColorSpace from, ColorSpace to) {
    if (from == to) {
        return true;  // No conversion needed
    }

    if (width <= 0 || height <= 0) {
        spdlog::error("ImageUtils: Invalid dimensions for color conversion");
        return false;
    }

    // Determine source channels
    int src_channels;
    switch (from) {
        case ColorSpace::RGB:
        case ColorSpace::BGR:
        case ColorSpace::HSV:
        case ColorSpace::Lab:
        case ColorSpace::YCbCr:
            src_channels = 3;
            break;
        case ColorSpace::Grayscale:
            src_channels = 1;
            break;
        default:
            spdlog::error("ImageUtils: Unknown source color space");
            return false;
    }

    // Check data size
    size_t expected_size = width * height * src_channels;
    if (data.size() != expected_size) {
        spdlog::error("ImageUtils: Data size mismatch for color conversion");
        return false;
    }

    // Create source Mat
    cv::Mat src(height, width, CV_32FC(src_channels), data.data());
    cv::Mat dst;

    // Perform conversion using OpenCV
    int conversion_code = -1;

    // Handle conversions
    if (from == ColorSpace::RGB && to == ColorSpace::BGR) {
        conversion_code = cv::COLOR_RGB2BGR;
    } else if (from == ColorSpace::BGR && to == ColorSpace::RGB) {
        conversion_code = cv::COLOR_BGR2RGB;
    } else if (from == ColorSpace::RGB && to == ColorSpace::Grayscale) {
        conversion_code = cv::COLOR_RGB2GRAY;
    } else if (from == ColorSpace::BGR && to == ColorSpace::Grayscale) {
        conversion_code = cv::COLOR_BGR2GRAY;
    } else if (from == ColorSpace::Grayscale && to == ColorSpace::RGB) {
        conversion_code = cv::COLOR_GRAY2RGB;
    } else if (from == ColorSpace::Grayscale && to == ColorSpace::BGR) {
        conversion_code = cv::COLOR_GRAY2BGR;
    } else if (from == ColorSpace::RGB && to == ColorSpace::HSV) {
        conversion_code = cv::COLOR_RGB2HSV;
    } else if (from == ColorSpace::HSV && to == ColorSpace::RGB) {
        conversion_code = cv::COLOR_HSV2RGB;
    } else if (from == ColorSpace::RGB && to == ColorSpace::Lab) {
        conversion_code = cv::COLOR_RGB2Lab;
    } else if (from == ColorSpace::Lab && to == ColorSpace::RGB) {
        conversion_code = cv::COLOR_Lab2RGB;
    } else if (from == ColorSpace::RGB && to == ColorSpace::YCbCr) {
        conversion_code = cv::COLOR_RGB2YCrCb;
    } else if (from == ColorSpace::YCbCr && to == ColorSpace::RGB) {
        conversion_code = cv::COLOR_YCrCb2RGB;
    } else {
        spdlog::error("ImageUtils: Unsupported color conversion: {} -> {}",
                      static_cast<int>(from), static_cast<int>(to));
        return false;
    }

    cv::cvtColor(src, dst, conversion_code);

    // Copy back to output vector
    int dst_channels = dst.channels();
    size_t num_pixels = width * height * dst_channels;
    data.resize(num_pixels);
    std::memcpy(data.data(), dst.data, num_pixels * sizeof(float));

    return true;
}

// =============================================================================
// Pixel Normalization
// =============================================================================

void ImageUtils::NormalizePixels(const unsigned char* src, float* dst,
                                 size_t count, float scale) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = static_cast<float>(src[i]) * scale;
    }
}

void ImageUtils::FloatToUint8(const float* src, unsigned char* dst,
                              size_t count, float scale) {
    for (size_t i = 0; i < count; ++i) {
        float val = src[i] * scale;
        // Clamp to [0, 255]
        val = std::max(0.0f, std::min(255.0f, val));
        dst[i] = static_cast<unsigned char>(val);
    }
}

// =============================================================================
// Supported Formats
// =============================================================================

std::vector<std::string> ImageUtils::GetSupportedFormats() {
    // OpenCV supports many formats via imread
    return {
        "jpg", "jpeg", "png", "bmp", "tiff", "tif",
        "webp", "gif", "ppm", "pgm", "pbm", "sr", "ras"
    };
}

// =============================================================================
// Helper Functions (Private)
// =============================================================================

void ImageUtils::MatToFloatVector(const void* mat_ptr, std::vector<float>& data) {
    const cv::Mat* mat = static_cast<const cv::Mat*>(mat_ptr);
    size_t num_elements = mat->total() * mat->channels();
    data.resize(num_elements);
    std::memcpy(data.data(), mat->data, num_elements * sizeof(float));
}

void ImageUtils::FloatVectorToMat(const std::vector<float>& data,
                                  int width, int height, int channels,
                                  void* mat_ptr) {
    cv::Mat* mat = static_cast<cv::Mat*>(mat_ptr);
    *mat = cv::Mat(height, width, CV_32FC(channels), const_cast<float*>(data.data()));
}

} // namespace cyxwiz
