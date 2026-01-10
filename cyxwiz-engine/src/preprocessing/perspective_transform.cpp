#include "perspective_transform.h"
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <cstring>
#include <cmath>

namespace cyxwiz {

// ============================================================================
// Enum String Conversions
// ============================================================================

std::string CoordinateTransformToString(CoordinateTransform type) {
    switch (type) {
        case CoordinateTransform::Perspective: return "perspective";
        case CoordinateTransform::Affine:      return "affine";
        case CoordinateTransform::LogPolar:    return "log_polar";
        case CoordinateTransform::LinearPolar: return "linear_polar";
        default:                               return "perspective";
    }
}

CoordinateTransform StringToCoordinateTransform(const std::string& str) {
    if (str == "perspective")  return CoordinateTransform::Perspective;
    if (str == "affine")       return CoordinateTransform::Affine;
    if (str == "log_polar")    return CoordinateTransform::LogPolar;
    if (str == "linear_polar") return CoordinateTransform::LinearPolar;
    return CoordinateTransform::Perspective;
}

std::string WarpInterpolationToString(WarpInterpolation interp) {
    switch (interp) {
        case WarpInterpolation::Nearest: return "nearest";
        case WarpInterpolation::Linear:  return "linear";
        case WarpInterpolation::Cubic:   return "cubic";
        case WarpInterpolation::Lanczos: return "lanczos";
        default:                         return "linear";
    }
}

WarpInterpolation StringToWarpInterpolation(const std::string& str) {
    if (str == "nearest") return WarpInterpolation::Nearest;
    if (str == "linear")  return WarpInterpolation::Linear;
    if (str == "cubic")   return WarpInterpolation::Cubic;
    if (str == "lanczos") return WarpInterpolation::Lanczos;
    return WarpInterpolation::Linear;
}

// ============================================================================
// PerspectiveConfig Implementation
// ============================================================================

nlohmann::json PerspectiveConfig::ToJson() const {
    return {
        {"enabled", enabled},
        {"type", CoordinateTransformToString(type)},
        {"interpolation", WarpInterpolationToString(interpolation)},
        {"src_points", src_points},
        {"dst_points", dst_points},
        {"center_x", center_x},
        {"center_y", center_y},
        {"max_radius", max_radius},
        {"inverse_polar", inverse_polar},
        {"output_width", output_width},
        {"output_height", output_height}
    };
}

PerspectiveConfig PerspectiveConfig::FromJson(const nlohmann::json& j) {
    PerspectiveConfig config;
    if (j.contains("enabled")) config.enabled = j["enabled"].get<bool>();
    if (j.contains("type")) config.type = StringToCoordinateTransform(j["type"].get<std::string>());
    if (j.contains("interpolation")) config.interpolation = StringToWarpInterpolation(j["interpolation"].get<std::string>());
    if (j.contains("src_points")) config.src_points = j["src_points"].get<std::array<float, 8>>();
    if (j.contains("dst_points")) config.dst_points = j["dst_points"].get<std::array<float, 8>>();
    if (j.contains("center_x")) config.center_x = j["center_x"].get<float>();
    if (j.contains("center_y")) config.center_y = j["center_y"].get<float>();
    if (j.contains("max_radius")) config.max_radius = j["max_radius"].get<float>();
    if (j.contains("inverse_polar")) config.inverse_polar = j["inverse_polar"].get<bool>();
    if (j.contains("output_width")) config.output_width = j["output_width"].get<int>();
    if (j.contains("output_height")) config.output_height = j["output_height"].get<int>();
    return config;
}

bool PerspectiveConfig::operator==(const PerspectiveConfig& other) const {
    return enabled == other.enabled &&
           type == other.type &&
           interpolation == other.interpolation &&
           src_points == other.src_points &&
           dst_points == other.dst_points &&
           std::abs(center_x - other.center_x) < 0.001f &&
           std::abs(center_y - other.center_y) < 0.001f &&
           std::abs(max_radius - other.max_radius) < 0.001f &&
           inverse_polar == other.inverse_polar &&
           output_width == other.output_width &&
           output_height == other.output_height;
}

// ============================================================================
// PerspectiveTransform Implementation
// ============================================================================

bool PerspectiveTransform::ValidateInput(const std::vector<float>& image_data,
                                          int width, int height, int channels) {
    if (width <= 0 || height <= 0 || channels <= 0) {
        spdlog::error("PerspectiveTransform: Invalid dimensions ({}x{}, {} channels)",
                      width, height, channels);
        return false;
    }

    size_t expected_size = static_cast<size_t>(width) * height * channels;
    if (image_data.size() != expected_size) {
        spdlog::error("PerspectiveTransform: Data size mismatch. Expected {}, got {}",
                      expected_size, image_data.size());
        return false;
    }

    if (channels != 1 && channels != 3) {
        spdlog::error("PerspectiveTransform: Only 1 or 3 channels supported, got {}", channels);
        return false;
    }

    return true;
}

int PerspectiveTransform::GetCVInterpolation(WarpInterpolation interp) {
    switch (interp) {
        case WarpInterpolation::Nearest: return cv::INTER_NEAREST;
        case WarpInterpolation::Linear:  return cv::INTER_LINEAR;
        case WarpInterpolation::Cubic:   return cv::INTER_CUBIC;
        case WarpInterpolation::Lanczos: return cv::INTER_LANCZOS4;
        default:                         return cv::INTER_LINEAR;
    }
}

bool PerspectiveTransform::Apply(
    std::vector<float>& image_data,
    int& width, int& height, int channels,
    const PerspectiveConfig& config
) {
    if (!config.enabled) {
        return true;
    }

    switch (config.type) {
        case CoordinateTransform::Perspective:
            return WarpPerspective(image_data, width, height, channels,
                                   config.src_points, config.dst_points,
                                   config.output_width, config.output_height,
                                   config.interpolation);

        case CoordinateTransform::Affine:
            return WarpAffine(image_data, width, height, channels,
                             config.src_points, config.dst_points,
                             config.output_width, config.output_height,
                             config.interpolation);

        case CoordinateTransform::LogPolar:
            return LogPolar(image_data, width, height, channels,
                           config.center_x, config.center_y,
                           config.max_radius, config.inverse_polar);

        case CoordinateTransform::LinearPolar:
            return LinearPolar(image_data, width, height, channels,
                              config.center_x, config.center_y,
                              config.max_radius, config.inverse_polar);

        default:
            spdlog::error("PerspectiveTransform: Unknown transform type");
            return false;
    }
}

bool PerspectiveTransform::WarpPerspective(
    std::vector<float>& image_data,
    int& width, int& height, int channels,
    const std::array<float, 8>& src_pts,
    const std::array<float, 8>& dst_pts,
    int out_w, int out_h,
    WarpInterpolation interp
) {
    if (!ValidateInput(image_data, width, height, channels)) {
        return false;
    }

    // Determine output size
    int new_width = out_w > 0 ? out_w : width;
    int new_height = out_h > 0 ? out_h : height;

    try {
        // Convert normalized points to pixel coordinates
        std::vector<cv::Point2f> src_points(4);
        std::vector<cv::Point2f> dst_points(4);

        for (int i = 0; i < 4; ++i) {
            src_points[i] = cv::Point2f(src_pts[i * 2] * width, src_pts[i * 2 + 1] * height);
            dst_points[i] = cv::Point2f(dst_pts[i * 2] * new_width, dst_pts[i * 2 + 1] * new_height);
        }

        // Get perspective transformation matrix
        cv::Mat transform = cv::getPerspectiveTransform(src_points, dst_points);

        int cv_interp = GetCVInterpolation(interp);

        if (channels == 1) {
            cv::Mat img(height, width, CV_32FC1, image_data.data());
            cv::Mat result;
            cv::warpPerspective(img, result, transform, cv::Size(new_width, new_height),
                               cv_interp, cv::BORDER_CONSTANT, cv::Scalar(0));

            // Resize output vector and copy
            image_data.resize(static_cast<size_t>(new_width) * new_height);
            std::memcpy(image_data.data(), result.data, image_data.size() * sizeof(float));
        } else {
            cv::Mat img(height, width, CV_32FC3, image_data.data());
            cv::Mat result;
            cv::warpPerspective(img, result, transform, cv::Size(new_width, new_height),
                               cv_interp, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

            image_data.resize(static_cast<size_t>(new_width) * new_height * channels);
            std::memcpy(image_data.data(), result.data, image_data.size() * sizeof(float));
        }

        width = new_width;
        height = new_height;

        spdlog::debug("PerspectiveTransform: Applied perspective warp ({}x{})",
                      new_width, new_height);
        return true;

    } catch (const cv::Exception& e) {
        spdlog::error("PerspectiveTransform: OpenCV exception in WarpPerspective: {}", e.what());
        return false;
    }
}

bool PerspectiveTransform::WarpAffine(
    std::vector<float>& image_data,
    int& width, int& height, int channels,
    const std::array<float, 8>& src_pts,
    const std::array<float, 8>& dst_pts,
    int out_w, int out_h,
    WarpInterpolation interp
) {
    if (!ValidateInput(image_data, width, height, channels)) {
        return false;
    }

    int new_width = out_w > 0 ? out_w : width;
    int new_height = out_h > 0 ? out_h : height;

    try {
        // Convert normalized points to pixel coordinates (only use first 3 points)
        std::vector<cv::Point2f> src_points(3);
        std::vector<cv::Point2f> dst_points(3);

        for (int i = 0; i < 3; ++i) {
            src_points[i] = cv::Point2f(src_pts[i * 2] * width, src_pts[i * 2 + 1] * height);
            dst_points[i] = cv::Point2f(dst_pts[i * 2] * new_width, dst_pts[i * 2 + 1] * new_height);
        }

        // Get affine transformation matrix
        cv::Mat transform = cv::getAffineTransform(src_points, dst_points);

        int cv_interp = GetCVInterpolation(interp);

        if (channels == 1) {
            cv::Mat img(height, width, CV_32FC1, image_data.data());
            cv::Mat result;
            cv::warpAffine(img, result, transform, cv::Size(new_width, new_height),
                          cv_interp, cv::BORDER_CONSTANT, cv::Scalar(0));

            image_data.resize(static_cast<size_t>(new_width) * new_height);
            std::memcpy(image_data.data(), result.data, image_data.size() * sizeof(float));
        } else {
            cv::Mat img(height, width, CV_32FC3, image_data.data());
            cv::Mat result;
            cv::warpAffine(img, result, transform, cv::Size(new_width, new_height),
                          cv_interp, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

            image_data.resize(static_cast<size_t>(new_width) * new_height * channels);
            std::memcpy(image_data.data(), result.data, image_data.size() * sizeof(float));
        }

        width = new_width;
        height = new_height;

        spdlog::debug("PerspectiveTransform: Applied affine warp ({}x{})",
                      new_width, new_height);
        return true;

    } catch (const cv::Exception& e) {
        spdlog::error("PerspectiveTransform: OpenCV exception in WarpAffine: {}", e.what());
        return false;
    }
}

bool PerspectiveTransform::LogPolar(
    std::vector<float>& image_data,
    int& width, int& height, int channels,
    float center_x, float center_y,
    float max_radius, bool inverse
) {
    if (!ValidateInput(image_data, width, height, channels)) {
        return false;
    }

    try {
        // Calculate center in pixel coordinates
        cv::Point2f center(center_x * width, center_y * height);

        // Calculate maximum radius in pixels
        float radius = max_radius * std::min(width, height);

        int flags = cv::WARP_FILL_OUTLIERS;
        if (inverse) {
            flags |= cv::WARP_INVERSE_MAP;
        }

        if (channels == 1) {
            cv::Mat img(height, width, CV_32FC1, image_data.data());

            // Convert to 8-bit for warpPolar
            cv::Mat uint8_img;
            img.convertTo(uint8_img, CV_8UC1, 255.0);

            cv::Mat result;
            cv::warpPolar(uint8_img, result, cv::Size(width, height),
                         center, radius, cv::WARP_POLAR_LOG | flags);

            result.convertTo(img, CV_32FC1, 1.0 / 255.0);
            std::memcpy(image_data.data(), img.data, image_data.size() * sizeof(float));
        } else {
            cv::Mat img(height, width, CV_32FC3, image_data.data());

            cv::Mat uint8_img;
            img.convertTo(uint8_img, CV_8UC3, 255.0);

            cv::Mat result;
            cv::warpPolar(uint8_img, result, cv::Size(width, height),
                         center, radius, cv::WARP_POLAR_LOG | flags);

            result.convertTo(img, CV_32FC3, 1.0 / 255.0);
            std::memcpy(image_data.data(), img.data, image_data.size() * sizeof(float));
        }

        spdlog::debug("PerspectiveTransform: Applied log-polar transform (center=({},{}), radius={})",
                      center_x, center_y, max_radius);
        return true;

    } catch (const cv::Exception& e) {
        spdlog::error("PerspectiveTransform: OpenCV exception in LogPolar: {}", e.what());
        return false;
    }
}

bool PerspectiveTransform::LinearPolar(
    std::vector<float>& image_data,
    int& width, int& height, int channels,
    float center_x, float center_y,
    float max_radius, bool inverse
) {
    if (!ValidateInput(image_data, width, height, channels)) {
        return false;
    }

    try {
        cv::Point2f center(center_x * width, center_y * height);
        float radius = max_radius * std::min(width, height);

        int flags = cv::WARP_FILL_OUTLIERS;
        if (inverse) {
            flags |= cv::WARP_INVERSE_MAP;
        }

        if (channels == 1) {
            cv::Mat img(height, width, CV_32FC1, image_data.data());

            cv::Mat uint8_img;
            img.convertTo(uint8_img, CV_8UC1, 255.0);

            cv::Mat result;
            cv::warpPolar(uint8_img, result, cv::Size(width, height),
                         center, radius, cv::WARP_POLAR_LINEAR | flags);

            result.convertTo(img, CV_32FC1, 1.0 / 255.0);
            std::memcpy(image_data.data(), img.data, image_data.size() * sizeof(float));
        } else {
            cv::Mat img(height, width, CV_32FC3, image_data.data());

            cv::Mat uint8_img;
            img.convertTo(uint8_img, CV_8UC3, 255.0);

            cv::Mat result;
            cv::warpPolar(uint8_img, result, cv::Size(width, height),
                         center, radius, cv::WARP_POLAR_LINEAR | flags);

            result.convertTo(img, CV_32FC3, 1.0 / 255.0);
            std::memcpy(image_data.data(), img.data, image_data.size() * sizeof(float));
        }

        spdlog::debug("PerspectiveTransform: Applied linear-polar transform (center=({},{}), radius={})",
                      center_x, center_y, max_radius);
        return true;

    } catch (const cv::Exception& e) {
        spdlog::error("PerspectiveTransform: OpenCV exception in LinearPolar: {}", e.what());
        return false;
    }
}

bool PerspectiveTransform::Rotate(
    std::vector<float>& image_data,
    int& width, int& height, int channels,
    float angle, float scale, bool crop
) {
    if (!ValidateInput(image_data, width, height, channels)) {
        return false;
    }

    try {
        cv::Point2f center(width / 2.0f, height / 2.0f);
        cv::Mat rotation = cv::getRotationMatrix2D(center, angle, scale);

        int new_width = width;
        int new_height = height;

        if (!crop) {
            // Calculate new bounding box size
            double abs_cos = std::abs(rotation.at<double>(0, 0));
            double abs_sin = std::abs(rotation.at<double>(0, 1));
            new_width = static_cast<int>(height * abs_sin + width * abs_cos);
            new_height = static_cast<int>(height * abs_cos + width * abs_sin);

            // Adjust rotation matrix for new center
            rotation.at<double>(0, 2) += (new_width - width) / 2.0;
            rotation.at<double>(1, 2) += (new_height - height) / 2.0;
        }

        if (channels == 1) {
            cv::Mat img(height, width, CV_32FC1, image_data.data());
            cv::Mat result;
            cv::warpAffine(img, result, rotation, cv::Size(new_width, new_height),
                          cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

            image_data.resize(static_cast<size_t>(new_width) * new_height);
            std::memcpy(image_data.data(), result.data, image_data.size() * sizeof(float));
        } else {
            cv::Mat img(height, width, CV_32FC3, image_data.data());
            cv::Mat result;
            cv::warpAffine(img, result, rotation, cv::Size(new_width, new_height),
                          cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

            image_data.resize(static_cast<size_t>(new_width) * new_height * channels);
            std::memcpy(image_data.data(), result.data, image_data.size() * sizeof(float));
        }

        width = new_width;
        height = new_height;

        spdlog::debug("PerspectiveTransform: Rotated by {} degrees (scale={}, crop={})",
                      angle, scale, crop);
        return true;

    } catch (const cv::Exception& e) {
        spdlog::error("PerspectiveTransform: OpenCV exception in Rotate: {}", e.what());
        return false;
    }
}

std::array<float, 8> PerspectiveTransform::DetectDocumentCorners(
    const std::vector<float>& image_data,
    int width, int height, int channels
) {
    // Default identity corners
    std::array<float, 8> corners = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};

    if (width <= 0 || height <= 0 || channels <= 0) {
        return corners;
    }

    try {
        // Convert to grayscale
        cv::Mat img;
        if (channels == 1) {
            img = cv::Mat(height, width, CV_32FC1, const_cast<float*>(image_data.data()));
        } else {
            cv::Mat color(height, width, CV_32FC3, const_cast<float*>(image_data.data()));
            cv::cvtColor(color, img, cv::COLOR_RGB2GRAY);
        }

        // Convert to 8-bit
        cv::Mat uint8_img;
        img.convertTo(uint8_img, CV_8UC1, 255.0);

        // Apply edge detection
        cv::Mat edges;
        cv::Canny(uint8_img, edges, 50, 150);

        // Dilate edges to close gaps
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(edges, edges, kernel);

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (contours.empty()) {
            spdlog::debug("PerspectiveTransform: No contours found for document detection");
            return corners;
        }

        // Find the largest contour
        double max_area = 0;
        int max_idx = 0;
        for (size_t i = 0; i < contours.size(); ++i) {
            double area = cv::contourArea(contours[i]);
            if (area > max_area) {
                max_area = area;
                max_idx = static_cast<int>(i);
            }
        }

        // Approximate to polygon
        std::vector<cv::Point> approx;
        double epsilon = 0.02 * cv::arcLength(contours[max_idx], true);
        cv::approxPolyDP(contours[max_idx], approx, epsilon, true);

        // Check if we have 4 corners
        if (approx.size() != 4) {
            spdlog::debug("PerspectiveTransform: Document detection found {} corners, need 4",
                          approx.size());
            return corners;
        }

        // Sort corners: top-left, top-right, bottom-right, bottom-left
        std::sort(approx.begin(), approx.end(), [](const cv::Point& a, const cv::Point& b) {
            return a.y < b.y;
        });

        cv::Point tl, tr, bl, br;
        if (approx[0].x < approx[1].x) {
            tl = approx[0];
            tr = approx[1];
        } else {
            tl = approx[1];
            tr = approx[0];
        }
        if (approx[2].x < approx[3].x) {
            bl = approx[2];
            br = approx[3];
        } else {
            bl = approx[3];
            br = approx[2];
        }

        // Normalize to 0-1 range
        corners[0] = static_cast<float>(tl.x) / width;
        corners[1] = static_cast<float>(tl.y) / height;
        corners[2] = static_cast<float>(tr.x) / width;
        corners[3] = static_cast<float>(tr.y) / height;
        corners[4] = static_cast<float>(br.x) / width;
        corners[5] = static_cast<float>(br.y) / height;
        corners[6] = static_cast<float>(bl.x) / width;
        corners[7] = static_cast<float>(bl.y) / height;

        spdlog::debug("PerspectiveTransform: Detected document corners");
        return corners;

    } catch (const cv::Exception& e) {
        spdlog::error("PerspectiveTransform: Exception in DetectDocumentCorners: {}", e.what());
        return corners;
    }
}

} // namespace cyxwiz
