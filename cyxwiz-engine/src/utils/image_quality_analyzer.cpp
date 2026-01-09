#include "image_quality_analyzer.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

namespace cyxwiz {

// ============================================================================
// Static Members
// ============================================================================

ImageQualityAnalyzer::Thresholds ImageQualityAnalyzer::thresholds_;

// ============================================================================
// ImageQualityAnalyzer - Public API
// ============================================================================

QualityMetrics ImageQualityAnalyzer::Analyze(
    const std::vector<float>& image_data,
    int width, int height, int channels
) {
    QualityMetrics metrics;

    if (image_data.empty() || width <= 0 || height <= 0 || channels <= 0) {
        spdlog::error("ImageQualityAnalyzer::Analyze: Invalid input parameters");
        metrics.issue_flags = static_cast<uint32_t>(QualityIssue::Corrupted);
        return metrics;
    }

    size_t expected_size = static_cast<size_t>(width) * height * channels;
    if (image_data.size() != expected_size) {
        spdlog::error("ImageQualityAnalyzer::Analyze: Image data size mismatch (expected {}, got {})",
                      expected_size, image_data.size());
        metrics.issue_flags = static_cast<uint32_t>(QualityIssue::Corrupted);
        return metrics;
    }

    try {
        // Convert to OpenCV Mat (float 0-1 range → uint8 0-255 range)
        cv::Mat image;
        if (channels == 1) {
            image = cv::Mat(height, width, CV_32FC1, const_cast<float*>(image_data.data())).clone();
            image.convertTo(image, CV_8UC1, 255.0);
        } else if (channels == 3) {
            image = cv::Mat(height, width, CV_32FC3, const_cast<float*>(image_data.data())).clone();
            image.convertTo(image, CV_8UC3, 255.0);
        } else {
            spdlog::error("ImageQualityAnalyzer::Analyze: Unsupported channel count ({})", channels);
            metrics.issue_flags = static_cast<uint32_t>(QualityIssue::Corrupted);
            return metrics;
        }

        // Convert to grayscale for analysis
        cv::Mat gray;
        if (channels == 1) {
            gray = image.clone();
        } else {
            cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
        }

        // 1. Compute blur score (Laplacian variance)
        metrics.blur_score = ComputeBlurScore(image_data, width, height, channels);
        if (metrics.blur_score < thresholds_.blur_threshold) {
            metrics.issue_flags |= static_cast<uint32_t>(QualityIssue::Blurry);
        }

        // 2. Compute sharpness (Canny edge strength)
        metrics.sharpness_score = ComputeSharpness(image_data, width, height, channels);
        if (metrics.sharpness_score < thresholds_.sharpness_threshold) {
            metrics.issue_flags |= static_cast<uint32_t>(QualityIssue::LowSharpness);
        }

        // 3. Compute exposure score (histogram analysis)
        metrics.exposure_score = ComputeExposure(image_data, width, height, channels);

        // Check for overexposure (too many bright pixels)
        cv::Mat hist;
        int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};
        cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

        float total_pixels = static_cast<float>(width * height);
        float bright_pixels = 0.0f;
        float dark_pixels = 0.0f;

        for (int i = 230; i < 256; ++i) {
            bright_pixels += hist.at<float>(i);
        }
        for (int i = 0; i < 25; ++i) {
            dark_pixels += hist.at<float>(i);
        }

        float bright_ratio = bright_pixels / total_pixels;
        float dark_ratio = dark_pixels / total_pixels;

        if (bright_ratio > thresholds_.overexposure_pct) {
            metrics.issue_flags |= static_cast<uint32_t>(QualityIssue::Overexposed);
        }
        if (dark_ratio > thresholds_.underexposure_pct) {
            metrics.issue_flags |= static_cast<uint32_t>(QualityIssue::Underexposed);
        }

        // 4. Compute mean brightness
        metrics.mean_brightness = ComputeMeanBrightness(image_data, width, height, channels);

        // Check for unusual brightness values
        if (metrics.mean_brightness < thresholds_.brightness_low ||
            metrics.mean_brightness > thresholds_.brightness_high) {
            // Already flagged via exposure analysis
        }

    } catch (const cv::Exception& e) {
        spdlog::error("ImageQualityAnalyzer::Analyze: OpenCV exception: {}", e.what());
        metrics.issue_flags = static_cast<uint32_t>(QualityIssue::Corrupted);
    } catch (const std::exception& e) {
        spdlog::error("ImageQualityAnalyzer::Analyze: Exception: {}", e.what());
        metrics.issue_flags = static_cast<uint32_t>(QualityIssue::Corrupted);
    }

    return metrics;
}

std::vector<QualityMetrics> ImageQualityAnalyzer::AnalyzeBatch(
    const std::vector<std::vector<float>>& images,
    int width, int height, int channels,
    std::function<void(float progress, const std::string& msg)> progress_callback
) {
    std::vector<QualityMetrics> results;
    results.reserve(images.size());

    if (progress_callback) {
        progress_callback(0.0f, "Starting quality analysis...");
    }

    for (size_t i = 0; i < images.size(); ++i) {
        QualityMetrics metrics = Analyze(images[i], width, height, channels);
        results.push_back(metrics);

        if (progress_callback && (i % 10 == 0 || i == images.size() - 1)) {
            float progress = static_cast<float>(i + 1) / images.size();
            std::string msg = std::format("Analyzed {}/{} images", i + 1, images.size());
            progress_callback(progress, msg);
        }
    }

    if (progress_callback) {
        progress_callback(1.0f, "Quality analysis complete!");
    }

    return results;
}

// ============================================================================
// Private Helper Methods
// ============================================================================

float ImageQualityAnalyzer::ComputeBlurScore(
    const std::vector<float>& image,
    int width, int height, int channels
) {
    try {
        // Convert to OpenCV Mat
        cv::Mat img;
        if (channels == 1) {
            img = cv::Mat(height, width, CV_32FC1, const_cast<float*>(image.data())).clone();
            img.convertTo(img, CV_8UC1, 255.0);
        } else {
            img = cv::Mat(height, width, CV_32FC3, const_cast<float*>(image.data())).clone();
            cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);
            img.convertTo(img, CV_8UC1, 255.0);
        }

        // Compute Laplacian
        cv::Mat laplacian;
        cv::Laplacian(img, laplacian, CV_64F);

        // Compute variance (standard deviation squared)
        cv::Scalar mean, stddev;
        cv::meanStdDev(laplacian, mean, stddev);

        return static_cast<float>(stddev[0] * stddev[0]);

    } catch (const cv::Exception& e) {
        spdlog::error("ComputeBlurScore: OpenCV exception: {}", e.what());
        return 0.0f;
    }
}

float ImageQualityAnalyzer::ComputeSharpness(
    const std::vector<float>& image,
    int width, int height, int channels
) {
    try {
        // Convert to OpenCV Mat
        cv::Mat img;
        if (channels == 1) {
            img = cv::Mat(height, width, CV_32FC1, const_cast<float*>(image.data())).clone();
            img.convertTo(img, CV_8UC1, 255.0);
        } else {
            img = cv::Mat(height, width, CV_32FC3, const_cast<float*>(image.data())).clone();
            cv::Mat gray_temp;
            cv::cvtColor(img, gray_temp, cv::COLOR_RGB2GRAY);
            img = gray_temp;
            img.convertTo(img, CV_8UC1, 255.0);
        }

        // Apply Canny edge detection
        cv::Mat edges;
        cv::Canny(img, edges, 50, 150);

        // Compute ratio of edge pixels to total pixels
        int edge_count = cv::countNonZero(edges);
        float total_pixels = static_cast<float>(width * height);

        return edge_count / total_pixels;

    } catch (const cv::Exception& e) {
        spdlog::error("ComputeSharpness: OpenCV exception: {}", e.what());
        return 0.0f;
    }
}

float ImageQualityAnalyzer::ComputeExposure(
    const std::vector<float>& image,
    int width, int height, int channels
) {
    try {
        // Compute mean brightness
        float brightness = ComputeMeanBrightness(image, width, height, channels);

        // Normalize to 0-1 range (0.5 = ideal exposure)
        // brightness is in 0-255 range, so 127.5 = ideal
        return brightness / 255.0f;

    } catch (const std::exception& e) {
        spdlog::error("ComputeExposure: Exception: {}", e.what());
        return 0.5f;
    }
}

float ImageQualityAnalyzer::ComputeMeanBrightness(
    const std::vector<float>& image,
    int width, int height, int channels
) {
    if (image.empty()) return 0.0f;

    float sum = 0.0f;
    size_t pixel_count = width * height;

    if (channels == 1) {
        // Grayscale: average all pixels
        for (size_t i = 0; i < pixel_count; ++i) {
            sum += image[i] * 255.0f;  // Convert to 0-255 range
        }
        return sum / pixel_count;
    } else if (channels == 3) {
        // RGB: compute luminosity (weighted average)
        for (size_t i = 0; i < pixel_count; ++i) {
            float r = image[i * 3 + 0] * 255.0f;
            float g = image[i * 3 + 1] * 255.0f;
            float b = image[i * 3 + 2] * 255.0f;
            sum += 0.299f * r + 0.587f * g + 0.114f * b;
        }
        return sum / pixel_count;
    }

    return 0.0f;
}

} // namespace cyxwiz
