#pragma once

#include <vector>
#include <string>
#include <functional>
#include <cstdint>

namespace cyxwiz {

/**
 * Image quality issues (bitmask flags)
 */
enum class QualityIssue : uint32_t {
    None = 0,
    Blurry = 1 << 0,          // Low Laplacian variance
    Overexposed = 1 << 1,     // Too many bright pixels
    Underexposed = 1 << 2,    // Too many dark pixels
    LowSharpness = 1 << 3,    // Low edge strength
    Corrupted = 1 << 4        // Unusual pixel distribution
};

// Bitwise operators for QualityIssue
inline QualityIssue operator|(QualityIssue a, QualityIssue b) {
    return static_cast<QualityIssue>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline QualityIssue operator&(QualityIssue a, QualityIssue b) {
    return static_cast<QualityIssue>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

inline uint32_t operator~(QualityIssue a) {
    return ~static_cast<uint32_t>(a);
}

/**
 * Quality metrics for a single image
 */
struct QualityMetrics {
    float blur_score = 0.0f;           // Laplacian variance (higher = sharper)
    float sharpness_score = 0.0f;      // Edge strength (0-1)
    float exposure_score = 0.0f;       // 0-1 (0.5 = ideal, <0.2 = underexposed, >0.8 = overexposed)
    float mean_brightness = 0.0f;      // 0-255
    uint32_t issue_flags = 0;          // QualityIssue bitmask

    // Check for specific issues
    bool IsBlurry() const { return (issue_flags & static_cast<uint32_t>(QualityIssue::Blurry)) != 0; }
    bool IsOverexposed() const { return (issue_flags & static_cast<uint32_t>(QualityIssue::Overexposed)) != 0; }
    bool IsUnderexposed() const { return (issue_flags & static_cast<uint32_t>(QualityIssue::Underexposed)) != 0; }
    bool HasLowSharpness() const { return (issue_flags & static_cast<uint32_t>(QualityIssue::LowSharpness)) != 0; }
    bool IsCorrupted() const { return (issue_flags & static_cast<uint32_t>(QualityIssue::Corrupted)) != 0; }
    bool HasIssues() const { return issue_flags != 0; }

    // Get issue count
    int GetIssueCount() const {
        int count = 0;
        if (IsBlurry()) count++;
        if (IsOverexposed()) count++;
        if (IsUnderexposed()) count++;
        if (HasLowSharpness()) count++;
        if (IsCorrupted()) count++;
        return count;
    }

    // Get issue description
    std::string GetIssueDescription() const {
        if (!HasIssues()) return "No issues";
        std::string desc;
        if (IsBlurry()) desc += "Blurry, ";
        if (IsOverexposed()) desc += "Overexposed, ";
        if (IsUnderexposed()) desc += "Underexposed, ";
        if (HasLowSharpness()) desc += "Low Sharpness, ";
        if (IsCorrupted()) desc += "Corrupted, ";
        if (!desc.empty()) desc = desc.substr(0, desc.length() - 2);  // Remove trailing ", "
        return desc;
    }
};

/**
 * Image quality analyzer using OpenCV
 */
class ImageQualityAnalyzer {
public:
    /**
     * Quality detection thresholds
     */
    struct Thresholds {
        float blur_threshold = 100.0f;          // Laplacian variance (< threshold = blurry)
        float sharpness_threshold = 0.3f;       // Edge strength (< threshold = low sharpness)
        float overexposure_pct = 0.80f;         // Histogram % (> threshold = overexposed)
        float underexposure_pct = 0.80f;        // Histogram % (> threshold = underexposed)
        float brightness_low = 30.0f;           // Mean brightness (< threshold = dark)
        float brightness_high = 225.0f;         // Mean brightness (> threshold = bright)
    };

    /**
     * Analyze a single image for quality issues
     * @param image_data Float image data (0-1 range) [HWC format]
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels (1=grayscale, 3=RGB)
     * @return Quality metrics
     */
    static QualityMetrics Analyze(
        const std::vector<float>& image_data,
        int width, int height, int channels
    );

    /**
     * Analyze a batch of images
     * @param images Vector of image data
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @param progress_callback Optional progress callback (progress, message)
     * @return Vector of quality metrics
     */
    static std::vector<QualityMetrics> AnalyzeBatch(
        const std::vector<std::vector<float>>& images,
        int width, int height, int channels,
        std::function<void(float progress, const std::string& msg)> progress_callback = nullptr
    );

    /**
     * Set quality detection thresholds
     */
    static void SetThresholds(const Thresholds& thresholds) {
        thresholds_ = thresholds;
    }

    /**
     * Get current thresholds
     */
    static Thresholds GetThresholds() {
        return thresholds_;
    }

private:
    // Compute blur score using Laplacian variance
    static float ComputeBlurScore(const std::vector<float>& image, int width, int height, int channels);

    // Compute sharpness using Canny edge detection
    static float ComputeSharpness(const std::vector<float>& image, int width, int height, int channels);

    // Compute exposure score from histogram
    static float ComputeExposure(const std::vector<float>& image, int width, int height, int channels);

    // Compute mean brightness
    static float ComputeMeanBrightness(const std::vector<float>& image, int width, int height, int channels);

    // Quality thresholds
    static Thresholds thresholds_;
};

} // namespace cyxwiz
