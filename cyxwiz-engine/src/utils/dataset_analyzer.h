#pragma once

#include <vector>
#include <map>
#include <string>
#include <functional>
#include <cstdint>

namespace cyxwiz {

// Forward declarations
class DataRegistry;

/**
 * Color histogram statistics
 */
struct ColorHistogram {
    std::vector<int> red_bins;      // 256 bins
    std::vector<int> green_bins;    // 256 bins
    std::vector<int> blue_bins;     // 256 bins
    std::vector<int> gray_bins;     // 256 bins
    size_t total_pixels = 0;

    bool is_grayscale = false;      // True if image is grayscale

    ColorHistogram() {
        red_bins.resize(256, 0);
        green_bins.resize(256, 0);
        blue_bins.resize(256, 0);
        gray_bins.resize(256, 0);
    }
};

/**
 * Class distribution statistics
 */
struct ClassDistribution {
    std::map<int, size_t> counts;           // label → count
    std::map<int, float> percentages;       // label → percentage
    size_t total_samples = 0;
    int num_classes = 0;

    // Compute imbalance ratio (max_count / min_count)
    float GetImbalanceRatio() const;

    // Check if balanced (imbalance ratio < threshold)
    bool IsBalanced(float threshold = 2.0f) const;
};

/**
 * Dimension statistics
 */
struct DimensionStatistics {
    int min_width = 0, max_width = 0, mean_width = 0;
    int min_height = 0, max_height = 0, mean_height = 0;
    int min_channels = 0, max_channels = 0, mode_channels = 0;

    // Dimension frequency map: (width, height) → count
    std::map<std::pair<int, int>, size_t> dimension_counts;

    // Check if all images have same dimensions
    bool IsUniform() const {
        return dimension_counts.size() == 1;
    }
};

/**
 * Outlier detection information
 */
struct OutlierInfo {
    enum class OutlierReason {
        Brightness,     // Unusually bright/dark
        Contrast,       // Very high/low contrast
        Size,           // Different dimensions
        ChannelCount    // Different number of channels
    };

    std::vector<size_t> outlier_indices;                // Sample indices
    std::vector<float> outlier_scores;                  // Outlier scores (0-1)
    std::map<size_t, OutlierReason> outlier_reasons;    // Index → reason

    size_t GetCount() const { return outlier_indices.size(); }
};

/**
 * Complete dataset analytics
 */
struct DatasetAnalytics {
    ColorHistogram color_histogram;
    ClassDistribution class_distribution;
    DimensionStatistics dimension_stats;
    OutlierInfo outliers;

    float mean_brightness = 0.0f;
    float std_brightness = 0.0f;
    float mean_contrast = 0.0f;
    float std_contrast = 0.0f;

    bool success = false;
    std::string error_message;
};

/**
 * Dataset analyzer for computing statistics
 */
class DatasetAnalyzer {
public:
    /**
     * Compute complete analytics for a loaded dataset
     * @param dataset_name Name of dataset in registry
     * @param registry Pointer to DataRegistry
     * @param progress_callback Optional progress callback (progress, message)
     * @return Complete analytics results
     */
    static DatasetAnalytics ComputeAnalytics(
        const std::string& dataset_name,
        DataRegistry* registry,
        std::function<void(float progress, const std::string& msg)> progress_callback = nullptr
    );

    /**
     * Compute color histogram from image batch
     * @param images Vector of image data (each image is float vector [0-1])
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels (1=grayscale, 3=RGB)
     * @return Color histogram
     */
    static ColorHistogram ComputeColorHistogram(
        const std::vector<std::vector<float>>& images,
        int width, int height, int channels
    );

    /**
     * Detect outliers using IQR method
     * @param images Vector of image data
     * @param channels Number of channels
     * @param iqr_multiplier IQR multiplier for outlier threshold (default 1.5)
     * @return Outlier information
     */
    static OutlierInfo DetectOutliers(
        const std::vector<std::vector<float>>& images,
        int channels,
        float iqr_multiplier = 1.5f
    );

    /**
     * Compute class distribution from labels
     * @param labels Vector of class labels
     * @return Class distribution statistics
     */
    static ClassDistribution ComputeClassDistribution(
        const std::vector<int>& labels
    );

    /**
     * Compute dimension statistics
     * @param dataset_name Name of dataset in registry
     * @param registry Pointer to DataRegistry
     * @return Dimension statistics
     */
    static DimensionStatistics ComputeDimensionStats(
        const std::string& dataset_name,
        DataRegistry* registry
    );

private:
    // Helper: Compute brightness of a single image
    static float ComputeBrightness(const std::vector<float>& image, int channels);

    // Helper: Compute contrast of a single image
    static float ComputeContrast(const std::vector<float>& image);
};

} // namespace cyxwiz
