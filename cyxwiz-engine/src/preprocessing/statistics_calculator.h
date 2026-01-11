#pragma once

#include "preprocessing_config.h"
#include "../core/data_registry.h"
#include <cyxwiz/tensor.h>
#include <vector>
#include <unordered_map>
#include <functional>
#include <mutex>
#include <string>

namespace cyxwiz {

/**
 * Dataset statistics for preprocessing
 */
struct DatasetStatistics {
    std::vector<float> mean;        // Per-channel mean
    std::vector<float> std;         // Per-channel std
    std::vector<float> min;         // Per-channel min
    std::vector<float> max;         // Per-channel max
    std::vector<float> median;      // Per-channel median
    std::vector<float> q25;         // 25th percentile
    std::vector<float> q75;         // 75th percentile

    // Histograms (per-channel, 256 bins for images)
    std::vector<std::vector<int>> histograms;

    // PCA whitening components
    std::vector<std::vector<float>> pca_components;  // Eigenvectors [n_features, n_components]
    std::vector<float> pca_eigenvalues;              // Eigenvalues (variance explained)
    std::vector<float> pca_mean;                     // Mean to center data before PCA
    bool pca_computed = false;

    size_t num_samples = 0;
    std::vector<size_t> shape;      // Data shape (e.g., [H, W, C])

    bool is_valid = false;

    // Helper: Get number of channels
    size_t GetNumChannels() const {
        return mean.size();
    }

    // Helper: Check if data is in [0, 1] range
    bool IsNormalized() const {
        for (size_t i = 0; i < min.size(); ++i) {
            if (min[i] < -0.1f || max[i] > 1.1f) {
                return false;
            }
        }
        return true;
    }

    // Helper: Get average mean across channels
    float GetAverageMean() const {
        if (mean.empty()) return 0.0f;
        float sum = 0.0f;
        for (float m : mean) sum += m;
        return sum / mean.size();
    }

    // Helper: Get average std across channels
    float GetAverageStd() const {
        if (std.empty()) return 1.0f;
        float sum = 0.0f;
        for (float s : std) sum += s;
        return sum / std.size();
    }
};

/**
 * Statistics calculator for datasets
 * Computes statistical properties needed for preprocessing
 */
class StatisticsCalculator {
public:
    /**
     * Compute statistics from dataset (progress callback for UI)
     * @param dataset_id ID of dataset to compute statistics for
     * @param registry DataRegistry containing the dataset
     * @param progress_callback Optional callback for progress updates (0.0 to 1.0)
     * @return Computed statistics
     */
    static DatasetStatistics Compute(
        const std::string& dataset_id,
        DataRegistry* registry,
        std::function<void(float)> progress_callback = nullptr
    );

    /**
     * Auto-detect dataset type based on statistics and metadata
     * @param stats Computed statistics
     * @param dataset_name Name of dataset (for keyword matching)
     * @return Recommended normalization strategy
     */
    static NormalizationStrategy AutoDetectStrategy(
        const DatasetStatistics& stats,
        const std::string& dataset_name
    );

    /**
     * Get cached statistics (returns invalid stats if not cached)
     * @param dataset_id ID of dataset
     * @return Cached statistics or invalid stats
     */
    static DatasetStatistics GetCached(const std::string& dataset_id);

    /**
     * Clear cache for dataset
     * @param dataset_id ID of dataset to clear cache for
     */
    static void ClearCache(const std::string& dataset_id);

    /**
     * Clear all cached statistics
     */
    static void ClearAllCache();

private:
    // Cache statistics to avoid recomputation
    static std::unordered_map<std::string, DatasetStatistics> cache_;
    static std::mutex cache_mutex_;

    // Helper: Compute histogram for single channel
    static std::vector<int> ComputeHistogram(const std::vector<float>& data, int num_bins = 256);

    // Helper: Compute percentile
    static float ComputePercentile(std::vector<float> data, float percentile);

    // Helper: Compute mean
    static float ComputeMean(const std::vector<float>& data);

    // Helper: Compute standard deviation
    static float ComputeStd(const std::vector<float>& data, float mean);

    // Helper: Convert dataset name to lowercase
    static std::string ToLowerCase(const std::string& str);
};

} // namespace cyxwiz
