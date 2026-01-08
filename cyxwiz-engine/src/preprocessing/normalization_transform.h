#pragma once

#include "preprocessing_config.h"
#include "statistics_calculator.h"
#include <cyxwiz/tensor.h>
#include <vector>
#include <memory>

namespace cyxwiz {

/**
 * Normalization transform for preprocessing
 * Applies (x - mean) / std normalization
 */
class NormalizationTransform {
public:
    /**
     * Create normalization transform with config
     * @param config Normalization configuration
     */
    explicit NormalizationTransform(const NormalizationConfig& config);

    /**
     * Initialize transform with dataset statistics
     * @param stats Dataset statistics (for Custom strategy)
     */
    void Initialize(const DatasetStatistics& stats);

    /**
     * Apply normalization to input tensor
     * @param input Input tensor [batch_size, ...] or [..., channels]
     * @return Normalized tensor
     */
    Tensor Apply(const Tensor& input);

    /**
     * Get current mean values
     */
    const std::vector<float>& GetMean() const { return mean_; }

    /**
     * Get current std values
     */
    const std::vector<float>& GetStd() const { return std_; }

    /**
     * Check if transform is initialized
     */
    bool IsInitialized() const { return initialized_; }

private:
    NormalizationConfig config_;
    std::vector<float> mean_;  // Computed or preset mean
    std::vector<float> std_;   // Computed or preset std
    bool initialized_ = false;

    /**
     * Load preset mean/std values based on strategy
     * @param strategy Normalization strategy
     */
    void LoadPreset(NormalizationStrategy strategy);
};

} // namespace cyxwiz
