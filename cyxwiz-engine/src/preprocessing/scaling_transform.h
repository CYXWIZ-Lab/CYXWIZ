#pragma once

#include "preprocessing_config.h"
#include "statistics_calculator.h"
#include <cyxwiz/tensor.h>
#include <vector>
#include <memory>

namespace cyxwiz {

/**
 * Scaling transform for preprocessing
 * Supports MinMax, Standard, Robust, and PCA Whitening strategies
 */
class ScalingTransform {
public:
    /**
     * Create scaling transform with config
     * @param config Scaling configuration
     */
    explicit ScalingTransform(const ScalingConfig& config);

    /**
     * Initialize transform with dataset statistics
     * @param stats Dataset statistics (required for all strategies)
     */
    void Initialize(const DatasetStatistics& stats);

    /**
     * Apply scaling to input tensor
     * @param input Input tensor [batch_size, ...] or [..., channels]
     * @return Scaled tensor
     */
    Tensor Apply(const Tensor& input);

    /**
     * Check if transform is initialized
     */
    bool IsInitialized() const { return initialized_; }

private:
    ScalingConfig config_;

    // Cached statistics for scaling
    std::vector<float> data_min_;
    std::vector<float> data_max_;
    std::vector<float> data_mean_;
    std::vector<float> data_std_;
    std::vector<float> data_median_;
    std::vector<float> data_q25_;
    std::vector<float> data_q75_;
    std::vector<float> data_max_abs_;  // For MaxAbs scaling

    // PCA whitening parameters (not implemented yet - placeholder)
    // TODO: Implement PCA whitening using ArrayFire or Eigen
    bool pca_computed_ = false;

    bool initialized_ = false;

    /**
     * Apply MinMax scaling
     */
    Tensor ApplyMinMax(const Tensor& input);

    /**
     * Apply Standard scaling (z-score)
     */
    Tensor ApplyStandard(const Tensor& input);

    /**
     * Apply Robust scaling (median-based)
     */
    Tensor ApplyRobust(const Tensor& input);

    /**
     * Apply MaxAbs scaling (scale by max absolute value)
     * Formula: x_scaled = x / max(|x|)
     * Result range: [-1, 1]
     */
    Tensor ApplyMaxAbs(const Tensor& input);

    /**
     * Apply Quantile transform
     * Maps data to uniform [0, 1] or normal N(0, 1) distribution
     */
    Tensor ApplyQuantile(const Tensor& input);

    /**
     * Apply PCA Whitening (TODO: implement)
     */
    Tensor ApplyPCAWhitening(const Tensor& input);
};

} // namespace cyxwiz
