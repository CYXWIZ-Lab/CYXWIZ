#include "scaling_transform.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

namespace cyxwiz {

ScalingTransform::ScalingTransform(const ScalingConfig& config)
    : config_(config) {
}

void ScalingTransform::Initialize(const DatasetStatistics& stats) {
    if (config_.strategy == ScalingStrategy::None) {
        initialized_ = false;
        return;
    }

    if (!stats.is_valid) {
        spdlog::warn("ScalingTransform: Invalid statistics, cannot initialize");
        initialized_ = false;
        return;
    }

    // Cache statistics based on strategy
    switch (config_.strategy) {
        case ScalingStrategy::MinMax:
            data_min_ = stats.min;
            data_max_ = stats.max;
            spdlog::info("ScalingTransform: Initialized MinMax scaling ({} channels)", data_min_.size());
            break;

        case ScalingStrategy::Standard:
            data_mean_ = stats.mean;
            data_std_ = stats.std;
            spdlog::info("ScalingTransform: Initialized Standard scaling ({} channels)", data_mean_.size());
            break;

        case ScalingStrategy::Robust:
            data_median_ = stats.median;
            data_q25_ = stats.q25;
            data_q75_ = stats.q75;
            spdlog::info("ScalingTransform: Initialized Robust scaling ({} channels)", data_median_.size());
            break;

        case ScalingStrategy::PCAWhitening:
            spdlog::warn("ScalingTransform: PCA Whitening not yet implemented, falling back to Standard scaling");
            data_mean_ = stats.mean;
            data_std_ = stats.std;
            break;

        default:
            initialized_ = false;
            return;
    }

    initialized_ = true;
}

Tensor ScalingTransform::Apply(const Tensor& input) {
    if (!initialized_) {
        spdlog::warn("ScalingTransform: Not initialized, returning input unchanged");
        return input;
    }

    switch (config_.strategy) {
        case ScalingStrategy::MinMax:
            return ApplyMinMax(input);
        case ScalingStrategy::Standard:
            return ApplyStandard(input);
        case ScalingStrategy::Robust:
            return ApplyRobust(input);
        case ScalingStrategy::PCAWhitening:
            return ApplyPCAWhitening(input);
        default:
            return input;
    }
}

Tensor ScalingTransform::ApplyMinMax(const Tensor& input) {
    // Create output tensor (copy of input)
    Tensor output = input.Clone();
    const auto& shape = output.Shape();
    size_t num_elements = output.NumElements();
    float* data = output.Data<float>();
    if (!data) {
        spdlog::error("ScalingTransform: Failed to get tensor data");
        return input;
    }

    // Determine number of channels
    size_t num_channels = 1;
    if (shape.size() >= 3) {
        num_channels = shape[2];
    }

    // Ensure min/max match number of channels
    if (data_min_.size() != num_channels) {
        if (data_min_.size() == 1 && num_channels > 1) {
            data_min_.resize(num_channels, data_min_[0]);
            data_max_.resize(num_channels, data_max_[0]);
        } else {
            spdlog::error("ScalingTransform: Min/Max size mismatch");
            return input;
        }
    }

    // Apply MinMax scaling: (x - min) / (max - min) * (target_max - target_min) + target_min
    size_t values_per_channel = num_elements / num_channels;
    for (size_t i = 0; i < values_per_channel; ++i) {
        for (size_t ch = 0; ch < num_channels; ++ch) {
            size_t idx = i * num_channels + ch;
            if (idx < num_elements) {
                float min_val = data_min_[ch];
                float max_val = data_max_[ch];
                float range = max_val - min_val;

                if (range < config_.epsilon) {
                    // Avoid division by zero
                    data[idx] = config_.min_value;
                } else {
                    // Scale to [0, 1] then to [min_value, max_value]
                    float normalized = (data[idx] - min_val) / range;
                    data[idx] = normalized * (config_.max_value - config_.min_value) + config_.min_value;
                }
            }
        }
    }

    return output;
}

Tensor ScalingTransform::ApplyStandard(const Tensor& input) {
    // Create output tensor (copy of input)
    Tensor output = input.Clone();
    const auto& shape = output.Shape();
    size_t num_elements = output.NumElements();
    float* data = output.Data<float>();
    if (!data) {
        spdlog::error("ScalingTransform: Failed to get tensor data");
        return input;
    }

    // Determine number of channels
    size_t num_channels = 1;
    if (shape.size() >= 3) {
        num_channels = shape[2];
    }

    // Ensure mean/std match number of channels
    if (data_mean_.size() != num_channels) {
        if (data_mean_.size() == 1 && num_channels > 1) {
            data_mean_.resize(num_channels, data_mean_[0]);
            data_std_.resize(num_channels, data_std_[0]);
        } else {
            spdlog::error("ScalingTransform: Mean/Std size mismatch");
            return input;
        }
    }

    // Apply Standard scaling: (x - mean) / std
    size_t values_per_channel = num_elements / num_channels;
    for (size_t i = 0; i < values_per_channel; ++i) {
        for (size_t ch = 0; ch < num_channels; ++ch) {
            size_t idx = i * num_channels + ch;
            if (idx < num_elements) {
                float mean = data_mean_[ch];
                float std = data_std_[ch];

                if (std < config_.epsilon) {
                    // Avoid division by zero
                    data[idx] = 0.0f;
                } else {
                    data[idx] = (data[idx] - mean) / std;
                }
            }
        }
    }

    return output;
}

Tensor ScalingTransform::ApplyRobust(const Tensor& input) {
    // Create output tensor (copy of input)
    Tensor output = input.Clone();
    const auto& shape = output.Shape();
    size_t num_elements = output.NumElements();
    float* data = output.Data<float>();
    if (!data) {
        spdlog::error("ScalingTransform: Failed to get tensor data");
        return input;
    }

    // Determine number of channels
    size_t num_channels = 1;
    if (shape.size() >= 3) {
        num_channels = shape[2];
    }

    // Ensure percentiles match number of channels
    if (data_median_.size() != num_channels) {
        if (data_median_.size() == 1 && num_channels > 1) {
            data_median_.resize(num_channels, data_median_[0]);
            data_q25_.resize(num_channels, data_q25_[0]);
            data_q75_.resize(num_channels, data_q75_[0]);
        } else {
            spdlog::error("ScalingTransform: Percentile size mismatch");
            return input;
        }
    }

    // Apply Robust scaling: (x - median) / (Q75 - Q25)
    size_t values_per_channel = num_elements / num_channels;
    for (size_t i = 0; i < values_per_channel; ++i) {
        for (size_t ch = 0; ch < num_channels; ++ch) {
            size_t idx = i * num_channels + ch;
            if (idx < num_elements) {
                float median = data_median_[ch];
                float iqr = data_q75_[ch] - data_q25_[ch];  // Interquartile range

                if (iqr < config_.epsilon) {
                    // Avoid division by zero
                    data[idx] = 0.0f;
                } else {
                    data[idx] = (data[idx] - median) / iqr;
                }
            }
        }
    }

    return output;
}

Tensor ScalingTransform::ApplyPCAWhitening(const Tensor& input) {
    // TODO: Implement PCA Whitening
    // Requires eigenvalue decomposition and matrix operations
    // For now, fall back to Standard scaling
    spdlog::warn("ScalingTransform: PCA Whitening not implemented, using Standard scaling");
    return ApplyStandard(input);
}

} // namespace cyxwiz
