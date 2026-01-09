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

        case ScalingStrategy::MaxAbs:
            // Compute max absolute value per channel
            data_max_abs_.resize(stats.mean.size());
            for (size_t ch = 0; ch < stats.mean.size(); ++ch) {
                data_max_abs_[ch] = std::max(std::abs(stats.min[ch]), std::abs(stats.max[ch]));
            }
            spdlog::info("ScalingTransform: Initialized MaxAbs scaling ({} channels)", data_max_abs_.size());
            break;

        case ScalingStrategy::Quantile:
            // Quantile transform doesn't need pre-computed statistics
            // It computes ranks dynamically per batch
            spdlog::info("ScalingTransform: Initialized Quantile transform (mode={})",
                         config_.quantile_use_normal ? "normal" : "uniform");
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
        case ScalingStrategy::MaxAbs:
            return ApplyMaxAbs(input);
        case ScalingStrategy::Quantile:
            return ApplyQuantile(input);
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

Tensor ScalingTransform::ApplyMaxAbs(const Tensor& input) {
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

    // Ensure max_abs matches number of channels
    if (data_max_abs_.size() != num_channels) {
        if (data_max_abs_.size() == 1 && num_channels > 1) {
            data_max_abs_.resize(num_channels, data_max_abs_[0]);
        } else {
            spdlog::error("ScalingTransform: MaxAbs size mismatch");
            return input;
        }
    }

    // Apply MaxAbs scaling: x_scaled = x / max(|x|)
    size_t values_per_channel = num_elements / num_channels;
    for (size_t i = 0; i < values_per_channel; ++i) {
        for (size_t ch = 0; ch < num_channels; ++ch) {
            size_t idx = i * num_channels + ch;
            if (idx < num_elements) {
                float max_abs = data_max_abs_[ch];

                if (max_abs < config_.epsilon) {
                    // All values are zero
                    data[idx] = 0.0f;
                } else {
                    data[idx] = data[idx] / max_abs;
                }
            }
        }
    }

    return output;
}

Tensor ScalingTransform::ApplyQuantile(const Tensor& input) {
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

    size_t values_per_channel = num_elements / num_channels;

    // Process each channel independently
    for (size_t ch = 0; ch < num_channels; ++ch) {
        // Collect channel data
        std::vector<std::pair<float, size_t>> channel_data;
        channel_data.reserve(values_per_channel);

        for (size_t i = 0; i < values_per_channel; ++i) {
            size_t idx = i * num_channels + ch;
            if (idx < num_elements) {
                channel_data.push_back({data[idx], idx});
            }
        }

        // Sort by value
        std::sort(channel_data.begin(), channel_data.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        // Assign ranks and transform
        for (size_t i = 0; i < channel_data.size(); ++i) {
            size_t idx = channel_data[i].second;
            double quantile = static_cast<double>(i) / (channel_data.size() - 1);

            if (config_.quantile_use_normal) {
                // Transform to standard normal distribution
                // Using Beasley-Springer-Moro approximation for inverse normal CDF
                double t = quantile;
                if (t <= 0.0) t = 0.0001;
                if (t >= 1.0) t = 0.9999;

                // Approximation coefficients
                double a0 = 2.515517, a1 = 0.802853, a2 = 0.010328;
                double b1 = 1.432788, b2 = 0.189269, b3 = 0.001308;

                double p = t < 0.5 ? t : 1 - t;
                double s = std::sqrt(-2.0 * std::log(p));
                double z = s - (a0 + a1 * s + a2 * s * s) /
                              (1 + b1 * s + b2 * s * s + b3 * s * s * s);

                data[idx] = static_cast<float>((t < 0.5) ? -z : z);
            } else {
                // Transform to uniform distribution [0, 1]
                data[idx] = static_cast<float>(quantile);
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
