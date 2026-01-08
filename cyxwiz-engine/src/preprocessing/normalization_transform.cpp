#include "normalization_transform.h"
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace cyxwiz {

NormalizationTransform::NormalizationTransform(const NormalizationConfig& config)
    : config_(config) {

    // For non-Custom strategies, load preset immediately
    if (config_.strategy != NormalizationStrategy::None &&
        config_.strategy != NormalizationStrategy::AutoDetect &&
        config_.strategy != NormalizationStrategy::Custom) {
        LoadPreset(config_.strategy);
        initialized_ = true;
    }
}

void NormalizationTransform::Initialize(const DatasetStatistics& stats) {
    if (config_.strategy == NormalizationStrategy::None) {
        initialized_ = false;
        return;
    }

    if (config_.strategy == NormalizationStrategy::Custom) {
        // Use custom mean/std from config, or fall back to computed stats
        if (!config_.custom_mean.empty() && !config_.custom_std.empty()) {
            mean_ = config_.custom_mean;
            std_ = config_.custom_std;
            spdlog::info("NormalizationTransform: Using custom mean/std (size: {})", mean_.size());
        } else if (stats.is_valid) {
            // Use computed statistics
            mean_ = stats.mean;
            std_ = stats.std;
            spdlog::info("NormalizationTransform: Using computed statistics (size: {})", mean_.size());
        } else {
            spdlog::warn("NormalizationTransform: No custom values or valid stats, using default (0, 1)");
            mean_ = {0.0f};
            std_ = {1.0f};
        }
        initialized_ = true;
    } else if (config_.strategy == NormalizationStrategy::AutoDetect) {
        // Auto-detect was already resolved to specific strategy in UI
        // This shouldn't happen, but handle it gracefully
        spdlog::warn("NormalizationTransform: AutoDetect strategy not resolved, defaulting to None");
        initialized_ = false;
    } else {
        // Preset strategies were already loaded in constructor
        initialized_ = true;
    }
}

Tensor NormalizationTransform::Apply(const Tensor& input) {
    if (!initialized_ || mean_.empty() || std_.empty()) {
        spdlog::warn("NormalizationTransform: Not initialized, returning input unchanged");
        return input;
    }

    // Create output tensor (copy of input)
    Tensor output = input.Clone();

    // Get tensor data and shape
    const auto& shape = output.Shape();
    size_t num_elements = output.NumElements();
    float* data = output.Data<float>();
    if (!data) {
        spdlog::error("NormalizationTransform: Failed to get tensor data");
        return input;
    }

    // Determine number of channels
    size_t num_channels = 1;
    if (shape.size() >= 3) {
        num_channels = shape[2];  // Assuming [batch, H, W, C] or [H, W, C]
    }

    // Ensure mean/std match number of channels
    if (mean_.size() != num_channels) {
        if (mean_.size() == 1 && num_channels > 1) {
            // Broadcast single-channel stats to all channels
            mean_.resize(num_channels, mean_[0]);
            std_.resize(num_channels, std_[0]);
        } else {
            spdlog::error("NormalizationTransform: Mean/Std size ({}) doesn't match channels ({})",
                         mean_.size(), num_channels);
            return input;
        }
    }

    // Apply normalization
    if (config_.per_channel && num_channels > 1) {
        // Per-channel normalization
        size_t values_per_channel = num_elements / num_channels;
        for (size_t i = 0; i < values_per_channel; ++i) {
            for (size_t ch = 0; ch < num_channels; ++ch) {
                size_t idx = i * num_channels + ch;
                if (idx < num_elements) {
                    // Avoid division by zero
                    float std_val = std_[ch] > 1e-6f ? std_[ch] : 1.0f;
                    data[idx] = (data[idx] - mean_[ch]) / std_val;
                }
            }
        }
    } else {
        // Global normalization (use first channel's stats for all)
        float mean_val = mean_[0];
        float std_val = std_[0] > 1e-6f ? std_[0] : 1.0f;
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = (data[i] - mean_val) / std_val;
        }
    }

    return output;
}

void NormalizationTransform::LoadPreset(NormalizationStrategy strategy) {
    switch (strategy) {
        case NormalizationStrategy::MNIST:
            // MNIST dataset statistics
            mean_ = {0.1307f};
            std_ = {0.3081f};
            spdlog::info("NormalizationTransform: Loaded MNIST preset (mean=0.1307, std=0.3081)");
            break;

        case NormalizationStrategy::CIFAR10:
            // CIFAR-10 dataset statistics (per-channel)
            mean_ = {0.4914f, 0.4822f, 0.4465f};
            std_ = {0.2470f, 0.2435f, 0.2616f};
            spdlog::info("NormalizationTransform: Loaded CIFAR-10 preset (3 channels)");
            break;

        case NormalizationStrategy::ImageNet:
            // ImageNet dataset statistics (per-channel)
            mean_ = {0.485f, 0.456f, 0.406f};
            std_ = {0.229f, 0.224f, 0.225f};
            spdlog::info("NormalizationTransform: Loaded ImageNet preset (3 channels)");
            break;

        default:
            // No preset for None, AutoDetect, Custom
            mean_ = {0.0f};
            std_ = {1.0f};
            spdlog::warn("NormalizationTransform: No preset available for strategy, using default");
            break;
    }
}

} // namespace cyxwiz
