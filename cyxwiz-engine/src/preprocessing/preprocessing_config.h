#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace cyxwiz {

/**
 * Normalization strategies for preprocessing
 */
enum class NormalizationStrategy {
    None,           // No normalization
    AutoDetect,     // Detect dataset type and apply preset
    MNIST,          // MNIST preset (mean=0.1307, std=0.3081)
    CIFAR10,        // CIFAR-10 preset (per-channel ImageNet stats)
    ImageNet,       // ImageNet preset (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    Custom          // User-specified mean/std
};

/**
 * Scaling strategies for preprocessing
 */
enum class ScalingStrategy {
    None,           // No scaling
    MinMax,         // Scale to [min, max] range
    Standard,       // Z-score normalization (mean=0, std=1)
    Robust,         // Median-based scaling (outlier-resistant)
    MaxAbs,         // Scale by max absolute value to [-1, 1]
    Quantile,       // Transform to uniform or normal distribution
    PCAWhitening    // Decorrelate and normalize variance
};

/**
 * Resize modes for image preprocessing
 */
enum class ResizeMode {
    None,           // No resizing
    Exact,          // Resize to exact dimensions (may distort aspect ratio)
    AspectFit,      // Fit inside dimensions (preserve aspect ratio, pad)
    AspectFill,     // Fill dimensions (preserve aspect ratio, crop)
    Center          // Center crop to dimensions
};

/**
 * Image preprocessing configuration
 */
struct ImagePreprocessingConfig {
    ResizeMode resize_mode = ResizeMode::None;
    int target_width = 0;
    int target_height = 0;
    bool convert_to_grayscale = false;
    bool convert_to_rgb = false;  // For grayscale→RGB conversion

    // Image enhancement (applied before normalization/scaling)
    bool enable_clahe = false;
    float clahe_clip_limit = 2.0f;
    int clahe_tile_size = 8;

    bool enable_denoise = false;
    float denoise_strength = 10.0f;

    bool enable_sharpen = false;
    float sharpen_amount = 1.0f;

    nlohmann::json ToJson() const;
    static ImagePreprocessingConfig FromJson(const nlohmann::json& j);

    bool operator==(const ImagePreprocessingConfig& other) const {
        return resize_mode == other.resize_mode &&
               target_width == other.target_width &&
               target_height == other.target_height &&
               convert_to_grayscale == other.convert_to_grayscale &&
               convert_to_rgb == other.convert_to_rgb &&
               enable_clahe == other.enable_clahe &&
               clahe_clip_limit == other.clahe_clip_limit &&
               clahe_tile_size == other.clahe_tile_size &&
               enable_denoise == other.enable_denoise &&
               denoise_strength == other.denoise_strength &&
               enable_sharpen == other.enable_sharpen &&
               sharpen_amount == other.sharpen_amount;
    }
};

/**
 * Normalization configuration
 */
struct NormalizationConfig {
    NormalizationStrategy strategy = NormalizationStrategy::None;
    std::vector<float> custom_mean;   // Per-channel mean (if Custom)
    std::vector<float> custom_std;    // Per-channel std (if Custom)
    bool per_channel = true;          // Apply per-channel or global

    nlohmann::json ToJson() const;
    static NormalizationConfig FromJson(const nlohmann::json& j);

    bool operator==(const NormalizationConfig& other) const {
        return strategy == other.strategy &&
               custom_mean == other.custom_mean &&
               custom_std == other.custom_std &&
               per_channel == other.per_channel;
    }
};

/**
 * Scaling configuration
 */
struct ScalingConfig {
    ScalingStrategy strategy = ScalingStrategy::None;
    float min_value = 0.0f;           // Target min for MinMax
    float max_value = 1.0f;           // Target max for MinMax
    float epsilon = 1e-8f;            // Numerical stability

    // Quantile transform options
    bool quantile_use_normal = false; // true=normal, false=uniform distribution
    int quantile_n_quantiles = 1000;  // Number of quantiles (not used in simple implementation)

    nlohmann::json ToJson() const;
    static ScalingConfig FromJson(const nlohmann::json& j);

    bool operator==(const ScalingConfig& other) const {
        return strategy == other.strategy &&
               min_value == other.min_value &&
               max_value == other.max_value &&
               epsilon == other.epsilon &&
               quantile_use_normal == other.quantile_use_normal &&
               quantile_n_quantiles == other.quantile_n_quantiles;
    }
};

/**
 * Complete preprocessing configuration
 * Applied in order: Image → Normalization → Scaling
 */
struct PreprocessingConfig {
    // Execution order (applied in this sequence):
    ImagePreprocessingConfig image_config;
    NormalizationConfig normalization_config;
    ScalingConfig scaling_config;

    bool enabled = false;
    std::string dataset_id;  // Associated dataset

    nlohmann::json ToJson() const;
    static PreprocessingConfig FromJson(const nlohmann::json& j);

    bool operator==(const PreprocessingConfig& other) const {
        return image_config == other.image_config &&
               normalization_config == other.normalization_config &&
               scaling_config == other.scaling_config &&
               enabled == other.enabled &&
               dataset_id == other.dataset_id;
    }
};

// Helper functions for enum conversion
std::string NormalizationStrategyToString(NormalizationStrategy strategy);
NormalizationStrategy StringToNormalizationStrategy(const std::string& str);

std::string ScalingStrategyToString(ScalingStrategy strategy);
ScalingStrategy StringToScalingStrategy(const std::string& str);

std::string ResizeModeToString(ResizeMode mode);
ResizeMode StringToResizeMode(const std::string& str);

} // namespace cyxwiz
