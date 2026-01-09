#include "preprocessing_config.h"
#include <stdexcept>

namespace cyxwiz {

using json = nlohmann::json;

// ============================================================================
// Enum Conversion Functions
// ============================================================================

std::string NormalizationStrategyToString(NormalizationStrategy strategy) {
    switch (strategy) {
        case NormalizationStrategy::None: return "None";
        case NormalizationStrategy::AutoDetect: return "AutoDetect";
        case NormalizationStrategy::MNIST: return "MNIST";
        case NormalizationStrategy::CIFAR10: return "CIFAR10";
        case NormalizationStrategy::ImageNet: return "ImageNet";
        case NormalizationStrategy::Custom: return "Custom";
        default: return "None";
    }
}

NormalizationStrategy StringToNormalizationStrategy(const std::string& str) {
    if (str == "None") return NormalizationStrategy::None;
    if (str == "AutoDetect") return NormalizationStrategy::AutoDetect;
    if (str == "MNIST") return NormalizationStrategy::MNIST;
    if (str == "CIFAR10") return NormalizationStrategy::CIFAR10;
    if (str == "ImageNet") return NormalizationStrategy::ImageNet;
    if (str == "Custom") return NormalizationStrategy::Custom;
    return NormalizationStrategy::None;
}

std::string ScalingStrategyToString(ScalingStrategy strategy) {
    switch (strategy) {
        case ScalingStrategy::None: return "None";
        case ScalingStrategy::MinMax: return "MinMax";
        case ScalingStrategy::Standard: return "Standard";
        case ScalingStrategy::Robust: return "Robust";
        case ScalingStrategy::MaxAbs: return "MaxAbs";
        case ScalingStrategy::Quantile: return "Quantile";
        case ScalingStrategy::PCAWhitening: return "PCAWhitening";
        default: return "None";
    }
}

ScalingStrategy StringToScalingStrategy(const std::string& str) {
    if (str == "None") return ScalingStrategy::None;
    if (str == "MinMax") return ScalingStrategy::MinMax;
    if (str == "Standard") return ScalingStrategy::Standard;
    if (str == "Robust") return ScalingStrategy::Robust;
    if (str == "MaxAbs") return ScalingStrategy::MaxAbs;
    if (str == "Quantile") return ScalingStrategy::Quantile;
    if (str == "PCAWhitening") return ScalingStrategy::PCAWhitening;
    return ScalingStrategy::None;
}

std::string ResizeModeToString(ResizeMode mode) {
    switch (mode) {
        case ResizeMode::None: return "None";
        case ResizeMode::Exact: return "Exact";
        case ResizeMode::AspectFit: return "AspectFit";
        case ResizeMode::AspectFill: return "AspectFill";
        case ResizeMode::Center: return "Center";
        default: return "None";
    }
}

ResizeMode StringToResizeMode(const std::string& str) {
    if (str == "None") return ResizeMode::None;
    if (str == "Exact") return ResizeMode::Exact;
    if (str == "AspectFit") return ResizeMode::AspectFit;
    if (str == "AspectFill") return ResizeMode::AspectFill;
    if (str == "Center") return ResizeMode::Center;
    return ResizeMode::None;
}

// ============================================================================
// ImagePreprocessingConfig Serialization
// ============================================================================

json ImagePreprocessingConfig::ToJson() const {
    json j;
    j["resize_mode"] = ResizeModeToString(resize_mode);
    j["target_width"] = target_width;
    j["target_height"] = target_height;
    j["convert_to_grayscale"] = convert_to_grayscale;
    j["convert_to_rgb"] = convert_to_rgb;
    return j;
}

ImagePreprocessingConfig ImagePreprocessingConfig::FromJson(const json& j) {
    ImagePreprocessingConfig config;
    if (j.contains("resize_mode")) {
        config.resize_mode = StringToResizeMode(j["resize_mode"].get<std::string>());
    }
    if (j.contains("target_width")) {
        config.target_width = j["target_width"].get<int>();
    }
    if (j.contains("target_height")) {
        config.target_height = j["target_height"].get<int>();
    }
    if (j.contains("convert_to_grayscale")) {
        config.convert_to_grayscale = j["convert_to_grayscale"].get<bool>();
    }
    if (j.contains("convert_to_rgb")) {
        config.convert_to_rgb = j["convert_to_rgb"].get<bool>();
    }
    return config;
}

// ============================================================================
// NormalizationConfig Serialization
// ============================================================================

json NormalizationConfig::ToJson() const {
    json j;
    j["strategy"] = NormalizationStrategyToString(strategy);
    j["custom_mean"] = custom_mean;
    j["custom_std"] = custom_std;
    j["per_channel"] = per_channel;
    return j;
}

NormalizationConfig NormalizationConfig::FromJson(const json& j) {
    NormalizationConfig config;
    if (j.contains("strategy")) {
        config.strategy = StringToNormalizationStrategy(j["strategy"].get<std::string>());
    }
    if (j.contains("custom_mean")) {
        config.custom_mean = j["custom_mean"].get<std::vector<float>>();
    }
    if (j.contains("custom_std")) {
        config.custom_std = j["custom_std"].get<std::vector<float>>();
    }
    if (j.contains("per_channel")) {
        config.per_channel = j["per_channel"].get<bool>();
    }
    return config;
}

// ============================================================================
// ScalingConfig Serialization
// ============================================================================

json ScalingConfig::ToJson() const {
    json j;
    j["strategy"] = ScalingStrategyToString(strategy);
    j["min_value"] = min_value;
    j["max_value"] = max_value;
    j["epsilon"] = epsilon;
    j["quantile_use_normal"] = quantile_use_normal;
    j["quantile_n_quantiles"] = quantile_n_quantiles;
    return j;
}

ScalingConfig ScalingConfig::FromJson(const json& j) {
    ScalingConfig config;
    if (j.contains("strategy")) {
        config.strategy = StringToScalingStrategy(j["strategy"].get<std::string>());
    }
    if (j.contains("min_value")) {
        config.min_value = j["min_value"].get<float>();
    }
    if (j.contains("max_value")) {
        config.max_value = j["max_value"].get<float>();
    }
    if (j.contains("epsilon")) {
        config.epsilon = j["epsilon"].get<float>();
    }
    if (j.contains("quantile_use_normal")) {
        config.quantile_use_normal = j["quantile_use_normal"].get<bool>();
    }
    if (j.contains("quantile_n_quantiles")) {
        config.quantile_n_quantiles = j["quantile_n_quantiles"].get<int>();
    }
    return config;
}

// ============================================================================
// PreprocessingConfig Serialization
// ============================================================================

json PreprocessingConfig::ToJson() const {
    json j;
    j["enabled"] = enabled;
    j["dataset_id"] = dataset_id;
    j["image_config"] = image_config.ToJson();
    j["normalization_config"] = normalization_config.ToJson();
    j["scaling_config"] = scaling_config.ToJson();
    return j;
}

PreprocessingConfig PreprocessingConfig::FromJson(const json& j) {
    PreprocessingConfig config;
    if (j.contains("enabled")) {
        config.enabled = j["enabled"].get<bool>();
    }
    if (j.contains("dataset_id")) {
        config.dataset_id = j["dataset_id"].get<std::string>();
    }
    if (j.contains("image_config")) {
        config.image_config = ImagePreprocessingConfig::FromJson(j["image_config"]);
    }
    if (j.contains("normalization_config")) {
        config.normalization_config = NormalizationConfig::FromJson(j["normalization_config"]);
    }
    if (j.contains("scaling_config")) {
        config.scaling_config = ScalingConfig::FromJson(j["scaling_config"]);
    }
    return config;
}

} // namespace cyxwiz
