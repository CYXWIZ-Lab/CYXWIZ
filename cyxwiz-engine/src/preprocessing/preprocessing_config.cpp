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

std::string EdgeDetectorTypeToString(EdgeDetectorType type) {
    switch (type) {
        case EdgeDetectorType::Canny: return "Canny";
        case EdgeDetectorType::Sobel: return "Sobel";
        case EdgeDetectorType::Laplacian: return "Laplacian";
        case EdgeDetectorType::Scharr: return "Scharr";
        default: return "Canny";
    }
}

EdgeDetectorType StringToEdgeDetectorType(const std::string& str) {
    if (str == "Canny") return EdgeDetectorType::Canny;
    if (str == "Sobel") return EdgeDetectorType::Sobel;
    if (str == "Laplacian") return EdgeDetectorType::Laplacian;
    if (str == "Scharr") return EdgeDetectorType::Scharr;
    return EdgeDetectorType::Canny;
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
    j["enable_clahe"] = enable_clahe;
    j["clahe_clip_limit"] = clahe_clip_limit;
    j["clahe_tile_size"] = clahe_tile_size;
    j["enable_denoise"] = enable_denoise;
    j["denoise_strength"] = denoise_strength;
    j["enable_sharpen"] = enable_sharpen;
    j["sharpen_amount"] = sharpen_amount;
    j["enable_edge_detection"] = enable_edge_detection;
    j["edge_detector_type"] = EdgeDetectorTypeToString(edge_detector_type);
    j["edge_threshold1"] = edge_threshold1;
    j["edge_threshold2"] = edge_threshold2;
    j["edge_kernel_size"] = edge_kernel_size;
    // OpenCV Transform Configs (Phase 1)
    j["morphology_config"] = morphology_config.ToJson();
    j["blur_config"] = blur_config.ToJson();
    j["perspective_config"] = perspective_config.ToJson();
    j["pyramid_config"] = pyramid_config.ToJson();
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
    if (j.contains("enable_clahe")) {
        config.enable_clahe = j["enable_clahe"].get<bool>();
    }
    if (j.contains("clahe_clip_limit")) {
        config.clahe_clip_limit = j["clahe_clip_limit"].get<float>();
    }
    if (j.contains("clahe_tile_size")) {
        config.clahe_tile_size = j["clahe_tile_size"].get<int>();
    }
    if (j.contains("enable_denoise")) {
        config.enable_denoise = j["enable_denoise"].get<bool>();
    }
    if (j.contains("denoise_strength")) {
        config.denoise_strength = j["denoise_strength"].get<float>();
    }
    if (j.contains("enable_sharpen")) {
        config.enable_sharpen = j["enable_sharpen"].get<bool>();
    }
    if (j.contains("sharpen_amount")) {
        config.sharpen_amount = j["sharpen_amount"].get<float>();
    }
    if (j.contains("enable_edge_detection")) {
        config.enable_edge_detection = j["enable_edge_detection"].get<bool>();
    }
    if (j.contains("edge_detector_type")) {
        config.edge_detector_type = StringToEdgeDetectorType(j["edge_detector_type"].get<std::string>());
    }
    if (j.contains("edge_threshold1")) {
        config.edge_threshold1 = j["edge_threshold1"].get<float>();
    }
    if (j.contains("edge_threshold2")) {
        config.edge_threshold2 = j["edge_threshold2"].get<float>();
    }
    if (j.contains("edge_kernel_size")) {
        config.edge_kernel_size = j["edge_kernel_size"].get<int>();
    }
    // OpenCV Transform Configs (Phase 1)
    if (j.contains("morphology_config")) {
        config.morphology_config = MorphologyConfig::FromJson(j["morphology_config"]);
    }
    if (j.contains("blur_config")) {
        config.blur_config = BlurConfig::FromJson(j["blur_config"]);
    }
    if (j.contains("perspective_config")) {
        config.perspective_config = PerspectiveConfig::FromJson(j["perspective_config"]);
    }
    if (j.contains("pyramid_config")) {
        config.pyramid_config = PyramidConfig::FromJson(j["pyramid_config"]);
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
