#pragma once

/**
 * CyxWiz Image Transform Library
 *
 * A comprehensive data augmentation system for machine learning.
 * Supports geometric transforms, color adjustments, noise/blur,
 * and advanced techniques like Mixup, CutMix, and RandAugment.
 *
 * Usage:
 * ```cpp
 * #include "transforms/transforms.h"
 * using namespace cyxwiz::transforms;
 *
 * // Single transform
 * RandomHorizontalFlip flip(0.5f);
 * Image augmented = flip.apply(input);
 *
 * // Compose multiple transforms
 * auto pipeline = Compose();
 * pipeline.add(std::make_unique<Resize>(224, 224));
 * pipeline.add(std::make_unique<RandomHorizontalFlip>(0.5f));
 * pipeline.add(std::make_unique<ColorJitter>(0.4f, 0.4f, 0.4f, 0.1f));
 * pipeline.add(std::make_unique<Normalize::ImageNet>());
 *
 * Image result = pipeline.apply(input);
 *
 * // Use presets
 * auto train_transform = TransformFactory::createTrainTransform(224);
 * auto val_transform = TransformFactory::createValTransform(224);
 * ```
 */

#include "transform.h"
#include "geometric.h"
#include "color.h"
#include "noise.h"
#include "advanced.h"

#include <memory>
#include <string>
#include <map>
#include <functional>

namespace cyxwiz {
namespace transforms {

/**
 * TransformFactory - Create transforms by name or preset pipelines
 */
class TransformFactory {
public:
    /**
     * Create a transform by name
     * Supported names:
     *   Geometric: Resize, CenterCrop, RandomCrop, RandomResizedCrop,
     *              HorizontalFlip, VerticalFlip, RandomHorizontalFlip,
     *              RandomVerticalFlip, Rotate, RandomRotation, RandomAffine
     *   Color:     Normalize, AdjustBrightness, AdjustContrast, AdjustSaturation,
     *              AdjustHue, ColorJitter, Grayscale, RandomGrayscale,
     *              Invert, RandomInvert, Solarize, Posterize
     *   Noise:     GaussianNoise, SaltPepperNoise, GaussianBlur, RandomGaussianBlur,
     *              BoxBlur, Sharpen, RandomAdjustSharpness, RandomErasing, Cutout
     *   Advanced:  RandAugment, AutoAugment, TrivialAugmentWide
     *   Utility:   Compose, RandomChoice, RandomApply
     */
    static std::unique_ptr<Transform> create(const std::string& name,
                                             const std::map<std::string, ParamValue>& params = {});

    /**
     * Create standard training augmentation pipeline
     * Includes: RandomResizedCrop, RandomHorizontalFlip, ColorJitter, Normalize
     */
    static std::unique_ptr<Compose> createTrainTransform(
        int size = 224,
        bool use_color_jitter = true,
        bool use_random_erasing = false);

    /**
     * Create standard validation/test pipeline
     * Includes: Resize, CenterCrop, Normalize
     */
    static std::unique_ptr<Compose> createValTransform(int size = 224);

    /**
     * Create ImageNet standard transforms
     */
    static std::unique_ptr<Compose> createImageNetTrain(int size = 224);
    static std::unique_ptr<Compose> createImageNetVal(int size = 224);

    /**
     * Create CIFAR-10 standard transforms
     */
    static std::unique_ptr<Compose> createCIFAR10Train();
    static std::unique_ptr<Compose> createCIFAR10Val();

    /**
     * Create medical imaging transforms (conservative augmentation)
     */
    static std::unique_ptr<Compose> createMedicalTrain(int size = 224);
    static std::unique_ptr<Compose> createMedicalVal(int size = 224);

    /**
     * Get list of all available transform names
     */
    static std::vector<std::string> getAvailableTransforms();

    /**
     * Get default parameters for a transform
     */
    static std::map<std::string, ParamValue> getDefaultParams(const std::string& name);
};

// ============================================================================
// TransformFactory Implementation
// ============================================================================

inline std::unique_ptr<Transform> TransformFactory::create(
    const std::string& name,
    const std::map<std::string, ParamValue>& params) {

    // Helper to get param with default
    auto getInt = [&params](const std::string& key, int def) -> int {
        auto it = params.find(key);
        if (it != params.end() && std::holds_alternative<int>(it->second)) {
            return std::get<int>(it->second);
        }
        return def;
    };

    auto getFloat = [&params](const std::string& key, float def) -> float {
        auto it = params.find(key);
        if (it != params.end()) {
            if (std::holds_alternative<float>(it->second)) return std::get<float>(it->second);
            if (std::holds_alternative<double>(it->second)) return static_cast<float>(std::get<double>(it->second));
        }
        return def;
    };

    // Geometric transforms
    if (name == "Resize") {
        return std::make_unique<Resize>(
            getInt("width", 224), getInt("height", 224));
    }
    if (name == "CenterCrop") {
        return std::make_unique<CenterCrop>(
            getInt("width", 224), getInt("height", 224));
    }
    if (name == "RandomCrop") {
        return std::make_unique<RandomCrop>(
            getInt("width", 224), getInt("height", 224), getInt("padding", 0));
    }
    if (name == "RandomResizedCrop") {
        return std::make_unique<RandomResizedCrop>(
            getInt("size", 224),
            getFloat("scale_min", 0.08f), getFloat("scale_max", 1.0f),
            getFloat("ratio_min", 0.75f), getFloat("ratio_max", 1.33f));
    }
    if (name == "HorizontalFlip") {
        return std::make_unique<HorizontalFlip>();
    }
    if (name == "VerticalFlip") {
        return std::make_unique<VerticalFlip>();
    }
    if (name == "RandomHorizontalFlip") {
        return std::make_unique<RandomHorizontalFlip>(getFloat("p", 0.5f));
    }
    if (name == "RandomVerticalFlip") {
        return std::make_unique<RandomVerticalFlip>(getFloat("p", 0.5f));
    }
    if (name == "Rotate") {
        return std::make_unique<Rotate>(getFloat("degrees", 0.0f));
    }
    if (name == "RandomRotation") {
        return std::make_unique<RandomRotation>(getFloat("degrees", 15.0f));
    }

    // Color transforms
    if (name == "Normalize") {
        return std::make_unique<Normalize>(
            std::vector<float>{0.485f, 0.456f, 0.406f},
            std::vector<float>{0.229f, 0.224f, 0.225f});
    }
    if (name == "ColorJitter") {
        return std::make_unique<ColorJitter>(
            getFloat("brightness", 0.0f), getFloat("contrast", 0.0f),
            getFloat("saturation", 0.0f), getFloat("hue", 0.0f));
    }
    if (name == "Grayscale") {
        return std::make_unique<Grayscale>(getInt("num_output_channels", 1));
    }
    if (name == "RandomGrayscale") {
        return std::make_unique<RandomGrayscale>(getFloat("p", 0.1f));
    }
    if (name == "Invert") {
        return std::make_unique<Invert>();
    }
    if (name == "RandomInvert") {
        return std::make_unique<RandomInvert>(getFloat("p", 0.5f));
    }
    if (name == "Solarize") {
        return std::make_unique<Solarize>(getFloat("threshold", 0.5f));
    }
    if (name == "Posterize") {
        return std::make_unique<Posterize>(getInt("bits", 4));
    }

    // Noise transforms
    if (name == "GaussianNoise") {
        return std::make_unique<GaussianNoise>(
            getFloat("mean", 0.0f), getFloat("std", 0.1f));
    }
    if (name == "GaussianBlur") {
        return std::make_unique<GaussianBlur>(
            getInt("kernel_size", 3), getFloat("sigma", 1.0f));
    }
    if (name == "Sharpen") {
        return std::make_unique<Sharpen>(getFloat("factor", 1.0f));
    }
    if (name == "RandomErasing") {
        return std::make_unique<RandomErasing>(getFloat("p", 0.5f));
    }
    if (name == "Cutout") {
        return std::make_unique<Cutout>(
            getInt("n_holes", 1), getInt("length", 16));
    }

    // Advanced transforms
    if (name == "RandAugment") {
        return std::make_unique<RandAugment>(
            getInt("n", 2), getInt("m", 9));
    }
    if (name == "AutoAugment") {
        return std::make_unique<AutoAugment>(AutoAugmentPolicy::ImageNet);
    }
    if (name == "TrivialAugmentWide") {
        return std::make_unique<TrivialAugmentWide>();
    }

    // Utility
    if (name == "Compose") {
        return std::make_unique<Compose>();
    }

    return nullptr;
}

inline std::unique_ptr<Compose> TransformFactory::createTrainTransform(
    int size, bool use_color_jitter, bool use_random_erasing) {

    auto pipeline = std::make_unique<Compose>();

    pipeline->add(std::make_unique<RandomResizedCrop>(size));
    pipeline->add(std::make_unique<RandomHorizontalFlip>(0.5f));

    if (use_color_jitter) {
        pipeline->add(std::make_unique<ColorJitter>(0.4f, 0.4f, 0.4f, 0.1f));
    }

    pipeline->add(std::make_unique<Normalize>(
        std::vector<float>{0.485f, 0.456f, 0.406f},
        std::vector<float>{0.229f, 0.224f, 0.225f}));

    if (use_random_erasing) {
        pipeline->add(std::make_unique<RandomErasing>(0.25f));
    }

    return pipeline;
}

inline std::unique_ptr<Compose> TransformFactory::createValTransform(int size) {
    auto pipeline = std::make_unique<Compose>();

    int resize_size = static_cast<int>(size * 256.0f / 224.0f);
    pipeline->add(std::make_unique<Resize>(resize_size, resize_size));
    pipeline->add(std::make_unique<CenterCrop>(size, size));
    pipeline->add(std::make_unique<Normalize>(
        std::vector<float>{0.485f, 0.456f, 0.406f},
        std::vector<float>{0.229f, 0.224f, 0.225f}));

    return pipeline;
}

inline std::unique_ptr<Compose> TransformFactory::createImageNetTrain(int size) {
    return createTrainTransform(size, true, false);
}

inline std::unique_ptr<Compose> TransformFactory::createImageNetVal(int size) {
    return createValTransform(size);
}

inline std::unique_ptr<Compose> TransformFactory::createCIFAR10Train() {
    auto pipeline = std::make_unique<Compose>();

    pipeline->add(std::make_unique<RandomCrop>(32, 32, 4));
    pipeline->add(std::make_unique<RandomHorizontalFlip>(0.5f));
    pipeline->add(std::make_unique<Normalize>(
        std::vector<float>{0.4914f, 0.4822f, 0.4465f},
        std::vector<float>{0.2470f, 0.2435f, 0.2616f}));

    return pipeline;
}

inline std::unique_ptr<Compose> TransformFactory::createCIFAR10Val() {
    auto pipeline = std::make_unique<Compose>();

    pipeline->add(std::make_unique<Normalize>(
        std::vector<float>{0.4914f, 0.4822f, 0.4465f},
        std::vector<float>{0.2470f, 0.2435f, 0.2616f}));

    return pipeline;
}

inline std::unique_ptr<Compose> TransformFactory::createMedicalTrain(int size) {
    auto pipeline = std::make_unique<Compose>();

    // Conservative augmentation for medical images
    pipeline->add(std::make_unique<Resize>(size, size, InterpolationMode::Bilinear));
    pipeline->add(std::make_unique<RandomHorizontalFlip>(0.5f));
    pipeline->add(std::make_unique<RandomVerticalFlip>(0.5f));
    pipeline->add(std::make_unique<RandomRotation>(15.0f));  // Small rotation
    pipeline->add(std::make_unique<ColorJitter>(0.1f, 0.1f, 0.1f, 0.0f));  // Subtle color changes
    pipeline->add(std::make_unique<Normalize>(
        std::vector<float>{0.485f, 0.456f, 0.406f},
        std::vector<float>{0.229f, 0.224f, 0.225f}));

    return pipeline;
}

inline std::unique_ptr<Compose> TransformFactory::createMedicalVal(int size) {
    auto pipeline = std::make_unique<Compose>();

    pipeline->add(std::make_unique<Resize>(size, size, InterpolationMode::Bilinear));
    pipeline->add(std::make_unique<Normalize>(
        std::vector<float>{0.485f, 0.456f, 0.406f},
        std::vector<float>{0.229f, 0.224f, 0.225f}));

    return pipeline;
}

inline std::vector<std::string> TransformFactory::getAvailableTransforms() {
    return {
        // Geometric
        "Resize", "CenterCrop", "RandomCrop", "RandomResizedCrop",
        "HorizontalFlip", "VerticalFlip", "RandomHorizontalFlip", "RandomVerticalFlip",
        "Rotate", "RandomRotation", "RandomAffine",
        // Color
        "Normalize", "AdjustBrightness", "AdjustContrast", "AdjustSaturation",
        "AdjustHue", "ColorJitter", "Grayscale", "RandomGrayscale",
        "Invert", "RandomInvert", "Solarize", "Posterize",
        // Noise
        "GaussianNoise", "SaltPepperNoise", "GaussianBlur", "RandomGaussianBlur",
        "BoxBlur", "Sharpen", "RandomAdjustSharpness", "RandomErasing", "Cutout",
        // Advanced
        "RandAugment", "AutoAugment", "TrivialAugmentWide",
        // Utility
        "Compose", "RandomChoice", "RandomApply"
    };
}

inline std::map<std::string, ParamValue> TransformFactory::getDefaultParams(const std::string& name) {
    std::map<std::string, ParamValue> params;

    if (name == "Resize" || name == "CenterCrop" || name == "RandomCrop") {
        params["width"] = 224;
        params["height"] = 224;
    }
    if (name == "RandomCrop") {
        params["padding"] = 0;
    }
    if (name == "RandomResizedCrop") {
        params["size"] = 224;
        params["scale_min"] = 0.08f;
        params["scale_max"] = 1.0f;
    }
    if (name == "RandomHorizontalFlip" || name == "RandomVerticalFlip" ||
        name == "RandomGrayscale" || name == "RandomInvert" || name == "RandomErasing") {
        params["p"] = 0.5f;
    }
    if (name == "RandomRotation" || name == "Rotate") {
        params["degrees"] = 15.0f;
    }
    if (name == "ColorJitter") {
        params["brightness"] = 0.4f;
        params["contrast"] = 0.4f;
        params["saturation"] = 0.4f;
        params["hue"] = 0.1f;
    }
    if (name == "GaussianBlur") {
        params["kernel_size"] = 3;
        params["sigma"] = 1.0f;
    }
    if (name == "RandAugment") {
        params["n"] = 2;
        params["m"] = 9;
    }

    return params;
}

} // namespace transforms
} // namespace cyxwiz
