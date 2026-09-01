#pragma once

#include "../gui/node_editor.h"

#include <cyxwiz/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string_view>

namespace cyxwiz {

enum class ClassificationDecisionMode {
    MulticlassScores,
    BinaryProbability,
    BinaryLogit,
};

inline bool UsesScalarBinaryTargets(gui::NodeType loss_type) {
    return loss_type == gui::NodeType::BCELoss ||
           loss_type == gui::NodeType::BCEWithLogits;
}

inline bool UsesClassIndexTargets(gui::NodeType loss_type) {
    return loss_type == gui::NodeType::CrossEntropyLoss;
}

inline ClassificationDecisionMode ClassificationDecisionModeForLoss(
    gui::NodeType loss_type) {
    if (loss_type == gui::NodeType::BCEWithLogits) {
        return ClassificationDecisionMode::BinaryLogit;
    }
    if (loss_type == gui::NodeType::BCELoss) {
        return ClassificationDecisionMode::BinaryProbability;
    }
    return ClassificationDecisionMode::MulticlassScores;
}

inline int ClassificationPredictedClass(
    const float* scores,
    size_t width,
    ClassificationDecisionMode mode) {
    if (!scores || width == 0) return 0;
    if (mode == ClassificationDecisionMode::BinaryLogit) {
        return scores[0] >= 0.0f ? 1 : 0;
    }
    if (mode == ClassificationDecisionMode::BinaryProbability) {
        return scores[0] >= 0.5f ? 1 : 0;
    }
    return static_cast<int>(std::distance(
        scores, std::max_element(scores, scores + width)));
}

inline int ClassificationTargetClass(
    const float* targets,
    size_t width,
    ClassificationDecisionMode mode) {
    if (!targets || width == 0) return 0;
    if (mode != ClassificationDecisionMode::MulticlassScores) {
        return targets[0] >= 0.5f ? 1 : 0;
    }
    return static_cast<int>(std::distance(
        targets, std::max_element(targets, targets + width)));
}

struct ClassificationDecisionCount {
    size_t correct = 0;
    size_t total = 0;
};

struct ClassificationDecisionScalar {
    // Float32 [2] kept device-resident as {correct, valid_target_count}.
    Tensor counts;
};

inline ClassificationDecisionCount CountClassificationDecisions(
    const float* predictions,
    const float* targets,
    size_t batch_size,
    size_t output_width,
    ClassificationDecisionMode mode) {
    ClassificationDecisionCount result;
    if (!predictions || !targets || output_width == 0) return result;

    for (size_t row = 0; row < batch_size; ++row) {
        const float* prediction = predictions + row * output_width;
        const float* target = targets + row * output_width;
        if (ClassificationPredictedClass(prediction, output_width, mode) ==
            ClassificationTargetClass(target, output_width, mode)) {
            ++result.correct;
        }
        ++result.total;
    }
    return result;
}

ClassificationDecisionCount CountClassificationDecisionScalars(
    const Tensor& predictions,
    const Tensor& targets,
    size_t batch_size,
    size_t output_width,
    ClassificationDecisionMode mode,
    std::optional<int> ignore_index = std::nullopt);

ClassificationDecisionScalar BuildClassificationDecisionScalar(
    const Tensor& predictions,
    const Tensor& targets,
    size_t batch_size,
    size_t output_width,
    ClassificationDecisionMode mode,
    std::optional<int> ignore_index = std::nullopt);

ClassificationDecisionCount ReadClassificationDecisionScalar(
    const ClassificationDecisionScalar& scalar,
    std::string_view operation = "ClassificationDecision::CorrectCount");

inline float ClassificationAccuracy(
    const float* predictions,
    const float* targets,
    size_t batch_size,
    size_t output_width,
    ClassificationDecisionMode mode) {
    const auto count = CountClassificationDecisions(
        predictions, targets, batch_size, output_width, mode);
    return count.total > 0
        ? static_cast<float>(count.correct) / static_cast<float>(count.total)
        : 0.0f;
}

inline float ClassificationConfidence(
    const float* scores,
    size_t width,
    int predicted_class,
    ClassificationDecisionMode mode) {
    if (!scores || width == 0) return 0.0f;
    if (mode == ClassificationDecisionMode::BinaryLogit) {
        const float logit = scores[0];
        const float positive = logit >= 0.0f
            ? 1.0f / (1.0f + std::exp(-logit))
            : std::exp(logit) / (1.0f + std::exp(logit));
        return predicted_class == 1 ? positive : 1.0f - positive;
    }
    if (mode == ClassificationDecisionMode::BinaryProbability) {
        const float positive = std::clamp(scores[0], 0.0f, 1.0f);
        return predicted_class == 1 ? positive : 1.0f - positive;
    }
    if (predicted_class < 0 ||
        static_cast<size_t>(predicted_class) >= width) {
        return 0.0f;
    }

    const float max_score = *std::max_element(scores, scores + width);
    float exp_sum = 0.0f;
    for (size_t i = 0; i < width; ++i) {
        exp_sum += std::exp(scores[i] - max_score);
    }
    return exp_sum > 0.0f
        ? std::exp(scores[predicted_class] - max_score) / exp_sum
        : 0.0f;
}

} // namespace cyxwiz
