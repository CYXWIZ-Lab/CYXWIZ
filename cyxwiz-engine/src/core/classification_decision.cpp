#include "classification_decision.h"

#include <cyxwiz/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#include <spdlog/spdlog.h>
#endif

namespace cyxwiz {

namespace {

ClassificationDecisionCount CountClassificationDecisionsCpu(
    const Tensor& predictions,
    const Tensor& targets,
    size_t batch_size,
    size_t output_width,
    ClassificationDecisionMode mode) {
    return CountClassificationDecisions(
        predictions.Data<float>(),
        targets.Data<float>(),
        batch_size,
        output_width,
        mode);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
bool CanUseArrayFireDecisionCount(const Tensor& predictions,
                                  const Tensor& targets,
                                  size_t batch_size,
                                  size_t output_width) {
    if (batch_size == 0 || output_width == 0) {
        return false;
    }
    if (predictions.GetDataType() != DataType::Float32 ||
        targets.GetDataType() != DataType::Float32) {
        return false;
    }
    const auto& pred_shape = predictions.Shape();
    const auto& target_shape = targets.Shape();
    return pred_shape.size() == 2 &&
           target_shape.size() == 2 &&
           pred_shape[0] == batch_size &&
           pred_shape[1] == output_width &&
           target_shape[0] == batch_size &&
           target_shape[1] == output_width;
}

size_t ReadBoundedCorrectCount(const af::array& correct_mask,
                               size_t batch_size) {
    af::array correct_scalar = af::sum(af::flat(correct_mask.as(f32)));
    correct_scalar.eval();

    float correct = 0.0f;
    correct_scalar.host(&correct);
    if (!std::isfinite(correct) || correct <= 0.0f) {
        return 0;
    }

    const auto rounded = static_cast<size_t>(std::llround(correct));
    return std::min(rounded, batch_size);
}

ClassificationDecisionCount CountClassificationDecisionsArrayFire(
    const Tensor& predictions,
    const Tensor& targets,
    size_t batch_size,
    size_t output_width,
    ClassificationDecisionMode mode) {
    (void)output_width;
    const af::array pred = predictions.GetArrayRowMajor2D();
    const af::array target = targets.GetArrayRowMajor2D();

    af::array correct_mask;
    if (mode == ClassificationDecisionMode::BinaryLogit) {
        correct_mask = (pred >= 0.0f) == (target >= 0.5f);
    } else if (mode == ClassificationDecisionMode::BinaryProbability) {
        correct_mask = (pred >= 0.5f) == (target >= 0.5f);
    } else {
        af::array pred_values;
        af::array pred_indices;
        af::array target_values;
        af::array target_indices;
        af::max(pred_values, pred_indices, pred, 1);
        af::max(target_values, target_indices, target, 1);
        correct_mask = pred_indices == target_indices;
    }
    correct_mask.eval();

    return {ReadBoundedCorrectCount(correct_mask, batch_size), batch_size};
}
#endif

} // namespace

ClassificationDecisionCount CountClassificationDecisionScalars(
    const Tensor& predictions,
    const Tensor& targets,
    size_t batch_size,
    size_t output_width,
    ClassificationDecisionMode mode) {
    if (batch_size == 0 || output_width == 0) {
        return {};
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (CanUseArrayFireDecisionCount(
            predictions, targets, batch_size, output_width)) {
        try {
            return CountClassificationDecisionsArrayFire(
                predictions, targets, batch_size, output_width, mode);
        } catch (const af::exception& e) {
            spdlog::warn(
                "Classification metric ArrayFire count failed, using CPU fallback: {}",
                e.what());
        }
    }
#endif

    return CountClassificationDecisionsCpu(
        predictions, targets, batch_size, output_width, mode);
}

} // namespace cyxwiz
