#include "classification_decision.h"

#include "algorithms/arrayfire_backend_utils.h"

#include <cyxwiz/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

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
        predictions.ReadData<float>(),
        targets.ReadData<float>(),
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

ClassificationDecisionScalar BuildClassificationDecisionScalarArrayFire(
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

    af::array correct_scalar = af::sum(af::flat(correct_mask.as(f32)));
    correct_scalar.eval();
    return {
        Tensor::FromSemanticArray(correct_scalar, {1}),
        batch_size,
    };
}
#endif

} // namespace

ClassificationDecisionScalar BuildClassificationDecisionScalar(
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
            return BuildClassificationDecisionScalarArrayFire(
                predictions, targets, batch_size, output_width, mode);
        } catch (const af::exception& e) {
            const BackendFallbackReason reason =
                ClassifyArrayFireBackendFallbackReason(e.what());
            const std::string context = BuildArrayFireBackendFallbackContext(
                BuildTensorShapeContext("predictions", predictions.Shape()) +
                "; " +
                BuildTensorShapeContext("targets", targets.Shape()));
            ThrowIfArrayFireNativeCpuFallbackForbidden(
                "ClassificationDecisionCount",
                reason,
                e.what(),
                context);
            spdlog::warn("{}",
                         BuildArrayFireBackendFallbackMessage(
                             "ClassificationDecisionCount",
                             reason,
                             reason !=
                                 BackendFallbackReason::CudaJitParamOverflow,
                             e.what(),
                             context));
        }
    }
#endif

    const ClassificationDecisionCount cpu_count =
        CountClassificationDecisionsCpu(
        predictions, targets, batch_size, output_width, mode);
    const float correct = static_cast<float>(cpu_count.correct);
    return {
        Tensor({1}, &correct, DataType::Float32),
        cpu_count.total,
    };
}

ClassificationDecisionCount ReadClassificationDecisionScalar(
    const ClassificationDecisionScalar& scalar,
    std::string_view operation) {
    if (scalar.total == 0 || scalar.correct.NumElements() != 1 ||
        scalar.correct.GetDataType() != DataType::Float32) {
        return {};
    }

    const ScopedArrayFireHostSyncAttribution sync_attribution(
        ArrayFireHostSyncCategory::MetricScalarReadback,
        std::string(operation));
    const float correct = scalar.correct.ReadData<float>()[0];
    if (!std::isfinite(correct) || correct <= 0.0f) {
        return {0, scalar.total};
    }
    const auto rounded = static_cast<size_t>(std::llround(correct));
    return {std::min(rounded, scalar.total), scalar.total};
}

ClassificationDecisionCount CountClassificationDecisionScalars(
    const Tensor& predictions,
    const Tensor& targets,
    size_t batch_size,
    size_t output_width,
    ClassificationDecisionMode mode) {
    return ReadClassificationDecisionScalar(
        BuildClassificationDecisionScalar(
            predictions, targets, batch_size, output_width, mode));
}

} // namespace cyxwiz
