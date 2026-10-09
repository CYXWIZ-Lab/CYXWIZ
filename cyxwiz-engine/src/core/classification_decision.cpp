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

// predictions [3T, D]: rows 0..T-1 anchors, T..2T-1 positives, 2T..3T-1
// negatives (squared distances order the same as distances).
ClassificationDecisionScalar BuildTripletOrderScalar(
    const Tensor& predictions,
    size_t triplets) {
    const auto& shape = predictions.Shape();
    if (shape.size() != 2 || shape[0] != 3 * triplets ||
        predictions.GetDataType() != DataType::Float32) {
        return {};
    }
    const size_t width = shape[1];
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array all = predictions.GetArrayRowMajor2D();
        const double count = static_cast<double>(triplets);
        const af::array a = all(af::seq(0.0, count - 1.0), af::span);
        const af::array p = all(af::seq(count, 2.0 * count - 1.0), af::span);
        const af::array n = all(af::seq(2.0 * count, 3.0 * count - 1.0), af::span);
        const af::array d_ap = af::sum((a - p) * (a - p), 1);
        const af::array d_an = af::sum((a - n) * (a - n), 1);
        af::array correct = af::sum(af::flat((d_ap < d_an).as(f32)));
        af::array counts = af::join(
            0, correct, af::constant(static_cast<float>(triplets), 1));
        counts.eval();
        return {Tensor::FromSemanticArray(counts, {2})};
    } catch (const af::exception& e) {
        const BackendFallbackReason reason = ClassifyArrayFireBackendFallbackReason(e.what());
        const std::string context = BuildArrayFireBackendFallbackContext(
            BuildTensorShapeContext("predictions", predictions.Shape()));
        ThrowIfArrayFireNativeCpuFallbackForbidden("TripletDecisionCount", reason, e.what(), context);
        spdlog::warn("{}", BuildArrayFireBackendFallbackMessage(
                               "TripletDecisionCount", reason,
                               reason != BackendFallbackReason::CudaJitParamOverflow, e.what(), context));
    }
#endif
    const float* values = predictions.ReadData<float>();
    size_t correct = 0;
    for (size_t t = 0; t < triplets; ++t) {
        const float* a = values + t * width;
        const float* p = values + (triplets + t) * width;
        const float* n = values + (2 * triplets + t) * width;
        double d_ap = 0.0;
        double d_an = 0.0;
        for (size_t j = 0; j < width; ++j) {
            d_ap += static_cast<double>(a[j] - p[j]) * (a[j] - p[j]);
            d_an += static_cast<double>(a[j] - n[j]) * (a[j] - n[j]);
        }
        if (d_ap < d_an) ++correct;
    }
    const float counts[] = {static_cast<float>(correct), static_cast<float>(triplets)};
    return {Tensor({2}, counts, DataType::Float32)};
}

ClassificationDecisionCount CountClassificationDecisionsCpu(
    const Tensor& predictions,
    const Tensor& targets,
    size_t batch_size,
    size_t output_width,
    ClassificationDecisionMode mode,
    std::optional<int> ignore_index) {
    if (mode == ClassificationDecisionMode::MulticlassScores &&
        (targets.GetDataType() == DataType::Int32 ||
         targets.GetDataType() == DataType::Int64) &&
        targets.Shape() == std::vector<size_t>{batch_size}) {
        const float* scores = predictions.ReadData<float>();
        const int32_t* labels32 = targets.GetDataType() == DataType::Int32
            ? targets.ReadData<int32_t>()
            : nullptr;
        const int64_t* labels64 = targets.GetDataType() == DataType::Int64
            ? targets.ReadData<int64_t>()
            : nullptr;
        ClassificationDecisionCount result;
        for (size_t row = 0; row < batch_size; ++row) {
            const int64_t label = labels32
                ? static_cast<int64_t>(labels32[row])
                : labels64[row];
            if (ignore_index && label == *ignore_index) {
                continue;
            }
            if (ClassificationPredictedClass(
                    scores + row * output_width,
                    output_width,
                    mode) == label) {
                ++result.correct;
            }
            ++result.total;
        }
        return result;
    }
    return CountClassificationDecisions(
        predictions.ReadData<float>(), targets.ReadData<float>(),
        batch_size, output_width, mode);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
bool CanUseArrayFireDecisionCount(const Tensor& predictions,
                                  const Tensor& targets,
                                  size_t batch_size,
                                  size_t output_width) {
    if (batch_size == 0 || output_width == 0) {
        return false;
    }
    if (predictions.GetDataType() != DataType::Float32) {
        return false;
    }
    const auto& pred_shape = predictions.Shape();
    const auto& target_shape = targets.Shape();
    const bool probability_targets =
        targets.GetDataType() == DataType::Float32 &&
        target_shape == pred_shape;
    const bool class_index_targets =
        (targets.GetDataType() == DataType::Int32 ||
         targets.GetDataType() == DataType::Int64) &&
        target_shape == std::vector<size_t>{batch_size};
    return pred_shape.size() == 2 &&
           pred_shape[0] == batch_size &&
           pred_shape[1] == output_width &&
           (probability_targets || class_index_targets);
}

ClassificationDecisionScalar BuildClassificationDecisionScalarArrayFire(
    const Tensor& predictions,
    const Tensor& targets,
    size_t batch_size,
    size_t output_width,
    ClassificationDecisionMode mode,
    std::optional<int> ignore_index) {
    (void)output_width;
    const af::array pred = predictions.GetArrayRowMajor2D();
    const bool class_index_targets = targets.Shape().size() == 1;
    const af::array target = class_index_targets
        ? targets.GetSemanticArray()
        : targets.GetArrayRowMajor2D();

    af::array correct_mask;
    af::array valid_mask = af::constant(1.0f, batch_size);
    if (mode == ClassificationDecisionMode::BinaryLogit) {
        correct_mask = (pred >= 0.0f) == (target >= 0.5f);
    } else if (mode == ClassificationDecisionMode::BinaryProbability) {
        correct_mask = (pred >= 0.5f) == (target >= 0.5f);
    } else if (class_index_targets) {
        af::array pred_values;
        af::array pred_indices;
        af::max(pred_values, pred_indices, pred, 1);
        const af::array target_indices = af::flat(target).as(s32);
        if (ignore_index) {
            valid_mask = (target_indices != *ignore_index).as(f32);
        }
        correct_mask =
            (pred_indices.as(s32) == target_indices).as(f32) * valid_mask;
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
    af::array valid_scalar = af::sum(af::flat(valid_mask));
    valid_scalar.eval();
    af::array counts = af::join(0, correct_scalar, valid_scalar);
    counts.eval();
    return {
        Tensor::FromSemanticArray(counts, {2}),
    };
}
#endif

} // namespace

ClassificationDecisionScalar BuildClassificationDecisionScalar(
    const Tensor& predictions,
    const Tensor& targets,
    size_t batch_size,
    size_t output_width,
    ClassificationDecisionMode mode,
    std::optional<int> ignore_index) {
    if (batch_size == 0 || output_width == 0) {
        return {};
    }
    if (mode == ClassificationDecisionMode::TripletOrder) {
        return BuildTripletOrderScalar(predictions, batch_size);
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (CanUseArrayFireDecisionCount(
            predictions, targets, batch_size, output_width)) {
        try {
            return BuildClassificationDecisionScalarArrayFire(
                predictions, targets, batch_size, output_width, mode,
                ignore_index);
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
        predictions, targets, batch_size, output_width, mode, ignore_index);
    const float counts[] = {
        static_cast<float>(cpu_count.correct),
        static_cast<float>(cpu_count.total),
    };
    return {
        Tensor({2}, counts, DataType::Float32),
    };
}

ClassificationDecisionCount ReadClassificationDecisionScalar(
    const ClassificationDecisionScalar& scalar,
    std::string_view operation) {
    if (scalar.counts.NumElements() != 2 ||
        scalar.counts.GetDataType() != DataType::Float32) {
        return {};
    }

    const ScopedArrayFireHostSyncAttribution sync_attribution(
        ArrayFireHostSyncCategory::MetricScalarReadback,
        std::string(operation));
    const float* counts = scalar.counts.ReadData<float>();
    const float correct = counts[0];
    const float total = counts[1];
    if (!std::isfinite(total) || total <= 0.0f) {
        return {};
    }
    const auto rounded_total = static_cast<size_t>(std::llround(total));
    if (!std::isfinite(correct) || correct <= 0.0f) {
        return {0, rounded_total};
    }
    const auto rounded_correct = static_cast<size_t>(std::llround(correct));
    return {std::min(rounded_correct, rounded_total), rounded_total};
}

ClassificationDecisionCount CountClassificationDecisionScalars(
    const Tensor& predictions,
    const Tensor& targets,
    size_t batch_size,
    size_t output_width,
    ClassificationDecisionMode mode,
    std::optional<int> ignore_index) {
    return ReadClassificationDecisionScalar(
        BuildClassificationDecisionScalar(
            predictions, targets, batch_size, output_width, mode,
            ignore_index));
}

} // namespace cyxwiz
