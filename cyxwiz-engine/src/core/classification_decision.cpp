#include "classification_decision.h"

#include "algorithms/arrayfire_backend_utils.h"
#include "metric_learning_mining.h"

#include <cyxwiz/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#include "algorithms/arrayfire_host_values.h"
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
        throw std::runtime_error(DeviceError("TripletDecisionCount", e));
    }
#else
    throw std::runtime_error("TripletDecisionCount runs on ArrayFire, and this build has no ArrayFire");
#endif
}

// predictions [2P, D]: rows 0..P-1 firsts, P..2P-1 seconds; similar [P] is 1
// for a same-class pair. A pair is called similar when its distance is below
// the threshold (PairDistance) or its cosine above it (PairCosine).
ClassificationDecisionScalar BuildPairDecisionScalar(
    const Tensor& predictions,
    const Tensor& similar,
    size_t pairs,
    ClassificationDecisionMode mode,
    float threshold) {
    const auto& shape = predictions.Shape();
    if (shape.size() != 2 || shape[0] != 2 * pairs || similar.NumElements() != pairs ||
        predictions.GetDataType() != DataType::Float32 || similar.GetDataType() != DataType::Float32) {
        return {};
    }
    const bool by_distance = mode == ClassificationDecisionMode::PairDistance;
    // As the backend Cosine Embedding loss.
    constexpr float kCosineEpsilon = 1.0e-12f;
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array all = predictions.GetArrayRowMajor2D();
        const double count = static_cast<double>(pairs);
        const af::array a = all(af::seq(0.0, count - 1.0), af::span);
        const af::array b = all(af::seq(count, 2.0 * count - 1.0), af::span);
        const af::array called_similar = by_distance
            ? af::sqrt(af::sum((a - b) * (a - b), 1)) < threshold
            : af::sum(a * b, 1) / af::sqrt((af::sum(a * a, 1) + kCosineEpsilon) *
                                           (af::sum(b * b, 1) + kCosineEpsilon)) > threshold;
        const af::array truth = af::flat(similar.GetSemanticArray()) == 1.0f;
        af::array correct = af::sum(af::flat((af::flat(called_similar) == truth).as(f32)));
        af::array counts = af::join(0, correct, af::constant(static_cast<float>(pairs), 1));
        counts.eval();
        return {Tensor::FromSemanticArray(counts, {2})};
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("PairDecisionCount", e));
    }
#else
    (void)by_distance;
    (void)threshold;
    throw std::runtime_error("PairDecisionCount runs on ArrayFire, and this build has no ArrayFire");
#endif
}

// predictions [N, D] and the batch's class ids: the triplets / pairs the loss
// mines, decided as in the stacked modes, on the device like the mining.
ClassificationDecisionScalar BuildMinedDecisionScalar(
    const Tensor& predictions,
    const Tensor& class_ids,
    ClassificationDecisionMode mode,
    float threshold) {
    const auto& shape = predictions.Shape();
    if (shape.size() != 2 || shape[0] == 0 || predictions.GetDataType() != DataType::Float32) {
        return {};
    }
    const bool cosine = mode == ClassificationDecisionMode::PairCosineHard;
    const Tensor ids = BatchClassIds(class_ids, shape[0]);
    const Tensor distances = MiningDistances(predictions, cosine ? MiningDistance::Cosine : MiningDistance::Euclidean);
    Tensor correct;
    size_t total = 0;
    if (mode == ClassificationDecisionMode::TripletOrderHard ||
        mode == ClassificationDecisionMode::TripletOrderSemiHard) {
        const MinedTriplets triplets = MineTriplets(
            ids, distances,
            mode == ClassificationDecisionMode::TripletOrderHard ? MetricMining::Hard : MetricMining::SemiHard);
        total = triplets.count;
        if (total > 0) {
            correct = DistancesAt(distances, triplets.anchors, triplets.positives) <
                      DistancesAt(distances, triplets.anchors, triplets.negatives);
        }
    } else {
        // Cosine distance is 1 - cos: cos > threshold is distance < 1 - threshold.
        const float cut = cosine ? 1.0f - threshold : threshold;
        const MinedPairs pairs = MinePairs(ids, distances);
        total = pairs.count;
        if (total > 0) {
            correct = (DistancesAt(distances, pairs.firsts, pairs.seconds) < cut) == (pairs.similar == 1.0f);
        }
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array hits = total > 0 ? af::sum(af::flat(correct.GetRowMajorFlatArray()).as(f32))
                                         : af::constant(0.0f, 1);
        af::array counts = af::join(0, hits, af::constant(static_cast<float>(total), 1));
        counts.eval();
        return {Tensor::FromSemanticArray(counts, {2})};
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("MinedDecisionCount", e));
    }
#else
    throw std::runtime_error("MinedDecisionCount runs on ArrayFire, and this build has no ArrayFire");
#endif
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
// [batch, width] rows on the device from any layout with that element count.
af::array DecisionRows(const Tensor& values, size_t batch, size_t width, const char* what) {
    if (values.NumElements() != batch * width) {
        throw std::invalid_argument(std::string("ClassificationDecisionCount: ") + what + " hold " +
                                    std::to_string(values.NumElements()) + " values, expected " +
                                    std::to_string(batch) + " x " + std::to_string(width));
    }
    return values.Reshape({batch, width}).GetArrayRowMajor2D().as(f32);
}

// First index of each row's maximum (std::max_element order); a row without a
// maximum (all NaN) gives 0.
af::array FirstArgmaxRows(const af::array& rows) {
    const unsigned width = static_cast<unsigned>(rows.dims(1));
    const af::array maximum = af::tile((af::max)(rows, 1), 1, width);
    const af::array index = af::range(rows.dims(), 1, s32);
    const af::array first = (af::min)(af::select(rows == maximum, index, static_cast<double>(width)), 1);
    return af::select(first >= static_cast<double>(width), 0.0, first).as(s32);
}

ClassificationDecisionScalar BuildClassificationDecisionScalarArrayFire(
    const Tensor& predictions,
    const Tensor& targets,
    size_t batch_size,
    size_t output_width,
    ClassificationDecisionMode mode,
    std::optional<int> ignore_index) {
    const af::array pred = DecisionRows(predictions, batch_size, output_width, "predictions");
    const bool class_index_targets =
        (targets.GetDataType() == DataType::Int32 || targets.GetDataType() == DataType::Int64) &&
        targets.NumElements() == batch_size;
    af::array correct_mask;
    af::array valid_mask = af::constant(1.0f, static_cast<dim_t>(batch_size));
    if (mode == ClassificationDecisionMode::BinaryLogit ||
        mode == ClassificationDecisionMode::BinaryProbability) {
        // Column 0 decides, as the host helpers do.
        const float cut = mode == ClassificationDecisionMode::BinaryLogit ? 0.0f : 0.5f;
        const af::array target = class_index_targets
            ? af::flat(targets.GetRowMajorFlatArray()).as(f32)
            : DecisionRows(targets, batch_size, output_width, "targets").col(0);
        correct_mask = (pred.col(0) >= cut) == (target >= 0.5f);
    } else if (class_index_targets) {
        const af::array target_indices = af::flat(targets.GetRowMajorFlatArray()).as(s32);
        if (ignore_index) {
            valid_mask = (target_indices != *ignore_index).as(f32);
        }
        correct_mask = (FirstArgmaxRows(pred) == target_indices).as(f32) * valid_mask;
    } else {
        const af::array target = DecisionRows(targets, batch_size, output_width, "targets");
        correct_mask = FirstArgmaxRows(pred) == FirstArgmaxRows(target);
    }
    af::array counts = af::join(0, af::sum(af::flat(correct_mask.as(f32))), af::sum(af::flat(valid_mask)));
    counts.eval();
    return {Tensor::FromSemanticArray(counts, {2})};
}
#endif

} // namespace

ClassificationDecisionScalar BuildClassificationDecisionScalar(
    const Tensor& predictions,
    const Tensor& targets,
    size_t batch_size,
    size_t output_width,
    ClassificationDecisionMode mode,
    std::optional<int> ignore_index,
    float pair_threshold) {
    if (batch_size == 0 || output_width == 0) {
        return {};
    }
    if (mode == ClassificationDecisionMode::TripletOrder) {
        return BuildTripletOrderScalar(predictions, batch_size);
    }
    if (mode == ClassificationDecisionMode::PairDistance ||
        mode == ClassificationDecisionMode::PairCosine) {
        return BuildPairDecisionScalar(predictions, targets, batch_size, mode, pair_threshold);
    }
    if (IsMinedMetricDecision(mode)) {
        return BuildMinedDecisionScalar(predictions, targets, mode, pair_threshold);
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        return BuildClassificationDecisionScalarArrayFire(
            predictions, targets, batch_size, output_width, mode, ignore_index);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("ClassificationDecisionCount", e));
    }
#else
    (void)ignore_index;
    throw std::runtime_error("ClassificationDecisionCount runs on ArrayFire, and this build has no ArrayFire");
#endif
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
    std::optional<int> ignore_index,
    float pair_threshold) {
    return ReadClassificationDecisionScalar(
        BuildClassificationDecisionScalar(
            predictions, targets, batch_size, output_width, mode,
            ignore_index, pair_threshold));
}

} // namespace cyxwiz
