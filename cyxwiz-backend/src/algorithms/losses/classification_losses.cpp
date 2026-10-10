#include "cyxwiz/losses/classification.h"
#include "../arrayfire_backend_utils.h"
#include "loss_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#include <af/internal.h>
#endif

// Undefine Windows macros that conflict with std::max/min and ArrayFire helpers.
// Must be AFTER all includes (Windows headers define these).
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz {

using namespace loss_detail;

CrossEntropyLoss::CrossEntropyLoss(Reduction reduction, int ignore_index)
    : CrossEntropyLoss(reduction, ignore_index, {}) {}

CrossEntropyLoss::CrossEntropyLoss(Reduction reduction,
                                   int ignore_index,
                                   std::vector<float> class_weights)
    : CrossEntropyLoss(reduction, ignore_index, std::move(class_weights), 0.0f) {}

CrossEntropyLoss::CrossEntropyLoss(Reduction reduction,
                                   int ignore_index,
                                   std::vector<float> class_weights,
                                   float label_smoothing)
    : Loss(reduction),
      ignore_index_(ignore_index),
      class_weights_(std::move(class_weights)),
      label_smoothing_(label_smoothing) {
    if (!std::isfinite(label_smoothing_) ||
        label_smoothing_ < 0.0f || label_smoothing_ >= 1.0f) {
        throw std::runtime_error(
            "CrossEntropy label_smoothing must be finite and in [0, 1)");
    }
}

NLLLoss::NLLLoss(Reduction reduction, int ignore_index)
    : Loss(reduction), ignore_index_(ignore_index) {}

FocalLoss::FocalLoss(float alpha, float gamma, Reduction reduction)
    : Loss(reduction), alpha_(alpha), gamma_(gamma) {
    SetAlpha(alpha);
    SetGamma(gamma);
}

void FocalLoss::SetAlpha(float alpha) {
    if (!std::isfinite(alpha) || alpha < 0.0f) {
        throw std::invalid_argument("FocalLoss alpha must be finite and >= 0");
    }
    alpha_ = alpha;
}

void FocalLoss::SetGamma(float gamma) {
    if (!std::isfinite(gamma) || gamma < 0.0f) {
        throw std::invalid_argument("FocalLoss gamma must be finite and >= 0");
    }
    gamma_ = gamma;
}

namespace {

struct ClassAxisShape {
    size_t batch = 1;
    size_t classes = 0;
    bool batched = false;
    std::vector<size_t> class_index_target_shape;
    std::vector<size_t> unreduced_shape;
};

ClassAxisShape ValidateClassAxisPredictions(const Tensor& predictions, const char* name) {
    if (predictions.GetDataType() != DataType::Float32) {
        throw std::runtime_error(std::string(name) + " only supports Float32 predictions");
    }
    const std::vector<size_t>& shape = predictions.Shape();
    if (shape.size() == 1) {
        return {1, shape[0], false, {}, {1}};
    }
    if (shape.size() == 2) {
        return {shape[0], shape[1], true, {shape[0]}, {shape[0]}};
    }
    if (shape.size() == 3) {
        return {
            shape[0] * shape[1],
            shape[2],
            true,
            {shape[0], shape[1]},
            {shape[0], shape[1]},
        };
    }
    throw std::runtime_error(
        std::string(name) +
        " supports 1D, 2D, or [batch, seq, classes] predictions");
}

bool TargetsAreClassIndices(const Tensor& predictions, const Tensor& targets) {
    return targets.Shape() != predictions.Shape();
}

void ValidateClassIndexTargets(const Tensor& targets, const ClassAxisShape& shape, const char* name) {
    if (targets.GetDataType() != DataType::Int32 && targets.GetDataType() != DataType::Int64) {
        throw std::runtime_error(std::string(name) + " class-index targets must be Int32 or Int64");
    }
    const std::vector<size_t>& target_shape = targets.Shape();
    const bool valid = shape.batched
                           ? target_shape == shape.class_index_target_shape
                           : targets.NumElements() == 1;
    if (!valid) {
        throw std::runtime_error(std::string(name) + " class-index target shape is invalid");
    }
}



void ValidateClassWeights(const std::vector<float>& class_weights,
                          size_t classes,
                          const char* name) {
    if (!class_weights.empty() && class_weights.size() != classes) {
        throw std::runtime_error(
            std::string(name) + " class_weights size must match class count");
    }
}









#ifdef CYXWIZ_HAS_ARRAYFIRE
af::array ToCrossEntropyRows(const af::array& values,
                             const std::vector<size_t>& semantic_shape) {
    if (semantic_shape.size() == 1) {
        return af::moddims(
            values,
            1,
            static_cast<dim_t>(semantic_shape[0]));
    }
    if (semantic_shape.size() == 2) {
        return values;
    }

    const dim_t batch = static_cast<dim_t>(semantic_shape[0]);
    const dim_t sequence = static_cast<dim_t>(semantic_shape[1]);
    const dim_t classes = static_cast<dim_t>(semantic_shape[2]);
    af::array class_first = af::reorder(values, 2, 1, 0);
    return af::transpose(af::moddims(
        class_first, classes, batch * sequence));
}

af::array ToCrossEntropyIndexRows(
    const af::array& targets,
    const std::vector<size_t>& prediction_shape) {
    if (prediction_shape.size() < 3) {
        return af::flat(targets);
    }
    return af::flat(af::transpose(targets));
}

af::array RestoreCrossEntropyClassLast(
    const af::array& rows,
    const std::vector<size_t>& semantic_shape) {
    if (semantic_shape.size() == 1) {
        return af::flat(rows);
    }
    if (semantic_shape.size() == 2) {
        return rows;
    }

    const dim_t batch = static_cast<dim_t>(semantic_shape[0]);
    const dim_t sequence = static_cast<dim_t>(semantic_shape[1]);
    const dim_t classes = static_cast<dim_t>(semantic_shape[2]);
    af::array class_first = af::moddims(
        af::transpose(rows), classes, sequence, batch);
    return af::reorder(class_first, 2, 1, 0);
}

af::array RestoreCrossEntropyUnreduced(
    const af::array& rows,
    const std::vector<size_t>& prediction_shape) {
    if (prediction_shape.size() < 3) {
        return af::flat(rows);
    }
    const dim_t batch = static_cast<dim_t>(prediction_shape[0]);
    const dim_t sequence = static_cast<dim_t>(prediction_shape[1]);
    return af::transpose(
        af::moddims(af::flat(rows), sequence, batch));
}

struct ArrayFireLogSoftmax {
    af::array log_probabilities;
    af::array probabilities;
};

ArrayFireLogSoftmax StableLogSoftmaxRows(const af::array& predictions) {
    const unsigned classes =
        static_cast<unsigned>(predictions.dims(1));
    const af::array row_max = af::max(predictions, 1);
    const af::array shifted =
        predictions - af::tile(row_max, 1, classes);
    const af::array log_denominator = af::log(af::sum(af::exp(shifted), 1));
    af::array log_probabilities =
        shifted - af::tile(log_denominator, 1, classes);
    af::array probabilities = af::exp(log_probabilities);
    log_probabilities.eval();
    probabilities.eval();
    return {log_probabilities, probabilities};
}

struct ArrayFireCrossEntropyTargets {
    af::array weighted_targets;
    af::array mean_denominator_rows;
};

ArrayFireCrossEntropyTargets BuildArrayFireCrossEntropyTargets(
    const af::array& predictions,
    const af::array& targets,
    bool targets_are_class_indices,
    const std::vector<float>& class_weights,
    float label_smoothing,
    int ignore_index,
    Tensor& cached_class_weights) {
    const dim_t batch_size = predictions.dims(0);
    const dim_t classes = predictions.dims(1);

    af::array target_distribution;
    af::array valid_rows = af::constant(1.0f, batch_size, 1, f32);
    if (targets_are_class_indices) {
        const af::array target_indices = af::flat(targets.as(s32));
        valid_rows = (target_indices != ignore_index).as(f32);
        const af::array safe_target_indices =
            target_indices * valid_rows.as(s32);
        const af::array identity = af::identity(classes, classes, f32);
        target_distribution =
            af::transpose(identity(af::span, safe_target_indices));
    } else {
        target_distribution = targets.as(f32);
    }

    af::array mean_denominator_rows;
    af::array tiled_weights;
    if (!class_weights.empty()) {
        const std::vector<size_t> expected_shape = {
            1, static_cast<size_t>(classes)};
        if (cached_class_weights.Shape() != expected_shape) {
            cached_class_weights = Tensor(
                expected_shape,
                class_weights.data(),
                DataType::Float32);
        }
        const af::array weights = cached_class_weights.GetSemanticArray();
        tiled_weights =
            af::tile(weights, static_cast<unsigned>(batch_size), 1);
    }
    if (targets_are_class_indices) {
        mean_denominator_rows = class_weights.empty()
            ? valid_rows
            : af::sum(target_distribution * tiled_weights, 1) * valid_rows;
    } else {
        mean_denominator_rows = valid_rows;
    }

    if (label_smoothing > 0.0f) {
        target_distribution =
            target_distribution * (1.0f - label_smoothing) +
            label_smoothing / static_cast<float>(classes);
    }
    if (targets_are_class_indices) {
        target_distribution =
            target_distribution *
            af::tile(valid_rows, 1, static_cast<unsigned>(classes));
    }

    if (!class_weights.empty()) {
        target_distribution = target_distribution * tiled_weights;
    }

    target_distribution.eval();
    mean_denominator_rows.eval();
    return {target_distribution, mean_denominator_rows};
}

// Class-index cross entropy along the class axis, in the tensor's own
// layout (TOFIX118 P8). The general path reorders the logits into rows and
// back (full copies) and builds one-hot targets from a classes x classes
// identity matrix (8192^2 floats per call for a Berean vocabulary); this one
// reads the logits twice. Used when targets are class ids, no class weights,
// no label smoothing, reduction Mean or Sum.
struct ClassIndexCrossEntropy {
    af::array log_sum_exp;   // per position, shaped like the logits minus the class axis
    af::array linear_target; // flat index of each position's target logit
    af::array valid;         // f32 1/0 per position (ignore_index -> 0)
    af::array denominator;   // valid position count (Mean)
    int class_axis = 0;
    dim_t classes = 0;
};

ClassIndexCrossEntropy PrepareClassIndexCrossEntropy(const af::array& logits, const af::array& targets,
                                                     size_t rank, int ignore_index) {
    ClassIndexCrossEntropy prepared;
    prepared.class_axis = static_cast<int>(rank) - 1;
    prepared.classes = logits.dims(prepared.class_axis);
    const dim_t positions = logits.elements() / prepared.classes;
    af::dim4 tile_dims(1, 1, 1, 1);
    tile_dims[prepared.class_axis] = static_cast<dim_t>(prepared.classes);

    const af::array row_max = af::max(logits, prepared.class_axis);
    prepared.log_sum_exp =
        row_max + af::log(af::sum(af::exp(logits - af::tile(row_max, tile_dims)), prepared.class_axis));

    const af::array ids = af::flat(targets).as(s32);
    const af::array valid_mask = ids != ignore_index;
    prepared.valid = valid_mask.as(f32);
    const af::array safe_ids = af::select(valid_mask, ids, 0.0).as(s32);
    // Column-major: position p's class c sits at p + c * positions.
    prepared.linear_target = af::range(af::dim4(positions), 0, s32) + safe_ids * static_cast<int>(positions);
    prepared.denominator = af::sum(prepared.valid);
    return prepared;
}

// The same device buffer (not just equal values): Backward reuses Forward's
// log-sum-exp only for the logits and targets it was computed from. The
// cache holds both arrays, so their buffers cannot be recycled meanwhile.
bool SameDeviceBuffer(const af::array& a, const af::array& b) {
    return a.dims() == b.dims() && a.type() == b.type() && af::getRawPtr(a) == af::getRawPtr(b);
}

af::array ClassIndexCrossEntropyLoss(const af::array& logits, const ClassIndexCrossEntropy& prepared,
                                     bool mean) {
    const af::array target_logit = af::lookup(af::flat(logits), prepared.linear_target);
    const af::array per_position = (af::flat(prepared.log_sum_exp) - target_logit) * prepared.valid;
    const af::array total = af::sum(per_position);
    return mean ? total / prepared.denominator : total;
}

af::array ClassIndexCrossEntropyGradient(const af::array& logits, const ClassIndexCrossEntropy& prepared,
                                         bool mean) {
    af::dim4 tile_dims(1, 1, 1, 1);
    tile_dims[prepared.class_axis] = static_cast<dim_t>(prepared.classes);
    af::dim4 position_dims = logits.dims();
    position_dims[prepared.class_axis] = 1;
    // Same guard as the general path: an all-ignored batch divides by 1.
    const af::array scale = mean
        ? 1.0f / (prepared.denominator + (prepared.denominator == 0.0f).as(f32))
        : af::constant(1.0f, 1, f32);
    const af::array position_scale = af::moddims(prepared.valid, position_dims) *
                                     af::tile(scale, position_dims);
    af::array gradient = af::exp(logits - af::tile(prepared.log_sum_exp, tile_dims)) *
                         af::tile(position_scale, tile_dims);
    af::array flat_gradient = af::flat(gradient);
    flat_gradient(prepared.linear_target) =
        flat_gradient(prepared.linear_target) - af::flat(position_scale);
    return af::moddims(flat_gradient, logits.dims());
}

af::array ApplyWeightedCrossEntropyReduction(
    const af::array& per_sample_loss,
    const af::array& mean_denominator_rows,
    Reduction reduction,
    af::array* mean_denominator) {
    if (reduction == Reduction::None) {
        return per_sample_loss;
    }

    af::array total = af::sum(af::flat(per_sample_loss));
    total.eval();
    if (reduction != Reduction::Mean) {
        return total;
    }

    af::array denominator = af::sum(af::flat(mean_denominator_rows));
    denominator.eval();
    if (mean_denominator != nullptr) {
        *mean_denominator = denominator;
    }
    return total / denominator;
}
#endif

} // namespace

// ============================================================================
// Cross Entropy Loss Implementation
// ============================================================================

#ifdef CYXWIZ_HAS_ARRAYFIRE
struct CrossEntropyLoss::ClassIndexForwardCache {
    af::array logits;
    af::array target_ids;
    ClassIndexCrossEntropy prepared;
};
#else
struct CrossEntropyLoss::ClassIndexForwardCache {};
#endif

Tensor CrossEntropyLoss::Forward(const Tensor& predictions, const Tensor& targets) {
    has_cached_mean_denominator_ = false;
    const ClassAxisShape shape =
        ValidateClassAxisPredictions(predictions, "CrossEntropy");
    ValidateClassWeights(class_weights_, shape.classes, "CrossEntropy");
    const bool class_indices = TargetsAreClassIndices(predictions, targets);
    if (class_indices) {
        ValidateClassIndexTargets(targets, shape, "CrossEntropy");
    } else {
        ValidateFloat32Pair(predictions, targets, "CrossEntropy");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        if (class_indices && class_weights_.empty() && label_smoothing_ == 0.0f &&
            (reduction_ == Reduction::Mean || reduction_ == Reduction::Sum)) {
            const af::array logits = TensorToAf(predictions);
            const af::array target_ids = TensorToAf(targets);
            auto prepared = PrepareClassIndexCrossEntropy(
                logits, target_ids, predictions.Shape().size(), ignore_index_);
            af::array loss = ClassIndexCrossEntropyLoss(logits, prepared, reduction_ == Reduction::Mean);
            // One eval per array: af::eval of several arrays requires equal
            // shapes ("Invalid input size" on every backend).
            loss.eval();
            prepared.log_sum_exp.eval();
            prepared.linear_target.eval();
            prepared.valid.eval();
            cached_softmax_ = Tensor();  // the fast path does not store probabilities
            class_index_cache_ = std::make_shared<ClassIndexForwardCache>(
                ClassIndexForwardCache{logits, target_ids, prepared});
            if (reduction_ == Reduction::Mean) {
                af::array denominator = prepared.denominator;
                denominator.eval();
                cached_mean_denominator_ = Tensor::FromSemanticArray(denominator, {1});
                has_cached_mean_denominator_ = true;
            }
            return Tensor::FromSemanticArray(loss, {1});
        }
        const af::array prediction_rows = ToCrossEntropyRows(
            TensorToAf(predictions), predictions.Shape());
        const af::array target_rows = class_indices
            ? ToCrossEntropyIndexRows(TensorToAf(targets), predictions.Shape())
            : ToCrossEntropyRows(TensorToAf(targets), targets.Shape());
        const auto normalized = StableLogSoftmaxRows(prediction_rows);
        const af::array semantic_softmax = RestoreCrossEntropyClassLast(
            normalized.probabilities, predictions.Shape());
        cached_softmax_ = Tensor::FromSemanticArray(
            semantic_softmax, predictions.Shape());

        const auto weighted = BuildArrayFireCrossEntropyTargets(
            prediction_rows,
            target_rows,
            class_indices,
            class_weights_,
            label_smoothing_,
            ignore_index_,
            cached_class_weights_);
        af::array per_sample_loss = -af::sum(
            weighted.weighted_targets * normalized.log_probabilities, 1);
        per_sample_loss.eval();
        af::array mean_denominator;
        af::array loss = ApplyWeightedCrossEntropyReduction(
            per_sample_loss,
            weighted.mean_denominator_rows,
            reduction_,
            reduction_ == Reduction::Mean ? &mean_denominator : nullptr);
        loss.eval();
        if (reduction_ == Reduction::Mean) {
            cached_mean_denominator_ = Tensor::FromSemanticArray(
                mean_denominator, {1});
            has_cached_mean_denominator_ = true;
        }
        if (reduction_ == Reduction::None) {
            const af::array semantic_loss = RestoreCrossEntropyUnreduced(
                loss, predictions.Shape());
            return Tensor::FromSemanticArray(
                semantic_loss, shape.unreduced_shape);
        }
        return Tensor::FromSemanticArray(loss, {1});
    } catch (const af::exception& e) {
        loss_detail::ThrowLossDeviceError("CrossEntropyLoss::Forward", e);
    }
#else
    loss_detail::ThrowLossNeedsArrayFire("CrossEntropyLoss::Forward");
#endif
}

Tensor CrossEntropyLoss::Backward(const Tensor& predictions, const Tensor& targets) {
    const ClassAxisShape shape =
        ValidateClassAxisPredictions(predictions, "CrossEntropy");
    ValidateClassWeights(class_weights_, shape.classes, "CrossEntropy");
    const bool class_indices = TargetsAreClassIndices(predictions, targets);
    if (class_indices) {
        ValidateClassIndexTargets(targets, shape, "CrossEntropy");
    } else {
        ValidateFloat32Pair(predictions, targets, "CrossEntropy");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        if (class_indices && class_weights_.empty() && label_smoothing_ == 0.0f &&
            (reduction_ == Reduction::Mean || reduction_ == Reduction::Sum)) {
            ScopedProfileSpan span("CrossEntropy.gradient");
            const af::array logits = TensorToAf(predictions);
            const af::array target_ids = TensorToAf(targets);
            const auto cache = std::move(class_index_cache_);
            const bool reuse = cache && SameDeviceBuffer(cache->logits, logits) &&
                               SameDeviceBuffer(cache->target_ids, target_ids);
            const auto prepared = reuse ? cache->prepared
                                        : PrepareClassIndexCrossEntropy(logits, target_ids,
                                                                        predictions.Shape().size(), ignore_index_);
            af::array gradient = ClassIndexCrossEntropyGradient(logits, prepared, reduction_ == Reduction::Mean);
            gradient.eval();
            return Tensor::FromSemanticArray(gradient, predictions.Shape());
        }
        const af::array prediction_rows = ToCrossEntropyRows(
            TensorToAf(predictions), predictions.Shape());
        af::array softmax_rows;
        if (cached_softmax_.Shape() == predictions.Shape()) {
            softmax_rows = ToCrossEntropyRows(
                TensorToAf(cached_softmax_), predictions.Shape());
        } else {
            softmax_rows = StableLogSoftmaxRows(prediction_rows).probabilities;
        }
        const af::array target_rows = class_indices
            ? ToCrossEntropyIndexRows(TensorToAf(targets), predictions.Shape())
            : ToCrossEntropyRows(TensorToAf(targets), targets.Shape());
        const auto weighted = BuildArrayFireCrossEntropyTargets(
            prediction_rows,
            target_rows,
            class_indices,
            class_weights_,
            label_smoothing_,
            ignore_index_,
            cached_class_weights_);
        af::array grad_rows =
            softmax_rows *
                af::tile(
                    af::sum(weighted.weighted_targets, 1),
                    1,
                    static_cast<unsigned>(prediction_rows.dims(1))) -
            weighted.weighted_targets;
        if (reduction_ == Reduction::Mean) {
            af::array denominator = af::sum(
                af::flat(weighted.mean_denominator_rows));
            denominator = denominator + (denominator == 0.0f).as(f32);
            denominator.eval();
            grad_rows = grad_rows / denominator;
        }
        grad_rows.eval();
        const af::array semantic_gradient = RestoreCrossEntropyClassLast(
            grad_rows, predictions.Shape());
        return Tensor::FromSemanticArray(
            semantic_gradient, predictions.Shape());
    } catch (const af::exception& e) {
        loss_detail::ThrowLossDeviceError("CrossEntropyLoss::Backward", e);
    }
#else
    loss_detail::ThrowLossNeedsArrayFire("CrossEntropyLoss::Backward");
#endif
}

// ============================================================================
// NLL Loss Implementation
// ============================================================================

Tensor NLLLoss::Forward(const Tensor& predictions, const Tensor& targets) {
    const ClassAxisShape shape =
        ValidateClassAxisPredictions(predictions, "NLL");
    ValidateClassIndexTargets(targets, shape, "NLL");

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array log_probability_rows = ToCrossEntropyRows(
            TensorToAf(predictions), predictions.Shape());
        const af::array target_indices = ToCrossEntropyIndexRows(
            TensorToAf(targets), predictions.Shape()).as(s32);
        const af::array valid_rows =
            (target_indices != ignore_index_).as(f32);
        const af::array safe_target_indices =
            target_indices * valid_rows.as(s32);
        const af::array identity = af::identity(
            static_cast<dim_t>(shape.classes),
            static_cast<dim_t>(shape.classes), f32);
        af::array target_rows =
            af::transpose(identity(af::span, safe_target_indices));
        target_rows = target_rows * af::tile(
            valid_rows, 1, static_cast<unsigned>(shape.classes));
        af::array per_sample_loss =
            -af::sum(log_probability_rows * target_rows, 1);
        per_sample_loss.eval();

        if (reduction_ == Reduction::None) {
            const af::array semantic_loss = RestoreCrossEntropyUnreduced(
                per_sample_loss, predictions.Shape());
            return Tensor::FromSemanticArray(
                semantic_loss, shape.unreduced_shape);
        }
        af::array loss = af::sum(af::flat(per_sample_loss));
        if (reduction_ == Reduction::Mean) {
            loss = loss / af::sum(af::flat(valid_rows));
        }
        loss.eval();
        return Tensor::FromSemanticArray(loss, {1});
    } catch (const af::exception& e) {
        loss_detail::ThrowLossDeviceError("NLLLoss::Forward", e);
    }
#else
    loss_detail::ThrowLossNeedsArrayFire("NLLLoss::Forward");
#endif
}

Tensor NLLLoss::Backward(const Tensor& predictions, const Tensor& targets) {
    const ClassAxisShape shape =
        ValidateClassAxisPredictions(predictions, "NLL");
    ValidateClassIndexTargets(targets, shape, "NLL");

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array target_indices = ToCrossEntropyIndexRows(
            TensorToAf(targets), predictions.Shape()).as(s32);
        const af::array valid_rows =
            (target_indices != ignore_index_).as(f32);
        const af::array safe_target_indices =
            target_indices * valid_rows.as(s32);
        const af::array identity = af::identity(
            static_cast<dim_t>(shape.classes),
            static_cast<dim_t>(shape.classes), f32);
        af::array grad_rows = -af::transpose(
            identity(af::span, safe_target_indices));
        grad_rows = grad_rows * af::tile(
            valid_rows, 1, static_cast<unsigned>(shape.classes));
        if (reduction_ == Reduction::Mean) {
            af::array denominator = af::sum(af::flat(valid_rows));
            denominator = denominator + (denominator == 0.0f).as(f32);
            grad_rows = grad_rows / denominator;
        }
        grad_rows.eval();
        return Tensor::FromSemanticArray(
            RestoreCrossEntropyClassLast(grad_rows, predictions.Shape()),
            predictions.Shape());
    } catch (const af::exception& e) {
        loss_detail::ThrowLossDeviceError("NLLLoss::Backward", e);
    }
#else
    loss_detail::ThrowLossNeedsArrayFire("NLLLoss::Backward");
#endif
}

// ============================================================================
// Focal Loss Implementation
// ============================================================================

Tensor FocalLoss::Forward(const Tensor& predictions, const Tensor& targets) {
    const ClassAxisShape shape =
        ValidateClassAxisPredictions(predictions, "Focal");
    ValidateClassIndexTargets(targets, shape, "Focal");
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array prediction_rows = ToCrossEntropyRows(
            TensorToAf(predictions), predictions.Shape());
        const ArrayFireLogSoftmax normalized =
            StableLogSoftmaxRows(prediction_rows);
        const af::array target_indices = ToCrossEntropyIndexRows(
            TensorToAf(targets), predictions.Shape()).as(s32);
        const af::array identity = af::identity(
            static_cast<dim_t>(shape.classes),
            static_cast<dim_t>(shape.classes), f32);
        const af::array target_rows =
            af::transpose(identity(af::span, target_indices));
        const af::array pt = af::sum(
            normalized.probabilities * target_rows, 1);
        const af::array log_pt = af::sum(
            normalized.log_probabilities * target_rows, 1);

        // Focal loss: -alpha * (1 - pt)^gamma * log(pt)
        af::array focal_weight = af::pow(1.0f - pt, gamma_);
        af::array per_sample_loss = -alpha_ * focal_weight * log_pt;
        per_sample_loss.eval();
        if (reduction_ == Reduction::None) {
            return Tensor::FromSemanticArray(
                RestoreCrossEntropyUnreduced(
                    per_sample_loss, predictions.Shape()),
                shape.unreduced_shape);
        }
        const af::array loss = ApplyReduction(per_sample_loss, reduction_);
        return Tensor::FromSemanticArray(loss, {1});
    } catch (const af::exception& e) {
        loss_detail::ThrowLossDeviceError("FocalLoss::Forward", e);
    }
#else
    loss_detail::ThrowLossNeedsArrayFire("FocalLoss::Forward");
#endif
}

Tensor FocalLoss::Backward(const Tensor& predictions, const Tensor& targets) {
    const ClassAxisShape shape =
        ValidateClassAxisPredictions(predictions, "Focal");
    ValidateClassIndexTargets(targets, shape, "Focal");
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array prediction_rows = ToCrossEntropyRows(
            TensorToAf(predictions), predictions.Shape());
        const ArrayFireLogSoftmax normalized =
            StableLogSoftmaxRows(prediction_rows);
        const af::array target_indices = ToCrossEntropyIndexRows(
            TensorToAf(targets), predictions.Shape()).as(s32);
        const af::array identity = af::identity(
            static_cast<dim_t>(shape.classes),
            static_cast<dim_t>(shape.classes), f32);
        const af::array target_rows =
            af::transpose(identity(af::span, target_indices));
        const af::array pt = af::sum(
            normalized.probabilities * target_rows, 1);
        af::array log_pt = af::sum(
            normalized.log_probabilities * target_rows, 1);

        // d_loss/d_pred = alpha * [(1-pt)^gamma - gamma*pt*(1-pt)^(gamma-1)*log(pt)] * (p - y)
        log_pt.eval();
        af::array one_minus_pt = 1.0f - pt;
        one_minus_pt.eval();
        af::array scale = gamma_ == 0.0f
            ? af::constant(alpha_, pt.dims(), f32)
            : alpha_ * (af::pow(one_minus_pt, gamma_) -
                        gamma_ * pt *
                            af::pow(one_minus_pt, gamma_ - 1.0f) * log_pt);
        scale.eval();

        af::array grad_rows = af::tile(
            scale, 1, static_cast<unsigned>(shape.classes)) *
            (normalized.probabilities - target_rows);
        grad_rows.eval();

        if (reduction_ == Reduction::Mean) {
            grad_rows = grad_rows / static_cast<float>(shape.batch);
            grad_rows.eval();
        }

        return Tensor::FromSemanticArray(
            RestoreCrossEntropyClassLast(
                grad_rows, predictions.Shape()),
            predictions.Shape());
    } catch (const af::exception& e) {
        loss_detail::ThrowLossDeviceError("FocalLoss::Backward", e);
    }
#else
    loss_detail::ThrowLossNeedsArrayFire("FocalLoss::Backward");
#endif
}

} // namespace cyxwiz
