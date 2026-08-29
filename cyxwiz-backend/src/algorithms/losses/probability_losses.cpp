#include "cyxwiz/losses/probability.h"
#include "loss_utils.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
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

BCELoss::BCELoss(Reduction reduction, float denominator_epsilon)
    : Loss(reduction), denominator_epsilon_(denominator_epsilon) {
    if (!std::isfinite(denominator_epsilon_) ||
        denominator_epsilon_ <= 0.0f) {
        throw std::invalid_argument(
            "BCELoss denominator epsilon must be finite and positive");
    }
}

BCEWithLogitsLoss::BCEWithLogitsLoss(Reduction reduction, float pos_weight)
    : Loss(reduction), pos_weight_(pos_weight) {
    SetPosWeight(pos_weight);
}

void BCEWithLogitsLoss::SetPosWeight(float pos_weight) {
    if (!std::isfinite(pos_weight) || pos_weight <= 0.0f) {
        throw std::invalid_argument(
            "BCEWithLogitsLoss pos_weight must be finite and positive");
    }
    pos_weight_ = pos_weight;
}

SoftDiceLoss::SoftDiceLoss(Reduction reduction, float smooth)
    : Loss(reduction), smooth_(smooth) {
    if (!std::isfinite(smooth_) || smooth_ < 0.0f) {
        throw std::invalid_argument(
            "SoftDiceLoss smooth must be finite and non-negative");
    }
}

JaccardLoss::JaccardLoss(Reduction reduction, float smooth)
    : Loss(reduction), smooth_(smooth) {
    if (!std::isfinite(smooth_) || smooth_ < 0.0f) {
        throw std::invalid_argument(
            "JaccardLoss smooth must be finite and non-negative");
    }
}

// ============================================================================
// BCE Loss Implementation
// ============================================================================

Tensor BCELoss::Forward(const Tensor& predictions, const Tensor& targets) {
    ValidateFloat32Pair(predictions, targets, "BCE");
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array pred = TensorToAf(predictions);
        af::array target = TensorToAf(targets);

        // PyTorch BCELoss clamps the logarithm to -100 while retaining the
        // public probability value for its separately bounded derivative.
        af::array log_prediction = af::max(af::log(pred), -100.0f);
        af::array log_complement =
            af::max(af::log(1.0f - pred), -100.0f);
        af::array loss = -(target * log_prediction +
                          (1.0f - target) * log_complement);
        loss.eval();

        loss = ApplyReduction(loss, reduction_);

        return AfToTensor(
            loss,
            reduction_ == Reduction::None
                ? predictions.Shape()
                : std::vector<size_t>{1});
    } catch (const af::exception& e) {
        LogArrayFireLossFallbackOnce(
            "BCELoss::Forward", e.what(), predictions, targets, reduction_);
    }
#endif
    return CpuBCEForward(predictions, targets, reduction_);
}

Tensor BCELoss::Backward(const Tensor& predictions, const Tensor& targets) {
    ValidateFloat32Pair(predictions, targets, "BCE");
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array pred = TensorToAf(predictions);
        af::array target = TensorToAf(targets);

        // Match PyTorch's bounded BCE derivative denominator.
        af::array denominator = af::max(
            pred * (1.0f - pred), denominator_epsilon_);
        af::array grad = (pred - target) / denominator;
        grad.eval();

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(pred.elements());
            grad.eval();
        }

        return AfToTensor(grad, predictions.Shape());
    } catch (const af::exception& e) {
        LogArrayFireLossFallbackOnce(
            "BCELoss::Backward", e.what(), predictions, targets, reduction_);
    }
#endif
    return CpuBCEBackward(
        predictions, targets, denominator_epsilon_, reduction_);
}

// ============================================================================
// BCE With Logits Loss Implementation
// ============================================================================

Tensor BCEWithLogitsLoss::Forward(const Tensor& predictions, const Tensor& targets) {
    ValidateFloat32Pair(predictions, targets, "BCEWithLogits");
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array logits = TensorToAf(predictions);
        af::array target = TensorToAf(targets);

        // Stable weighted BCEWithLogits, matching the CPU reference and
        // supporting binary as well as fractional targets.
        af::array log_weight =
            1.0f + (pos_weight_ - 1.0f) * target;
        af::array softplus_negative =
            af::max(-logits, 0.0f) +
            af::log(1.0f + af::exp(-af::abs(logits)));
        af::array loss =
            (1.0f - target) * logits +
            log_weight * softplus_negative;
        loss.eval();

        loss = ApplyReduction(loss, reduction_);

        return AfToTensor(
            loss,
            reduction_ == Reduction::None
                ? predictions.Shape()
                : std::vector<size_t>{1});
    } catch (const af::exception& e) {
        LogArrayFireLossFallbackOnce(
            "BCEWithLogitsLoss::Forward", e.what(), predictions, targets, reduction_);
    }
#endif
    return CpuBCEWithLogitsForward(predictions, targets, reduction_, pos_weight_);
}

Tensor BCEWithLogitsLoss::Backward(const Tensor& predictions, const Tensor& targets) {
    ValidateFloat32Pair(predictions, targets, "BCEWithLogits");
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array logits = TensorToAf(predictions);
        af::array target = TensorToAf(targets);

        af::array log_weight =
            1.0f + (pos_weight_ - 1.0f) * target;
        af::array sigmoid_logits = af::sigmoid(logits);
        af::array grad =
            (1.0f - target) +
            log_weight * (sigmoid_logits - 1.0f);
        grad.eval();

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(logits.elements());
            grad.eval();
        }

        return AfToTensor(grad, predictions.Shape());
    } catch (const af::exception& e) {
        LogArrayFireLossFallbackOnce(
            "BCEWithLogitsLoss::Backward", e.what(), predictions, targets, reduction_);
    }
#endif
    return CpuBCEWithLogitsBackward(predictions, targets, reduction_, pos_weight_);
}

// ============================================================================
// KL Divergence Loss Implementation
// ============================================================================

Tensor KLDivLoss::Forward(const Tensor& predictions, const Tensor& targets) {
    ValidateFloat32Pair(predictions, targets, "KLDiv");
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array log_pred = TensorToAf(predictions);  // Log probabilities
        af::array target = TensorToAf(targets);        // Probabilities or log probabilities

        af::array loss;
        if (log_target_) {
            // KL = exp(target) * (target - pred)
            af::array target_prob = af::exp(target);
            target_prob.eval();
            loss = target_prob * (target - log_pred);
        } else {
            // KL = target * (log(target) - pred)
            // xlogy semantics define the zero-target contribution as zero;
            // negative targets remain NaN, matching PyTorch.
            loss = af::select(
                target == 0.0f,
                0.0f,
                target * (af::log(target) - log_pred));
        }
        loss.eval();
        loss = ApplyReduction(loss, reduction_);

        return AfToTensor(
            loss,
            reduction_ == Reduction::None
                ? predictions.Shape()
                : std::vector<size_t>{1});
    } catch (const af::exception& e) {
        LogArrayFireLossFallbackOnce(
            "KLDivLoss::Forward", e.what(), predictions, targets, reduction_);
    }
#endif
    return CpuKLDivForward(predictions, targets, log_target_, reduction_);
}

Tensor KLDivLoss::Backward(const Tensor& predictions, const Tensor& targets) {
    ValidateFloat32Pair(predictions, targets, "KLDiv");
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array log_pred = TensorToAf(predictions);
        af::array target = TensorToAf(targets);

        // Gradient w.r.t. log_pred: -target (or -exp(target) if log_target)
        af::array grad;
        if (log_target_) {
            grad = -af::exp(target);
        } else {
            grad = -target;
        }
        grad.eval();

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(log_pred.elements());
            grad.eval();
        }

        return AfToTensor(grad, predictions.Shape());
    } catch (const af::exception& e) {
        LogArrayFireLossFallbackOnce(
            "KLDivLoss::Backward", e.what(), predictions, targets, reduction_);
    }
#endif
    return CpuKLDivBackward(predictions, targets, log_target_, reduction_);
}

// ============================================================================
// Soft Dice Loss Implementation
// ============================================================================

namespace {

std::vector<size_t> SoftDiceLossShape(size_t batch) {
    return batch == 1 ? std::vector<size_t>{1} : std::vector<size_t>{batch};
}

size_t SoftDiceBatchSize(const Tensor& predictions) {
    const auto& shape = predictions.Shape();
    if (shape.empty() || shape.size() == 1) {
        return 1;
    }
    return shape[0];
}

size_t SoftDiceSampleSize(const Tensor& predictions) {
    return predictions.NumElements() / SoftDiceBatchSize(predictions);
}

void ValidateSoftDiceInputs(const Tensor& predictions,
                            const Tensor& targets,
                            float smooth) {
    if (predictions.GetDataType() != DataType::Float32 ||
        targets.GetDataType() != DataType::Float32) {
        throw std::runtime_error(
            "SoftDiceLoss requires Float32 predictions and targets");
    }
    if (predictions.Shape() != targets.Shape()) {
        throw std::runtime_error(
            "SoftDiceLoss predictions and targets must have identical shapes");
    }
    if (predictions.NumElements() == 0) {
        throw std::runtime_error("SoftDiceLoss requires non-empty tensors");
    }
    if (!std::isfinite(smooth) || smooth < 0.0f) {
        throw std::runtime_error("SoftDiceLoss smooth must be finite and non-negative");
    }
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
af::array ReduceOverlapSamples(
    const af::array& values,
    const std::vector<size_t>& semantic_shape) {
    if (semantic_shape.size() <= 1) {
        af::array result = af::sum(af::flat(values));
        result.eval();
        return result;
    }

    af::array result = values;
    for (int axis = static_cast<int>(semantic_shape.size()) - 1;
         axis >= 1;
         --axis) {
        result = af::sum(result, axis);
        result.eval();
    }
    return result;
}

af::array TileOverlapSamples(
    const af::array& per_sample,
    const std::vector<size_t>& semantic_shape) {
    af::dim4 factors(1, 1, 1, 1);
    if (semantic_shape.size() == 1) {
        factors[0] = static_cast<dim_t>(semantic_shape[0]);
    } else {
        for (size_t axis = 1; axis < semantic_shape.size(); ++axis) {
            factors[static_cast<unsigned>(axis)] =
                static_cast<dim_t>(semantic_shape[axis]);
        }
    }
    af::array result = af::tile(per_sample, factors);
    result.eval();
    return result;
}

Tensor WrapOverlapLoss(const af::array& per_sample,
                       Reduction reduction,
                       size_t batch) {
    const af::array reduced = ApplyReduction(per_sample, reduction);
    return Tensor::FromSemanticArray(
        reduced,
        reduction == Reduction::None
            ? SoftDiceLossShape(batch)
            : std::vector<size_t>{1});
}
#endif

}  // namespace

Tensor SoftDiceLoss::Forward(const Tensor& predictions, const Tensor& targets) {
    constexpr const char* kOperation = "SoftDiceLoss::Forward";
    ValidateSoftDiceInputs(predictions, targets, smooth_);

    const size_t batch = SoftDiceBatchSize(predictions);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const bool use_native_cpu = PrepareLossNativeCpuFallback(
        kOperation, predictions, targets, reduction_);
    if (!use_native_cpu) try {
        const af::array prediction_values = TensorToAf(predictions);
        const af::array target_values = TensorToAf(targets);
        const af::array intersection = ReduceOverlapSamples(
            prediction_values * target_values, predictions.Shape());
        const af::array prediction_sum = ReduceOverlapSamples(
            prediction_values, predictions.Shape());
        const af::array target_sum = ReduceOverlapSamples(
            target_values, targets.Shape());
        af::array per_sample = 1.0f -
            (2.0f * intersection + smooth_) /
                (prediction_sum + target_sum + smooth_);
        per_sample.eval();
        return WrapOverlapLoss(per_sample, reduction_, batch);
    } catch (const af::exception& e) {
        LogArrayFireLossFallbackOnce(
            kOperation, e.what(), predictions, targets, reduction_);
    }
#endif

    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LossCpuPath, kOperation);
    const size_t sample_size = SoftDiceSampleSize(predictions);
    const float* pred = predictions.Data<float>();
    const float* target = targets.Data<float>();
    std::vector<float> per_sample(batch, 0.0f);

    for (size_t b = 0; b < batch; ++b) {
        const size_t base = b * sample_size;
        float intersection = 0.0f;
        float pred_sum = 0.0f;
        float target_sum = 0.0f;
        for (size_t i = 0; i < sample_size; ++i) {
            const float p = pred[base + i];
            const float t = target[base + i];
            intersection += p * t;
            pred_sum += p;
            target_sum += t;
        }

        const float numerator = 2.0f * intersection + smooth_;
        const float denominator = pred_sum + target_sum + smooth_;
        per_sample[b] = 1.0f - numerator / denominator;
    }

    if (reduction_ == Reduction::None) {
        return Tensor(SoftDiceLossShape(batch), per_sample.data(), DataType::Float32);
    }

    float total = 0.0f;
    for (float value : per_sample) {
        total += value;
    }
    if (reduction_ == Reduction::Mean && batch > 0) {
        total /= static_cast<float>(batch);
    }
    return Tensor({1}, &total, DataType::Float32);
}

Tensor SoftDiceLoss::Backward(const Tensor& predictions, const Tensor& targets) {
    constexpr const char* kOperation = "SoftDiceLoss::Backward";
    ValidateSoftDiceInputs(predictions, targets, smooth_);

    const size_t batch = SoftDiceBatchSize(predictions);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const bool use_native_cpu = PrepareLossNativeCpuFallback(
        kOperation, predictions, targets, reduction_);
    if (!use_native_cpu) try {
        const af::array prediction_values = TensorToAf(predictions);
        const af::array target_values = TensorToAf(targets);
        const af::array intersection = ReduceOverlapSamples(
            prediction_values * target_values, predictions.Shape());
        const af::array prediction_sum = ReduceOverlapSamples(
            prediction_values, predictions.Shape());
        const af::array target_sum = ReduceOverlapSamples(
            target_values, targets.Shape());
        const af::array numerator = 2.0f * intersection + smooth_;
        const af::array denominator =
            prediction_sum + target_sum + smooth_;
        const af::array numerator_tiled = TileOverlapSamples(
            numerator, predictions.Shape());
        const af::array denominator_tiled = TileOverlapSamples(
            denominator, predictions.Shape());
        af::array gradient =
            -((2.0f * target_values * denominator_tiled) -
              numerator_tiled) /
            (denominator_tiled * denominator_tiled);
        if (reduction_ == Reduction::Mean) {
            gradient = gradient / static_cast<float>(batch);
        }
        gradient.eval();
        return Tensor::FromSemanticArray(gradient, predictions.Shape());
    } catch (const af::exception& e) {
        LogArrayFireLossFallbackOnce(
            kOperation, e.what(), predictions, targets, reduction_);
    }
#endif

    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LossCpuPath, kOperation);
    const size_t sample_size = SoftDiceSampleSize(predictions);
    const float* pred = predictions.Data<float>();
    const float* target = targets.Data<float>();
    Tensor grad(predictions.Shape(), DataType::Float32);
    float* out = grad.Data<float>();

    for (size_t b = 0; b < batch; ++b) {
        const size_t base = b * sample_size;
        float intersection = 0.0f;
        float pred_sum = 0.0f;
        float target_sum = 0.0f;
        for (size_t i = 0; i < sample_size; ++i) {
            const float p = pred[base + i];
            const float t = target[base + i];
            intersection += p * t;
            pred_sum += p;
            target_sum += t;
        }

        const float numerator = 2.0f * intersection + smooth_;
        const float denominator = pred_sum + target_sum + smooth_;
        const float denom_sq = denominator * denominator;
        const float reduction_scale =
            reduction_ == Reduction::Mean && batch > 0
                ? 1.0f / static_cast<float>(batch)
                : 1.0f;

        for (size_t i = 0; i < sample_size; ++i) {
            const float t = target[base + i];
            out[base + i] =
                -((2.0f * t * denominator) - numerator) / denom_sq *
                reduction_scale;
        }
    }

    return grad;
}

TverskyLoss::TverskyLoss(Reduction reduction,
                         float alpha,
                         float beta,
                         float smooth)
    : Loss(reduction), alpha_(alpha), beta_(beta), smooth_(smooth) {
    if (!std::isfinite(alpha_) || alpha_ < 0.0f) {
        throw std::invalid_argument(
            "TverskyLoss alpha must be finite and non-negative");
    }
    if (!std::isfinite(beta_) || beta_ < 0.0f) {
        throw std::invalid_argument(
            "TverskyLoss beta must be finite and non-negative");
    }
    if (!std::isfinite(smooth_) || smooth_ < 0.0f) {
        throw std::invalid_argument(
            "TverskyLoss smooth must be finite and non-negative");
    }
}

Tensor TverskyLoss::Forward(const Tensor& predictions, const Tensor& targets) {
    constexpr const char* kOperation = "TverskyLoss::Forward";
    ValidateSoftDiceInputs(predictions, targets, smooth_);

    const size_t batch = SoftDiceBatchSize(predictions);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const bool use_native_cpu = PrepareLossNativeCpuFallback(
        kOperation, predictions, targets, reduction_);
    if (!use_native_cpu) try {
        const af::array prediction_values = TensorToAf(predictions);
        const af::array target_values = TensorToAf(targets);
        const af::array true_positive = ReduceOverlapSamples(
            prediction_values * target_values, predictions.Shape());
        const af::array false_positive = ReduceOverlapSamples(
            prediction_values * (1.0f - target_values),
            predictions.Shape());
        const af::array false_negative = ReduceOverlapSamples(
            (1.0f - prediction_values) * target_values,
            predictions.Shape());
        af::array per_sample = 1.0f -
            (true_positive + smooth_) /
                (true_positive + alpha_ * false_positive +
                 beta_ * false_negative + smooth_);
        per_sample.eval();
        return WrapOverlapLoss(per_sample, reduction_, batch);
    } catch (const af::exception& e) {
        LogArrayFireLossFallbackOnce(
            kOperation, e.what(), predictions, targets, reduction_);
    }
#endif

    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LossCpuPath, kOperation);
    const size_t sample_size = SoftDiceSampleSize(predictions);
    const float* pred = predictions.Data<float>();
    const float* target = targets.Data<float>();
    std::vector<float> per_sample(batch, 0.0f);

    for (size_t b = 0; b < batch; ++b) {
        const size_t base = b * sample_size;
        float true_positive = 0.0f;
        float false_positive = 0.0f;
        float false_negative = 0.0f;

        for (size_t i = 0; i < sample_size; ++i) {
            const float p = pred[base + i];
            const float t = target[base + i];
            true_positive += p * t;
            false_positive += p * (1.0f - t);
            false_negative += (1.0f - p) * t;
        }

        const float numerator = true_positive + smooth_;
        const float denominator = true_positive +
            alpha_ * false_positive + beta_ * false_negative + smooth_;
        per_sample[b] = 1.0f - numerator / denominator;
    }

    if (reduction_ == Reduction::None) {
        return Tensor(SoftDiceLossShape(batch), per_sample.data(), DataType::Float32);
    }

    float total = 0.0f;
    for (float value : per_sample) {
        total += value;
    }
    if (reduction_ == Reduction::Mean && batch > 0) {
        total /= static_cast<float>(batch);
    }
    return Tensor({1}, &total, DataType::Float32);
}

Tensor TverskyLoss::Backward(const Tensor& predictions, const Tensor& targets) {
    constexpr const char* kOperation = "TverskyLoss::Backward";
    ValidateSoftDiceInputs(predictions, targets, smooth_);

    const size_t batch = SoftDiceBatchSize(predictions);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const bool use_native_cpu = PrepareLossNativeCpuFallback(
        kOperation, predictions, targets, reduction_);
    if (!use_native_cpu) try {
        const af::array prediction_values = TensorToAf(predictions);
        const af::array target_values = TensorToAf(targets);
        const af::array true_positive = ReduceOverlapSamples(
            prediction_values * target_values, predictions.Shape());
        const af::array false_positive = ReduceOverlapSamples(
            prediction_values * (1.0f - target_values),
            predictions.Shape());
        const af::array false_negative = ReduceOverlapSamples(
            (1.0f - prediction_values) * target_values,
            predictions.Shape());
        const af::array numerator = true_positive + smooth_;
        const af::array denominator = true_positive +
            alpha_ * false_positive + beta_ * false_negative + smooth_;
        const af::array numerator_tiled = TileOverlapSamples(
            numerator, predictions.Shape());
        const af::array denominator_tiled = TileOverlapSamples(
            denominator, predictions.Shape());
        const af::array denominator_derivative =
            alpha_ + (1.0f - alpha_ - beta_) * target_values;
        af::array gradient =
            -((target_values * denominator_tiled) -
              (numerator_tiled * denominator_derivative)) /
            (denominator_tiled * denominator_tiled);
        if (reduction_ == Reduction::Mean) {
            gradient = gradient / static_cast<float>(batch);
        }
        gradient.eval();
        return Tensor::FromSemanticArray(gradient, predictions.Shape());
    } catch (const af::exception& e) {
        LogArrayFireLossFallbackOnce(
            kOperation, e.what(), predictions, targets, reduction_);
    }
#endif

    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LossCpuPath, kOperation);
    const size_t sample_size = SoftDiceSampleSize(predictions);
    const float* pred = predictions.Data<float>();
    const float* target = targets.Data<float>();
    Tensor grad(predictions.Shape(), DataType::Float32);
    float* out = grad.Data<float>();

    for (size_t b = 0; b < batch; ++b) {
        const size_t base = b * sample_size;
        float true_positive = 0.0f;
        float false_positive = 0.0f;
        float false_negative = 0.0f;

        for (size_t i = 0; i < sample_size; ++i) {
            const float p = pred[base + i];
            const float t = target[base + i];
            true_positive += p * t;
            false_positive += p * (1.0f - t);
            false_negative += (1.0f - p) * t;
        }

        const float numerator = true_positive + smooth_;
        const float denominator = true_positive +
            alpha_ * false_positive + beta_ * false_negative + smooth_;
        const float denom_sq = denominator * denominator;
        const float reduction_scale =
            reduction_ == Reduction::Mean && batch > 0
                ? 1.0f / static_cast<float>(batch)
                : 1.0f;

        for (size_t i = 0; i < sample_size; ++i) {
            const float t = target[base + i];
            const float d_numerator = t;
            const float d_denominator =
                alpha_ + (1.0f - alpha_ - beta_) * t;
            out[base + i] =
                -((d_numerator * denominator) -
                  (numerator * d_denominator)) /
                denom_sq * reduction_scale;
        }
    }

    return grad;
}

Tensor JaccardLoss::Forward(const Tensor& predictions, const Tensor& targets) {
    constexpr const char* kOperation = "JaccardLoss::Forward";
    ValidateSoftDiceInputs(predictions, targets, smooth_);

    const size_t batch = SoftDiceBatchSize(predictions);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const bool use_native_cpu = PrepareLossNativeCpuFallback(
        kOperation, predictions, targets, reduction_);
    if (!use_native_cpu) try {
        const af::array prediction_values = TensorToAf(predictions);
        const af::array target_values = TensorToAf(targets);
        const af::array intersection = ReduceOverlapSamples(
            prediction_values * target_values, predictions.Shape());
        const af::array prediction_sum = ReduceOverlapSamples(
            prediction_values, predictions.Shape());
        const af::array target_sum = ReduceOverlapSamples(
            target_values, targets.Shape());
        const af::array numerator = intersection + smooth_;
        const af::array denominator =
            prediction_sum + target_sum - intersection + smooth_;
        af::array per_sample = 1.0f - numerator / denominator;
        per_sample.eval();
        return WrapOverlapLoss(per_sample, reduction_, batch);
    } catch (const af::exception& e) {
        LogArrayFireLossFallbackOnce(
            kOperation, e.what(), predictions, targets, reduction_);
    }
#endif

    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LossCpuPath, kOperation);
    const size_t sample_size = SoftDiceSampleSize(predictions);
    const float* pred = predictions.Data<float>();
    const float* target = targets.Data<float>();
    std::vector<float> per_sample(batch, 0.0f);

    for (size_t b = 0; b < batch; ++b) {
        const size_t base = b * sample_size;
        float intersection = 0.0f;
        float pred_sum = 0.0f;
        float target_sum = 0.0f;

        for (size_t i = 0; i < sample_size; ++i) {
            const float p = pred[base + i];
            const float t = target[base + i];
            intersection += p * t;
            pred_sum += p;
            target_sum += t;
        }

        const float numerator = intersection + smooth_;
        const float union_value = pred_sum + target_sum - intersection;
        const float denominator = union_value + smooth_;
        per_sample[b] = 1.0f - numerator / denominator;
    }

    if (reduction_ == Reduction::None) {
        return Tensor(SoftDiceLossShape(batch), per_sample.data(), DataType::Float32);
    }

    float total = 0.0f;
    for (float value : per_sample) {
        total += value;
    }
    if (reduction_ == Reduction::Mean && batch > 0) {
        total /= static_cast<float>(batch);
    }
    return Tensor({1}, &total, DataType::Float32);
}

Tensor JaccardLoss::Backward(const Tensor& predictions, const Tensor& targets) {
    constexpr const char* kOperation = "JaccardLoss::Backward";
    ValidateSoftDiceInputs(predictions, targets, smooth_);

    const size_t batch = SoftDiceBatchSize(predictions);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const bool use_native_cpu = PrepareLossNativeCpuFallback(
        kOperation, predictions, targets, reduction_);
    if (!use_native_cpu) try {
        const af::array prediction_values = TensorToAf(predictions);
        const af::array target_values = TensorToAf(targets);
        const af::array intersection = ReduceOverlapSamples(
            prediction_values * target_values, predictions.Shape());
        const af::array prediction_sum = ReduceOverlapSamples(
            prediction_values, predictions.Shape());
        const af::array target_sum = ReduceOverlapSamples(
            target_values, targets.Shape());
        const af::array numerator = intersection + smooth_;
        const af::array denominator =
            prediction_sum + target_sum - intersection + smooth_;
        const af::array numerator_tiled = TileOverlapSamples(
            numerator, predictions.Shape());
        const af::array denominator_tiled = TileOverlapSamples(
            denominator, predictions.Shape());
        af::array gradient =
            -((target_values * denominator_tiled) -
              (numerator_tiled * (1.0f - target_values))) /
            (denominator_tiled * denominator_tiled);
        if (reduction_ == Reduction::Mean) {
            gradient = gradient / static_cast<float>(batch);
        }
        gradient.eval();
        return Tensor::FromSemanticArray(gradient, predictions.Shape());
    } catch (const af::exception& e) {
        LogArrayFireLossFallbackOnce(
            kOperation, e.what(), predictions, targets, reduction_);
    }
#endif

    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LossCpuPath, kOperation);
    const size_t sample_size = SoftDiceSampleSize(predictions);
    const float* pred = predictions.Data<float>();
    const float* target = targets.Data<float>();
    Tensor grad(predictions.Shape(), DataType::Float32);
    float* out = grad.Data<float>();

    for (size_t b = 0; b < batch; ++b) {
        const size_t base = b * sample_size;
        float intersection = 0.0f;
        float pred_sum = 0.0f;
        float target_sum = 0.0f;

        for (size_t i = 0; i < sample_size; ++i) {
            const float p = pred[base + i];
            const float t = target[base + i];
            intersection += p * t;
            pred_sum += p;
            target_sum += t;
        }

        const float numerator = intersection + smooth_;
        const float union_value = pred_sum + target_sum - intersection;
        const float denominator = union_value + smooth_;
        const float denom_sq = denominator * denominator;
        const float reduction_scale =
            reduction_ == Reduction::Mean && batch > 0
                ? 1.0f / static_cast<float>(batch)
                : 1.0f;

        for (size_t i = 0; i < sample_size; ++i) {
            const float t = target[base + i];
            const float d_numerator = t;
            const float d_denominator = 1.0f - t;
            out[base + i] =
                -((d_numerator * denominator) -
                  (numerator * d_denominator)) /
                denom_sq * reduction_scale;
        }
    }

    return grad;
}

} // namespace cyxwiz
