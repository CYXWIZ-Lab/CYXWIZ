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

}  // namespace

Tensor SoftDiceLoss::Forward(const Tensor& predictions, const Tensor& targets) {
    ValidateSoftDiceInputs(predictions, targets, smooth_);

    const size_t batch = SoftDiceBatchSize(predictions);
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
    ValidateSoftDiceInputs(predictions, targets, smooth_);

    const size_t batch = SoftDiceBatchSize(predictions);
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
    ValidateSoftDiceInputs(predictions, targets, smooth_);

    const size_t batch = SoftDiceBatchSize(predictions);
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
    ValidateSoftDiceInputs(predictions, targets, smooth_);

    const size_t batch = SoftDiceBatchSize(predictions);
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
    ValidateSoftDiceInputs(predictions, targets, smooth_);

    const size_t batch = SoftDiceBatchSize(predictions);
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
    ValidateSoftDiceInputs(predictions, targets, smooth_);

    const size_t batch = SoftDiceBatchSize(predictions);
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
