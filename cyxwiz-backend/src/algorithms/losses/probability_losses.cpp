#include "cyxwiz/losses/probability.h"
#include "loss_utils.h"

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

// ============================================================================
// BCE Loss Implementation
// ============================================================================

Tensor BCELoss::Forward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array pred = TensorToAf(predictions);
        af::array target = TensorToAf(targets);

        // Clamp predictions for numerical stability
        af::array pred_clamped = af::clamp(pred, eps_, 1.0f - eps_);
        pred_clamped.eval();

        // BCE: -[target * log(pred) + (1 - target) * log(1 - pred)]
        af::array loss = -(target * af::log(pred_clamped) +
                          (1.0f - target) * af::log(1.0f - pred_clamped));
        loss.eval();

        loss = ApplyReduction(loss, reduction_);

        return AfToTensor(loss);
    } catch (const af::exception& e) {
        LogArrayFireLossFallbackOnce(
            "BCELoss::Forward", e.what(), predictions, "predictions");
    }
#endif
    return CpuBCEForward(predictions, targets, eps_, reduction_);
}

Tensor BCELoss::Backward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array pred = TensorToAf(predictions);
        af::array target = TensorToAf(targets);

        // Clamp predictions for numerical stability
        af::array pred_clamped = af::clamp(pred, eps_, 1.0f - eps_);
        pred_clamped.eval();

        // Gradient: -target/pred + (1-target)/(1-pred)
        //         = (pred - target) / (pred * (1 - pred))
        af::array grad = (pred_clamped - target) / (pred_clamped * (1.0f - pred_clamped) + eps_);
        grad.eval();

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(pred.elements());
            grad.eval();
        }

        return AfToTensor(grad);
    } catch (const af::exception& e) {
        LogArrayFireLossFallbackOnce(
            "BCELoss::Backward", e.what(), predictions, "predictions");
    }
#endif
    return CpuBCEBackward(predictions, targets, eps_, reduction_);
}

// ============================================================================
// BCE With Logits Loss Implementation
// ============================================================================

Tensor BCEWithLogitsLoss::Forward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array logits = TensorToAf(predictions);
        af::array target = TensorToAf(targets);

        // Numerically stable BCE with logits:
        // max(logits, 0) - logits * target + log(1 + exp(-|logits|))
        af::array loss = af::max(logits, 0.0f) - logits * target +
                         af::log(1.0f + af::exp(-af::abs(logits)));
        loss.eval();

        loss = ApplyReduction(loss, reduction_);

        return AfToTensor(loss);
    } catch (const af::exception& e) {
        LogArrayFireLossFallbackOnce(
            "BCEWithLogitsLoss::Forward", e.what(), predictions, "predictions");
    }
#endif
    return CpuBCEWithLogitsForward(predictions, targets, reduction_);
}

Tensor BCEWithLogitsLoss::Backward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array logits = TensorToAf(predictions);
        af::array target = TensorToAf(targets);

        // Gradient: sigmoid(logits) - target
        af::array sigmoid_logits = af::sigmoid(logits);
        sigmoid_logits.eval();
        af::array grad = sigmoid_logits - target;
        grad.eval();

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(logits.elements());
            grad.eval();
        }

        return AfToTensor(grad);
    } catch (const af::exception& e) {
        LogArrayFireLossFallbackOnce(
            "BCEWithLogitsLoss::Backward", e.what(), predictions, "predictions");
    }
#endif
    return CpuBCEWithLogitsBackward(predictions, targets, reduction_);
}

// ============================================================================
// KL Divergence Loss Implementation
// ============================================================================

Tensor KLDivLoss::Forward(const Tensor& predictions, const Tensor& targets) {
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
            // Avoid log(0) by adding small epsilon
            af::array log_target = af::log(target + 1e-10f);
            log_target.eval();
            loss = target * (log_target - log_pred);
        }
        loss.eval();

        // Only consider positive targets
        loss = af::select(target > 0, loss, 0.0f);
        loss.eval();
        loss = ApplyReduction(loss, reduction_);

        return AfToTensor(loss);
    } catch (const af::exception& e) {
        LogArrayFireLossFallbackOnce(
            "KLDivLoss::Forward", e.what(), predictions, "predictions");
    }
#endif
    return CpuKLDivForward(predictions, targets, log_target_, reduction_);
}

Tensor KLDivLoss::Backward(const Tensor& predictions, const Tensor& targets) {
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

        // Only consider positive targets
        grad = af::select(target > 0, grad, 0.0f);
        grad.eval();

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(log_pred.elements());
            grad.eval();
        }

        return AfToTensor(grad);
    } catch (const af::exception& e) {
        LogArrayFireLossFallbackOnce(
            "KLDivLoss::Backward", e.what(), predictions, "predictions");
    }
#endif
    return CpuKLDivBackward(predictions, targets, log_target_, reduction_);
}

} // namespace cyxwiz
