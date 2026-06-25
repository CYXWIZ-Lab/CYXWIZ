#include "cyxwiz/losses/regression.h"
#include "loss_utils.h"

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

Tensor MSELoss::Forward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array pred = loss_detail::TensorToAf(predictions);
        af::array target = loss_detail::TensorToAf(targets);

        af::array diff = pred - target;
        diff.eval();
        af::array squared = diff * diff;
        squared.eval();
        af::array loss = loss_detail::ApplyReduction(squared, reduction_);

        return loss_detail::AfToTensor(loss);
    } catch (const af::exception& e) {
        loss_detail::LogArrayFireLossFallbackOnce(
            "MSELoss::Forward", e.what(), predictions, "predictions");
    }
#endif
    return loss_detail::CpuMSEForward(predictions, targets, reduction_);
}

Tensor MSELoss::Backward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array pred = loss_detail::TensorToAf(predictions);
        af::array target = loss_detail::TensorToAf(targets);

        af::array diff = pred - target;
        float scale = 2.0f;

        if (reduction_ == Reduction::Mean) {
            scale /= static_cast<float>(pred.elements());
        }

        af::array grad = diff * scale;
        grad.eval();

        return loss_detail::AfToTensor(grad);
    } catch (const af::exception& e) {
        loss_detail::LogArrayFireLossFallbackOnce(
            "MSELoss::Backward", e.what(), predictions, "predictions");
    }
#endif
    return loss_detail::CpuMSEBackward(predictions, targets, reduction_);
}

Tensor L1Loss::Forward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array pred = loss_detail::TensorToAf(predictions);
        af::array target = loss_detail::TensorToAf(targets);

        af::array diff = af::abs(pred - target);
        diff.eval();
        af::array loss = loss_detail::ApplyReduction(diff, reduction_);

        return loss_detail::AfToTensor(loss);
    } catch (const af::exception& e) {
        loss_detail::LogArrayFireLossFallbackOnce(
            "L1Loss::Forward", e.what(), predictions, "predictions");
    }
#endif
    return loss_detail::CpuL1Forward(predictions, targets, reduction_);
}

Tensor L1Loss::Backward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array pred = loss_detail::TensorToAf(predictions);
        af::array target = loss_detail::TensorToAf(targets);

        af::array diff = pred - target;
        diff.eval();
        af::array grad = loss_detail::SignLike(diff);
        grad.eval();

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(pred.elements());
            grad.eval();
        }

        return loss_detail::AfToTensor(grad);
    } catch (const af::exception& e) {
        loss_detail::LogArrayFireLossFallbackOnce(
            "L1Loss::Backward", e.what(), predictions, "predictions");
    }
#endif
    return loss_detail::CpuL1Backward(predictions, targets, reduction_);
}

Tensor SmoothL1Loss::Forward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array pred = loss_detail::TensorToAf(predictions);
        af::array target = loss_detail::TensorToAf(targets);

        af::array diff = pred - target;
        diff.eval();
        af::array abs_diff = af::abs(diff);
        abs_diff.eval();

        af::array quadratic = 0.5f * diff * diff / delta_;
        quadratic.eval();
        af::array linear = abs_diff - 0.5f * delta_;
        linear.eval();

        af::array loss = af::select(abs_diff < delta_, quadratic, linear);
        loss.eval();
        loss = loss_detail::ApplyReduction(loss, reduction_);

        return loss_detail::AfToTensor(loss);
    } catch (const af::exception& e) {
        loss_detail::LogArrayFireLossFallbackOnce(
            "SmoothL1Loss::Forward", e.what(), predictions, "predictions");
    }
#endif
    return loss_detail::CpuSmoothL1Forward(predictions, targets, delta_, reduction_);
}

Tensor SmoothL1Loss::Backward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array pred = loss_detail::TensorToAf(predictions);
        af::array target = loss_detail::TensorToAf(targets);

        af::array diff = pred - target;
        diff.eval();
        af::array abs_diff = af::abs(diff);
        abs_diff.eval();

        af::array grad_quadratic = diff / delta_;
        grad_quadratic.eval();
        af::array grad_linear = loss_detail::SignLike(diff);
        grad_linear.eval();

        af::array grad = af::select(abs_diff < delta_, grad_quadratic, grad_linear);
        grad.eval();

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(pred.elements());
            grad.eval();
        }

        return loss_detail::AfToTensor(grad);
    } catch (const af::exception& e) {
        loss_detail::LogArrayFireLossFallbackOnce(
            "SmoothL1Loss::Backward", e.what(), predictions, "predictions");
    }
#endif
    return loss_detail::CpuSmoothL1Backward(predictions, targets, delta_, reduction_);
}

} // namespace cyxwiz
