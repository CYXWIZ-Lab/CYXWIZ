#include "cyxwiz/loss.h"
#include "loss_utils.h"

#include <spdlog/spdlog.h>

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
        af::array squared = diff * diff;
        af::array loss = loss_detail::ApplyReduction(squared, reduction_);

        return loss_detail::AfToTensor(loss);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire MSELoss::Forward failed: {}", e.what());
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

        return loss_detail::AfToTensor(grad);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire MSELoss::Backward failed: {}", e.what());
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
        af::array loss = loss_detail::ApplyReduction(diff, reduction_);

        return loss_detail::AfToTensor(loss);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire L1Loss::Forward failed: {}", e.what());
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
        af::array grad = loss_detail::SignLike(diff);

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(pred.elements());
        }

        return loss_detail::AfToTensor(grad);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire L1Loss::Backward failed: {}", e.what());
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
        af::array abs_diff = af::abs(diff);

        af::array quadratic = 0.5f * diff * diff / delta_;
        af::array linear = abs_diff - 0.5f * delta_;

        af::array loss = af::select(abs_diff < delta_, quadratic, linear);
        loss = loss_detail::ApplyReduction(loss, reduction_);

        return loss_detail::AfToTensor(loss);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire SmoothL1Loss::Forward failed: {}", e.what());
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
        af::array abs_diff = af::abs(diff);

        af::array grad_quadratic = diff / delta_;
        af::array grad_linear = loss_detail::SignLike(diff);

        af::array grad = af::select(abs_diff < delta_, grad_quadratic, grad_linear);

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(pred.elements());
        }

        return loss_detail::AfToTensor(grad);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire SmoothL1Loss::Backward failed: {}", e.what());
    }
#endif
    return loss_detail::CpuSmoothL1Backward(predictions, targets, delta_, reduction_);
}

} // namespace cyxwiz
