#include "cyxwiz/losses/regression.h"
#include "../arrayfire_backend_utils.h"
#include "loss_utils.h"

#include <cmath>
#include <stdexcept>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

namespace {

template <typename CpuFunction>
Tensor RunNativeCpuLoss(const char* operation_name, CpuFunction&& compute) {
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LossCpuPath,
        operation_name);
    return compute();
}

void ValidateSmoothL1Delta(float delta) {
    if (!std::isfinite(delta) || delta < 0.0f) {
        throw std::invalid_argument(
            "SmoothL1Loss beta must be finite and non-negative");
    }
}

void ValidateHuberDelta(float delta) {
    if (!std::isfinite(delta) || delta <= 0.0f) {
        throw std::invalid_argument(
            "HuberLoss delta must be finite and positive");
    }
}

} // namespace

SmoothL1Loss::SmoothL1Loss(float delta, Reduction reduction)
    : Loss(reduction), delta_(delta) {
    ValidateSmoothL1Delta(delta_);
}

HuberLoss::HuberLoss(float delta, Reduction reduction)
    : Loss(reduction), delta_(delta) {
    ValidateHuberDelta(delta_);
}

Tensor MSELoss::Forward(const Tensor& predictions, const Tensor& targets) {
    constexpr const char* kOperation = "MSELoss::Forward";
    const bool use_native_cpu = loss_detail::PrepareLossNativeCpuFallback(
        kOperation, predictions, targets, reduction_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (!use_native_cpu) try {
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
            kOperation, e.what(), predictions, targets, reduction_);
    }
#endif
    return RunNativeCpuLoss(kOperation, [&] {
        return loss_detail::CpuMSEForward(predictions, targets, reduction_);
    });
}

Tensor MSELoss::Backward(const Tensor& predictions, const Tensor& targets) {
    constexpr const char* kOperation = "MSELoss::Backward";
    const bool use_native_cpu = loss_detail::PrepareLossNativeCpuFallback(
        kOperation, predictions, targets, reduction_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (!use_native_cpu) try {
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
            kOperation, e.what(), predictions, targets, reduction_);
    }
#endif
    return RunNativeCpuLoss(kOperation, [&] {
        return loss_detail::CpuMSEBackward(predictions, targets, reduction_);
    });
}

Tensor L1Loss::Forward(const Tensor& predictions, const Tensor& targets) {
    constexpr const char* kOperation = "L1Loss::Forward";
    const bool use_native_cpu = loss_detail::PrepareLossNativeCpuFallback(
        kOperation, predictions, targets, reduction_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (!use_native_cpu) try {
        af::array pred = loss_detail::TensorToAf(predictions);
        af::array target = loss_detail::TensorToAf(targets);

        af::array diff = af::abs(pred - target);
        diff.eval();
        af::array loss = loss_detail::ApplyReduction(diff, reduction_);

        return loss_detail::AfToTensor(loss);
    } catch (const af::exception& e) {
        loss_detail::LogArrayFireLossFallbackOnce(
            kOperation, e.what(), predictions, targets, reduction_);
    }
#endif
    return RunNativeCpuLoss(kOperation, [&] {
        return loss_detail::CpuL1Forward(predictions, targets, reduction_);
    });
}

Tensor L1Loss::Backward(const Tensor& predictions, const Tensor& targets) {
    constexpr const char* kOperation = "L1Loss::Backward";
    const bool use_native_cpu = loss_detail::PrepareLossNativeCpuFallback(
        kOperation, predictions, targets, reduction_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (!use_native_cpu) try {
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
            kOperation, e.what(), predictions, targets, reduction_);
    }
#endif
    return RunNativeCpuLoss(kOperation, [&] {
        return loss_detail::CpuL1Backward(predictions, targets, reduction_);
    });
}

Tensor SmoothL1Loss::Forward(const Tensor& predictions, const Tensor& targets) {
    constexpr const char* kOperation = "SmoothL1Loss::Forward";
    const bool use_native_cpu = loss_detail::PrepareLossNativeCpuFallback(
        kOperation, predictions, targets, reduction_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (!use_native_cpu) try {
        af::array pred = loss_detail::TensorToAf(predictions);
        af::array target = loss_detail::TensorToAf(targets);

        af::array diff = pred - target;
        diff.eval();
        af::array abs_diff = af::abs(diff);
        abs_diff.eval();

        af::array loss;
        if (delta_ == 0.0f) {
            loss = abs_diff;
        } else {
            af::array quadratic = 0.5f * diff * diff / delta_;
            quadratic.eval();
            af::array linear = abs_diff - 0.5f * delta_;
            linear.eval();
            loss = af::select(abs_diff < delta_, quadratic, linear);
        }
        loss.eval();
        loss = loss_detail::ApplyReduction(loss, reduction_);

        return loss_detail::AfToTensor(loss);
    } catch (const af::exception& e) {
        loss_detail::LogArrayFireLossFallbackOnce(
            kOperation, e.what(), predictions, targets, reduction_);
    }
#endif
    return RunNativeCpuLoss(kOperation, [&] {
        return loss_detail::CpuSmoothL1Forward(
            predictions, targets, delta_, reduction_);
    });
}

Tensor SmoothL1Loss::Backward(const Tensor& predictions, const Tensor& targets) {
    constexpr const char* kOperation = "SmoothL1Loss::Backward";
    const bool use_native_cpu = loss_detail::PrepareLossNativeCpuFallback(
        kOperation, predictions, targets, reduction_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (!use_native_cpu) try {
        af::array pred = loss_detail::TensorToAf(predictions);
        af::array target = loss_detail::TensorToAf(targets);

        af::array diff = pred - target;
        diff.eval();
        af::array abs_diff = af::abs(diff);
        abs_diff.eval();

        af::array grad;
        if (delta_ == 0.0f) {
            grad = loss_detail::SignLike(diff);
        } else {
            af::array grad_quadratic = diff / delta_;
            grad_quadratic.eval();
            af::array grad_linear = loss_detail::SignLike(diff);
            grad_linear.eval();
            grad = af::select(
                abs_diff < delta_, grad_quadratic, grad_linear);
        }
        grad.eval();

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(pred.elements());
            grad.eval();
        }

        return loss_detail::AfToTensor(grad);
    } catch (const af::exception& e) {
        loss_detail::LogArrayFireLossFallbackOnce(
            kOperation, e.what(), predictions, targets, reduction_);
    }
#endif
    return RunNativeCpuLoss(kOperation, [&] {
        return loss_detail::CpuSmoothL1Backward(
            predictions, targets, delta_, reduction_);
    });
}

Tensor HuberLoss::Forward(const Tensor& predictions, const Tensor& targets) {
    constexpr const char* kOperation = "HuberLoss::Forward";
    const bool use_native_cpu = loss_detail::PrepareLossNativeCpuFallback(
        kOperation, predictions, targets, reduction_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (!use_native_cpu) try {
        af::array pred = loss_detail::TensorToAf(predictions);
        af::array target = loss_detail::TensorToAf(targets);
        af::array diff = pred - target;
        diff.eval();
        af::array abs_diff = af::abs(diff);
        abs_diff.eval();
        af::array quadratic = 0.5f * diff * diff;
        quadratic.eval();
        af::array linear = delta_ * (abs_diff - 0.5f * delta_);
        linear.eval();
        af::array loss = af::select(abs_diff < delta_, quadratic, linear);
        loss.eval();
        loss = loss_detail::ApplyReduction(loss, reduction_);
        return loss_detail::AfToTensor(loss);
    } catch (const af::exception& e) {
        loss_detail::LogArrayFireLossFallbackOnce(
            kOperation, e.what(), predictions, targets, reduction_);
    }
#endif
    return RunNativeCpuLoss(kOperation, [&] {
        return loss_detail::CpuHuberForward(
            predictions, targets, delta_, reduction_);
    });
}

Tensor HuberLoss::Backward(const Tensor& predictions, const Tensor& targets) {
    constexpr const char* kOperation = "HuberLoss::Backward";
    const bool use_native_cpu = loss_detail::PrepareLossNativeCpuFallback(
        kOperation, predictions, targets, reduction_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (!use_native_cpu) try {
        af::array pred = loss_detail::TensorToAf(predictions);
        af::array target = loss_detail::TensorToAf(targets);
        af::array diff = pred - target;
        diff.eval();
        af::array abs_diff = af::abs(diff);
        abs_diff.eval();
        af::array clipped = af::select(
            diff > delta_, delta_, af::select(diff < -delta_, -delta_, diff));
        clipped.eval();
        af::array grad = af::select(abs_diff < delta_, diff, clipped);
        grad.eval();
        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(pred.elements());
            grad.eval();
        }
        return loss_detail::AfToTensor(grad);
    } catch (const af::exception& e) {
        loss_detail::LogArrayFireLossFallbackOnce(
            kOperation, e.what(), predictions, targets, reduction_);
    }
#endif
    return RunNativeCpuLoss(kOperation, [&] {
        return loss_detail::CpuHuberBackward(
            predictions, targets, delta_, reduction_);
    });
}

} // namespace cyxwiz
