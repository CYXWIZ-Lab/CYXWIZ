#include "loss_utils.h"
#include "cyxwiz/backend_placement_observation.h"
#include "../arrayfire_backend_utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <spdlog/spdlog.h>

// Undefine Windows macros that conflict with ArrayFire functions.
// Must be AFTER all includes (Windows headers define these).
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz {
namespace loss_detail {

void ValidateFloat32Pair(const Tensor& predictions, const Tensor& targets, const char* name) {
    if (predictions.GetDataType() != DataType::Float32 || targets.GetDataType() != DataType::Float32) {
        throw std::runtime_error(std::string(name) + " only supports Float32 tensors");
    }
    if (predictions.Shape() != targets.Shape()) {
        throw std::runtime_error(std::string(name) + " requires matching prediction and target shapes");
    }
}

Tensor ApplyCpuReduction(const std::vector<size_t>& input_shape,
                         const std::vector<float>& values,
                         Reduction reduction) {
    if (reduction == Reduction::None) {
        return Tensor(input_shape, values.data(), DataType::Float32);
    }

    float total = 0.0f;
    for (float value : values) {
        total += value;
    }
    if (reduction == Reduction::Mean && !values.empty()) {
        total /= static_cast<float>(values.size());
    }
    return Tensor({1}, &total, DataType::Float32);
}

Tensor ApplyClassReduction(const std::vector<float>& per_sample,
                           size_t batch,
                           Reduction reduction) {
    if (reduction == Reduction::None) {
        return Tensor({batch}, per_sample.data(), DataType::Float32);
    }

    float total = 0.0f;
    for (float value : per_sample) {
        total += value;
    }
    if (reduction == Reduction::Mean && batch > 0) {
        total /= static_cast<float>(batch);
    }
    return Tensor({1}, &total, DataType::Float32);
}

Tensor CpuMSEForward(const Tensor& predictions, const Tensor& targets, Reduction reduction) {
    ValidateFloat32Pair(predictions, targets, "MSE");
    const size_t count = predictions.NumElements();
    const float* pred = predictions.ReadData<float>();
    const float* target = targets.ReadData<float>();
    std::vector<float> losses(count);
    for (size_t i = 0; i < count; ++i) {
        const float diff = pred[i] - target[i];
        losses[i] = diff * diff;
    }
    return ApplyCpuReduction(predictions.Shape(), losses, reduction);
}

Tensor CpuMSEBackward(const Tensor& predictions, const Tensor& targets, Reduction reduction) {
    ValidateFloat32Pair(predictions, targets, "MSE");
    Tensor grad(predictions.Shape(), DataType::Float32);
    const size_t count = predictions.NumElements();
    const float scale = reduction == Reduction::Mean && count > 0
                            ? 2.0f / static_cast<float>(count)
                            : 2.0f;
    const float* pred = predictions.ReadData<float>();
    const float* target = targets.ReadData<float>();
    float* out = grad.MutableData<float>();
    for (size_t i = 0; i < count; ++i) {
        out[i] = (pred[i] - target[i]) * scale;
    }
    return grad;
}

Tensor CpuL1Forward(const Tensor& predictions, const Tensor& targets, Reduction reduction) {
    ValidateFloat32Pair(predictions, targets, "L1");
    const size_t count = predictions.NumElements();
    const float* pred = predictions.ReadData<float>();
    const float* target = targets.ReadData<float>();
    std::vector<float> losses(count);
    for (size_t i = 0; i < count; ++i) {
        losses[i] = std::fabs(pred[i] - target[i]);
    }
    return ApplyCpuReduction(predictions.Shape(), losses, reduction);
}

Tensor CpuL1Backward(const Tensor& predictions, const Tensor& targets, Reduction reduction) {
    ValidateFloat32Pair(predictions, targets, "L1");
    Tensor grad(predictions.Shape(), DataType::Float32);
    const size_t count = predictions.NumElements();
    const float scale = reduction == Reduction::Mean && count > 0 ? 1.0f / static_cast<float>(count) : 1.0f;
    const float* pred = predictions.ReadData<float>();
    const float* target = targets.ReadData<float>();
    float* out = grad.MutableData<float>();
    for (size_t i = 0; i < count; ++i) {
        const float diff = pred[i] - target[i];
        out[i] = (diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f)) * scale;
    }
    return grad;
}

Tensor CpuSmoothL1Forward(const Tensor& predictions,
                          const Tensor& targets,
                          float delta,
                          Reduction reduction) {
    ValidateFloat32Pair(predictions, targets, "SmoothL1");
    const size_t count = predictions.NumElements();
    const float* pred = predictions.ReadData<float>();
    const float* target = targets.ReadData<float>();
    std::vector<float> losses(count);
    for (size_t i = 0; i < count; ++i) {
        const float diff = pred[i] - target[i];
        const float abs_diff = std::fabs(diff);
        losses[i] = abs_diff < delta ? 0.5f * diff * diff / delta : abs_diff - 0.5f * delta;
    }
    return ApplyCpuReduction(predictions.Shape(), losses, reduction);
}

Tensor CpuSmoothL1Backward(const Tensor& predictions,
                           const Tensor& targets,
                           float delta,
                           Reduction reduction) {
    ValidateFloat32Pair(predictions, targets, "SmoothL1");
    Tensor grad(predictions.Shape(), DataType::Float32);
    const size_t count = predictions.NumElements();
    const float scale = reduction == Reduction::Mean && count > 0 ? 1.0f / static_cast<float>(count) : 1.0f;
    const float* pred = predictions.ReadData<float>();
    const float* target = targets.ReadData<float>();
    float* out = grad.MutableData<float>();
    for (size_t i = 0; i < count; ++i) {
        const float diff = pred[i] - target[i];
        const float abs_diff = std::fabs(diff);
        const float base_grad = abs_diff < delta ? diff / delta
                              : (diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f));
        out[i] = base_grad * scale;
    }
    return grad;
}

Tensor CpuHuberForward(const Tensor& predictions,
                       const Tensor& targets,
                       float delta,
                       Reduction reduction) {
    ValidateFloat32Pair(predictions, targets, "Huber");
    const size_t count = predictions.NumElements();
    const float* pred = predictions.ReadData<float>();
    const float* target = targets.ReadData<float>();
    std::vector<float> losses(count);
    for (size_t i = 0; i < count; ++i) {
        const float diff = pred[i] - target[i];
        const float abs_diff = std::fabs(diff);
        losses[i] = abs_diff < delta
                        ? 0.5f * diff * diff
                        : delta * (abs_diff - 0.5f * delta);
    }
    return ApplyCpuReduction(predictions.Shape(), losses, reduction);
}

Tensor CpuHuberBackward(const Tensor& predictions,
                        const Tensor& targets,
                        float delta,
                        Reduction reduction) {
    ValidateFloat32Pair(predictions, targets, "Huber");
    Tensor grad(predictions.Shape(), DataType::Float32);
    const size_t count = predictions.NumElements();
    const float scale = reduction == Reduction::Mean && count > 0
                            ? 1.0f / static_cast<float>(count)
                            : 1.0f;
    const float* pred = predictions.ReadData<float>();
    const float* target = targets.ReadData<float>();
    float* out = grad.MutableData<float>();
    for (size_t i = 0; i < count; ++i) {
        const float diff = pred[i] - target[i];
        const float abs_diff = std::fabs(diff);
        const float base_grad = abs_diff < delta
                                    ? diff
                                    : (diff > 0.0f ? delta
                                                   : (diff < 0.0f ? -delta
                                                                  : 0.0f));
        out[i] = base_grad * scale;
    }
    return grad;
}

Tensor CpuBCEForward(const Tensor& predictions,
                     const Tensor& targets,
                     Reduction reduction) {
    ValidateFloat32Pair(predictions, targets, "BCE");
    const size_t count = predictions.NumElements();
    const float* pred = predictions.Data<float>();
    const float* target = targets.Data<float>();
    std::vector<float> losses(count);
    for (size_t i = 0; i < count; ++i) {
        const float log_prediction =
            std::max(std::log(pred[i]), -100.0f);
        const float log_complement =
            std::max(std::log(1.0f - pred[i]), -100.0f);
        losses[i] = -(target[i] * log_prediction +
                      (1.0f - target[i]) * log_complement);
    }
    return ApplyCpuReduction(predictions.Shape(), losses, reduction);
}

Tensor CpuBCEBackward(const Tensor& predictions, const Tensor& targets, float eps, Reduction reduction) {
    ValidateFloat32Pair(predictions, targets, "BCE");
    Tensor grad(predictions.Shape(), DataType::Float32);
    const size_t count = predictions.NumElements();
    const float scale = reduction == Reduction::Mean && count > 0 ? 1.0f / static_cast<float>(count) : 1.0f;
    const float* pred = predictions.Data<float>();
    const float* target = targets.Data<float>();
    float* out = grad.Data<float>();
    for (size_t i = 0; i < count; ++i) {
        const float denominator =
            std::max(pred[i] * (1.0f - pred[i]), eps);
        out[i] = ((pred[i] - target[i]) / denominator) * scale;
    }
    return grad;
}

float CpuSigmoidValue(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    const float exp_x = std::exp(x);
    return exp_x / (1.0f + exp_x);
}

Tensor CpuBCEWithLogitsForward(const Tensor& predictions,
                               const Tensor& targets,
                               Reduction reduction,
                               float pos_weight) {
    ValidateFloat32Pair(predictions, targets, "BCEWithLogits");
    const size_t count = predictions.NumElements();
    const float* logits = predictions.Data<float>();
    const float* target = targets.Data<float>();
    std::vector<float> losses(count);
    for (size_t i = 0; i < count; ++i) {
        const float logit = logits[i];
        const float log_weight = 1.0f + (pos_weight - 1.0f) * target[i];
        losses[i] = (1.0f - target[i]) * logit +
                    log_weight *
                        (std::max(-logit, 0.0f) +
                         std::log1p(std::exp(-std::fabs(logit))));
    }
    return ApplyCpuReduction(predictions.Shape(), losses, reduction);
}

Tensor CpuBCEWithLogitsBackward(const Tensor& predictions,
                                const Tensor& targets,
                                Reduction reduction,
                                float pos_weight) {
    ValidateFloat32Pair(predictions, targets, "BCEWithLogits");
    Tensor grad(predictions.Shape(), DataType::Float32);
    const size_t count = predictions.NumElements();
    const float scale = reduction == Reduction::Mean && count > 0 ? 1.0f / static_cast<float>(count) : 1.0f;
    const float* logits = predictions.Data<float>();
    const float* target = targets.Data<float>();
    float* out = grad.Data<float>();
    for (size_t i = 0; i < count; ++i) {
        const float log_weight = 1.0f + (pos_weight - 1.0f) * target[i];
        out[i] =
            ((1.0f - target[i]) +
             log_weight * (CpuSigmoidValue(logits[i]) - 1.0f)) *
            scale;
    }
    return grad;
}

Tensor CpuKLDivForward(const Tensor& predictions,
                       const Tensor& targets,
                       bool log_target,
                       Reduction reduction) {
    ValidateFloat32Pair(predictions, targets, "KLDiv");
    const size_t count = predictions.NumElements();
    const float* pred = predictions.Data<float>();
    const float* target = targets.Data<float>();
    std::vector<float> losses(count, 0.0f);
    for (size_t i = 0; i < count; ++i) {
        if (log_target) {
            losses[i] = std::exp(target[i]) * (target[i] - pred[i]);
        } else if (target[i] == 0.0f) {
            losses[i] = 0.0f;
        } else {
            losses[i] = target[i] * (std::log(target[i]) - pred[i]);
        }
    }
    return ApplyCpuReduction(predictions.Shape(), losses, reduction);
}

Tensor CpuKLDivBackward(const Tensor& predictions,
                        const Tensor& targets,
                        bool log_target,
                        Reduction reduction) {
    ValidateFloat32Pair(predictions, targets, "KLDiv");
    Tensor grad(predictions.Shape(), DataType::Float32);
    const size_t count = predictions.NumElements();
    const float scale = reduction == Reduction::Mean && count > 0 ? 1.0f / static_cast<float>(count) : 1.0f;
    const float* target = targets.Data<float>();
    float* out = grad.Data<float>();
    for (size_t i = 0; i < count; ++i) {
        out[i] = (log_target ? -std::exp(target[i]) : -target[i]) * scale;
    }
    return grad;
}

const char* ReductionName(Reduction reduction) {
    switch (reduction) {
        case Reduction::None:
            return "none";
        case Reduction::Mean:
            return "mean";
        case Reduction::Sum:
            return "sum";
        default:
            return "unknown";
    }
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
af::array TensorToAf(const Tensor& t) {
    return t.GetSemanticArray();
}

Tensor AfToTensor(const af::array& arr) {
    int ndims = 0;
    for (unsigned int i = 0; i < 4; i++) {
        if (arr.dims(i) > 1) {
            ndims = i + 1;
        } else if (i == 0) {
            ndims = 1;
        }
    }

    if (ndims == 2) {
        return Tensor::FromArrayRowMajor2D(arr);
    }

    return Tensor(arr);
}

Tensor AfToTensor(const af::array& arr,
                  const std::vector<size_t>& semantic_shape) {
    return Tensor::FromSemanticArray(arr, semantic_shape);
}
#endif

bool PrepareLossNativeCpuFallback(
    const char* operation_name,
    const Tensor& predictions,
    const Tensor& targets,
    Reduction reduction) {
    ValidateFloat32Pair(predictions, targets, operation_name);
    if (ShouldForceArrayFireBackendFallbackForTesting(operation_name)) {
        LogArrayFireLossFallbackOnce(
            operation_name,
            BackendFallbackReason::GpuBackendException,
            "forced ArrayFire backend fallback test hook",
            predictions,
            targets,
            reduction);
        return true;
    }
    if (!IsCurrentArrayFireBackendAvailable()) {
        LogArrayFireLossFallbackOnce(
            operation_name,
            BackendFallbackReason::BackendUnavailable,
            "ArrayFire backend unavailable",
            predictions,
            targets,
            reduction);
        return true;
    }
    return false;
}

void LogArrayFireLossFallbackOnce(
    const char* operation_name,
    const char* error_message,
    const Tensor& tensor,
    const char* tensor_name) {
    const BackendFallbackReason reason =
        ClassifyArrayFireBackendFallbackReason(error_message);
    const std::string context =
        BuildArrayFireBackendFallbackContext(
            BuildTensorShapeContext(tensor_name, tensor.Shape()));
    const std::string message = BuildArrayFireBackendFallbackMessage(
        operation_name,
        reason,
        reason != BackendFallbackReason::CudaJitParamOverflow,
        error_message,
        context);
    RecordBackendPlacementObservationForActiveDevice(
        operation_name ? operation_name : "Loss",
        CurrentArrayFireBackendName(),
        "float32",
        BuildLossPlacementShapeSignature(
            tensor.Shape(),
            {},
            "unknown",
            "float32"),
        BackendFallbackReasonName(reason),
        BackendPlacementObservationSource::RuntimeFallback,
        message);
    ThrowIfArrayFireNativeCpuFallbackForbidden(
        operation_name,
        reason,
        error_message,
        context);
    const bool log_fallback =
        ShouldLogArrayFireBackendFallbackOnce(
            operation_name, reason, context);
    if (log_fallback) {
        spdlog::warn("{}", message);
    }
}

void LogArrayFireLossFallbackOnce(
    const char* operation_name,
    const char* error_message,
    const Tensor& predictions,
    const Tensor& targets,
    Reduction reduction) {
    LogArrayFireLossFallbackOnce(
        operation_name,
        ClassifyArrayFireBackendFallbackReason(error_message),
        error_message,
        predictions,
        targets,
        reduction);
}

void LogArrayFireLossFallbackOnce(
    const char* operation_name,
    BackendFallbackReason reason,
    const char* error_message,
    const Tensor& predictions,
    const Tensor& targets,
    Reduction reduction) {
    const std::string context =
        BuildArrayFireBackendFallbackContext(
            BuildTensorShapeContext("predictions", predictions.Shape()) +
            "; " +
            BuildTensorShapeContext("targets", targets.Shape()));
    const std::string message = BuildArrayFireBackendFallbackMessage(
        operation_name,
        reason,
        reason != BackendFallbackReason::CudaJitParamOverflow,
        error_message,
        context);
    RecordBackendPlacementObservationForActiveDevice(
        operation_name ? operation_name : "Loss",
        CurrentArrayFireBackendName(),
        "float32",
        BuildLossPlacementShapeSignature(
            predictions.Shape(),
            targets.Shape(),
            ReductionName(reduction),
            "float32"),
        BackendFallbackReasonName(reason),
        BackendPlacementObservationSource::RuntimeFallback,
        message);
    ThrowIfArrayFireNativeCpuFallbackForbidden(
        operation_name,
        reason,
        error_message,
        context);
    const bool log_fallback =
        ShouldLogArrayFireBackendFallbackOnce(
            operation_name, reason, context);
    if (log_fallback) {
        spdlog::warn("{}", message);
    }
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
af::array ApplyReduction(const af::array& loss, Reduction reduction) {
    switch (reduction) {
        case Reduction::None:
            return loss;
        case Reduction::Mean: {
            // ArrayFire's single-argument reduction operates on the first
            // non-singleton dimension. Loss::Mean is a global reduction, so
            // flatten first to produce exactly one scalar for tensors such as
            // [batch, forecast_horizon].
            af::array result = af::mean(af::flat(loss));
            result.eval();
            return result;
        }
        case Reduction::Sum: {
            af::array result = af::sum(af::flat(loss));
            result.eval();
            return result;
        }
        default: {
            af::array result = af::mean(af::flat(loss));
            result.eval();
            return result;
        }
    }
}

af::array StableSoftmax(const af::array& x, int axis) {
    af::array max_val = af::max(x, axis);
    max_val.eval();
    af::dim4 tile_dims(1, 1, 1, 1);
    tile_dims[axis] = x.dims(axis);
    af::array x_stable = x - af::tile(max_val, tile_dims);
    x_stable.eval();
    af::array exp_x = af::exp(x_stable);
    exp_x.eval();
    af::array sum_exp = af::sum(exp_x, axis);
    sum_exp.eval();
    af::array result = exp_x / af::tile(sum_exp, tile_dims);
    result.eval();
    return result;
}

af::array SignLike(const af::array& x) {
    const af::array ones = af::constant(1.0f, x.dims(), f32);
    const af::array minus_ones = af::constant(-1.0f, x.dims(), f32);
    const af::array zeros = af::constant(0.0f, x.dims(), f32);
    return af::select(x > 0.0f, ones, af::select(x < 0.0f, minus_ones, zeros));
}

#endif // CYXWIZ_HAS_ARRAYFIRE

} // namespace loss_detail
} // namespace cyxwiz

