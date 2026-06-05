#include "cyxwiz/loss.h"
#include "cyxwiz/tensor.h"
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <vector>
#include <spdlog/spdlog.h>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

// Undefine Windows macros that conflict with ArrayFire functions
// Must be AFTER all includes (Windows headers define these)
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz {

namespace {

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

Tensor CpuMSEForward(const Tensor& predictions, const Tensor& targets, Reduction reduction) {
    ValidateFloat32Pair(predictions, targets, "MSE");
    const size_t count = predictions.NumElements();
    const float* pred = predictions.Data<float>();
    const float* target = targets.Data<float>();
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
    const float* pred = predictions.Data<float>();
    const float* target = targets.Data<float>();
    float* out = grad.Data<float>();
    for (size_t i = 0; i < count; ++i) {
        out[i] = (pred[i] - target[i]) * scale;
    }
    return grad;
}

Tensor CpuL1Forward(const Tensor& predictions, const Tensor& targets, Reduction reduction) {
    ValidateFloat32Pair(predictions, targets, "L1");
    const size_t count = predictions.NumElements();
    const float* pred = predictions.Data<float>();
    const float* target = targets.Data<float>();
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
    const float* pred = predictions.Data<float>();
    const float* target = targets.Data<float>();
    float* out = grad.Data<float>();
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
    const float* pred = predictions.Data<float>();
    const float* target = targets.Data<float>();
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
    const float* pred = predictions.Data<float>();
    const float* target = targets.Data<float>();
    float* out = grad.Data<float>();
    for (size_t i = 0; i < count; ++i) {
        const float diff = pred[i] - target[i];
        const float abs_diff = std::fabs(diff);
        const float base_grad = abs_diff < delta ? diff / delta
                              : (diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f));
        out[i] = base_grad * scale;
    }
    return grad;
}

Tensor CpuBCEForward(const Tensor& predictions, const Tensor& targets, float eps, Reduction reduction) {
    ValidateFloat32Pair(predictions, targets, "BCE");
    const size_t count = predictions.NumElements();
    const float* pred = predictions.Data<float>();
    const float* target = targets.Data<float>();
    std::vector<float> losses(count);
    for (size_t i = 0; i < count; ++i) {
        const float clamped = std::clamp(pred[i], eps, 1.0f - eps);
        losses[i] = -(target[i] * std::log(clamped) +
                      (1.0f - target[i]) * std::log(1.0f - clamped));
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
        const float clamped = std::clamp(pred[i], eps, 1.0f - eps);
        out[i] = ((clamped - target[i]) / (clamped * (1.0f - clamped) + eps)) * scale;
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

Tensor CpuBCEWithLogitsForward(const Tensor& predictions, const Tensor& targets, Reduction reduction) {
    ValidateFloat32Pair(predictions, targets, "BCEWithLogits");
    const size_t count = predictions.NumElements();
    const float* logits = predictions.Data<float>();
    const float* target = targets.Data<float>();
    std::vector<float> losses(count);
    for (size_t i = 0; i < count; ++i) {
        const float logit = logits[i];
        losses[i] = std::max(logit, 0.0f) - logit * target[i] +
                    std::log1p(std::exp(-std::fabs(logit)));
    }
    return ApplyCpuReduction(predictions.Shape(), losses, reduction);
}

Tensor CpuBCEWithLogitsBackward(const Tensor& predictions, const Tensor& targets, Reduction reduction) {
    ValidateFloat32Pair(predictions, targets, "BCEWithLogits");
    Tensor grad(predictions.Shape(), DataType::Float32);
    const size_t count = predictions.NumElements();
    const float scale = reduction == Reduction::Mean && count > 0 ? 1.0f / static_cast<float>(count) : 1.0f;
    const float* logits = predictions.Data<float>();
    const float* target = targets.Data<float>();
    float* out = grad.Data<float>();
    for (size_t i = 0; i < count; ++i) {
        out[i] = (CpuSigmoidValue(logits[i]) - target[i]) * scale;
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
        if (target[i] <= 0.0f) {
            continue;
        }
        if (log_target) {
            losses[i] = std::exp(target[i]) * (target[i] - pred[i]);
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
        if (target[i] <= 0.0f) {
            out[i] = 0.0f;
            continue;
        }
        out[i] = (log_target ? -std::exp(target[i]) : -target[i]) * scale;
    }
    return grad;
}

struct ClassAxisShape {
    size_t batch = 1;
    size_t classes = 0;
    bool batched = false;
};

ClassAxisShape ValidateClassAxisPredictions(const Tensor& predictions, const char* name) {
    if (predictions.GetDataType() != DataType::Float32) {
        throw std::runtime_error(std::string(name) + " only supports Float32 predictions");
    }
    const std::vector<size_t>& shape = predictions.Shape();
    if (shape.size() == 1) {
        return {1, shape[0], false};
    }
    if (shape.size() == 2) {
        return {shape[0], shape[1], true};
    }
    throw std::runtime_error(std::string(name) + " CPU fallback supports 1D or 2D predictions");
}

bool TargetsAreClassIndices(const Tensor& predictions, const Tensor& targets) {
    return targets.Shape() != predictions.Shape();
}

void ValidateClassIndexTargets(const Tensor& targets, const ClassAxisShape& shape, const char* name) {
    if (targets.GetDataType() != DataType::Int32 && targets.GetDataType() != DataType::Int64) {
        throw std::runtime_error(std::string(name) + " class-index targets must be Int32 or Int64");
    }
    const std::vector<size_t>& target_shape = targets.Shape();
    const bool valid = shape.batched ? target_shape == std::vector<size_t>{shape.batch}
                                     : targets.NumElements() == 1;
    if (!valid) {
        throw std::runtime_error(std::string(name) + " class-index target shape is invalid");
    }
}

int64_t ClassIndexAt(const Tensor& targets, size_t index) {
    if (targets.GetDataType() == DataType::Int32) {
        return static_cast<int64_t>(targets.Data<int32_t>()[index]);
    }
    return targets.Data<int64_t>()[index];
}

void ValidateClassIndex(int64_t class_index, size_t classes, const char* name) {
    if (class_index < 0 || class_index >= static_cast<int64_t>(classes)) {
        throw std::runtime_error(std::string(name) + " target class index is out of range");
    }
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

Tensor CpuSoftmaxRows(const Tensor& predictions, const ClassAxisShape& shape) {
    Tensor softmax(predictions.Shape(), DataType::Float32);
    const float* pred = predictions.Data<float>();
    float* out = softmax.Data<float>();
    for (size_t batch = 0; batch < shape.batch; ++batch) {
        const size_t base = batch * shape.classes;
        float max_value = pred[base];
        for (size_t c = 1; c < shape.classes; ++c) {
            max_value = std::max(max_value, pred[base + c]);
        }

        float sum_exp = 0.0f;
        for (size_t c = 0; c < shape.classes; ++c) {
            const float value = std::exp(pred[base + c] - max_value);
            out[base + c] = value;
            sum_exp += value;
        }
        for (size_t c = 0; c < shape.classes; ++c) {
            out[base + c] /= sum_exp;
        }
    }
    return softmax;
}

Tensor CpuCrossEntropyForward(const Tensor& predictions,
                              const Tensor& targets,
                              Reduction reduction,
                              int ignore_index,
                              Tensor* cached_softmax) {
    const ClassAxisShape shape = ValidateClassAxisPredictions(predictions, "CrossEntropy");
    Tensor softmax = CpuSoftmaxRows(predictions, shape);
    if (cached_softmax) {
        *cached_softmax = softmax;
    }

    const float* probs = softmax.Data<float>();
    std::vector<float> losses(shape.batch, 0.0f);
    if (TargetsAreClassIndices(predictions, targets)) {
        ValidateClassIndexTargets(targets, shape, "CrossEntropy");
        for (size_t batch = 0; batch < shape.batch; ++batch) {
            const int64_t class_index = ClassIndexAt(targets, shape.batched ? batch : 0);
            if (ignore_index >= 0 && class_index == ignore_index) {
                continue;
            }
            ValidateClassIndex(class_index, shape.classes, "CrossEntropy");
            losses[batch] = -std::log(probs[batch * shape.classes + static_cast<size_t>(class_index)] + 1e-10f);
        }
    } else {
        ValidateFloat32Pair(predictions, targets, "CrossEntropy");
        const float* target = targets.Data<float>();
        for (size_t batch = 0; batch < shape.batch; ++batch) {
            const size_t base = batch * shape.classes;
            for (size_t c = 0; c < shape.classes; ++c) {
                losses[batch] -= target[base + c] * std::log(probs[base + c] + 1e-10f);
            }
        }
    }
    return ApplyClassReduction(losses, shape.batch, reduction);
}

Tensor CpuCrossEntropyBackward(const Tensor& predictions,
                               const Tensor& targets,
                               Reduction reduction,
                               int ignore_index,
                               const Tensor& cached_softmax) {
    const ClassAxisShape shape = ValidateClassAxisPredictions(predictions, "CrossEntropy");
    Tensor softmax = cached_softmax.Shape() == predictions.Shape()
                         ? cached_softmax
                         : CpuSoftmaxRows(predictions, shape);

    Tensor grad(predictions.Shape(), DataType::Float32);
    const float* probs = softmax.Data<float>();
    float* out = grad.Data<float>();
    std::copy(probs, probs + predictions.NumElements(), out);

    if (TargetsAreClassIndices(predictions, targets)) {
        ValidateClassIndexTargets(targets, shape, "CrossEntropy");
        for (size_t batch = 0; batch < shape.batch; ++batch) {
            const int64_t class_index = ClassIndexAt(targets, shape.batched ? batch : 0);
            const size_t base = batch * shape.classes;
            if (ignore_index >= 0 && class_index == ignore_index) {
                std::fill(out + base, out + base + shape.classes, 0.0f);
                continue;
            }
            ValidateClassIndex(class_index, shape.classes, "CrossEntropy");
            out[base + static_cast<size_t>(class_index)] -= 1.0f;
        }
    } else {
        ValidateFloat32Pair(predictions, targets, "CrossEntropy");
        const float* target = targets.Data<float>();
        for (size_t i = 0; i < predictions.NumElements(); ++i) {
            out[i] -= target[i];
        }
    }

    if (reduction == Reduction::Mean && shape.batch > 0) {
        const float scale = 1.0f / static_cast<float>(shape.batch);
        for (size_t i = 0; i < predictions.NumElements(); ++i) {
            out[i] *= scale;
        }
    }
    return grad;
}

Tensor CpuNLLForward(const Tensor& predictions,
                     const Tensor& targets,
                     Reduction reduction,
                     int ignore_index) {
    const ClassAxisShape shape = ValidateClassAxisPredictions(predictions, "NLL");
    ValidateClassIndexTargets(targets, shape, "NLL");

    const float* log_probs = predictions.Data<float>();
    std::vector<float> losses(shape.batch, 0.0f);
    for (size_t batch = 0; batch < shape.batch; ++batch) {
        const int64_t class_index = ClassIndexAt(targets, shape.batched ? batch : 0);
        if (ignore_index >= 0 && class_index == ignore_index) {
            continue;
        }
        ValidateClassIndex(class_index, shape.classes, "NLL");
        losses[batch] = -log_probs[batch * shape.classes + static_cast<size_t>(class_index)];
    }
    return ApplyClassReduction(losses, shape.batch, reduction);
}

Tensor CpuNLLBackward(const Tensor& predictions,
                      const Tensor& targets,
                      Reduction reduction,
                      int ignore_index) {
    const ClassAxisShape shape = ValidateClassAxisPredictions(predictions, "NLL");
    ValidateClassIndexTargets(targets, shape, "NLL");

    Tensor grad = Tensor::Zeros(predictions.Shape(), DataType::Float32);
    float* out = grad.Data<float>();
    const float scale = reduction == Reduction::Mean && shape.batch > 0 ? 1.0f / static_cast<float>(shape.batch)
                                                                        : 1.0f;
    for (size_t batch = 0; batch < shape.batch; ++batch) {
        const int64_t class_index = ClassIndexAt(targets, shape.batched ? batch : 0);
        if (ignore_index >= 0 && class_index == ignore_index) {
            continue;
        }
        ValidateClassIndex(class_index, shape.classes, "NLL");
        out[batch * shape.classes + static_cast<size_t>(class_index)] = -scale;
    }
    return grad;
}

Tensor CpuFocalForward(const Tensor& predictions,
                       const Tensor& targets,
                       float alpha,
                       float gamma,
                       Reduction reduction,
                       Tensor* cached_probs) {
    const ClassAxisShape shape = ValidateClassAxisPredictions(predictions, "Focal");
    ValidateClassIndexTargets(targets, shape, "Focal");

    Tensor probs = CpuSoftmaxRows(predictions, shape);
    if (cached_probs) {
        *cached_probs = probs;
    }

    const float* prob_data = probs.Data<float>();
    std::vector<float> losses(shape.batch, 0.0f);
    for (size_t batch = 0; batch < shape.batch; ++batch) {
        const int64_t class_index = ClassIndexAt(targets, shape.batched ? batch : 0);
        ValidateClassIndex(class_index, shape.classes, "Focal");
        const float pt = std::max(prob_data[batch * shape.classes + static_cast<size_t>(class_index)], 1e-8f);
        losses[batch] = -alpha * std::pow(1.0f - pt, gamma) * std::log(pt);
    }
    return ApplyClassReduction(losses, shape.batch, reduction);
}

Tensor CpuFocalBackward(const Tensor& predictions,
                        const Tensor& targets,
                        float alpha,
                        float gamma,
                        Reduction reduction,
                        const Tensor& cached_probs) {
    const ClassAxisShape shape = ValidateClassAxisPredictions(predictions, "Focal");
    ValidateClassIndexTargets(targets, shape, "Focal");

    Tensor probs = cached_probs.Shape() == predictions.Shape()
                       ? cached_probs
                       : CpuSoftmaxRows(predictions, shape);

    Tensor grad(predictions.Shape(), DataType::Float32);
    const float* prob_data = probs.Data<float>();
    float* out = grad.Data<float>();

    for (size_t batch = 0; batch < shape.batch; ++batch) {
        const int64_t class_index = ClassIndexAt(targets, shape.batched ? batch : 0);
        ValidateClassIndex(class_index, shape.classes, "Focal");
        const size_t target_class = static_cast<size_t>(class_index);
        const size_t base = batch * shape.classes;
        const float pt = std::clamp(prob_data[base + target_class], 1e-8f, 1.0f - 1e-8f);
        const float one_minus_pt = 1.0f - pt;
        const float scale = alpha * (std::pow(one_minus_pt, gamma) -
                                    gamma * pt * std::pow(one_minus_pt, gamma - 1.0f) * std::log(pt));

        for (size_t c = 0; c < shape.classes; ++c) {
            const float target_value = c == target_class ? 1.0f : 0.0f;
            out[base + c] = scale * (prob_data[base + c] - target_value);
        }
    }

    if (reduction == Reduction::Mean && shape.batch > 0) {
        const float batch_scale = 1.0f / static_cast<float>(shape.batch);
        for (size_t i = 0; i < predictions.NumElements(); ++i) {
            out[i] *= batch_scale;
        }
    }
    return grad;
}

struct EmbeddingPairShape {
    size_t batch = 0;
    size_t dim = 0;
};

EmbeddingPairShape ValidateEmbeddingPair(const Tensor& x1, const Tensor& x2, const char* name) {
    if (x1.GetDataType() != DataType::Float32 || x2.GetDataType() != DataType::Float32) {
        throw std::runtime_error(std::string(name) + " only supports Float32 embeddings");
    }
    if (x1.Shape() != x2.Shape()) {
        throw std::runtime_error(std::string(name) + " requires matching embedding shapes");
    }
    const std::vector<size_t>& shape = x1.Shape();
    if (shape.size() != 2) {
        throw std::runtime_error(std::string(name) + " CPU fallback supports [batch, embedding_dim] tensors");
    }
    return {shape[0], shape[1]};
}

const float* ValidateEmbeddingLabels(const Tensor& labels, size_t batch, const char* name) {
    if (labels.GetDataType() != DataType::Float32) {
        throw std::runtime_error(std::string(name) + " labels must be Float32");
    }
    if (labels.Shape() != std::vector<size_t>{batch}) {
        throw std::runtime_error(std::string(name) + " labels must have shape [batch]");
    }
    return labels.Data<float>();
}

Tensor CpuCosineEmbeddingForward(const Tensor& x1,
                                 const Tensor& x2,
                                 const Tensor& labels,
                                 float margin,
                                 Reduction reduction) {
    const EmbeddingPairShape shape = ValidateEmbeddingPair(x1, x2, "CosineEmbedding");
    const float* label_data = ValidateEmbeddingLabels(labels, shape.batch, "CosineEmbedding");
    const float* a = x1.Data<float>();
    const float* b = x2.Data<float>();

    std::vector<float> losses(shape.batch, 0.0f);
    for (size_t batch = 0; batch < shape.batch; ++batch) {
        const size_t base = batch * shape.dim;
        float dot = 0.0f;
        float norm1_sq = 0.0f;
        float norm2_sq = 0.0f;
        for (size_t d = 0; d < shape.dim; ++d) {
            dot += a[base + d] * b[base + d];
            norm1_sq += a[base + d] * a[base + d];
            norm2_sq += b[base + d] * b[base + d];
        }
        const float norm_product = std::sqrt(norm1_sq + 1e-8f) * std::sqrt(norm2_sq + 1e-8f);
        const float cos_sim = dot / norm_product;
        losses[batch] = label_data[batch] > 0.0f ? 1.0f - cos_sim
                                                 : std::max(cos_sim - margin, 0.0f);
    }
    return ApplyClassReduction(losses, shape.batch, reduction);
}

Tensor CpuCosineEmbeddingBackward(const Tensor& x1,
                                  const Tensor& x2,
                                  const Tensor& labels,
                                  float margin,
                                  Reduction reduction) {
    const EmbeddingPairShape shape = ValidateEmbeddingPair(x1, x2, "CosineEmbedding");
    const float* label_data = ValidateEmbeddingLabels(labels, shape.batch, "CosineEmbedding");
    const float* a = x1.Data<float>();
    const float* b = x2.Data<float>();

    Tensor grad(x1.Shape(), DataType::Float32);
    float* out = grad.Data<float>();
    for (size_t batch = 0; batch < shape.batch; ++batch) {
        const size_t base = batch * shape.dim;
        float dot = 0.0f;
        float norm1_sq = 0.0f;
        float norm2_sq = 0.0f;
        for (size_t d = 0; d < shape.dim; ++d) {
            dot += a[base + d] * b[base + d];
            norm1_sq += a[base + d] * a[base + d];
            norm2_sq += b[base + d] * b[base + d];
        }

        const float norm1 = std::sqrt(norm1_sq + 1e-8f);
        const float norm2 = std::sqrt(norm2_sq + 1e-8f);
        const float norm_product = norm1 * norm2;
        const float cos_sim = dot / norm_product;
        const float scale = label_data[batch] > 0.0f ? -1.0f
                          : (cos_sim > margin ? 1.0f : 0.0f);
        const float reduction_scale = reduction == Reduction::Mean && shape.batch > 0
                                          ? 1.0f / static_cast<float>(shape.batch)
                                          : 1.0f;

        for (size_t d = 0; d < shape.dim; ++d) {
            const float grad_cos = b[base + d] / norm_product -
                                   cos_sim * a[base + d] / (norm1_sq + 1e-8f);
            out[base + d] = scale * reduction_scale * grad_cos;
        }
    }
    return grad;
}

Tensor CpuTripletForward(const Tensor& anchor,
                         const Tensor& positive,
                         const Tensor& negative,
                         TripletLoss::DistanceType distance_type,
                         float margin,
                         Reduction reduction,
                         Tensor* cached_dist_ap,
                         Tensor* cached_dist_an) {
    const EmbeddingPairShape shape = ValidateEmbeddingPair(anchor, positive, "Triplet");
    ValidateEmbeddingPair(anchor, negative, "Triplet");

    const float* a = anchor.Data<float>();
    const float* p = positive.Data<float>();
    const float* n = negative.Data<float>();
    std::vector<float> dist_ap(shape.batch, 0.0f);
    std::vector<float> dist_an(shape.batch, 0.0f);
    std::vector<float> losses(shape.batch, 0.0f);

    for (size_t batch = 0; batch < shape.batch; ++batch) {
        const size_t base = batch * shape.dim;
        if (distance_type == TripletLoss::DistanceType::Euclidean) {
            float sum_ap = 0.0f;
            float sum_an = 0.0f;
            for (size_t d = 0; d < shape.dim; ++d) {
                const float diff_ap = a[base + d] - p[base + d];
                const float diff_an = a[base + d] - n[base + d];
                sum_ap += diff_ap * diff_ap;
                sum_an += diff_an * diff_an;
            }
            dist_ap[batch] = std::sqrt(sum_ap);
            dist_an[batch] = std::sqrt(sum_an);
        } else {
            float dot_ap = 0.0f;
            float dot_an = 0.0f;
            float norm_a_sq = 0.0f;
            float norm_p_sq = 0.0f;
            float norm_n_sq = 0.0f;
            for (size_t d = 0; d < shape.dim; ++d) {
                dot_ap += a[base + d] * p[base + d];
                dot_an += a[base + d] * n[base + d];
                norm_a_sq += a[base + d] * a[base + d];
                norm_p_sq += p[base + d] * p[base + d];
                norm_n_sq += n[base + d] * n[base + d];
            }
            const float norm_a = std::sqrt(norm_a_sq + 1e-8f);
            dist_ap[batch] = 1.0f - dot_ap / (norm_a * std::sqrt(norm_p_sq + 1e-8f));
            dist_an[batch] = 1.0f - dot_an / (norm_a * std::sqrt(norm_n_sq + 1e-8f));
        }
        losses[batch] = std::max(dist_ap[batch] - dist_an[batch] + margin, 0.0f);
    }

    if (cached_dist_ap) {
        *cached_dist_ap = Tensor({shape.batch}, dist_ap.data(), DataType::Float32);
    }
    if (cached_dist_an) {
        *cached_dist_an = Tensor({shape.batch}, dist_an.data(), DataType::Float32);
    }
    return ApplyClassReduction(losses, shape.batch, reduction);
}

Tensor CpuTripletBackward(const Tensor& anchor,
                          const Tensor& positive,
                          const Tensor& negative,
                          const Tensor& cached_dist_ap,
                          const Tensor& cached_dist_an,
                          TripletLoss::DistanceType distance_type,
                          float margin,
                          Reduction reduction) {
    const EmbeddingPairShape shape = ValidateEmbeddingPair(anchor, positive, "Triplet");
    ValidateEmbeddingPair(anchor, negative, "Triplet");

    Tensor dist_ap = cached_dist_ap.Shape() == std::vector<size_t>{shape.batch}
                         ? cached_dist_ap
                         : Tensor();
    Tensor dist_an = cached_dist_an.Shape() == std::vector<size_t>{shape.batch}
                         ? cached_dist_an
                         : Tensor();
    if (dist_ap.Shape().empty() || dist_an.Shape().empty()) {
        CpuTripletForward(anchor, positive, negative, distance_type, margin, Reduction::None, &dist_ap, &dist_an);
    }

    Tensor grad(anchor.Shape(), DataType::Float32);
    float* out = grad.Data<float>();
    const float* a = anchor.Data<float>();
    const float* p = positive.Data<float>();
    const float* n = negative.Data<float>();
    const float* dist_ap_data = dist_ap.Data<float>();
    const float* dist_an_data = dist_an.Data<float>();
    const float reduction_scale = reduction == Reduction::Mean && shape.batch > 0
                                      ? 1.0f / static_cast<float>(shape.batch)
                                      : 1.0f;

    for (size_t batch = 0; batch < shape.batch; ++batch) {
        const size_t base = batch * shape.dim;
        const bool active = dist_ap_data[batch] - dist_an_data[batch] + margin > 0.0f;
        if (!active) {
            std::fill(out + base, out + base + shape.dim, 0.0f);
            continue;
        }

        if (distance_type == TripletLoss::DistanceType::Euclidean) {
            const float safe_ap = std::max(dist_ap_data[batch], 1e-8f);
            const float safe_an = std::max(dist_an_data[batch], 1e-8f);
            for (size_t d = 0; d < shape.dim; ++d) {
                const float grad_ap = (a[base + d] - p[base + d]) / safe_ap;
                const float grad_an = (a[base + d] - n[base + d]) / safe_an;
                out[base + d] = (grad_ap - grad_an) * reduction_scale;
            }
        } else {
            float norm_a_sq = 0.0f;
            float norm_p_sq = 0.0f;
            float norm_n_sq = 0.0f;
            for (size_t d = 0; d < shape.dim; ++d) {
                norm_a_sq += a[base + d] * a[base + d];
                norm_p_sq += p[base + d] * p[base + d];
                norm_n_sq += n[base + d] * n[base + d];
            }
            const float norm_a = std::sqrt(norm_a_sq + 1e-8f);
            const float norm_p = std::sqrt(norm_p_sq + 1e-8f);
            const float norm_n = std::sqrt(norm_n_sq + 1e-8f);
            for (size_t d = 0; d < shape.dim; ++d) {
                const float grad_ap = -p[base + d] / (norm_a * norm_p + 1e-8f);
                const float grad_an = -n[base + d] / (norm_a * norm_n + 1e-8f);
                out[base + d] = (grad_ap - grad_an) * reduction_scale;
            }
        }
    }

    return grad;
}

Tensor CpuContrastiveForward(const Tensor& x1,
                             const Tensor& x2,
                             const Tensor& labels,
                             float margin,
                             Reduction reduction,
                             Tensor* cached_distances) {
    const EmbeddingPairShape shape = ValidateEmbeddingPair(x1, x2, "Contrastive");
    const float* label_data = ValidateEmbeddingLabels(labels, shape.batch, "Contrastive");
    const float* a = x1.Data<float>();
    const float* b = x2.Data<float>();
    std::vector<float> distances(shape.batch, 0.0f);
    std::vector<float> losses(shape.batch, 0.0f);

    for (size_t batch = 0; batch < shape.batch; ++batch) {
        const size_t base = batch * shape.dim;
        float distance_sq = 0.0f;
        for (size_t d = 0; d < shape.dim; ++d) {
            const float diff = a[base + d] - b[base + d];
            distance_sq += diff * diff;
        }

        distances[batch] = std::sqrt(distance_sq);
        const float margin_diff = std::max(margin - distances[batch], 0.0f);
        losses[batch] = label_data[batch] > 0.0f
                            ? margin_diff * margin_diff
                            : distance_sq;
    }

    if (cached_distances) {
        *cached_distances = Tensor({shape.batch}, distances.data(), DataType::Float32);
    }
    return ApplyClassReduction(losses, shape.batch, reduction);
}

Tensor CpuContrastiveBackward(const Tensor& x1,
                              const Tensor& x2,
                              const Tensor& labels,
                              const Tensor& cached_distances,
                              float margin,
                              Reduction reduction) {
    const EmbeddingPairShape shape = ValidateEmbeddingPair(x1, x2, "Contrastive");
    const float* label_data = ValidateEmbeddingLabels(labels, shape.batch, "Contrastive");

    Tensor distances = cached_distances.Shape() == std::vector<size_t>{shape.batch}
                           ? cached_distances
                           : Tensor();
    if (distances.Shape().empty()) {
        CpuContrastiveForward(x1, x2, labels, margin, Reduction::None, &distances);
    }

    Tensor grad(x1.Shape(), DataType::Float32);
    float* out = grad.Data<float>();
    const float* a = x1.Data<float>();
    const float* b = x2.Data<float>();
    const float* distance_data = distances.Data<float>();
    const float reduction_scale = reduction == Reduction::Mean && shape.batch > 0
                                      ? 1.0f / static_cast<float>(shape.batch)
                                      : 1.0f;

    for (size_t batch = 0; batch < shape.batch; ++batch) {
        const size_t base = batch * shape.dim;
        const bool dissimilar = label_data[batch] > 0.0f;
        const bool active_dissimilar = dissimilar && distance_data[batch] < margin;
        const float safe_distance = std::max(distance_data[batch], 1e-8f);
        const float dissimilar_scale = active_dissimilar
                                           ? -2.0f * (margin - distance_data[batch]) / safe_distance
                                           : 0.0f;
        const float scale = dissimilar ? dissimilar_scale : 2.0f;

        for (size_t d = 0; d < shape.dim; ++d) {
            out[base + d] = scale * (a[base + d] - b[base + d]) * reduction_scale;
        }
    }

    return grad;
}

} // namespace

// ============================================================================
// Helper Functions for ArrayFire Integration
// ============================================================================

#ifdef CYXWIZ_HAS_ARRAYFIRE

// Helper: Create ArrayFire array from Tensor
// Note: CyxWiz Tensor uses row-major (C-style), ArrayFire uses column-major (Fortran-style)
// For 2D arrays [rows, cols], we need to transpose after loading row-major data
static af::array TensorToAf(const Tensor& t) {
    return t.Shape().size() == 2 ? t.GetArrayRowMajor2D() : t.GetArray();
}

// Helper: Create Tensor from ArrayFire array
// Note: Transpose 2D arrays back to row-major for CyxWiz Tensor
static Tensor AfToTensor(const af::array& arr) {
    // Count significant dimensions
    int ndims = 0;
    for (unsigned int i = 0; i < 4; i++) {
        if (arr.dims(i) > 1) ndims = i + 1;
        else if (i == 0) ndims = 1;
    }

    // For 2D arrays, transpose to row-major before copying to Tensor
    if (ndims == 2) {
        return Tensor::FromArrayRowMajor2D(arr);
    }

    // For other dimensions, keep the ArrayFire result resident until host data is requested.
    return Tensor(arr);
}

// Helper: Apply reduction to loss tensor
static af::array ApplyReduction(const af::array& loss, Reduction reduction) {
    switch (reduction) {
        case Reduction::None:
            return loss;
        case Reduction::Mean:
            return af::mean(loss);
        case Reduction::Sum:
            return af::sum(loss);
        default:
            return af::mean(loss);
    }
}

// Helper: Numerically stable softmax for cross entropy
static af::array StableSoftmax(const af::array& x, int axis = 0) {
    af::array max_val = af::max(x, axis);
    af::dim4 tile_dims(1, 1, 1, 1);
    tile_dims[axis] = x.dims(axis);
    af::array x_stable = x - af::tile(max_val, tile_dims);
    af::array exp_x = af::exp(x_stable);
    af::array sum_exp = af::sum(exp_x, axis);
    return exp_x / af::tile(sum_exp, tile_dims);
}

static af::array SignLike(const af::array& x) {
    const af::array ones = af::constant(1.0f, x.dims(), f32);
    const af::array minus_ones = af::constant(-1.0f, x.dims(), f32);
    const af::array zeros = af::constant(0.0f, x.dims(), f32);
    return af::select(x > 0.0f, ones, af::select(x < 0.0f, minus_ones, zeros));
}

#endif // CYXWIZ_HAS_ARRAYFIRE

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<Loss> CreateLoss(LossType type, Reduction reduction, float delta) {
    switch (type) {
        case LossType::MSE:
            return std::make_unique<MSELoss>(reduction);
        case LossType::CrossEntropy:
            return std::make_unique<CrossEntropyLoss>(reduction);
        case LossType::BinaryCrossEntropy:
            return std::make_unique<BCELoss>(reduction);
        case LossType::BCEWithLogits:
            return std::make_unique<BCEWithLogitsLoss>(reduction);
        case LossType::NLLLoss:
            return std::make_unique<NLLLoss>(reduction);
        case LossType::L1:
            return std::make_unique<L1Loss>(reduction);
        case LossType::SmoothL1:
        case LossType::Huber:
            return std::make_unique<SmoothL1Loss>(delta, reduction);
        case LossType::KLDivergence:
            return std::make_unique<KLDivLoss>(reduction);
        case LossType::CosineEmbedding:
            return std::make_unique<CosineEmbeddingLoss>(0.0f, reduction);
        case LossType::Focal:
            return std::make_unique<FocalLoss>(0.25f, 2.0f, reduction);
        case LossType::Triplet:
            return std::make_unique<TripletLoss>(1.0f, TripletLoss::DistanceType::Euclidean, reduction);
        case LossType::Contrastive:
            return std::make_unique<ContrastiveLoss>(1.0f, reduction);
        default:
            throw std::runtime_error("Unknown loss type");
    }
}

// ============================================================================
// MSE Loss Implementation
// ============================================================================

Tensor MSELoss::Forward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array pred = TensorToAf(predictions);
        af::array target = TensorToAf(targets);

        // MSE: mean((pred - target)^2)
        af::array diff = pred - target;
        af::array squared = diff * diff;
        af::array loss = ApplyReduction(squared, reduction_);

        return AfToTensor(loss);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire MSELoss::Forward failed: {}", e.what());
    }
#endif
    return CpuMSEForward(predictions, targets, reduction_);
}

Tensor MSELoss::Backward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array pred = TensorToAf(predictions);
        af::array target = TensorToAf(targets);

        // Gradient: 2 * (pred - target) / N
        af::array diff = pred - target;
        float scale = 2.0f;

        if (reduction_ == Reduction::Mean) {
            scale /= static_cast<float>(pred.elements());
        }

        af::array grad = diff * scale;

        return AfToTensor(grad);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire MSELoss::Backward failed: {}", e.what());
    }
#endif
    return CpuMSEBackward(predictions, targets, reduction_);
}

// ============================================================================
// L1 Loss Implementation
// ============================================================================

Tensor L1Loss::Forward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array pred = TensorToAf(predictions);
        af::array target = TensorToAf(targets);

        // L1: mean(|pred - target|)
        af::array diff = af::abs(pred - target);
        af::array loss = ApplyReduction(diff, reduction_);

        return AfToTensor(loss);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire L1Loss::Forward failed: {}", e.what());
    }
#endif
    return CpuL1Forward(predictions, targets, reduction_);
}

Tensor L1Loss::Backward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array pred = TensorToAf(predictions);
        af::array target = TensorToAf(targets);

        // Gradient: sign(pred - target) / N
        af::array diff = pred - target;
        af::array grad = SignLike(diff);

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(pred.elements());
        }

        return AfToTensor(grad);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire L1Loss::Backward failed: {}", e.what());
    }
#endif
    return CpuL1Backward(predictions, targets, reduction_);
}

// ============================================================================
// Smooth L1 Loss (Huber Loss) Implementation
// ============================================================================

Tensor SmoothL1Loss::Forward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array pred = TensorToAf(predictions);
        af::array target = TensorToAf(targets);

        // SmoothL1: 0.5 * x^2 / delta     if |x| < delta
        //           |x| - 0.5 * delta     otherwise
        af::array diff = pred - target;
        af::array abs_diff = af::abs(diff);

        af::array quadratic = 0.5f * diff * diff / delta_;
        af::array linear = abs_diff - 0.5f * delta_;

        af::array loss = af::select(abs_diff < delta_, quadratic, linear);
        loss = ApplyReduction(loss, reduction_);

        return AfToTensor(loss);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire SmoothL1Loss::Forward failed: {}", e.what());
    }
#endif
    return CpuSmoothL1Forward(predictions, targets, delta_, reduction_);
}

Tensor SmoothL1Loss::Backward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array pred = TensorToAf(predictions);
        af::array target = TensorToAf(targets);

        // Gradient: x / delta    if |x| < delta
        //           sign(x)      otherwise
        af::array diff = pred - target;
        af::array abs_diff = af::abs(diff);

        af::array grad_quadratic = diff / delta_;
        af::array grad_linear = SignLike(diff);

        af::array grad = af::select(abs_diff < delta_, grad_quadratic, grad_linear);

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(pred.elements());
        }

        return AfToTensor(grad);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire SmoothL1Loss::Backward failed: {}", e.what());
    }
#endif
    return CpuSmoothL1Backward(predictions, targets, delta_, reduction_);
}

// ============================================================================
// Cross Entropy Loss Implementation
// ============================================================================

Tensor CrossEntropyLoss::Forward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array pred = TensorToAf(predictions);
        af::array target = TensorToAf(targets);

        // Apply softmax to predictions (assume logits input)
        // For numerical stability, use log-softmax
        int class_axis = 1;  // Assume predictions are [batch, classes]
        if (pred.numdims() == 1) {
            class_axis = 0;
        }

        af::array softmax_pred = StableSoftmax(pred, class_axis);
        cached_softmax_ = AfToTensor(softmax_pred);

        // Cross entropy: -sum(target * log(softmax))
        af::array log_softmax = af::log(softmax_pred + 1e-10f);
        af::array loss;

        // Check if target is one-hot encoded or class indices
        if (target.type() == af::dtype::s32 || target.type() == af::dtype::s64) {
            // Targets are class indices - GPU-optimized gather
            dim_t batch_size = pred.dims(0);
            
            // Create linear indices for gathering from flattened log_softmax
            af::array batch_indices = af::range(af::dim4(batch_size), 0, s32);
            af::array target_int = target.as(s32);
            af::array linear_indices = target_int * static_cast<int>(batch_size) + batch_indices;
            
            // Gather log probabilities at target indices (single GPU operation)
            af::array flat_log_softmax = af::flat(log_softmax);
            af::array gathered = flat_log_softmax(linear_indices);
            
            // Cross entropy loss: -log_softmax[target]
            af::array batch_loss = -gathered;
            
            // Handle ignore_index with mask (GPU operation)
            if (ignore_index_ >= 0) {
                af::array mask = (target_int != ignore_index_).as(f32);
                batch_loss = batch_loss * mask;
            }
            
            loss = ApplyReduction(batch_loss, reduction_);
        } else {
            // Targets are one-hot encoded or soft labels
            loss = -target * log_softmax;

            // Sum over class dimension
            loss = af::sum(loss, class_axis);
            loss = ApplyReduction(loss, reduction_);
        }

        return AfToTensor(loss);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire CrossEntropyLoss::Forward failed: {}", e.what());
    }
#endif
    return CpuCrossEntropyForward(predictions, targets, reduction_, ignore_index_, &cached_softmax_);
}

Tensor CrossEntropyLoss::Backward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array pred = TensorToAf(predictions);
        af::array softmax_pred;
        if (cached_softmax_.Shape() == predictions.Shape()) {
            softmax_pred = TensorToAf(cached_softmax_);
        } else {
            int class_axis = pred.numdims() == 1 ? 0 : 1;
            softmax_pred = StableSoftmax(pred, class_axis);
        }
        af::array target = TensorToAf(targets);

        af::array grad;

        // Check if target is one-hot encoded or class indices
        if (target.type() == af::dtype::s32 || target.type() == af::dtype::s64) {
            // Targets are class indices - GPU-optimized one-hot encoding
            dim_t num_classes = pred.dims(1);
            af::array target_int = target.as(s32);

            // Create one-hot using identity matrix indexing (GPU operation)
            af::array identity = af::identity(af::dim4(num_classes, num_classes), f32);
            af::array one_hot = identity(af::span, target_int);  // [num_classes, batch]
            one_hot = af::transpose(one_hot);  // [batch, num_classes]

            // Handle ignore_index with mask (GPU operation)
            if (ignore_index_ >= 0) {
                af::array mask = (target_int != ignore_index_).as(f32);
                af::array mask_tiled = af::tile(mask, 1, static_cast<unsigned>(num_classes));
                one_hot = one_hot * mask_tiled;
            }

            grad = softmax_pred - one_hot;
        } else {
            // Targets are one-hot encoded or soft labels
            grad = softmax_pred - target;
        }

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(pred.dims(0));
        }

        return AfToTensor(grad);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire CrossEntropyLoss::Backward failed: {}", e.what());
    }
#endif
    return CpuCrossEntropyBackward(predictions, targets, reduction_, ignore_index_, cached_softmax_);
}

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

        // BCE: -[target * log(pred) + (1 - target) * log(1 - pred)]
        af::array loss = -(target * af::log(pred_clamped) +
                          (1.0f - target) * af::log(1.0f - pred_clamped));

        loss = ApplyReduction(loss, reduction_);

        return AfToTensor(loss);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire BCELoss::Forward failed: {}", e.what());
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

        // Gradient: -target/pred + (1-target)/(1-pred)
        //         = (pred - target) / (pred * (1 - pred))
        af::array grad = (pred_clamped - target) / (pred_clamped * (1.0f - pred_clamped) + eps_);

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(pred.elements());
        }

        return AfToTensor(grad);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire BCELoss::Backward failed: {}", e.what());
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

        loss = ApplyReduction(loss, reduction_);

        return AfToTensor(loss);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire BCEWithLogitsLoss::Forward failed: {}", e.what());
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
        af::array grad = sigmoid_logits - target;

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(logits.elements());
        }

        return AfToTensor(grad);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire BCEWithLogitsLoss::Backward failed: {}", e.what());
    }
#endif
    return CpuBCEWithLogitsBackward(predictions, targets, reduction_);
}

// ============================================================================
// NLL Loss Implementation
// ============================================================================

Tensor NLLLoss::Forward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array log_probs = TensorToAf(predictions);  // Expects log probabilities [batch, classes]
        af::array target = TensorToAf(targets);         // Class indices [batch]

        dim_t batch_size = log_probs.dims(0);

        // GPU-optimized gather: compute linear indices and gather in one operation
        // Linear index = batch_idx * num_classes + class_idx
        af::array batch_indices = af::range(af::dim4(batch_size), 0, s32);
        af::array target_int = target.as(s32);

        // Compute linear indices for gathering from flattened log_probs
        af::array linear_indices = target_int * static_cast<int>(batch_size) + batch_indices;

        // Gather log probabilities at target indices (single GPU operation)
        af::array flat_log_probs = af::flat(log_probs);
        af::array gathered = flat_log_probs(linear_indices);

        // NLL loss: -log_probs[target]
        af::array batch_loss = -gathered;

        // Handle ignore_index with mask (GPU operation)
        if (ignore_index_ >= 0) {
            af::array mask = (target_int != ignore_index_).as(f32);
            batch_loss = batch_loss * mask;
        }

        af::array loss = ApplyReduction(batch_loss, reduction_);

        return AfToTensor(loss);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire NLLLoss::Forward failed: {}", e.what());
    }
#endif
    return CpuNLLForward(predictions, targets, reduction_, ignore_index_);
}

Tensor NLLLoss::Backward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array log_probs = TensorToAf(predictions);
        af::array target = TensorToAf(targets);

        dim_t batch_size = log_probs.dims(0);
        dim_t num_classes = log_probs.dims(1);

        // GPU-optimized: Create one-hot encoding using identity matrix indexing
        af::array target_int = target.as(s32);
        
        // Create one-hot gradient: -1 at target class, 0 elsewhere
        // Use identity matrix rows indexed by target classes
        af::array identity = af::identity(af::dim4(num_classes, num_classes), f32);
        
        // Gather rows from identity matrix at target indices (one-hot encoding)
        af::array one_hot = identity(af::span, target_int);  // [num_classes, batch]
        one_hot = af::transpose(one_hot);  // [batch, num_classes]
        
        // Gradient is -one_hot
        af::array grad = -one_hot;

        // Handle ignore_index with mask (GPU operation)
        if (ignore_index_ >= 0) {
            af::array mask = (target_int != ignore_index_).as(f32);
            // Tile mask to match grad dimensions
            af::array mask_tiled = af::tile(mask, 1, static_cast<unsigned>(num_classes));
            grad = grad * mask_tiled;
        }

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(batch_size);
        }

        return AfToTensor(grad);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire NLLLoss::Backward failed: {}", e.what());
    }
#endif
    return CpuNLLBackward(predictions, targets, reduction_, ignore_index_);
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
            loss = target_prob * (target - log_pred);
        } else {
            // KL = target * (log(target) - pred)
            // Avoid log(0) by adding small epsilon
            af::array log_target = af::log(target + 1e-10f);
            loss = target * (log_target - log_pred);
        }

        // Only consider positive targets
        loss = af::select(target > 0, loss, 0.0f);
        loss = ApplyReduction(loss, reduction_);

        return AfToTensor(loss);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire KLDivLoss::Forward failed: {}", e.what());
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

        // Only consider positive targets
        grad = af::select(target > 0, grad, 0.0f);

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(log_pred.elements());
        }

        return AfToTensor(grad);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire KLDivLoss::Backward failed: {}", e.what());
    }
#endif
    return CpuKLDivBackward(predictions, targets, log_target_, reduction_);
}

// ============================================================================
// Cosine Embedding Loss Implementation
// ============================================================================

Tensor CosineEmbeddingLoss::Forward(const Tensor& x1, const Tensor& x2) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array a1 = TensorToAf(x1);
        af::array a2 = TensorToAf(x2);
        af::array labels = TensorToAf(labels_);

        // Compute cosine similarity
        // cos(x1, x2) = (x1 . x2) / (||x1|| * ||x2||)
        af::array dot_product = af::sum(a1 * a2, 1);
        af::array norm1 = af::sqrt(af::sum(a1 * a1, 1) + 1e-8f);
        af::array norm2 = af::sqrt(af::sum(a2 * a2, 1) + 1e-8f);
        af::array cos_sim = dot_product / (norm1 * norm2);

        // Loss:
        // For similar pairs (y = 1): 1 - cos_sim
        // For dissimilar pairs (y = -1): max(0, cos_sim - margin)
        af::array loss_similar = 1.0f - cos_sim;
        af::array loss_dissimilar = af::max(cos_sim - margin_, 0.0f);

        af::array loss = af::select(labels > 0, loss_similar, loss_dissimilar);
        loss = ApplyReduction(loss, reduction_);

        return AfToTensor(loss);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire CosineEmbeddingLoss::Forward failed: {}", e.what());
    }
#endif
    return CpuCosineEmbeddingForward(x1, x2, labels_, margin_, reduction_);
}

Tensor CosineEmbeddingLoss::Backward(const Tensor& x1, const Tensor& x2) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array a1 = TensorToAf(x1);
        af::array a2 = TensorToAf(x2);
        af::array labels = TensorToAf(labels_);

        // Compute cosine similarity components
        af::array dot_product = af::sum(a1 * a2, 1);
        af::array norm1_sq = af::sum(a1 * a1, 1);
        af::array norm2_sq = af::sum(a2 * a2, 1);
        af::array norm1 = af::sqrt(norm1_sq + 1e-8f);
        af::array norm2 = af::sqrt(norm2_sq + 1e-8f);
        af::array norm_product = norm1 * norm2;
        af::array cos_sim = dot_product / norm_product;

        // Gradient of cosine similarity w.r.t x1
        // d(cos_sim)/dx1 = x2/(||x1||*||x2||) - cos_sim * x1/||x1||^2
        dim_t batch_size = a1.dims(0);
        af::dim4 tile_dims(1, static_cast<unsigned int>(a1.dims(1)));

        af::array grad_cos = a2 / af::tile(norm_product, tile_dims) -
                             a1 * af::tile(cos_sim / norm1_sq, tile_dims);

        // For similar pairs: d_loss = -d_cos_sim
        // For dissimilar pairs: d_loss = d_cos_sim (if cos_sim > margin)
        // Use mask-based approach instead of nested af::select with scalars
        af::array mask_similar = (labels > 0).as(af::dtype::f32);
        af::array mask_dissimilar = (1.0f - mask_similar);
        af::array mask_above_margin = (cos_sim > margin_).as(af::dtype::f32);
        af::array scale = mask_similar * (-1.0f) + mask_dissimilar * mask_above_margin;

        af::array grad = grad_cos * af::tile(scale, tile_dims);

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(batch_size);
        }

        return AfToTensor(grad);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire CosineEmbeddingLoss::Backward failed: {}", e.what());
    }
#endif
    return CpuCosineEmbeddingBackward(x1, x2, labels_, margin_, reduction_);
}

// ============================================================================
// Focal Loss Implementation
// ============================================================================

Tensor FocalLoss::Forward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array pred = TensorToAf(predictions);
        af::array target = TensorToAf(targets);

        // Apply softmax to get probabilities
        int class_axis = pred.numdims() == 1 ? 0 : 1;
        af::array probs = StableSoftmax(pred, class_axis);
        cached_probs_ = AfToTensor(probs);

        // Get probability of true class
        dim_t batch_size = probs.dims(0);
        af::array target_indices = target.as(af::dtype::s32);
        af::array batch_indices = af::range(af::dim4(batch_size), 0, s32);
        af::array linear_indices = target_indices * static_cast<int>(batch_size) + batch_indices;
        af::array pt = af::flat(probs)(linear_indices);

        // Focal loss: -alpha * (1 - pt)^gamma * log(pt)
        af::array focal_weight = af::pow(1.0f - pt, gamma_);
        af::array log_pt = af::log(af::max(pt, 1e-8f));
        af::array loss = -alpha_ * focal_weight * log_pt;

        loss = ApplyReduction(loss, reduction_);
        return AfToTensor(loss);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire FocalLoss::Forward failed: {}", e.what());
    }
#endif
    return CpuFocalForward(predictions, targets, alpha_, gamma_, reduction_, &cached_probs_);
}

Tensor FocalLoss::Backward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array probs;
        if (cached_probs_.Shape() == predictions.Shape()) {
            probs = TensorToAf(cached_probs_);
        } else {
            af::array pred = TensorToAf(predictions);
            int class_axis = pred.numdims() == 1 ? 0 : 1;
            probs = StableSoftmax(pred, class_axis);
        }
        af::array target = TensorToAf(targets);

        dim_t batch_size = probs.dims(0);
        dim_t num_classes = probs.dims(1);

        // Get probability of true class
        af::array target_indices = target.as(af::dtype::s32);
        af::array batch_indices = af::range(af::dim4(batch_size), 0, s32);
        af::array linear_indices = target_indices * static_cast<int>(batch_size) + batch_indices;
        af::array pt = af::flat(probs)(linear_indices);

        // Create one-hot target
        af::array one_hot = af::constant(0.0f, batch_size, num_classes);
        for (int i = 0; i < batch_size; ++i) {
            int class_idx = target(i).scalar<int>();
            one_hot(i, class_idx) = 1.0f;
        }

        // d_loss/d_pred = alpha * [(1-pt)^gamma - gamma*pt*(1-pt)^(gamma-1)*log(pt)] * (p - y)
        af::dim4 tile_dims(1, static_cast<unsigned int>(num_classes));
        af::array log_pt = af::log(af::max(pt, 1e-8f));
        af::array one_minus_pt = 1.0f - pt;
        af::array scale = alpha_ * (af::pow(one_minus_pt, gamma_) -
                                    gamma_ * pt * af::pow(one_minus_pt, gamma_ - 1.0f) * log_pt);

        af::array grad = af::tile(scale, tile_dims) * (probs - one_hot);

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(batch_size);
        }

        return AfToTensor(grad);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire FocalLoss::Backward failed: {}", e.what());
    }
#endif
    return CpuFocalBackward(predictions, targets, alpha_, gamma_, reduction_, cached_probs_);
}

// ============================================================================
// Triplet Loss Implementation
// ============================================================================

Tensor TripletLoss::Forward(const Tensor& anchor, const Tensor& positive) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array a = TensorToAf(anchor);
        af::array p = TensorToAf(positive);
        af::array n = TensorToAf(negative_);

        af::array dist_ap, dist_an;

        if (distance_type_ == DistanceType::Euclidean) {
            // Euclidean distance
            af::array diff_ap = a - p;
            af::array diff_an = a - n;
            dist_ap = af::sqrt(af::sum(diff_ap * diff_ap, 1));
            dist_an = af::sqrt(af::sum(diff_an * diff_an, 1));
        } else {
            // Cosine distance: 1 - cosine_similarity
            af::array norm_a = af::sqrt(af::sum(a * a, 1));
            af::array norm_p = af::sqrt(af::sum(p * p, 1));
            af::array norm_n = af::sqrt(af::sum(n * n, 1));
            af::array cos_ap = af::sum(a * p, 1) / (norm_a * norm_p + 1e-8f);
            af::array cos_an = af::sum(a * n, 1) / (norm_a * norm_n + 1e-8f);
            dist_ap = 1.0f - cos_ap;
            dist_an = 1.0f - cos_an;
        }

        cached_dist_ap_ = AfToTensor(dist_ap);
        cached_dist_an_ = AfToTensor(dist_an);

        // Triplet loss: max(d_ap - d_an + margin, 0)
        af::array loss = af::max(dist_ap - dist_an + margin_, 0.0f);
        loss = ApplyReduction(loss, reduction_);

        return AfToTensor(loss);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire TripletLoss::Forward failed: {}", e.what());
    }
#endif
    return CpuTripletForward(anchor, positive, negative_, distance_type_, margin_, reduction_,
                             &cached_dist_ap_, &cached_dist_an_);
}

Tensor TripletLoss::Backward(const Tensor& anchor, const Tensor& positive) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array a = TensorToAf(anchor);
        af::array p = TensorToAf(positive);
        af::array n = TensorToAf(negative_);

        dim_t batch_size = a.dims(0);
        dim_t embed_dim = a.dims(1);
        const std::vector<size_t> expected_cache_shape{static_cast<size_t>(batch_size)};

        af::array dist_ap, dist_an;
        if (cached_dist_ap_.Shape() == expected_cache_shape && cached_dist_an_.Shape() == expected_cache_shape) {
            dist_ap = TensorToAf(cached_dist_ap_);
            dist_an = TensorToAf(cached_dist_an_);
        } else if (distance_type_ == DistanceType::Euclidean) {
            af::array diff_ap = a - p;
            af::array diff_an = a - n;
            dist_ap = af::sqrt(af::sum(diff_ap * diff_ap, 1));
            dist_an = af::sqrt(af::sum(diff_an * diff_an, 1));
            cached_dist_ap_ = AfToTensor(dist_ap);
            cached_dist_an_ = AfToTensor(dist_an);
        } else {
            af::array norm_a = af::sqrt(af::sum(a * a, 1));
            af::array norm_p = af::sqrt(af::sum(p * p, 1));
            af::array norm_n = af::sqrt(af::sum(n * n, 1));
            af::array cos_ap = af::sum(a * p, 1) / (norm_a * norm_p + 1e-8f);
            af::array cos_an = af::sum(a * n, 1) / (norm_a * norm_n + 1e-8f);
            dist_ap = 1.0f - cos_ap;
            dist_an = 1.0f - cos_an;
            cached_dist_ap_ = AfToTensor(dist_ap);
            cached_dist_an_ = AfToTensor(dist_an);
        }

        // Gradient only non-zero where loss > 0
        af::array margin_violated = (dist_ap - dist_an + margin_ > 0).as(af::dtype::f32);
        af::dim4 tile_dims(1, static_cast<unsigned int>(embed_dim));

        af::array grad_a;
        if (distance_type_ == DistanceType::Euclidean) {
            // d(d_ap)/da = (a-p) / d_ap
            // d(d_an)/da = (a-n) / d_an
            // d_loss/da = d(d_ap)/da - d(d_an)/da = (a-p)/d_ap - (a-n)/d_an
            af::array safe_dist_ap = af::max(dist_ap, 1e-8f);
            af::array safe_dist_an = af::max(dist_an, 1e-8f);

            af::array grad_ap = (a - p) / af::tile(safe_dist_ap, tile_dims);
            af::array grad_an = (a - n) / af::tile(safe_dist_an, tile_dims);
            grad_a = (grad_ap - grad_an) * af::tile(margin_violated, tile_dims);
        } else {
            // Cosine distance gradient is more complex - simplified version
            af::array norm_a = af::sqrt(af::sum(a * a, 1));
            af::array norm_p = af::sqrt(af::sum(p * p, 1));
            af::array norm_n = af::sqrt(af::sum(n * n, 1));

            af::array grad_ap = -p / af::tile(norm_a * norm_p + 1e-8f, tile_dims);
            af::array grad_an = -n / af::tile(norm_a * norm_n + 1e-8f, tile_dims);
            grad_a = (grad_ap - grad_an) * af::tile(margin_violated, tile_dims);
        }

        if (reduction_ == Reduction::Mean) {
            grad_a = grad_a / static_cast<float>(batch_size);
        }

        return AfToTensor(grad_a);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire TripletLoss::Backward failed: {}", e.what());
    }
#endif
    return CpuTripletBackward(anchor, positive, negative_, cached_dist_ap_, cached_dist_an_,
                              distance_type_, margin_, reduction_);
}

// ============================================================================
// Contrastive Loss Implementation
// ============================================================================

Tensor ContrastiveLoss::Forward(const Tensor& x1, const Tensor& x2) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array a1 = TensorToAf(x1);
        af::array a2 = TensorToAf(x2);
        af::array labels = TensorToAf(labels_);

        // Compute pairwise Euclidean distance
        af::array diff = a1 - a2;
        af::array distances = af::sqrt(af::sum(diff * diff, 1));
        cached_distances_ = AfToTensor(distances);

        // Contrastive loss: y*d^2 + (1-y)*max(0, margin-d)^2
        // where y=0 for similar, y=1 for dissimilar
        af::array similar_loss = (1.0f - labels) * distances * distances;
        af::array margin_diff = af::max(0.0f, margin_ - distances);
        af::array dissimilar_loss = labels * margin_diff * margin_diff;

        af::array loss = similar_loss + dissimilar_loss;
        loss = ApplyReduction(loss, reduction_);

        return AfToTensor(loss);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire ContrastiveLoss::Forward failed: {}", e.what());
    }
#endif
    return CpuContrastiveForward(x1, x2, labels_, margin_, reduction_, &cached_distances_);
}

Tensor ContrastiveLoss::Backward(const Tensor& x1, const Tensor& x2) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array a1 = TensorToAf(x1);
        af::array a2 = TensorToAf(x2);
        af::array labels = TensorToAf(labels_);

        // Gradient w.r.t. x1
        // For similar: d_loss/dx1 = 2*(x1-x2) = 2*diff
        // For dissimilar: d_loss/dx1 = -2*(margin-d)/d * (x1-x2) if d < margin, else 0

        af::array diff = a1 - a2;
        dim_t batch_size = a1.dims(0);
        dim_t embed_dim = a1.dims(1);
        const std::vector<size_t> expected_cache_shape{static_cast<size_t>(batch_size)};
        af::array distances;
        if (cached_distances_.Shape() == expected_cache_shape) {
            distances = TensorToAf(cached_distances_);
        } else {
            distances = af::sqrt(af::sum(diff * diff, 1));
            cached_distances_ = AfToTensor(distances);
        }

        // Avoid division by zero
        af::array safe_distances = af::max(distances, 1e-8f);
        af::dim4 tile_dims(1, static_cast<unsigned int>(embed_dim));

        // Similar pairs gradient: 2 * diff
        af::array grad_similar = 2.0f * diff;

        // Dissimilar pairs gradient: -2 * (margin - d) / d * diff (when d < margin)
        af::array margin_diff = margin_ - safe_distances;
        af::array mask_in_margin = (distances < margin_).as(af::dtype::f32);
        af::array scale = -2.0f * margin_diff / safe_distances * mask_in_margin;
        af::array grad_dissimilar = diff * af::tile(scale, tile_dims);

        // Combine based on labels (0=similar, 1=dissimilar)
        af::array labels_tiled = af::tile(labels, tile_dims);
        af::array grad = (1.0f - labels_tiled) * grad_similar + labels_tiled * grad_dissimilar;

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(batch_size);
        }

        return AfToTensor(grad);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire ContrastiveLoss::Backward failed: {}", e.what());
    }
#endif
    return CpuContrastiveBackward(x1, x2, labels_, cached_distances_, margin_, reduction_);
}

} // namespace cyxwiz
