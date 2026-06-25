#include "cyxwiz/losses/classification.h"
#include "loss_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
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
        " CPU fallback supports 1D, 2D, or [batch, seq, classes] predictions");
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

int64_t ClassIndexAt(const Tensor& targets, size_t index) {
    if (targets.GetDataType() == DataType::Int32) {
        return static_cast<int64_t>(targets.Data<int32_t>()[index]);
    }
    return targets.Data<int64_t>()[index];
}

bool ClassIndexTargetsContain(const Tensor& targets, int64_t value) {
    if (targets.GetDataType() != DataType::Int32 &&
        targets.GetDataType() != DataType::Int64) {
        return false;
    }
    for (size_t i = 0; i < targets.NumElements(); ++i) {
        if (ClassIndexAt(targets, i) == value) {
            return true;
        }
    }
    return false;
}

void ValidateClassIndex(int64_t class_index, size_t classes, const char* name) {
    if (class_index < 0 || class_index >= static_cast<int64_t>(classes)) {
        throw std::runtime_error(std::string(name) + " target class index is out of range");
    }
}

Tensor ApplyClassReduction(const std::vector<float>& per_sample,
                           const ClassAxisShape& shape,
                           Reduction reduction,
                           size_t mean_count = 0) {
    if (reduction == Reduction::None) {
        return Tensor(shape.unreduced_shape, per_sample.data(), DataType::Float32);
    }

    float total = 0.0f;
    for (float value : per_sample) {
        total += value;
    }
    const size_t divisor = mean_count > 0 ? mean_count : shape.batch;
    if (reduction == Reduction::Mean && divisor > 0) {
        total /= static_cast<float>(divisor);
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
    size_t mean_count = shape.batch;
    if (TargetsAreClassIndices(predictions, targets)) {
        ValidateClassIndexTargets(targets, shape, "CrossEntropy");
        mean_count = 0;
        for (size_t batch = 0; batch < shape.batch; ++batch) {
            const int64_t class_index = ClassIndexAt(targets, shape.batched ? batch : 0);
            if (class_index == ignore_index) {
                continue;
            }
            ValidateClassIndex(class_index, shape.classes, "CrossEntropy");
            losses[batch] = -std::log(probs[batch * shape.classes + static_cast<size_t>(class_index)] + 1e-10f);
            ++mean_count;
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
    return ApplyClassReduction(losses, shape, reduction, mean_count);
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
    size_t mean_count = shape.batch;

    if (TargetsAreClassIndices(predictions, targets)) {
        ValidateClassIndexTargets(targets, shape, "CrossEntropy");
        mean_count = 0;
        for (size_t batch = 0; batch < shape.batch; ++batch) {
            const int64_t class_index = ClassIndexAt(targets, shape.batched ? batch : 0);
            const size_t base = batch * shape.classes;
            if (class_index == ignore_index) {
                std::fill(out + base, out + base + shape.classes, 0.0f);
                continue;
            }
            ValidateClassIndex(class_index, shape.classes, "CrossEntropy");
            out[base + static_cast<size_t>(class_index)] -= 1.0f;
            ++mean_count;
        }
    } else {
        ValidateFloat32Pair(predictions, targets, "CrossEntropy");
        const float* target = targets.Data<float>();
        for (size_t i = 0; i < predictions.NumElements(); ++i) {
            out[i] -= target[i];
        }
    }

    const size_t divisor = mean_count > 0 ? mean_count : shape.batch;
    if (reduction == Reduction::Mean && divisor > 0) {
        const float scale = 1.0f / static_cast<float>(divisor);
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
    size_t mean_count = 0;
    for (size_t batch = 0; batch < shape.batch; ++batch) {
        const int64_t class_index = ClassIndexAt(targets, shape.batched ? batch : 0);
        if (class_index == ignore_index) {
            continue;
        }
        ValidateClassIndex(class_index, shape.classes, "NLL");
        losses[batch] = -log_probs[batch * shape.classes + static_cast<size_t>(class_index)];
        ++mean_count;
    }
    return ApplyClassReduction(losses, shape, reduction, mean_count);
}

Tensor CpuNLLBackward(const Tensor& predictions,
                      const Tensor& targets,
                      Reduction reduction,
                      int ignore_index) {
    const ClassAxisShape shape = ValidateClassAxisPredictions(predictions, "NLL");
    ValidateClassIndexTargets(targets, shape, "NLL");

    Tensor grad = Tensor::Zeros(predictions.Shape(), DataType::Float32);
    float* out = grad.Data<float>();
    size_t mean_count = 0;
    for (size_t batch = 0; batch < shape.batch; ++batch) {
        const int64_t class_index = ClassIndexAt(targets, shape.batched ? batch : 0);
        if (class_index == ignore_index) {
            continue;
        }
        ValidateClassIndex(class_index, shape.classes, "NLL");
        ++mean_count;
    }

    const size_t divisor = mean_count > 0 ? mean_count : shape.batch;
    const float scale = reduction == Reduction::Mean && divisor > 0 ? 1.0f / static_cast<float>(divisor)
                                                                    : 1.0f;
    for (size_t batch = 0; batch < shape.batch; ++batch) {
        const int64_t class_index = ClassIndexAt(targets, shape.batched ? batch : 0);
        if (class_index == ignore_index) {
            continue;
        }
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
    return ApplyClassReduction(losses, shape, reduction);
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

} // namespace

// ============================================================================
// Cross Entropy Loss Implementation
// ============================================================================

Tensor CrossEntropyLoss::Forward(const Tensor& predictions, const Tensor& targets) {
    if (TargetsAreClassIndices(predictions, targets) &&
        ClassIndexTargetsContain(targets, ignore_index_)) {
        return CpuCrossEntropyForward(
            predictions, targets, reduction_, ignore_index_, &cached_softmax_);
    }

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
        LogArrayFireLossFallbackOnce(
            "CrossEntropyLoss::Forward", e.what(), predictions, "predictions");
    }
#endif
    return CpuCrossEntropyForward(predictions, targets, reduction_, ignore_index_, &cached_softmax_);
}

Tensor CrossEntropyLoss::Backward(const Tensor& predictions, const Tensor& targets) {
    if (TargetsAreClassIndices(predictions, targets) &&
        ClassIndexTargetsContain(targets, ignore_index_)) {
        return CpuCrossEntropyBackward(
            predictions, targets, reduction_, ignore_index_, cached_softmax_);
    }
    if (predictions.Shape().size() == 3) {
        return CpuCrossEntropyBackward(
            predictions, targets, reduction_, ignore_index_, cached_softmax_);
    }

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
            one_hot.eval();

            // Handle ignore_index with mask (GPU operation)
            if (ignore_index_ >= 0) {
                af::array mask = (target_int != ignore_index_).as(f32);
                mask.eval();
                af::array mask_tiled = af::tile(mask, 1, static_cast<unsigned>(num_classes));
                mask_tiled.eval();
                one_hot = one_hot * mask_tiled;
                one_hot.eval();
            }

            grad = softmax_pred - one_hot;
        } else {
            // Targets are one-hot encoded or soft labels
            grad = softmax_pred - target;
        }
        grad.eval();

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(pred.dims(0));
            grad.eval();
        }

        return AfToTensor(grad);
    } catch (const af::exception& e) {
        LogArrayFireLossFallbackOnce(
            "CrossEntropyLoss::Backward", e.what(), predictions, "predictions");
    }
#endif
    return CpuCrossEntropyBackward(predictions, targets, reduction_, ignore_index_, cached_softmax_);
}

// ============================================================================
// NLL Loss Implementation
// ============================================================================

Tensor NLLLoss::Forward(const Tensor& predictions, const Tensor& targets) {
    if (ClassIndexTargetsContain(targets, ignore_index_)) {
        return CpuNLLForward(predictions, targets, reduction_, ignore_index_);
    }

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
        LogArrayFireLossFallbackOnce(
            "NLLLoss::Forward", e.what(), predictions, "predictions");
    }
#endif
    return CpuNLLForward(predictions, targets, reduction_, ignore_index_);
}

Tensor NLLLoss::Backward(const Tensor& predictions, const Tensor& targets) {
    if (ClassIndexTargetsContain(targets, ignore_index_)) {
        return CpuNLLBackward(predictions, targets, reduction_, ignore_index_);
    }

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
        LogArrayFireLossFallbackOnce(
            "NLLLoss::Backward", e.what(), predictions, "predictions");
    }
#endif
    return CpuNLLBackward(predictions, targets, reduction_, ignore_index_);
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
        LogArrayFireLossFallbackOnce(
            "FocalLoss::Forward", e.what(), predictions, "predictions");
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
        log_pt.eval();
        af::array one_minus_pt = 1.0f - pt;
        one_minus_pt.eval();
        af::array scale = alpha_ * (af::pow(one_minus_pt, gamma_) -
                                    gamma_ * pt * af::pow(one_minus_pt, gamma_ - 1.0f) * log_pt);
        scale.eval();

        af::array grad = af::tile(scale, tile_dims) * (probs - one_hot);
        grad.eval();

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(batch_size);
            grad.eval();
        }

        return AfToTensor(grad);
    } catch (const af::exception& e) {
        LogArrayFireLossFallbackOnce(
            "FocalLoss::Backward", e.what(), predictions, "predictions");
    }
#endif
    return CpuFocalBackward(predictions, targets, alpha_, gamma_, reduction_, cached_probs_);
}

} // namespace cyxwiz
