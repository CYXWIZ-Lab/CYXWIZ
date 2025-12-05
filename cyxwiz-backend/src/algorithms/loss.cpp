#include "cyxwiz/loss.h"
#include "cyxwiz/tensor.h"
#include <stdexcept>
#include <cmath>
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

// ============================================================================
// Helper Functions for ArrayFire Integration
// ============================================================================

#ifdef CYXWIZ_HAS_ARRAYFIRE

// Helper: Convert CyxWiz DataType to ArrayFire dtype
static af::dtype ToAfType(DataType dtype) {
    switch (dtype) {
        case DataType::Float32: return af::dtype::f32;
        case DataType::Float64: return af::dtype::f64;
        case DataType::Int32: return af::dtype::s32;
        case DataType::Int64: return af::dtype::s64;
        case DataType::UInt8: return af::dtype::u8;
        default: throw std::runtime_error("Unsupported DataType for ArrayFire");
    }
}

// Helper: Create ArrayFire array from Tensor
static af::array TensorToAf(const Tensor& t) {
    const auto& shape = t.Shape();
    af::dim4 dims(1, 1, 1, 1);
    for (size_t i = 0; i < shape.size() && i < 4; i++) {
        dims[static_cast<unsigned int>(i)] = static_cast<dim_t>(shape[i]);
    }

    af::array arr(dims, ToAfType(t.GetDataType()));
    arr.write(t.Data(), arr.bytes(), afHost);
    return arr;
}

// Helper: Create Tensor from ArrayFire array
static Tensor AfToTensor(const af::array& arr) {
    std::vector<size_t> shape;
    for (unsigned int i = 0; i < 4; i++) {
        if (arr.dims(i) > 1 || i == 0) {
            shape.push_back(static_cast<size_t>(arr.dims(i)));
        } else if (i > 0 && arr.dims(i) == 1) {
            bool all_ones = true;
            for (unsigned int j = i; j < 4; j++) {
                if (arr.dims(j) != 1) {
                    all_ones = false;
                    break;
                }
            }
            if (all_ones) break;
            shape.push_back(static_cast<size_t>(arr.dims(i)));
        }
    }

    DataType dtype = DataType::Float32;
    switch (arr.type()) {
        case af::dtype::f32: dtype = DataType::Float32; break;
        case af::dtype::f64: dtype = DataType::Float64; break;
        case af::dtype::s32: dtype = DataType::Int32; break;
        case af::dtype::s64: dtype = DataType::Int64; break;
        case af::dtype::u8: dtype = DataType::UInt8; break;
        default: dtype = DataType::Float32;
    }

    Tensor result(shape, dtype);
    arr.host(result.Data());
    return result;
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
    throw std::runtime_error("MSE forward requires ArrayFire");
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
    throw std::runtime_error("MSE backward requires ArrayFire");
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
    throw std::runtime_error("L1 forward requires ArrayFire");
}

Tensor L1Loss::Backward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array pred = TensorToAf(predictions);
        af::array target = TensorToAf(targets);

        // Gradient: sign(pred - target) / N
        af::array diff = pred - target;
        af::array grad = af::sign(diff);

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(pred.elements());
        }

        return AfToTensor(grad);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire L1Loss::Backward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("L1 backward requires ArrayFire");
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
    throw std::runtime_error("SmoothL1 forward requires ArrayFire");
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
        af::array grad_linear = af::sign(diff);

        af::array grad = af::select(abs_diff < delta_, grad_quadratic, grad_linear);

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(pred.elements());
        }

        return AfToTensor(grad);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire SmoothL1Loss::Backward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("SmoothL1 backward requires ArrayFire");
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
            // Targets are class indices
            // Gather log probabilities at target indices
            dim_t batch_size = pred.dims(0);
            af::array batch_loss = af::constant(0.0f, af::dim4(batch_size));

            for (dim_t i = 0; i < batch_size; i++) {
                int class_idx = target(i).scalar<int>();
                if (class_idx != ignore_index_) {
                    // Cast array_proxy to scalar to enable unary minus
                    batch_loss(i) = 0.0f - log_softmax(i, class_idx).scalar<float>();
                }
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
    throw std::runtime_error("CrossEntropy forward requires ArrayFire");
}

Tensor CrossEntropyLoss::Backward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array softmax_pred = TensorToAf(cached_softmax_);
        af::array target = TensorToAf(targets);
        af::array pred = TensorToAf(predictions);

        af::array grad;

        // Check if target is one-hot encoded or class indices
        if (target.type() == af::dtype::s32 || target.type() == af::dtype::s64) {
            // Targets are class indices
            // Gradient: softmax - one_hot(target)
            dim_t batch_size = pred.dims(0);
            dim_t num_classes = pred.dims(1);

            // Create one-hot encoding
            af::array one_hot = af::constant(0.0f, pred.dims());
            for (dim_t i = 0; i < batch_size; i++) {
                int class_idx = target(i).scalar<int>();
                if (class_idx != ignore_index_ && class_idx >= 0 && class_idx < num_classes) {
                    one_hot(i, class_idx) = 1.0f;
                }
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
    throw std::runtime_error("CrossEntropy backward requires ArrayFire");
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
    throw std::runtime_error("BCE forward requires ArrayFire");
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
    throw std::runtime_error("BCE backward requires ArrayFire");
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
    throw std::runtime_error("BCEWithLogits forward requires ArrayFire");
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
    throw std::runtime_error("BCEWithLogits backward requires ArrayFire");
}

// ============================================================================
// NLL Loss Implementation
// ============================================================================

Tensor NLLLoss::Forward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array log_probs = TensorToAf(predictions);  // Expects log probabilities
        af::array target = TensorToAf(targets);         // Class indices

        // NLL: -log_probs[target]
        dim_t batch_size = log_probs.dims(0);
        af::array batch_loss = af::constant(0.0f, af::dim4(batch_size));

        for (dim_t i = 0; i < batch_size; i++) {
            int class_idx = target(i).scalar<int>();
            if (class_idx != ignore_index_) {
                // Cast array_proxy to scalar to enable unary minus
                batch_loss(i) = 0.0f - log_probs(i, class_idx).scalar<float>();
            }
        }

        af::array loss = ApplyReduction(batch_loss, reduction_);

        return AfToTensor(loss);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire NLLLoss::Forward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("NLL forward requires ArrayFire");
}

Tensor NLLLoss::Backward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array log_probs = TensorToAf(predictions);
        af::array target = TensorToAf(targets);

        dim_t batch_size = log_probs.dims(0);
        dim_t num_classes = log_probs.dims(1);

        // Gradient: -1 at target class, 0 elsewhere
        af::array grad = af::constant(0.0f, log_probs.dims());

        for (dim_t i = 0; i < batch_size; i++) {
            int class_idx = target(i).scalar<int>();
            if (class_idx != ignore_index_ && class_idx >= 0 && class_idx < num_classes) {
                grad(i, class_idx) = -1.0f;
            }
        }

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(batch_size);
        }

        return AfToTensor(grad);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire NLLLoss::Backward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("NLL backward requires ArrayFire");
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
    throw std::runtime_error("KLDiv forward requires ArrayFire");
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
    throw std::runtime_error("KLDiv backward requires ArrayFire");
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
    throw std::runtime_error("CosineEmbedding forward requires ArrayFire");
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
    throw std::runtime_error("CosineEmbedding backward requires ArrayFire");
}

} // namespace cyxwiz
