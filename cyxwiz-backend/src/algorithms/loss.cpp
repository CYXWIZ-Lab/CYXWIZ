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
// Note: CyxWiz Tensor uses row-major (C-style), ArrayFire uses column-major (Fortran-style)
// For 2D arrays [rows, cols], we need to transpose after loading row-major data
static af::array TensorToAf(const Tensor& t) {
    const auto& shape = t.Shape();
    af::dim4 dims(1, 1, 1, 1);
    for (size_t i = 0; i < shape.size() && i < 4; i++) {
        dims[static_cast<unsigned int>(i)] = static_cast<dim_t>(shape[i]);
    }

    // For 2D arrays, swap dimensions to account for row-major input
    // We load as [cols, rows] then transpose to get [rows, cols] in column-major
    if (shape.size() == 2) {
        af::dim4 swapped_dims(dims[1], dims[0], 1, 1);
        af::array arr(swapped_dims, ToAfType(t.GetDataType()));
        arr.write(t.Data(), arr.bytes(), afHost);
        return af::transpose(arr);  // Now [rows, cols] in column-major
    }

    af::array arr(dims, ToAfType(t.GetDataType()));
    arr.write(t.Data(), arr.bytes(), afHost);
    return arr;
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

    DataType dtype = DataType::Float32;
    switch (arr.type()) {
        case af::dtype::f32: dtype = DataType::Float32; break;
        case af::dtype::f64: dtype = DataType::Float64; break;
        case af::dtype::s32: dtype = DataType::Int32; break;
        case af::dtype::s64: dtype = DataType::Int64; break;
        case af::dtype::u8: dtype = DataType::UInt8; break;
        default: dtype = DataType::Float32;
    }

    // For 2D arrays, transpose to row-major before copying to Tensor
    if (ndims == 2) {
        af::array transposed = af::transpose(arr);
        std::vector<size_t> shape = {
            static_cast<size_t>(arr.dims(0)),
            static_cast<size_t>(arr.dims(1))
        };
        Tensor result(shape, dtype);
        transposed.host(result.Data());
        return result;
    }

    // For other dimensions, copy directly
    std::vector<size_t> shape;
    for (int i = 0; i < ndims; i++) {
        shape.push_back(static_cast<size_t>(arr.dims(i)));
    }
    if (shape.empty()) shape.push_back(1);

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
            // Targets are class indices - GPU-optimized gather
            dim_t batch_size = pred.dims(0);
            dim_t num_classes = pred.dims(1);
            
            // Create linear indices for gathering from flattened log_softmax
            af::array batch_indices = af::range(af::dim4(batch_size), 0, s32);
            af::array target_int = target.as(s32);
            af::array linear_indices = batch_indices * static_cast<int>(num_classes) + target_int;
            
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

            // DEBUG: Log shapes and sample values (first batch only)
            static bool logged = false;
            if (!logged) {
                logged = true;
                spdlog::info("DEBUG CrossEntropy: pred dims=({},{},{},{}), target dims=({},{},{},{})",
                    pred.dims(0), pred.dims(1), pred.dims(2), pred.dims(3),
                    target.dims(0), target.dims(1), target.dims(2), target.dims(3));
                spdlog::info("DEBUG CrossEntropy: loss before sum dims=({},{},{},{})",
                    loss.dims(0), loss.dims(1), loss.dims(2), loss.dims(3));

                // Sample first batch element
                af::array sample_softmax = softmax_pred(0, af::span);
                af::array sample_target = target(0, af::span);
                af::array sample_loss = loss(0, af::span);

                float* sm_data = sample_softmax.host<float>();
                float* tg_data = sample_target.host<float>();
                float* ls_data = sample_loss.host<float>();

                spdlog::info("DEBUG CrossEntropy: First sample softmax probs:");
                std::string sm_str = "  [";
                for (int i = 0; i < std::min((int)pred.dims(1), 10); i++) {
                    sm_str += fmt::format("{:.4f}", sm_data[i]);
                    if (i < std::min((int)pred.dims(1), 10) - 1) sm_str += ", ";
                }
                sm_str += "]";
                spdlog::info("{}", sm_str);

                spdlog::info("DEBUG CrossEntropy: First sample per-class loss:");
                std::string ls_str = "  [";
                for (int i = 0; i < std::min((int)pred.dims(1), 10); i++) {
                    ls_str += fmt::format("{:.4f}", ls_data[i]);
                    if (i < std::min((int)pred.dims(1), 10) - 1) ls_str += ", ";
                }
                ls_str += "]";
                spdlog::info("{}", ls_str);

                af::freeHost(sm_data);
                af::freeHost(tg_data);
                af::freeHost(ls_data);
            }

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
            // Targets are class indices - GPU-optimized one-hot encoding
            dim_t batch_size = pred.dims(0);
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
        af::array log_probs = TensorToAf(predictions);  // Expects log probabilities [batch, classes]
        af::array target = TensorToAf(targets);         // Class indices [batch]

        dim_t batch_size = log_probs.dims(0);
        dim_t num_classes = log_probs.dims(1);

        // GPU-optimized gather: compute linear indices and gather in one operation
        // Linear index = batch_idx * num_classes + class_idx
        af::array batch_indices = af::range(af::dim4(batch_size), 0, s32);
        af::array target_int = target.as(s32);

        // Compute linear indices for gathering from flattened log_probs
        af::array linear_indices = batch_indices * static_cast<int>(num_classes) + target_int;

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
    throw std::runtime_error("NLL forward requires ArrayFire");
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

// ============================================================================
// Focal Loss Implementation
// ============================================================================

Tensor FocalLoss::Forward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array pred = TensorToAf(predictions);
        af::array target = TensorToAf(targets);

        // Apply softmax to get probabilities
        af::array max_val = af::max(pred, 1);
        af::array pred_shifted = pred - af::tile(max_val, 1, pred.dims(1));
        af::array exp_pred = af::exp(pred_shifted);
        af::array sum_exp = af::sum(exp_pred, 1);
        af::array probs = exp_pred / af::tile(sum_exp, 1, pred.dims(1));
        cached_probs_ = AfToTensor(probs);

        // Get probability of true class
        dim_t batch_size = probs.dims(0);
        af::array target_indices = target.as(af::dtype::s32);
        af::array batch_indices = af::iota(af::dim4(batch_size), af::dim4(1), af::dtype::s32);
        af::array pt = probs(af::seq(batch_size), target_indices.T());
        pt = af::diag(pt);

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
    throw std::runtime_error("Focal forward requires ArrayFire");
}

Tensor FocalLoss::Backward(const Tensor& predictions, const Tensor& targets) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array probs = TensorToAf(cached_probs_);
        af::array target = TensorToAf(targets);

        dim_t batch_size = probs.dims(0);
        dim_t num_classes = probs.dims(1);

        // Get probability of true class
        af::array target_indices = target.as(af::dtype::s32);
        af::array pt = probs(af::seq(batch_size), target_indices.T());
        pt = af::diag(pt);

        // Create one-hot target
        af::array one_hot = af::constant(0.0f, batch_size, num_classes);
        for (int i = 0; i < batch_size; ++i) {
            int class_idx = target(i).scalar<int>();
            one_hot(i, class_idx) = 1.0f;
        }

        // Focal loss gradient is complex - simplified version:
        // d_loss/d_pred = alpha * (1-pt)^gamma * (gamma * pt * log(pt) / (1-pt) + 1) * (pt - 1_{y=c})
        af::dim4 tile_dims(1, static_cast<unsigned int>(num_classes));
        af::array focal_weight = af::pow(1.0f - pt, gamma_);
        af::array log_pt = af::log(af::max(pt, 1e-8f));
        af::array scale = alpha_ * focal_weight * (gamma_ * pt * log_pt / (1.0f - pt + 1e-8f) + 1.0f);

        af::array grad = af::tile(scale, tile_dims) * (probs - one_hot);

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(batch_size);
        }

        return AfToTensor(grad);
    } catch (const af::exception& e) {
        spdlog::warn("ArrayFire FocalLoss::Backward failed: {}", e.what());
    }
#endif
    throw std::runtime_error("Focal backward requires ArrayFire");
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
    throw std::runtime_error("Triplet forward requires ArrayFire");
}

Tensor TripletLoss::Backward(const Tensor& anchor, const Tensor& positive) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array a = TensorToAf(anchor);
        af::array p = TensorToAf(positive);
        af::array n = TensorToAf(negative_);
        af::array dist_ap = TensorToAf(cached_dist_ap_);
        af::array dist_an = TensorToAf(cached_dist_an_);

        dim_t batch_size = a.dims(0);
        dim_t embed_dim = a.dims(1);

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
    throw std::runtime_error("Triplet backward requires ArrayFire");
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
    throw std::runtime_error("Contrastive forward requires ArrayFire");
}

Tensor ContrastiveLoss::Backward(const Tensor& x1, const Tensor& x2) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array a1 = TensorToAf(x1);
        af::array a2 = TensorToAf(x2);
        af::array labels = TensorToAf(labels_);
        af::array distances = TensorToAf(cached_distances_);

        // Gradient w.r.t. x1
        // For similar: d_loss/dx1 = 2*(x1-x2) = 2*diff
        // For dissimilar: d_loss/dx1 = -2*(margin-d)/d * (x1-x2) if d < margin, else 0

        af::array diff = a1 - a2;
        dim_t batch_size = a1.dims(0);
        dim_t embed_dim = a1.dims(1);

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
    throw std::runtime_error("Contrastive backward requires ArrayFire");
}

} // namespace cyxwiz
