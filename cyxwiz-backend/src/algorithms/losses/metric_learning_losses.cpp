#include "cyxwiz/losses/metric_learning.h"
#include "loss_utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

// Undefine Windows macros that conflict with std::max/min and ArrayFire
// helpers. Must be AFTER all includes (Windows headers define these).
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz {

using namespace loss_detail;

namespace {

struct EmbeddingPairShape {
    size_t batch = 0;
    size_t dim = 0;
};

constexpr float kCosineEmbeddingEpsilon = 1.0e-12f;
constexpr float kTripletEuclideanEpsilon = 1.0e-6f;
constexpr float kTripletCosineEpsilon = 1.0e-8f;


void ValidateCosineMargin(float margin) {
    if (!std::isfinite(margin) || margin < -1.0f || margin > 1.0f) {
        throw std::invalid_argument("CosineEmbeddingLoss margin must be finite and in [-1, 1]");
    }
}

void ValidateTripletMargin(float margin) {
    if (!std::isfinite(margin) || margin <= 0.0f) {
        throw std::invalid_argument("TripletLoss margin must be finite and positive");
    }
}

void ValidateContrastiveMargin(float margin) {
    if (!std::isfinite(margin) || margin < 0.0f) {
        throw std::invalid_argument("ContrastiveLoss margin must be finite and non-negative");
    }
}

EmbeddingPairShape ValidateEmbeddingPair(const Tensor& x1, const Tensor& x2, const char* name) {
    if (x1.GetDataType() != DataType::Float32 || x2.GetDataType() != DataType::Float32) {
        throw std::runtime_error(std::string(name) + " only supports Float32 embeddings");
    }
    if (x1.Shape() != x2.Shape()) {
        throw std::runtime_error(std::string(name) + " requires matching embedding shapes");
    }
    const std::vector<size_t>& shape = x1.Shape();
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw std::runtime_error(std::string(name) +
                                 " requires non-empty [batch, embedding_dim] tensors");
    }
    return {shape[0], shape[1]};
}

void ValidateEmbeddingLabelShape(const Tensor& labels, size_t batch, const char* name) {
    if (labels.GetDataType() != DataType::Float32) {
        throw std::runtime_error(std::string(name) + " labels must be Float32");
    }
    const auto& shape = labels.Shape();
    const bool is_batch_vector =
        shape == std::vector<size_t>{batch} || shape == std::vector<size_t>{batch, 1};
    if (!is_batch_vector) {
        throw std::runtime_error(std::string(name) +
                                 " labels must have shape [batch] or [batch, 1]");
    }
}

const float* ValidateCosineEmbeddingLabelValues(const Tensor& labels, size_t batch) {
    const float* values = labels.ReadData<float>();
    for (size_t row = 0; row < batch; ++row) {
        if (values[row] != 1.0f && values[row] != -1.0f) {
            throw std::runtime_error("CosineEmbeddingLoss labels must be exactly +1 or -1");
        }
    }
    return values;
}

const float* ValidateContrastiveLabelValues(const Tensor& labels, size_t batch) {
    const float* values = labels.ReadData<float>();
    for (size_t row = 0; row < batch; ++row) {
        if (values[row] != 0.0f && values[row] != 1.0f) {
            throw std::runtime_error("ContrastiveLoss labels must be exactly 0 or 1");
        }
    }
    return values;
}

EmbeddingPairShape ValidateCosineEmbeddingInputs(const Tensor& x1, const Tensor& x2,
                                                 const Tensor& labels) {
    const EmbeddingPairShape shape = ValidateEmbeddingPair(x1, x2, "CosineEmbeddingLoss");
    ValidateEmbeddingLabelShape(labels, shape.batch, "CosineEmbeddingLoss");
    return shape;
}

EmbeddingPairShape ValidateContrastiveInputs(const Tensor& x1, const Tensor& x2,
                                             const Tensor& labels) {
    const EmbeddingPairShape shape = ValidateEmbeddingPair(x1, x2, "ContrastiveLoss");
    ValidateEmbeddingLabelShape(labels, shape.batch, "ContrastiveLoss");
    return shape;
}

EmbeddingPairShape ValidateTripletInputs(const Tensor& anchor, const Tensor& positive,
                                         const Tensor& negative) {
    const EmbeddingPairShape shape = ValidateEmbeddingPair(anchor, positive, "TripletLoss");
    ValidateEmbeddingPair(anchor, negative, "TripletLoss");
    return shape;
}







} // namespace

// ============================================================================
// Cosine Embedding Loss Implementation
// ============================================================================

CosineEmbeddingLoss::CosineEmbeddingLoss(float margin, Reduction reduction)
    : Loss(reduction), margin_(margin) {
    ValidateCosineMargin(margin_);
}

void CosineEmbeddingLoss::SetMargin(float margin) {
    ValidateCosineMargin(margin);
    margin_ = margin;
}

TripletLoss::TripletLoss(float margin, DistanceType distance_type, Reduction reduction)
    : Loss(reduction), margin_(margin), distance_type_(distance_type) {
    ValidateTripletMargin(margin_);
}

void TripletLoss::SetMargin(float margin) {
    ValidateTripletMargin(margin);
    margin_ = margin;
}

ContrastiveLoss::ContrastiveLoss(float margin, Reduction reduction)
    : Loss(reduction), margin_(margin) {
    ValidateContrastiveMargin(margin_);
}

void ContrastiveLoss::SetMargin(float margin) {
    ValidateContrastiveMargin(margin);
    margin_ = margin;
}

Tensor CosineEmbeddingLoss::Forward(const Tensor& x1, const Tensor& x2) {
    constexpr const char* kOperation = "CosineEmbeddingLoss::Forward";
    const EmbeddingPairShape shape = ValidateCosineEmbeddingInputs(x1, x2, labels_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    loss_detail::ValidateFloat32Pair(x1, x2, kOperation);
    try {
        {
            const ScopedArrayFireHostSyncAttribution validation(
                ArrayFireHostSyncCategory::LossInputValidation, kOperation);
            ValidateCosineEmbeddingLabelValues(labels_, shape.batch);
        }
        af::array a1 = TensorToAf(x1);
        af::array a2 = TensorToAf(x2);
        af::array labels = af::flat(TensorToAf(labels_));

        // Compute cosine similarity
        // cos(x1, x2) = (x1 . x2) / (||x1|| * ||x2||)
        af::array dot_product = af::sum(a1 * a2, 1);
        af::array norm1 = af::sqrt(af::sum(a1 * a1, 1) + kCosineEmbeddingEpsilon);
        af::array norm2 = af::sqrt(af::sum(a2 * a2, 1) + kCosineEmbeddingEpsilon);
        af::array cos_sim = dot_product / (norm1 * norm2);

        // Loss:
        // For similar pairs (y = 1): 1 - cos_sim
        // For dissimilar pairs (y = -1): max(0, cos_sim - margin)
        af::array loss_similar = 1.0f - cos_sim;
        af::array loss_dissimilar = af::max(cos_sim - margin_, 0.0f);

        af::array loss = af::select(labels == 1.0f, loss_similar, loss_dissimilar);
        loss = ApplyReduction(loss, reduction_);

        return AfToTensor(loss);
    } catch (const af::exception& e) {
        loss_detail::ThrowLossDeviceError(kOperation, e);
    }
#else
    loss_detail::ThrowLossNeedsArrayFire(kOperation);
#endif
}

Tensor CosineEmbeddingLoss::Backward(const Tensor& x1, const Tensor& x2) {
    constexpr const char* kOperation = "CosineEmbeddingLoss::Backward";
    const EmbeddingPairShape shape = ValidateCosineEmbeddingInputs(x1, x2, labels_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    loss_detail::ValidateFloat32Pair(x1, x2, kOperation);
    try {
        {
            const ScopedArrayFireHostSyncAttribution validation(
                ArrayFireHostSyncCategory::LossInputValidation, kOperation);
            ValidateCosineEmbeddingLabelValues(labels_, shape.batch);
        }
        af::array a1 = TensorToAf(x1);
        af::array a2 = TensorToAf(x2);
        af::array labels = af::flat(TensorToAf(labels_));

        // Compute cosine similarity components
        af::array dot_product = af::sum(a1 * a2, 1);
        af::array norm1_sq = af::sum(a1 * a1, 1);
        af::array norm2_sq = af::sum(a2 * a2, 1);
        af::array safe_norm1_sq = norm1_sq + kCosineEmbeddingEpsilon;
            af::array safe_norm2_sq = norm2_sq + kCosineEmbeddingEpsilon;
            af::array norm1 = af::sqrt(safe_norm1_sq);
        af::array norm2 = af::sqrt(safe_norm2_sq);
        af::array norm_product = norm1 * norm2;
        af::array cos_sim = dot_product / norm_product;

        // Gradient of cosine similarity w.r.t x1
        // d(cos_sim)/dx1 = x2/(||x1||*||x2||) - cos_sim * x1/||x1||^2
            af::dim4 tile_dims(1, static_cast<unsigned int>(a1.dims(1)));

        af::array grad_cos = a2 / af::tile(norm_product, tile_dims) -
                             a1 * af::tile(cos_sim / safe_norm1_sq, tile_dims);
        grad_cos.eval();

        // For similar pairs: d_loss = -d_cos_sim
        // For dissimilar pairs: d_loss = d_cos_sim (if cos_sim > margin)
        // Use mask-based approach instead of nested af::select with scalars
        af::array mask_similar = (labels == 1.0f).as(af::dtype::f32);
        af::array mask_dissimilar = (1.0f - mask_similar);
        af::array mask_above_margin = (cos_sim > margin_).as(af::dtype::f32);
        af::array scale = mask_similar * (-1.0f) + mask_dissimilar * mask_above_margin;
        scale.eval();

        af::array grad = grad_cos * af::tile(scale, tile_dims);
        grad.eval();

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(shape.batch);
            grad.eval();
        }

        return AfToTensor(grad, x1.Shape());
    } catch (const af::exception& e) {
        loss_detail::ThrowLossDeviceError(kOperation, e);
    }
#else
    loss_detail::ThrowLossNeedsArrayFire(kOperation);
#endif
}

// ============================================================================
// Triplet Loss Implementation
// ============================================================================

Tensor TripletLoss::Forward(const Tensor& anchor, const Tensor& positive) {
    constexpr const char* kOperation = "TripletLoss::Forward";
    ValidateTripletInputs(anchor, positive, negative_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    loss_detail::ValidateFloat32Pair(anchor, positive, kOperation);
    try {
        af::array a = TensorToAf(anchor);
        af::array p = TensorToAf(positive);
        af::array n = TensorToAf(negative_);

        af::array dist_ap, dist_an;

        if (distance_type_ == DistanceType::Euclidean) {
                // Match torch.nn.TripletMarginLoss defaults: p=2, eps=1e-6.
                af::array diff_ap = a - p + kTripletEuclideanEpsilon;
            af::array diff_an = a - n + kTripletEuclideanEpsilon;
            dist_ap = af::sqrt(af::sum(diff_ap * diff_ap, 1));
            dist_an = af::sqrt(af::sum(diff_an * diff_an, 1));
        } else {
                // Explicit PyTorch-autograd reference equation with smooth norms.
                af::array norm_a = af::sqrt(af::sum(a * a, 1) + kTripletCosineEpsilon);
            af::array norm_p = af::sqrt(af::sum(p * p, 1) + kTripletCosineEpsilon);
            af::array norm_n = af::sqrt(af::sum(n * n, 1) + kTripletCosineEpsilon);
            af::array cos_ap = af::sum(a * p, 1) / (norm_a * norm_p);
            af::array cos_an = af::sum(a * n, 1) / (norm_a * norm_n);
            dist_ap = 1.0f - cos_ap;
            dist_an = 1.0f - cos_an;
        }

            // Triplet loss: max(d_ap - d_an + margin, 0)
        af::array loss = af::max(dist_ap - dist_an + margin_, 0.0f);
        loss = ApplyReduction(loss, reduction_);

        return AfToTensor(loss);
    } catch (const af::exception& e) {
        loss_detail::ThrowLossDeviceError(kOperation, e);
    }
#else
    loss_detail::ThrowLossNeedsArrayFire(kOperation);
#endif
}

Tensor TripletLoss::Backward(const Tensor& anchor, const Tensor& positive) {
    return BackwardAll(anchor, positive).anchor;
}

TripletLossGradients TripletLoss::BackwardAll(const Tensor& anchor, const Tensor& positive) {
    constexpr const char* kOperation = "TripletLoss::Backward";
    const EmbeddingPairShape shape = ValidateTripletInputs(anchor, positive, negative_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    loss_detail::ValidateFloat32Pair(anchor, positive, kOperation);
    try {
        af::array a = TensorToAf(anchor);
        af::array p = TensorToAf(positive);
        af::array n = TensorToAf(negative_);

        dim_t embed_dim = a.dims(1);
            af::array dist_ap, dist_an;
            af::array grad_anchor;
            af::array grad_positive;
            af::array grad_negative;
            const af::dim4 tile_dims(1, static_cast<unsigned int>(embed_dim));

            if (distance_type_ == DistanceType::Euclidean) {
            af::array diff_ap = a - p + kTripletEuclideanEpsilon;
            af::array diff_an = a - n + kTripletEuclideanEpsilon;
            dist_ap = af::sqrt(af::sum(diff_ap * diff_ap, 1));
            dist_an = af::sqrt(af::sum(diff_an * diff_an, 1));
                af::array active = (dist_ap - dist_an + margin_ > 0.0f).as(af::dtype::f32);
                af::array active_tiled = af::tile(active, tile_dims);
                af::array unit_ap = diff_ap / af::tile(dist_ap, tile_dims);
                af::array unit_an = diff_an / af::tile(dist_an, tile_dims);
                grad_anchor = (unit_ap - unit_an) * active_tiled;
                grad_positive = -unit_ap * active_tiled;
                grad_negative = unit_an * active_tiled;
        } else {
                af::array safe_a_sq = af::sum(a * a, 1) + kTripletCosineEpsilon;
                af::array safe_p_sq = af::sum(p * p, 1) + kTripletCosineEpsilon;
                af::array safe_n_sq = af::sum(n * n, 1) + kTripletCosineEpsilon;
                af::array norm_a = af::sqrt(safe_a_sq);
            af::array norm_p = af::sqrt(safe_p_sq);
            af::array norm_n = af::sqrt(safe_n_sq);
                af::array norm_ap = norm_a * norm_p;
                af::array norm_an = norm_a * norm_n;
            af::array cos_ap = af::sum(a * p, 1) / norm_ap;
            af::array cos_an = af::sum(a * n, 1) / norm_an;
            dist_ap = 1.0f - cos_ap;
            dist_an = 1.0f - cos_an;
                af::array active = (dist_ap - dist_an + margin_ > 0.0f).as(af::dtype::f32);
        af::array active_tiled = af::tile(active, tile_dims);

        af::array grad_cos_ap_anchor =
                    p / af::tile(norm_ap, tile_dims) - a * af::tile(cos_ap / safe_a_sq, tile_dims);
            af::array grad_cos_an_anchor =
                    n / af::tile(norm_an, tile_dims) - a * af::tile(cos_an / safe_a_sq, tile_dims);
                af::array grad_cos_ap_positive =
                    a / af::tile(norm_ap, tile_dims) - p * af::tile(cos_ap / safe_p_sq, tile_dims);
                af::array grad_cos_an_negative =
                    a / af::tile(norm_an, tile_dims) - n * af::tile(cos_an / safe_n_sq, tile_dims);
                grad_anchor = (grad_cos_an_anchor - grad_cos_ap_anchor) * active_tiled;
                grad_positive = -grad_cos_ap_positive * active_tiled;
                grad_negative = grad_cos_an_negative * active_tiled;
        }

        if (reduction_ == Reduction::Mean) {
                const float divisor = static_cast<float>(shape.batch);
                grad_anchor = grad_anchor / divisor;
                grad_positive = grad_positive / divisor;
                grad_negative = grad_negative / divisor;
            }
            grad_anchor.eval();
            grad_positive.eval();
            grad_negative.eval();

            return TripletLossGradients{AfToTensor(grad_anchor, anchor.Shape()),
                                        AfToTensor(grad_positive, positive.Shape()),
                                        AfToTensor(grad_negative, negative_.Shape())};
    } catch (const af::exception& e) {
        loss_detail::ThrowLossDeviceError(kOperation, e);
    }
#else
    loss_detail::ThrowLossNeedsArrayFire(kOperation);
#endif
}

// ============================================================================
// Contrastive Loss Implementation
// ============================================================================

Tensor ContrastiveLoss::Forward(const Tensor& x1, const Tensor& x2) {
    constexpr const char* kOperation = "ContrastiveLoss::Forward";
    const EmbeddingPairShape shape = ValidateContrastiveInputs(x1, x2, labels_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    loss_detail::ValidateFloat32Pair(x1, x2, kOperation);
    try {
        {
            const ScopedArrayFireHostSyncAttribution validation(
                ArrayFireHostSyncCategory::LossInputValidation, kOperation);
            ValidateContrastiveLabelValues(labels_, shape.batch);
        }
        af::array a1 = TensorToAf(x1);
        af::array a2 = TensorToAf(x2);
        af::array labels = af::flat(TensorToAf(labels_));

        // Compute pairwise Euclidean distance
        af::array diff = a1 - a2;
        af::array distances = af::sqrt(af::sum(diff * diff, 1));

        // Contrastive loss: y*d^2 + (1-y)*max(0, margin-d)^2
        // where y=0 for similar, y=1 for dissimilar
        af::array similar_loss = (1.0f - labels) * distances * distances;
        af::array margin_diff = af::max(0.0f, margin_ - distances);
        af::array dissimilar_loss = labels * margin_diff * margin_diff;

        af::array loss = similar_loss + dissimilar_loss;
        loss = ApplyReduction(loss, reduction_);

        return AfToTensor(loss);
    } catch (const af::exception& e) {
        loss_detail::ThrowLossDeviceError(kOperation, e);
    }
#else
    loss_detail::ThrowLossNeedsArrayFire(kOperation);
#endif
}

Tensor ContrastiveLoss::Backward(const Tensor& x1, const Tensor& x2) {
    constexpr const char* kOperation = "ContrastiveLoss::Backward";
    const EmbeddingPairShape shape = ValidateContrastiveInputs(x1, x2, labels_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    loss_detail::ValidateFloat32Pair(x1, x2, kOperation);
    try {
        {
            const ScopedArrayFireHostSyncAttribution validation(
                ArrayFireHostSyncCategory::LossInputValidation, kOperation);
            ValidateContrastiveLabelValues(labels_, shape.batch);
        }
        af::array a1 = TensorToAf(x1);
        af::array a2 = TensorToAf(x2);
        af::array labels = af::flat(TensorToAf(labels_));

        // Gradient w.r.t. x1
        // For similar: d_loss/dx1 = 2*(x1-x2) = 2*diff
            // For dissimilar: d_loss/dx1 = -2*(margin-d)/d * (x1-x2) if d < margin,
            // else 0

            af::array diff = a1 - a2;
        dim_t embed_dim = a1.dims(1);
            af::array distances = af::sqrt(af::sum(diff * diff, 1));

            // Avoid division by zero
        af::array safe_distances = af::max(distances, 1e-8f);
        af::dim4 tile_dims(1, static_cast<unsigned int>(embed_dim));

        // Similar pairs gradient: 2 * diff
        af::array grad_similar = 2.0f * diff;

            // Dissimilar pairs gradient: -2 * (margin - d) / d * diff (when d <
            // margin)
            af::array margin_diff = margin_ - safe_distances;
        margin_diff.eval();
        af::array mask_in_margin = (distances < margin_).as(af::dtype::f32);
        mask_in_margin.eval();
        af::array scale = -2.0f * margin_diff / safe_distances * mask_in_margin;
        scale.eval();
        af::array grad_dissimilar = diff * af::tile(scale, tile_dims);
        grad_dissimilar.eval();

        // Combine based on labels (0=similar, 1=dissimilar)
        af::array labels_tiled = af::tile(labels, tile_dims);
        labels_tiled.eval();
        af::array grad = (1.0f - labels_tiled) * grad_similar + labels_tiled * grad_dissimilar;
        grad.eval();

        if (reduction_ == Reduction::Mean) {
            grad = grad / static_cast<float>(shape.batch);
            grad.eval();
        }

        return AfToTensor(grad, x1.Shape());
    } catch (const af::exception& e) {
        loss_detail::ThrowLossDeviceError(kOperation, e);
    }
#else
    loss_detail::ThrowLossNeedsArrayFire(kOperation);
#endif
}

} // namespace cyxwiz
