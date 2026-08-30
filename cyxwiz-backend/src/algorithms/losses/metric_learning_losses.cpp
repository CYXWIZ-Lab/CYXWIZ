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

template <typename CpuFunction>
auto RunNativeCpuMetricLoss(const char* operation_name, CpuFunction&& compute)
    -> decltype(compute()) {
    const ScopedArrayFireHostSyncAttribution attribution(ArrayFireHostSyncCategory::LossCpuPath,
                                                         operation_name);
    return compute();
}

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

const float* ValidateEmbeddingLabels(const Tensor& labels, size_t batch, const char* name) {
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
    return labels.Data<float>();
}

EmbeddingPairShape ValidateCosineEmbeddingInputs(const Tensor& x1, const Tensor& x2,
                                                 const Tensor& labels) {
    const EmbeddingPairShape shape = ValidateEmbeddingPair(x1, x2, "CosineEmbeddingLoss");
    const float* values = ValidateEmbeddingLabels(labels, shape.batch, "CosineEmbeddingLoss");
    for (size_t row = 0; row < shape.batch; ++row) {
        if (values[row] != 1.0f && values[row] != -1.0f) {
            throw std::runtime_error("CosineEmbeddingLoss labels must be exactly +1 or -1");
        }
    }
    return shape;
}

EmbeddingPairShape ValidateContrastiveInputs(const Tensor& x1, const Tensor& x2,
                                             const Tensor& labels) {
    const EmbeddingPairShape shape = ValidateEmbeddingPair(x1, x2, "ContrastiveLoss");
    const float* values = ValidateEmbeddingLabels(labels, shape.batch, "ContrastiveLoss");
    for (size_t row = 0; row < shape.batch; ++row) {
        if (values[row] != 0.0f && values[row] != 1.0f) {
            throw std::runtime_error("ContrastiveLoss labels must be exactly 0 or 1");
        }
    }
    return shape;
}

EmbeddingPairShape ValidateTripletInputs(const Tensor& anchor, const Tensor& positive,
                                         const Tensor& negative) {
    const EmbeddingPairShape shape = ValidateEmbeddingPair(anchor, positive, "TripletLoss");
    ValidateEmbeddingPair(anchor, negative, "TripletLoss");
    return shape;
}

Tensor CpuCosineEmbeddingForward(const Tensor& x1,
                                 const Tensor& x2,
                                 const Tensor& labels,
                                 float margin,
                                 Reduction reduction) {
    const EmbeddingPairShape shape = ValidateCosineEmbeddingInputs(x1, x2, labels);
    const float* label_data = labels.Data<float>();
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
        const float norm_product = std::sqrt(norm1_sq + kCosineEmbeddingEpsilon) * std::sqrt(norm2_sq + kCosineEmbeddingEpsilon);
        const float cos_sim = dot / norm_product;
        losses[batch] = label_data[batch] == 1.0f ? 1.0f - cos_sim
                                                 : std::max(cos_sim - margin, 0.0f);
    }
    return ApplyClassReduction(losses, shape.batch, reduction);
}

Tensor CpuCosineEmbeddingBackward(const Tensor& x1,
                                  const Tensor& x2,
                                  const Tensor& labels,
                                  float margin,
                                  Reduction reduction) {
    const EmbeddingPairShape shape = ValidateCosineEmbeddingInputs(x1, x2, labels);
    const float* label_data = labels.Data<float>();
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

        const float safe_norm1_sq = norm1_sq + kCosineEmbeddingEpsilon;
        const float safe_norm2_sq = norm2_sq + kCosineEmbeddingEpsilon;
        const float norm1 = std::sqrt(safe_norm1_sq);
        const float norm2 = std::sqrt(safe_norm2_sq);
        const float norm_product = norm1 * norm2;
        const float cos_sim = dot / norm_product;
        const float scale = label_data[batch] == 1.0f ? -1.0f
                          : (cos_sim > margin ? 1.0f : 0.0f);
        const float reduction_scale = reduction == Reduction::Mean && shape.batch > 0
                                          ? 1.0f / static_cast<float>(shape.batch)
                                          : 1.0f;

        for (size_t d = 0; d < shape.dim; ++d) {
            const float grad_cos = b[base + d] / norm_product -
                                   cos_sim * a[base + d] / safe_norm1_sq;
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
                         Reduction reduction) {
    const EmbeddingPairShape shape = ValidateTripletInputs(anchor, positive, negative);

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
                const float diff_ap = a[base + d] - p[base + d] + kTripletEuclideanEpsilon;
                const float diff_an = a[base + d] - n[base + d] + kTripletEuclideanEpsilon;
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
            const float norm_a = std::sqrt(norm_a_sq + kTripletCosineEpsilon);
            dist_ap[batch] = 1.0f - dot_ap / (norm_a * std::sqrt(norm_p_sq + kTripletCosineEpsilon));
            dist_an[batch] = 1.0f - dot_an / (norm_a * std::sqrt(norm_n_sq + kTripletCosineEpsilon));
        }
        losses[batch] = std::max(dist_ap[batch] - dist_an[batch] + margin, 0.0f);
    }
    return ApplyClassReduction(losses, shape.batch, reduction);
}

TripletLossGradients CpuTripletBackwardAll(const Tensor& anchor,
                          const Tensor& positive,
                          const Tensor& negative,
                          TripletLoss::DistanceType distance_type,
                          float margin,
                          Reduction reduction) {
    const EmbeddingPairShape shape = ValidateTripletInputs(anchor, positive, negative);

    TripletLossGradients gradients{Tensor(anchor.Shape(), DataType::Float32),
                                   Tensor(anchor.Shape(), DataType::Float32),
                                   Tensor(anchor.Shape(), DataType::Float32)};
    float* grad_anchor = gradients.anchor.Data<float>();
    float* grad_positive = gradients.positive.Data<float>();
    float* grad_negative = gradients.negative.Data<float>();
    const float* a = anchor.Data<float>();
    const float* p = positive.Data<float>();
    const float* n = negative.Data<float>();
    const float reduction_scale = reduction == Reduction::Mean && shape.batch > 0
                                      ? 1.0f / static_cast<float>(shape.batch)
                                      : 1.0f;

    for (size_t batch = 0; batch < shape.batch; ++batch) {
        const size_t base = batch * shape.dim;
        if (distance_type == TripletLoss::DistanceType::Euclidean) {
            float dist_ap_sq = 0.0f;
            float dist_an_sq = 0.0f;
            for (size_t d = 0; d < shape.dim; ++d) {
                const float diff_ap = a[base + d] - p[base + d] + kTripletEuclideanEpsilon;
                const float diff_an = a[base + d] - n[base + d] + kTripletEuclideanEpsilon;
                dist_ap_sq += diff_ap * diff_ap;
                dist_an_sq += diff_an * diff_an;
            }
            const float dist_ap = std::sqrt(dist_ap_sq);
            const float dist_an = std::sqrt(dist_an_sq);
            if (dist_ap - dist_an + margin <= 0.0f) {
                continue;
            }
            for (size_t d = 0; d < shape.dim; ++d) {
                const float diff_ap = a[base + d] - p[base + d] + kTripletEuclideanEpsilon;
                const float diff_an = a[base + d] - n[base + d] + kTripletEuclideanEpsilon;
                const float grad_ap = diff_ap / dist_ap;
                const float grad_an = diff_an / dist_an;
                grad_anchor[base + d] = (grad_ap - grad_an) * reduction_scale;
                grad_positive[base + d] = -grad_ap * reduction_scale;
                grad_negative[base + d] = grad_an * reduction_scale;
            }
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
            const float safe_a_sq = norm_a_sq + kTripletCosineEpsilon;
            const float safe_p_sq = norm_p_sq + kTripletCosineEpsilon;
            const float safe_n_sq = norm_n_sq + kTripletCosineEpsilon;
            const float norm_a = std::sqrt(safe_a_sq);
            const float norm_p = std::sqrt(safe_p_sq);
            const float norm_n = std::sqrt(safe_n_sq);
            const float norm_ap = norm_a * norm_p;
            const float norm_an = norm_a * norm_n;
            const float cos_ap = dot_ap / norm_ap;
            const float cos_an = dot_an / norm_an;
            const float dist_ap = 1.0f - cos_ap;
            const float dist_an = 1.0f - cos_an;
            if (dist_ap - dist_an + margin <= 0.0f) {
                continue;
            }
            for (size_t d = 0; d < shape.dim; ++d) {
                const float grad_cos_ap_a =
                    p[base + d] / norm_ap - cos_ap * a[base + d] / safe_a_sq;
                const float grad_cos_an_a =
                    n[base + d] / norm_an - cos_an * a[base + d] / safe_a_sq;
                const float grad_cos_ap_p =
                    a[base + d] / norm_ap - cos_ap * p[base + d] / safe_p_sq;
                const float grad_cos_an_n =
                    a[base + d] / norm_an - cos_an * n[base + d] / safe_n_sq;
                grad_anchor[base + d] = (grad_cos_an_a - grad_cos_ap_a) * reduction_scale;
                grad_positive[base + d] = -grad_cos_ap_p * reduction_scale;
                grad_negative[base + d] = grad_cos_an_n * reduction_scale;
            }
        }
    }

    return gradients;
}

Tensor CpuContrastiveForward(const Tensor& x1,
                             const Tensor& x2,
                             const Tensor& labels,
                             float margin,
                             Reduction reduction) {
    const EmbeddingPairShape shape = ValidateContrastiveInputs(x1, x2, labels);
    const float* label_data = labels.Data<float>();
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
        losses[batch] = label_data[batch] == 1.0f ? margin_diff * margin_diff
                            : distance_sq;
    }
    return ApplyClassReduction(losses, shape.batch, reduction);
}

Tensor CpuContrastiveBackward(const Tensor& x1,
                              const Tensor& x2,
                              const Tensor& labels,
                              float margin,
                              Reduction reduction) {
    const EmbeddingPairShape shape = ValidateContrastiveInputs(x1, x2, labels);
    const float* label_data = labels.Data<float>();

    Tensor grad(x1.Shape(), DataType::Float32);
    float* out = grad.Data<float>();
    const float* a = x1.Data<float>();
    const float* b = x2.Data<float>();
    const float reduction_scale = reduction == Reduction::Mean && shape.batch > 0
                                      ? 1.0f / static_cast<float>(shape.batch)
                                      : 1.0f;

    for (size_t batch = 0; batch < shape.batch; ++batch) {
        const size_t base = batch * shape.dim;
        float distance_sq = 0.0f;
        for (size_t d = 0; d < shape.dim; ++d) {
            const float diff = a[base + d] - b[base + d];
            distance_sq += diff * diff;
        }
        const float distance = std::sqrt(distance_sq);
        const bool dissimilar = label_data[batch] == 1.0f;
        const bool active_dissimilar = dissimilar && distance < margin;
        const float safe_distance = std::max(distance, 1e-8f);
        const float dissimilar_scale = active_dissimilar
                                           ? -2.0f * (margin - distance) / safe_distance
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
    ValidateCosineEmbeddingInputs(x1, x2, labels_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const bool use_native_cpu = PrepareLossNativeCpuFallback(kOperation, x1, x2, reduction_);
    if (!use_native_cpu)
        try {
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
        LogArrayFireLossFallbackOnce(kOperation, e.what(), x1, x2, reduction_);
    }
#endif
    return RunNativeCpuMetricLoss(kOperation, [&] {
        return CpuCosineEmbeddingForward(x1, x2, labels_, margin_, reduction_);
    });
}

Tensor CosineEmbeddingLoss::Backward(const Tensor& x1, const Tensor& x2) {
    constexpr const char* kOperation = "CosineEmbeddingLoss::Backward";
    const EmbeddingPairShape shape = ValidateCosineEmbeddingInputs(x1, x2, labels_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const bool use_native_cpu = PrepareLossNativeCpuFallback(kOperation, x1, x2, reduction_);
    if (!use_native_cpu)
        try {
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
        LogArrayFireLossFallbackOnce(kOperation, e.what(), x1, x2, reduction_);
    }
#endif
    return RunNativeCpuMetricLoss(kOperation, [&] {
        return CpuCosineEmbeddingBackward(x1, x2, labels_, margin_, reduction_);
    });
}

// ============================================================================
// Triplet Loss Implementation
// ============================================================================

Tensor TripletLoss::Forward(const Tensor& anchor, const Tensor& positive) {
    constexpr const char* kOperation = "TripletLoss::Forward";
    ValidateTripletInputs(anchor, positive, negative_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const bool use_native_cpu =
        PrepareLossNativeCpuFallback(kOperation, anchor, positive, reduction_);
    if (!use_native_cpu)
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
        LogArrayFireLossFallbackOnce(kOperation, e.what(), anchor, positive, reduction_);
    }
#endif
    return RunNativeCpuMetricLoss(kOperation, [&] {
        return CpuTripletForward(anchor, positive, negative_, distance_type_, margin_, reduction_);
    });
}

Tensor TripletLoss::Backward(const Tensor& anchor, const Tensor& positive) {
    return BackwardAll(anchor, positive).anchor;
}

TripletLossGradients TripletLoss::BackwardAll(const Tensor& anchor, const Tensor& positive) {
    constexpr const char* kOperation = "TripletLoss::Backward";
    const EmbeddingPairShape shape = ValidateTripletInputs(anchor, positive, negative_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const bool use_native_cpu =
        PrepareLossNativeCpuFallback(kOperation, anchor, positive, reduction_);
    if (!use_native_cpu)
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
        LogArrayFireLossFallbackOnce(kOperation, e.what(), anchor, positive, reduction_);
    }
#endif
    return RunNativeCpuMetricLoss(kOperation, [&] {
        return CpuTripletBackwardAll(anchor, positive, negative_, distance_type_, margin_, reduction_);
    });
}

// ============================================================================
// Contrastive Loss Implementation
// ============================================================================

Tensor ContrastiveLoss::Forward(const Tensor& x1, const Tensor& x2) {
    constexpr const char* kOperation = "ContrastiveLoss::Forward";
    ValidateContrastiveInputs(x1, x2, labels_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const bool use_native_cpu = PrepareLossNativeCpuFallback(kOperation, x1, x2, reduction_);
    if (!use_native_cpu)
        try {
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
        LogArrayFireLossFallbackOnce(kOperation, e.what(), x1, x2, reduction_);
    }
#endif
    return RunNativeCpuMetricLoss(
        kOperation, [&] { return CpuContrastiveForward(x1, x2, labels_, margin_, reduction_); });
}

Tensor ContrastiveLoss::Backward(const Tensor& x1, const Tensor& x2) {
    constexpr const char* kOperation = "ContrastiveLoss::Backward";
    const EmbeddingPairShape shape = ValidateContrastiveInputs(x1, x2, labels_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const bool use_native_cpu = PrepareLossNativeCpuFallback(kOperation, x1, x2, reduction_);
    if (!use_native_cpu)
        try {
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
        LogArrayFireLossFallbackOnce(kOperation, e.what(), x1, x2, reduction_);
    }
#endif
    return RunNativeCpuMetricLoss(
        kOperation, [&] { return CpuContrastiveBackward(x1, x2, labels_, margin_, reduction_);
});
}

} // namespace cyxwiz
