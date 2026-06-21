#include "cyxwiz/losses/metric_learning.h"
#include "loss_utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

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
