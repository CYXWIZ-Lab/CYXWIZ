#pragma once

#include "metric_learning_batch.h"

#include <cyxwiz/losses/metric_learning.h>
#include <cyxwiz/tensor.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace cyxwiz {

enum class MetricLearningPairLossKind {
    Contrastive,
    CosineEmbedding,
};

struct PairMetricLossResult {
    Tensor loss;
    Tensor grad_a;
    Tensor grad_b;
};

struct TripletMetricLossResult {
    Tensor loss;
    Tensor grad_anchor;
    Tensor grad_positive;
    Tensor grad_negative;
};

inline void ValidatePairLossInputs(const Tensor& embedding_a,
                                   const Tensor& embedding_b,
                                   const Tensor& labels,
                                   MetricLearningLabelConvention convention) {
    if (embedding_a.Shape() != embedding_b.Shape()) {
        throw std::invalid_argument(
            "metric-learning pair loss embeddings must have identical shapes");
    }
    const auto& shape = embedding_a.Shape();
    const size_t pair_count = shape.empty() ? 0 : shape[0];
    if (pair_count == 0 || shape.size() < 2) {
        throw std::invalid_argument(
            "metric-learning pair loss embeddings must be [batch, ...features]");
    }
    if (!TensorIsBatchVector(labels, pair_count)) {
        throw std::invalid_argument(
            "metric-learning pair loss labels must be [batch] or [batch, 1]");
    }
    const float* label_data = labels.Data<float>();
    for (size_t row = 0; row < pair_count; ++row) {
        if (!IsValidMetricLearningLabel(
                convention, static_cast<double>(label_data[row]))) {
            throw std::invalid_argument(
                "metric-learning pair loss label does not match convention");
        }
    }
}

inline PairMetricLossResult ComputePairMetricLoss(
    const Tensor& embedding_a,
    const Tensor& embedding_b,
    const Tensor& labels,
    MetricLearningPairLossKind kind,
    float margin,
    Reduction reduction = Reduction::Mean) {
    const auto convention =
        kind == MetricLearningPairLossKind::Contrastive
            ? MetricLearningLabelConvention::
                  ContrastiveZeroSimilarOneDissimilar
            : MetricLearningLabelConvention::
                  CosineOneSimilarNegativeOneDissimilar;
    ValidatePairLossInputs(embedding_a, embedding_b, labels, convention);

    PairMetricLossResult result;
    if (kind == MetricLearningPairLossKind::Contrastive) {
        ContrastiveLoss forward_loss(margin, reduction);
        forward_loss.SetLabels(labels);
        result.loss = forward_loss.Forward(embedding_a, embedding_b);
        result.grad_a = forward_loss.Backward(embedding_a, embedding_b);

        ContrastiveLoss swapped_loss(margin, reduction);
        swapped_loss.SetLabels(labels);
        (void)swapped_loss.Forward(embedding_b, embedding_a);
        result.grad_b = swapped_loss.Backward(embedding_b, embedding_a);
    } else {
        CosineEmbeddingLoss forward_loss(margin, reduction);
        forward_loss.SetLabels(labels);
        result.loss = forward_loss.Forward(embedding_a, embedding_b);
        result.grad_a = forward_loss.Backward(embedding_a, embedding_b);

        CosineEmbeddingLoss swapped_loss(margin, reduction);
        swapped_loss.SetLabels(labels);
        (void)swapped_loss.Forward(embedding_b, embedding_a);
        result.grad_b = swapped_loss.Backward(embedding_b, embedding_a);
    }

    return result;
}

inline void ValidateTripletLossInputs(const Tensor& anchor,
                                      const Tensor& positive,
                                      const Tensor& negative) {
    if (anchor.Shape() != positive.Shape() ||
        anchor.Shape() != negative.Shape()) {
        throw std::invalid_argument(
            "metric-learning triplet loss embeddings must have identical shapes");
    }
    const auto& shape = anchor.Shape();
    if (shape.empty() || shape[0] == 0 || shape.size() < 2) {
        throw std::invalid_argument(
            "metric-learning triplet loss embeddings must be [batch, ...features]");
    }
}

inline size_t TripletEmbeddingDim(const Tensor& embedding) {
    return embedding.NumElements() / embedding.Shape()[0];
}

inline TripletMetricLossResult ComputeEuclideanTripletMetricLoss(
    const Tensor& anchor,
    const Tensor& positive,
    const Tensor& negative,
    float margin,
    Reduction reduction = Reduction::Mean) {
    ValidateTripletLossInputs(anchor, positive, negative);

    TripletLoss loss(margin, TripletLoss::DistanceType::Euclidean, reduction);
    loss.SetNegative(negative);

    TripletMetricLossResult result;
    result.loss = loss.Forward(anchor, positive);
    result.grad_anchor = loss.Backward(anchor, positive);

    const auto& shape = anchor.Shape();
    const size_t batch_size = shape[0];
    const size_t dim = TripletEmbeddingDim(anchor);
    const float scale =
        reduction == Reduction::Mean ? 1.0f / static_cast<float>(batch_size)
                                     : 1.0f;

    std::vector<float> grad_positive(anchor.NumElements(), 0.0f);
    std::vector<float> grad_negative(anchor.NumElements(), 0.0f);
    const float* a = anchor.Data<float>();
    const float* p = positive.Data<float>();
    const float* n = negative.Data<float>();

    for (size_t row = 0; row < batch_size; ++row) {
        const size_t offset = row * dim;
        double dist_ap_sq = 0.0;
        double dist_an_sq = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            const double ap = static_cast<double>(a[offset + i]) -
                              static_cast<double>(p[offset + i]);
            const double an = static_cast<double>(a[offset + i]) -
                              static_cast<double>(n[offset + i]);
            dist_ap_sq += ap * ap;
            dist_an_sq += an * an;
        }

        const double dist_ap = std::sqrt(dist_ap_sq);
        const double dist_an = std::sqrt(dist_an_sq);
        if (dist_ap - dist_an + static_cast<double>(margin) <= 0.0) {
            continue;
        }

        const double safe_ap = std::max(dist_ap, 1e-8);
        const double safe_an = std::max(dist_an, 1e-8);
        for (size_t i = 0; i < dim; ++i) {
            const size_t index = offset + i;
            grad_positive[index] =
                static_cast<float>((static_cast<double>(p[index]) -
                                    static_cast<double>(a[index])) /
                                   safe_ap) *
                scale;
            grad_negative[index] =
                static_cast<float>((static_cast<double>(a[index]) -
                                    static_cast<double>(n[index])) /
                                   safe_an) *
                scale;
        }
    }

    result.grad_positive = Tensor(shape, grad_positive.data());
    result.grad_negative = Tensor(shape, grad_negative.data());
    return result;
}

}  // namespace cyxwiz
