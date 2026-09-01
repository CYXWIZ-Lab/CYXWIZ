#pragma once

#include "metric_learning_batch.h"

#include <cyxwiz/losses/metric_learning.h>
#include <cyxwiz/tensor.h>

#include <stdexcept>
#include <string>

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
    if (embedding_a.GetDataType() != DataType::Float32 ||
        embedding_b.GetDataType() != DataType::Float32 ||
        labels.GetDataType() != DataType::Float32) {
        throw std::invalid_argument("metric-learning pair loss inputs must be Float32");
    }
    if (embedding_a.Shape() != embedding_b.Shape()) {
        throw std::invalid_argument(
            "metric-learning pair loss embeddings must have identical shapes");
    }
    const auto& shape = embedding_a.Shape();
    const size_t pair_count = shape.empty() ? 0 : shape[0];
    if (shape.size() != 2 || pair_count == 0 || shape[1] == 0) {
        throw std::invalid_argument("metric-learning pair loss embeddings must be "
                                    "non-empty [batch, embedding_dim]");
    }
    if (!TensorIsBatchVector(labels, pair_count)) {
        throw std::invalid_argument(
            "metric-learning pair loss labels must be [batch] or [batch, 1]");
    }
    const float* label_data = labels.ReadData<float>();
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
        result.grad_b = swapped_loss.Backward(embedding_b, embedding_a);
    } else {
        CosineEmbeddingLoss forward_loss(margin, reduction);
        forward_loss.SetLabels(labels);
        result.loss = forward_loss.Forward(embedding_a, embedding_b);
        result.grad_a = forward_loss.Backward(embedding_a, embedding_b);

        CosineEmbeddingLoss swapped_loss(margin, reduction);
        swapped_loss.SetLabels(labels);
        result.grad_b = swapped_loss.Backward(embedding_b, embedding_a);
    }

    return result;
}

inline void ValidateTripletLossInputs(const Tensor& anchor,
                                      const Tensor& positive,
                                      const Tensor& negative) {
    if (anchor.GetDataType() != DataType::Float32 || positive.GetDataType() != DataType::Float32 ||
        negative.GetDataType() != DataType::Float32) {
        throw std::invalid_argument("metric-learning triplet loss embeddings must be Float32");
    }
    if (anchor.Shape() != positive.Shape() ||
        anchor.Shape() != negative.Shape()) {
        throw std::invalid_argument(
            "metric-learning triplet loss embeddings must have identical shapes");
    }
    const auto& shape = anchor.Shape();
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw std::invalid_argument("metric-learning triplet loss embeddings must "
                                    "be non-empty [batch, embedding_dim]");
    }
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
    const TripletLossGradients gradients = loss.BackwardAll(anchor, positive);
    result.grad_anchor = gradients.anchor;
    result.grad_positive = gradients.positive;
    result.grad_negative = gradients.negative;
    return result;
}

}  // namespace cyxwiz
