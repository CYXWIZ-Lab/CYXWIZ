#pragma once

#include "metric_learning_losses.h"
#include "metric_learning_shared_encoder.h"

#include <cyxwiz/optimizer.h>
#include <cyxwiz/tensor.h>

#include <stdexcept>
#include <utility>

namespace cyxwiz {

struct PairMetricTrainingStepConfig {
    MetricLearningPairLossKind loss_kind =
        MetricLearningPairLossKind::Contrastive;
    float margin = 1.0f;
    Reduction reduction = Reduction::Mean;
    bool update_parameters = false;
};

struct TripletMetricTrainingStepConfig {
    float margin = 1.0f;
    Reduction reduction = Reduction::Mean;
    bool update_parameters = false;
};

struct PairMetricTrainingStepResult {
    Tensor loss;
    PairEmbeddings embeddings;
    PairBranchGradients input_gradients;
};

struct TripletMetricTrainingStepResult {
    Tensor loss;
    TripletEmbeddings embeddings;
    TripletBranchGradients input_gradients;
};

inline void ValidateMetricTrainingStepUpdate(bool update_parameters,
                                             Optimizer* optimizer) {
    if (update_parameters && optimizer == nullptr) {
        throw std::invalid_argument(
            "metric-learning training step update requires an optimizer");
    }
}

inline PairMetricTrainingStepResult RunPairMetricTrainingStep(
    SharedEncoderRuntime& runtime,
    const PairBatch& batch,
    const PairMetricTrainingStepConfig& config,
    Optimizer* optimizer = nullptr) {
    if (!batch.IsValid()) {
        throw std::invalid_argument(
            "pair metric training step requires a valid PairBatch");
    }
    ValidateMetricTrainingStepUpdate(config.update_parameters, optimizer);

    PairMetricTrainingStepResult result;
    result.embeddings = runtime.ForwardPair(batch);
    const auto loss = ComputePairMetricLoss(
        result.embeddings.embedding_a,
        result.embeddings.embedding_b,
        batch.pair_label,
        config.loss_kind,
        config.margin,
        config.reduction);
    result.loss = loss.loss;
    result.input_gradients =
        runtime.BackwardPair(loss.grad_a, loss.grad_b);
    if (config.update_parameters) {
        runtime.UpdateParameters(optimizer);
    }
    return result;
}

inline TripletMetricTrainingStepResult RunTripletMetricTrainingStep(
    SharedEncoderRuntime& runtime,
    const TripletBatch& batch,
    const TripletMetricTrainingStepConfig& config,
    Optimizer* optimizer = nullptr) {
    if (!batch.IsValid()) {
        throw std::invalid_argument(
            "triplet metric training step requires a valid TripletBatch");
    }
    ValidateMetricTrainingStepUpdate(config.update_parameters, optimizer);

    TripletMetricTrainingStepResult result;
    result.embeddings = runtime.ForwardTriplet(batch);
    const auto loss = ComputeEuclideanTripletMetricLoss(
        result.embeddings.anchor,
        result.embeddings.positive,
        result.embeddings.negative,
        config.margin,
        config.reduction);
    result.loss = loss.loss;
    result.input_gradients = runtime.BackwardTriplet(
        loss.grad_anchor, loss.grad_positive, loss.grad_negative);
    if (config.update_parameters) {
        runtime.UpdateParameters(optimizer);
    }
    return result;
}

}  // namespace cyxwiz
