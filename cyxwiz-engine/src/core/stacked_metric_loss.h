#pragma once

// Metric-learning losses over a stacked batch (TOFIX140 A5). The model runs
// once over the rows MetricBatchSampler stacks, so its output is [3T, D]
// (anchors; positives; negatives) or [2P, D] (firsts; seconds). The loss
// splits the blocks, scores them with the backend loss, and returns the
// gradient stacked the same way, so a single backward sums the shared-weight
// gradients (torch: encoder(torch.cat(...))).

#include <cyxwiz/losses/metric_learning.h>

#include <utility>

namespace cyxwiz {

// The class ids the loss receives are the anchors' and are not used.
class StackedTripletLoss final : public Loss {
public:
    explicit StackedTripletLoss(float margin);

    Tensor Forward(const Tensor& embeddings, const Tensor& class_ids) override;
    Tensor Backward(const Tensor& embeddings, const Tensor& class_ids) override;
    std::string GetName() const override { return "Triplet"; }

    float GetMargin() const { return triplet_.GetMargin(); }

private:
    // Splits [3T, D] into the three [T, D] blocks and hands the negatives to
    // the backend loss; returns {anchors, positives}.
    std::pair<Tensor, Tensor> Split(const Tensor& embeddings);

    TripletLoss triplet_;
};

// Contrastive or Cosine Embedding loss over [firsts; seconds]. The labels are
// the sampler's: 1 for a same-class pair, 0 otherwise; each loss gets them in
// its own convention (Contrastive 0 = similar, 1 = dissimilar; Cosine
// Embedding +1 = similar, -1 = dissimilar).
class StackedPairLoss final : public Loss {
public:
    enum class Kind { Contrastive, CosineEmbedding };

    StackedPairLoss(Kind kind, float margin);

    Tensor Forward(const Tensor& embeddings, const Tensor& similar) override;
    Tensor Backward(const Tensor& embeddings, const Tensor& similar) override;
    std::string GetName() const override {
        return kind_ == Kind::Contrastive ? "Contrastive" : "CosineEmbedding";
    }

private:
    // Splits [2P, D] into the two [P, D] blocks and hands the pair labels to
    // the backend loss; returns {firsts, seconds}.
    std::pair<Tensor, Tensor> Split(const Tensor& embeddings, const Tensor& similar);

    Kind kind_;
    ContrastiveLoss contrastive_;
    CosineEmbeddingLoss cosine_;
};

}  // namespace cyxwiz
