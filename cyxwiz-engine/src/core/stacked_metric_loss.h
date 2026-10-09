#pragma once

// Metric-learning losses over a stacked batch (TOFIX140 A5). The model runs
// once over [anchors; positives; negatives] (TripletBatchSampler), so its
// output is [3T, D]; the loss splits the three blocks, scores them with the
// backend loss, and returns the gradient stacked the same way, so a single
// backward sums the shared-weight gradients (torch: encoder(torch.cat(...))).
// The class ids the loss receives are the anchors' and are not used.

#include <cyxwiz/losses/metric_learning.h>

#include <utility>

namespace cyxwiz {

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

}  // namespace cyxwiz
