#pragma once

// Metric-learning losses over one encoder pass (TOFIX140 A5).
//   Random mining: MetricBatchSampler stacked the picks, so the embeddings are
//     [3T, D] (anchors; positives; negatives) or [2P, D] (firsts; seconds);
//     the loss splits the blocks.
//   Hard / semi-hard mining: the embeddings are the batch's [N, D] and the
//     labels its class ids; the loss mines its triplets or pairs from their
//     distances (metric_learning_mining.h, on the device) and gathers the blocks.
// Either way the blocks are scored by the backend loss, and the gradient goes
// back to the encoder's rows (stacked, or summed per row through the picks),
// so a single backward sums the shared-weight gradients (torch:
// encoder(torch.cat(...)) or encoder(x) followed by indexing).

#include "metric_learning_batch.h"

#include <cyxwiz/losses/metric_learning.h>

#include <utility>

namespace cyxwiz {

class StackedTripletLoss final : public Loss {
public:
    explicit StackedTripletLoss(float margin, MetricMining mining = MetricMining::Random);

    // labels: unused anchors' ids (random) or the batch's class ids (mining).
    Tensor Forward(const Tensor& embeddings, const Tensor& labels) override;
    Tensor Backward(const Tensor& embeddings, const Tensor& labels) override;
    std::string GetName() const override { return "Triplet"; }

    float GetMargin() const { return triplet_.GetMargin(); }

private:
    // The three [T, D] blocks; hands the negatives to the backend loss and
    // returns {anchors, positives}. `source_rows` gets the mined rows of the
    // stacked blocks (anchors; positives; negatives) as device indices, or
    // stays empty for random mining.
    std::pair<Tensor, Tensor> Blocks(const Tensor& embeddings, const Tensor& labels, Tensor& source_rows);

    MetricMining mining_;
    TripletLoss triplet_;
};

// Contrastive or Cosine Embedding loss over pairs. With random mining the
// labels are the sampler's: 1 for a same-class pair, 0 otherwise; with hard
// mining they are the batch's class ids. Each loss gets the pair labels in its
// own convention (Contrastive 0 = similar, 1 = dissimilar; Cosine Embedding
// +1 = similar, -1 = dissimilar).
class StackedPairLoss final : public Loss {
public:
    enum class Kind { Contrastive, CosineEmbedding };

    StackedPairLoss(Kind kind, float margin, MetricMining mining = MetricMining::Random);

    Tensor Forward(const Tensor& embeddings, const Tensor& labels) override;
    Tensor Backward(const Tensor& embeddings, const Tensor& labels) override;
    std::string GetName() const override {
        return kind_ == Kind::Contrastive ? "Contrastive" : "CosineEmbedding";
    }

private:
    // The two [P, D] blocks; hands the pair labels to the backend loss and
    // returns {firsts, seconds}. `source_rows` gets the mined rows of the
    // stacked blocks (firsts; seconds) as device indices, or stays empty for
    // random mining.
    std::pair<Tensor, Tensor> Blocks(const Tensor& embeddings, const Tensor& labels, Tensor& source_rows);
    // similar: Float32 [P], 1 for a same-class pair.
    void SetPairLabels(const Tensor& similar);

    Kind kind_;
    MetricMining mining_;
    ContrastiveLoss contrastive_;
    CosineEmbeddingLoss cosine_;
};

}  // namespace cyxwiz
