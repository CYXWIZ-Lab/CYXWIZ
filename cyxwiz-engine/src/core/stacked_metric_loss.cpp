#include "stacked_metric_loss.h"

#include "metric_learning_mining.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

namespace {

std::string ShapeText(const std::vector<size_t>& shape) {
    std::string text;
    for (size_t dim : shape) text += (text.empty() ? "" : ", ") + std::to_string(dim);
    return "[" + text + "]";
}

void RequireBatchEmbeddings(const Tensor& embeddings, const std::string& loss_name) {
    const auto& shape = embeddings.Shape();
    if (shape.size() != 2 || shape[0] == 0) {
        throw std::runtime_error(loss_name + " needs [N, D] embeddings; got " + ShapeText(shape));
    }
}

}  // namespace

StackedTripletLoss::StackedTripletLoss(float margin, MetricMining mining)
    : Loss(Reduction::Mean),
      mining_(mining),
      triplet_(margin, TripletLoss::DistanceType::Euclidean, Reduction::Mean) {}

std::pair<Tensor, Tensor> StackedTripletLoss::Blocks(const Tensor& embeddings, const Tensor& labels,
                                                     Tensor& source_rows) {
    if (mining_ != MetricMining::Random) {
        RequireBatchEmbeddings(embeddings, "Triplet Loss");
        const Tensor class_ids = BatchClassIds(labels, embeddings.Shape()[0]);
        const MinedTriplets picks =
            MineTriplets(class_ids, MiningDistances(embeddings, MiningDistance::Euclidean), mining_);
        if (picks.count == 0) {
            throw std::runtime_error("Triplet Loss found no anchor with a same-class and an other-class row in the batch");
        }
        source_rows = Tensor::Cat({picks.anchors, picks.positives, picks.negatives}, 0);
        triplet_.SetNegative(GatherRows(embeddings, picks.negatives));
        return {GatherRows(embeddings, picks.anchors), GatherRows(embeddings, picks.positives)};
    }
    const auto& shape = embeddings.Shape();
    if (shape.size() != 2 || shape[0] == 0 || shape[0] % 3 != 0) {
        throw std::runtime_error(
            "Triplet Loss needs [3T, D] embeddings (anchors; positives; negatives) from the Triplet Dataset "
            "Builder; got " + ShapeText(shape));
    }
    const int count = static_cast<int>(shape[0] / 3);
    triplet_.SetNegative(embeddings.Slice(0, 2 * count, 3 * count));
    return {embeddings.Slice(0, 0, count), embeddings.Slice(0, count, 2 * count)};
}

Tensor StackedTripletLoss::Forward(const Tensor& embeddings, const Tensor& labels) {
    Tensor source_rows;
    const auto [anchors, positives] = Blocks(embeddings, labels, source_rows);
    return triplet_.Forward(anchors, positives);
}

Tensor StackedTripletLoss::Backward(const Tensor& embeddings, const Tensor& labels) {
    Tensor source_rows;
    const auto [anchors, positives] = Blocks(embeddings, labels, source_rows);
    const TripletLossGradients gradients = triplet_.BackwardAll(anchors, positives);
    const Tensor stacked = Tensor::Cat({gradients.anchor, gradients.positive, gradients.negative}, 0);
    if (source_rows.Shape().empty()) return stacked;  // random mining: blocks were stacked by the sampler
    return ScatterAddRows(stacked, source_rows, embeddings.Shape()[0]);
}

// The loss of the other kind is never used; it gets a margin it accepts.
StackedPairLoss::StackedPairLoss(Kind kind, float margin, MetricMining mining)
    : Loss(Reduction::Mean),
      kind_(kind),
      mining_(mining),
      contrastive_(kind == Kind::Contrastive ? margin : 1.0f, Reduction::Mean),
      cosine_(kind == Kind::CosineEmbedding ? margin : 0.0f, Reduction::Mean) {
    if (mining_ == MetricMining::SemiHard) {
        throw std::invalid_argument("semi-hard mining is for the Triplet Loss; pair losses take random or hard");
    }
}

void StackedPairLoss::SetPairLabels(const Tensor& similar) {
    // Contrastive: 0 = similar, 1 = dissimilar. Cosine Embedding: +1 / -1.
    if (kind_ == Kind::Contrastive) {
        contrastive_.SetLabels(similar * -1.0f + 1.0f);
    } else {
        cosine_.SetLabels(similar * 2.0f - 1.0f);
    }
}

std::pair<Tensor, Tensor> StackedPairLoss::Blocks(const Tensor& embeddings, const Tensor& labels,
                                                  Tensor& source_rows) {
    if (mining_ == MetricMining::Hard) {
        RequireBatchEmbeddings(embeddings, GetName() + " Loss");
        const Tensor class_ids = BatchClassIds(labels, embeddings.Shape()[0]);
        const MinedPairs picks = MinePairs(
            class_ids,
            MiningDistances(embeddings,
                            kind_ == Kind::Contrastive ? MiningDistance::Euclidean : MiningDistance::Cosine));
        if (picks.count == 0) {
            throw std::runtime_error(GetName() + " Loss found no row with a partner in the batch");
        }
        source_rows = Tensor::Cat({picks.firsts, picks.seconds}, 0);
        SetPairLabels(picks.similar);
        return {GatherRows(embeddings, picks.firsts), GatherRows(embeddings, picks.seconds)};
    }
    const auto& shape = embeddings.Shape();
    const size_t count = shape.size() == 2 ? shape[0] / 2 : 0;
    if (shape.size() != 2 || shape[0] == 0 || shape[0] % 2 != 0 || labels.NumElements() != count) {
        throw std::runtime_error(
            GetName() + " Loss needs [2P, D] embeddings (firsts; seconds) and P pair labels from the Pair "
            "Dataset Builder; got embeddings " + ShapeText(shape) + " and " +
            std::to_string(labels.NumElements()) + " labels");
    }
    // The sampler's labels: 1 for a same-class pair.
    SetPairLabels((labels.Reshape({count}) == 1.0f) * 1.0f);
    const int rows = static_cast<int>(count);
    return {embeddings.Slice(0, 0, rows), embeddings.Slice(0, rows, 2 * rows)};
}

Tensor StackedPairLoss::Forward(const Tensor& embeddings, const Tensor& labels) {
    Tensor source_rows;
    const auto [firsts, seconds] = Blocks(embeddings, labels, source_rows);
    return kind_ == Kind::Contrastive ? contrastive_.Forward(firsts, seconds) : cosine_.Forward(firsts, seconds);
}

Tensor StackedPairLoss::Backward(const Tensor& embeddings, const Tensor& labels) {
    Tensor source_rows;
    const auto [firsts, seconds] = Blocks(embeddings, labels, source_rows);
    Tensor first_gradient;
    Tensor second_gradient;
    if (kind_ == Kind::Contrastive) {
        // The loss depends on firsts - seconds only.
        first_gradient = contrastive_.Backward(firsts, seconds);
        second_gradient = -first_gradient;
    } else {
        // cos(a, b) is symmetric: the seconds' gradient is Backward(seconds, firsts).
        first_gradient = cosine_.Backward(firsts, seconds);
        second_gradient = cosine_.Backward(seconds, firsts);
    }
    const Tensor stacked = Tensor::Cat({first_gradient, second_gradient}, 0);
    if (source_rows.Shape().empty()) return stacked;  // random mining: blocks were stacked by the sampler
    return ScatterAddRows(stacked, source_rows, embeddings.Shape()[0]);
}

}  // namespace cyxwiz
