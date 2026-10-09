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

// The batch's [N, D] embeddings and class ids for mining, checked.
struct MiningInput {
    size_t rows = 0;
    std::vector<double> distances;
    std::vector<int64_t> class_ids;
};

MiningInput ReadMiningInput(const Tensor& embeddings, const Tensor& class_ids, MiningDistance kind,
                            const std::string& loss_name) {
    const auto& shape = embeddings.Shape();
    if (shape.size() != 2 || shape[0] == 0) {
        throw std::runtime_error(loss_name + " needs [N, D] embeddings; got " + ShapeText(shape));
    }
    MiningInput input;
    input.rows = shape[0];
    input.class_ids = ReadBatchClassIds(class_ids, input.rows);
    // [N, D] floats to the host once per call; the picks are host indices.
    input.distances = MiningDistances(embeddings.ReadData<float>(), shape[0], shape[1], kind);
    return input;
}

// The gradient of the gathered blocks summed back onto the N encoder rows:
// selection[row, j] = 1 when stacked row j was taken from `row`.
Tensor ScatterToRows(const Tensor& stacked_gradient, const std::vector<int>& source_rows, size_t rows) {
    const size_t stacked = source_rows.size();
    const size_t width = stacked_gradient.Shape()[1];
    std::vector<float> selection(rows * stacked, 0.0f);
    for (size_t j = 0; j < stacked; ++j) selection[static_cast<size_t>(source_rows[j]) * stacked + j] = 1.0f;
    const Tensor matrix({1, rows, stacked}, selection.data(), DataType::Float32);
    return matrix.BatchMatMul(stacked_gradient.Reshape({1, stacked, width})).Reshape({rows, width});
}

std::vector<int> BlockRows(const std::vector<std::array<int, 3>>& picks, size_t block) {
    std::vector<int> rows(picks.size());
    for (size_t i = 0; i < picks.size(); ++i) rows[i] = picks[i][block];
    return rows;
}

}  // namespace

StackedTripletLoss::StackedTripletLoss(float margin, MetricMining mining)
    : Loss(Reduction::Mean),
      mining_(mining),
      triplet_(margin, TripletLoss::DistanceType::Euclidean, Reduction::Mean) {}

std::pair<Tensor, Tensor> StackedTripletLoss::Blocks(const Tensor& embeddings, const Tensor& labels,
                                                     std::vector<std::array<int, 3>>& picks) {
    if (mining_ != MetricMining::Random) {
        const MiningInput input = ReadMiningInput(embeddings, labels, MiningDistance::Euclidean, "Triplet Loss");
        picks = MineTriplets(input.class_ids, input.distances, mining_);
        if (picks.empty()) {
            throw std::runtime_error("Triplet Loss found no anchor with a same-class and an other-class row in the batch");
        }
        triplet_.SetNegative(embeddings.IndexSelect(0, BlockRows(picks, 2)));
        return {embeddings.IndexSelect(0, BlockRows(picks, 0)), embeddings.IndexSelect(0, BlockRows(picks, 1))};
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
    std::vector<std::array<int, 3>> picks;
    const auto [anchors, positives] = Blocks(embeddings, labels, picks);
    return triplet_.Forward(anchors, positives);
}

Tensor StackedTripletLoss::Backward(const Tensor& embeddings, const Tensor& labels) {
    std::vector<std::array<int, 3>> picks;
    const auto [anchors, positives] = Blocks(embeddings, labels, picks);
    const TripletLossGradients gradients = triplet_.BackwardAll(anchors, positives);
    const Tensor stacked = Tensor::Cat({gradients.anchor, gradients.positive, gradients.negative}, 0);
    if (picks.empty()) return stacked;
    std::vector<int> source_rows = BlockRows(picks, 0);
    for (size_t block = 1; block < 3; ++block) {
        const auto rows = BlockRows(picks, block);
        source_rows.insert(source_rows.end(), rows.begin(), rows.end());
    }
    return ScatterToRows(stacked, source_rows, embeddings.Shape()[0]);
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

void StackedPairLoss::SetPairLabels(const std::vector<bool>& similar) {
    std::vector<float> labels(similar.size());
    for (size_t i = 0; i < similar.size(); ++i) {
        labels[i] = kind_ == Kind::Contrastive ? (similar[i] ? 0.0f : 1.0f) : (similar[i] ? 1.0f : -1.0f);
    }
    const Tensor label_tensor({labels.size()}, labels.data(), DataType::Float32);
    if (kind_ == Kind::Contrastive) {
        contrastive_.SetLabels(label_tensor);
    } else {
        cosine_.SetLabels(label_tensor);
    }
}

std::pair<Tensor, Tensor> StackedPairLoss::Blocks(const Tensor& embeddings, const Tensor& labels,
                                                  std::vector<std::array<int, 3>>& picks) {
    if (mining_ == MetricMining::Hard) {
        const MiningInput input = ReadMiningInput(
            embeddings, labels,
            kind_ == Kind::Contrastive ? MiningDistance::Euclidean : MiningDistance::Cosine, GetName() + " Loss");
        picks = MinePairs(input.class_ids, input.distances);
        if (picks.empty()) {
            throw std::runtime_error(GetName() + " Loss found no row with a partner in the batch");
        }
        std::vector<bool> similar(picks.size());
        for (size_t i = 0; i < picks.size(); ++i) similar[i] = picks[i][2] == 1;
        SetPairLabels(similar);
        return {embeddings.IndexSelect(0, BlockRows(picks, 0)), embeddings.IndexSelect(0, BlockRows(picks, 1))};
    }
    const auto& shape = embeddings.Shape();
    const size_t count = shape.size() == 2 ? shape[0] / 2 : 0;
    if (shape.size() != 2 || shape[0] == 0 || shape[0] % 2 != 0 || labels.NumElements() != count) {
        throw std::runtime_error(
            GetName() + " Loss needs [2P, D] embeddings (firsts; seconds) and P pair labels from the Pair "
            "Dataset Builder; got embeddings " + ShapeText(shape) + " and " +
            std::to_string(labels.NumElements()) + " labels");
    }
    // P values that came from the host sampler: convert there.
    const float* flags = labels.ReadData<float>();
    std::vector<bool> similar(count);
    for (size_t i = 0; i < count; ++i) similar[i] = flags[i] == 1.0f;
    SetPairLabels(similar);
    const int rows = static_cast<int>(count);
    return {embeddings.Slice(0, 0, rows), embeddings.Slice(0, rows, 2 * rows)};
}

Tensor StackedPairLoss::Forward(const Tensor& embeddings, const Tensor& labels) {
    std::vector<std::array<int, 3>> picks;
    const auto [firsts, seconds] = Blocks(embeddings, labels, picks);
    return kind_ == Kind::Contrastive ? contrastive_.Forward(firsts, seconds) : cosine_.Forward(firsts, seconds);
}

Tensor StackedPairLoss::Backward(const Tensor& embeddings, const Tensor& labels) {
    std::vector<std::array<int, 3>> picks;
    const auto [firsts, seconds] = Blocks(embeddings, labels, picks);
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
    if (picks.empty()) return stacked;
    std::vector<int> source_rows = BlockRows(picks, 0);
    const auto second_rows = BlockRows(picks, 1);
    source_rows.insert(source_rows.end(), second_rows.begin(), second_rows.end());
    return ScatterToRows(stacked, source_rows, embeddings.Shape()[0]);
}

}  // namespace cyxwiz
