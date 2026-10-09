#include "stacked_metric_loss.h"

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

}  // namespace

StackedTripletLoss::StackedTripletLoss(float margin)
    : Loss(Reduction::Mean), triplet_(margin, TripletLoss::DistanceType::Euclidean, Reduction::Mean) {}

std::pair<Tensor, Tensor> StackedTripletLoss::Split(const Tensor& embeddings) {
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

Tensor StackedTripletLoss::Forward(const Tensor& embeddings, const Tensor& /*class_ids*/) {
    const auto [anchors, positives] = Split(embeddings);
    return triplet_.Forward(anchors, positives);
}

Tensor StackedTripletLoss::Backward(const Tensor& embeddings, const Tensor& /*class_ids*/) {
    const auto [anchors, positives] = Split(embeddings);
    const TripletLossGradients gradients = triplet_.BackwardAll(anchors, positives);
    return Tensor::Cat({gradients.anchor, gradients.positive, gradients.negative}, 0);
}

// The loss of the other kind is never used; it gets a margin it accepts.
StackedPairLoss::StackedPairLoss(Kind kind, float margin)
    : Loss(Reduction::Mean),
      kind_(kind),
      contrastive_(kind == Kind::Contrastive ? margin : 1.0f, Reduction::Mean),
      cosine_(kind == Kind::CosineEmbedding ? margin : 0.0f, Reduction::Mean) {}

std::pair<Tensor, Tensor> StackedPairLoss::Split(const Tensor& embeddings, const Tensor& similar) {
    const auto& shape = embeddings.Shape();
    const size_t count = shape.size() == 2 ? shape[0] / 2 : 0;
    if (shape.size() != 2 || shape[0] == 0 || shape[0] % 2 != 0 || similar.NumElements() != count) {
        throw std::runtime_error(
            GetName() + " Loss needs [2P, D] embeddings (firsts; seconds) and P pair labels from the Pair "
            "Dataset Builder; got embeddings " + ShapeText(shape) + " and " +
            std::to_string(similar.NumElements()) + " labels");
    }
    // P values that came from the host sampler: convert there.
    const float* flags = similar.ReadData<float>();
    std::vector<float> labels(count);
    for (size_t i = 0; i < count; ++i) {
        const bool same = flags[i] == 1.0f;
        labels[i] = kind_ == Kind::Contrastive ? (same ? 0.0f : 1.0f) : (same ? 1.0f : -1.0f);
    }
    const Tensor label_tensor({count}, labels.data(), DataType::Float32);
    if (kind_ == Kind::Contrastive) {
        contrastive_.SetLabels(label_tensor);
    } else {
        cosine_.SetLabels(label_tensor);
    }
    const int rows = static_cast<int>(count);
    return {embeddings.Slice(0, 0, rows), embeddings.Slice(0, rows, 2 * rows)};
}

Tensor StackedPairLoss::Forward(const Tensor& embeddings, const Tensor& similar) {
    const auto [firsts, seconds] = Split(embeddings, similar);
    return kind_ == Kind::Contrastive ? contrastive_.Forward(firsts, seconds) : cosine_.Forward(firsts, seconds);
}

Tensor StackedPairLoss::Backward(const Tensor& embeddings, const Tensor& similar) {
    const auto [firsts, seconds] = Split(embeddings, similar);
    if (kind_ == Kind::Contrastive) {
        // The loss depends on firsts - seconds only.
        const Tensor gradient = contrastive_.Backward(firsts, seconds);
        return Tensor::Cat({gradient, -gradient}, 0);
    }
    // cos(a, b) is symmetric: the seconds' gradient is Backward(seconds, firsts).
    return Tensor::Cat({cosine_.Backward(firsts, seconds), cosine_.Backward(seconds, firsts)}, 0);
}

}  // namespace cyxwiz
