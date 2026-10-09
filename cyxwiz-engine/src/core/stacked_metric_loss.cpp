#include "stacked_metric_loss.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace cyxwiz {

StackedTripletLoss::StackedTripletLoss(float margin)
    : Loss(Reduction::Mean), triplet_(margin, TripletLoss::DistanceType::Euclidean, Reduction::Mean) {}

std::pair<Tensor, Tensor> StackedTripletLoss::Split(const Tensor& embeddings) {
    const auto& shape = embeddings.Shape();
    if (shape.size() != 2 || shape[0] == 0 || shape[0] % 3 != 0) {
        std::string text;
        for (size_t dim : shape) text += (text.empty() ? "" : ", ") + std::to_string(dim);
        throw std::runtime_error(
            "Triplet Loss needs [3T, D] embeddings (anchors; positives; negatives) from the Triplet Dataset "
            "Builder; got [" + text + "]");
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

}  // namespace cyxwiz
