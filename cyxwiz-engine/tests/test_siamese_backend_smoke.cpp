#include <cyxwiz/losses/metric_learning.h>
#include <cyxwiz/tensor.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr size_t kInputDim = 2;
constexpr size_t kEmbeddingDim = 2;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

float SquaredDistance(const std::vector<float>& a,
                      const std::vector<float>& b) {
    float total = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float diff = a[i] - b[i];
        total += diff * diff;
    }
    return total;
}

std::vector<float> Encode(const std::vector<float>& input,
                          const std::vector<float>& weights) {
    std::vector<float> embedding(kEmbeddingDim, 0.0f);
    for (size_t col = 0; col < kEmbeddingDim; ++col) {
        for (size_t row = 0; row < kInputDim; ++row) {
            embedding[col] += input[row] * weights[row * kEmbeddingDim + col];
        }
    }
    return embedding;
}

cyxwiz::Tensor EncodeBatch(const std::vector<std::vector<float>>& inputs,
                           const std::vector<float>& weights) {
    std::vector<float> encoded;
    encoded.reserve(inputs.size() * kEmbeddingDim);
    for (const auto& input : inputs) {
        const auto embedding = Encode(input, weights);
        encoded.insert(encoded.end(), embedding.begin(), embedding.end());
    }
    return cyxwiz::Tensor({inputs.size(), kEmbeddingDim}, encoded.data());
}

float ContrastiveLossValue(cyxwiz::ContrastiveLoss& loss,
                           const std::vector<std::vector<float>>& left,
                           const std::vector<std::vector<float>>& right,
                           const std::vector<float>& weights) {
    cyxwiz::Tensor left_embeddings = EncodeBatch(left, weights);
    cyxwiz::Tensor right_embeddings = EncodeBatch(right, weights);
    cyxwiz::Tensor loss_value = loss.Forward(left_embeddings, right_embeddings);
    return loss_value.Data<float>()[0];
}

float MeanPositiveDistance(const std::vector<std::vector<float>>& left,
                           const std::vector<std::vector<float>>& right,
                           const std::vector<float>& labels,
                           const std::vector<float>& weights) {
    float total = 0.0f;
    size_t count = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] != 0.0f) continue;
        total += std::sqrt(SquaredDistance(Encode(left[i], weights),
                                           Encode(right[i], weights)));
        ++count;
    }
    return count > 0 ? total / static_cast<float>(count) : 0.0f;
}

float MeanNegativeDistance(const std::vector<std::vector<float>>& left,
                           const std::vector<std::vector<float>>& right,
                           const std::vector<float>& labels,
                           const std::vector<float>& weights) {
    float total = 0.0f;
    size_t count = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] == 0.0f) continue;
        total += std::sqrt(SquaredDistance(Encode(left[i], weights),
                                           Encode(right[i], weights)));
        ++count;
    }
    return count > 0 ? total / static_cast<float>(count) : 0.0f;
}

void TrainSharedEncoderStep(cyxwiz::ContrastiveLoss& loss,
                            const std::vector<std::vector<float>>& left,
                            const std::vector<std::vector<float>>& right,
                            std::vector<float>& weights,
                            float learning_rate) {
    cyxwiz::Tensor left_embeddings = EncodeBatch(left, weights);
    cyxwiz::Tensor right_embeddings = EncodeBatch(right, weights);
    (void)loss.Forward(left_embeddings, right_embeddings);
    cyxwiz::Tensor grad_left = loss.Backward(left_embeddings, right_embeddings);
    const float* grad_left_data = grad_left.Data<float>();

    std::vector<float> grad_weights(weights.size(), 0.0f);
    for (size_t sample = 0; sample < left.size(); ++sample) {
        for (size_t out = 0; out < kEmbeddingDim; ++out) {
            const float g_left =
                grad_left_data[sample * kEmbeddingDim + out];
            const float g_right = -g_left;
            for (size_t in = 0; in < kInputDim; ++in) {
                grad_weights[in * kEmbeddingDim + out] +=
                    left[sample][in] * g_left + right[sample][in] * g_right;
            }
        }
    }

    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= learning_rate * grad_weights[i];
    }
}

}  // namespace

int main() {
    const std::vector<std::vector<float>> left = {
        {1.0f, 1.0f},
        {-1.0f, -1.0f},
        {1.0f, 1.0f},
        {-1.0f, -1.0f},
    };
    const std::vector<std::vector<float>> right = {
        {1.1f, 0.9f},
        {-1.1f, -0.9f},
        {-1.0f, -1.0f},
        {1.0f, 1.0f},
    };
    const std::vector<float> labels = {0.0f, 0.0f, 1.0f, 1.0f};

    cyxwiz::ContrastiveLoss loss(2.0f, cyxwiz::Reduction::Mean);
    cyxwiz::Tensor label_tensor({labels.size()}, labels.data());
    loss.SetLabels(label_tensor);

    std::vector<float> weights = {
        0.20f, -0.10f,
        0.10f,  0.20f,
    };

    const float initial_loss = ContrastiveLossValue(loss, left, right, weights);
    const float initial_positive_distance =
        MeanPositiveDistance(left, right, labels, weights);
    const float initial_negative_distance =
        MeanNegativeDistance(left, right, labels, weights);

    for (int step = 0; step < 120; ++step) {
        TrainSharedEncoderStep(loss, left, right, weights, 0.08f);
    }

    const float final_loss = ContrastiveLossValue(loss, left, right, weights);
    const float final_positive_distance =
        MeanPositiveDistance(left, right, labels, weights);
    const float final_negative_distance =
        MeanNegativeDistance(left, right, labels, weights);

    Check(std::isfinite(initial_loss) && std::isfinite(final_loss),
          "contrastive smoke losses should be finite");
    Check(final_loss < initial_loss * 0.40f,
          "shared encoder training should reduce contrastive loss");
    Check(final_positive_distance <= initial_positive_distance + 1e-4f,
          "positive pairs should not move farther apart");
    Check(final_negative_distance > initial_negative_distance + 0.75f,
          "negative pairs should move farther apart");

    std::cout << "Siamese backend contrastive smoke passed: loss "
              << initial_loss << " -> " << final_loss
              << ", positive_distance " << initial_positive_distance
              << " -> " << final_positive_distance
              << ", negative_distance " << initial_negative_distance
              << " -> " << final_negative_distance << "\n";
    return 0;
}
