#include "core/metric_learning_losses.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

void CheckNear(float actual,
               float expected,
               float tolerance,
               const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << message
                  << " actual=" << actual
                  << " expected=" << expected << "\n";
        std::exit(1);
    }
}

cyxwiz::Tensor FloatTensor(const std::vector<size_t>& shape,
                           const std::vector<float>& values) {
    return cyxwiz::Tensor(shape, values.data());
}

void TestContrastivePairLossBranchGradients() {
    const auto embedding_a = FloatTensor({1, 2}, {1.0f, 0.0f});
    const auto embedding_b = FloatTensor({1, 2}, {0.0f, 0.0f});
    const auto labels = FloatTensor({1}, {0.0f});

    const auto result = cyxwiz::ComputePairMetricLoss(
        embedding_a,
        embedding_b,
        labels,
        cyxwiz::MetricLearningPairLossKind::Contrastive,
        2.0f);

    Check(result.loss.Shape() == std::vector<size_t>({1}),
          "contrastive pair loss should reduce to scalar tensor");
    CheckNear(result.loss.Data<float>()[0], 1.0f, 1e-5f,
              "contrastive similar-pair loss should be squared distance");
    CheckNear(result.grad_a.Data<float>()[0], 2.0f, 1e-5f,
              "contrastive grad_a should point away from branch b");
    CheckNear(result.grad_a.Data<float>()[1], 0.0f, 1e-5f,
              "contrastive grad_a y component should be zero");
    CheckNear(result.grad_b.Data<float>()[0], -2.0f, 1e-5f,
              "contrastive grad_b should mirror grad_a");
}

void TestCosinePairLossBranchGradients() {
    const auto embedding_a = FloatTensor({1, 2}, {1.0f, 0.0f});
    const auto embedding_b = FloatTensor({1, 2}, {0.0f, 1.0f});
    const auto labels = FloatTensor({1}, {1.0f});

    const auto result = cyxwiz::ComputePairMetricLoss(
        embedding_a,
        embedding_b,
        labels,
        cyxwiz::MetricLearningPairLossKind::CosineEmbedding,
        0.0f);

    CheckNear(result.loss.Data<float>()[0], 1.0f, 1e-5f,
              "cosine similar-pair loss should be one minus cosine");
    CheckNear(result.grad_a.Data<float>()[0], 0.0f, 1e-5f,
              "cosine grad_a x component should be zero");
    CheckNear(result.grad_a.Data<float>()[1], -1.0f, 1e-5f,
              "cosine grad_a should pull toward branch b");
    CheckNear(result.grad_b.Data<float>()[0], -1.0f, 1e-5f,
              "cosine grad_b should pull toward branch a");
    CheckNear(result.grad_b.Data<float>()[1], 0.0f, 1e-5f,
              "cosine grad_b y component should be zero");
}

void TestTripletLossBranchGradients() {
    const auto anchor = FloatTensor({1, 2}, {0.0f, 0.0f});
    const auto positive = FloatTensor({1, 2}, {0.5f, 0.0f});
    const auto negative = FloatTensor({1, 2}, {1.0f, 0.0f});

    const auto result = cyxwiz::ComputeEuclideanTripletMetricLoss(
        anchor, positive, negative, 1.0f);

    CheckNear(result.loss.Data<float>()[0], 0.5f, 1e-5f,
              "triplet loss should be active margin violation");
    CheckNear(result.grad_anchor.Data<float>()[0], 0.0f, 1e-5f,
              "triplet anchor gradient should combine positive and negative terms");
    CheckNear(result.grad_positive.Data<float>()[0], 1.0f, 1e-5f,
              "triplet positive gradient should move positive toward anchor");
    CheckNear(result.grad_negative.Data<float>()[0], -1.0f, 1e-5f,
              "triplet negative gradient should move negative away from anchor");
}

void TestInvalidLabelConventionRejected() {
    bool rejected = false;
    try {
        (void)cyxwiz::ComputePairMetricLoss(
            FloatTensor({1, 2}, {1.0f, 0.0f}),
            FloatTensor({1, 2}, {0.0f, 0.0f}),
            FloatTensor({1}, {0.0f}),
            cyxwiz::MetricLearningPairLossKind::CosineEmbedding,
            0.0f);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Check(rejected,
          "cosine embedding loss should reject contrastive-style labels");
}

}  // namespace

int main() {
    TestContrastivePairLossBranchGradients();
    TestCosinePairLossBranchGradients();
    TestTripletLossBranchGradients();
    TestInvalidLabelConventionRejected();
    std::cout << "Metric-learning loss contracts passed\n";
    return 0;
}
