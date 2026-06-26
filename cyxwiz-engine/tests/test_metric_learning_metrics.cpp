#include "core/metric_learning_metrics.h"

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

void CheckNear(double actual,
               double expected,
               double tolerance,
               const std::string& message) {
    if (std::fabs(actual - expected) > tolerance) {
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

cyxwiz::Tensor IntTensor(const std::vector<size_t>& shape,
                         const std::vector<int64_t>& values) {
    return cyxwiz::Tensor(shape, values.data(), cyxwiz::DataType::Int64);
}

void TestContrastivePairDistanceMetrics() {
    const auto left = FloatTensor({4, 2}, {
        0.0f, 0.0f,
        0.0f, 0.0f,
        1.0f, 1.0f,
        2.0f, 2.0f,
    });
    const auto right = FloatTensor({4, 2}, {
        0.1f, 0.0f,
        2.0f, 0.0f,
        1.2f, 1.0f,
        2.0f, 4.0f,
    });
    const auto labels = FloatTensor({4}, {0.0f, 1.0f, 0.0f, 1.0f});

    const auto metrics = cyxwiz::ComputePairDistanceMetrics(
        left,
        right,
        labels,
        cyxwiz::MetricLearningLabelConvention::
            ContrastiveZeroSimilarOneDissimilar,
        0.5);

    Check(metrics.pair_count == 4, "pair metric should record pair count");
    CheckNear(metrics.accuracy, 1.0, 1e-9,
              "contrastive distance threshold should classify all pairs");
    Check(metrics.positive_count == 2 && metrics.negative_count == 2,
          "pair metric should count positive and negative pairs");
    CheckNear(metrics.positive_distance_mean, 0.15, 1e-6,
              "pair metric should average similar-pair distances");
    CheckNear(metrics.negative_distance_mean, 2.0, 1e-6,
              "pair metric should average dissimilar-pair distances");
}

void TestCosineConventionPairDistanceMetrics() {
    const auto left = FloatTensor({2, 2}, {
        0.0f, 0.0f,
        0.0f, 0.0f,
    });
    const auto right = FloatTensor({2, 2}, {
        0.0f, 0.25f,
        2.0f, 0.0f,
    });
    const auto labels = FloatTensor({2, 1}, {1.0f, -1.0f});

    const auto metrics = cyxwiz::ComputePairDistanceMetrics(
        left,
        right,
        labels,
        cyxwiz::MetricLearningLabelConvention::
            CosineOneSimilarNegativeOneDissimilar,
        0.5);

    CheckNear(metrics.accuracy, 1.0, 1e-9,
              "cosine label convention should treat 1 as similar");
    CheckNear(metrics.positive_distance_mean, 0.25, 1e-6,
              "cosine convention should compute positive distance mean");
    CheckNear(metrics.negative_distance_mean, 2.0, 1e-6,
              "cosine convention should compute negative distance mean");
}

void TestRetrievalMetrics() {
    const auto embeddings = FloatTensor({5, 2}, {
        0.0f, 0.0f,
        0.1f, 0.0f,
        5.0f, 5.0f,
        5.1f, 5.0f,
        0.0f, 5.0f,
    });
    const auto class_ids = IntTensor({5}, {1, 1, 2, 2, 3});

    const auto metrics = cyxwiz::ComputeRetrievalMetrics(
        embeddings, class_ids, 1);

    Check(metrics.query_count == 5,
          "retrieval metrics should record query count");
    Check(metrics.k == 1,
          "retrieval metrics should preserve effective k");
    CheckNear(metrics.recall_at_k, 0.8, 1e-9,
              "retrieval recall@1 should count nearest class hits");
    CheckNear(metrics.nearest_neighbor_class_agreement, 0.8, 1e-9,
              "nearest neighbor agreement should match nearest class hits");
    CheckNear(metrics.mean_reciprocal_rank, 0.8, 1e-9,
              "MRR should average first relevant rank");
}

void TestRetrievalEffectiveKAndMRR() {
    const auto embeddings = FloatTensor({3, 1}, {
        0.0f,
        1.0f,
        2.0f,
    });
    const auto class_ids = FloatTensor({3}, {1.0f, 2.0f, 1.0f});

    const auto metrics = cyxwiz::ComputeRetrievalMetrics(
        embeddings, class_ids, 10);

    Check(metrics.k == 2,
          "retrieval metrics should cap k at sample_count - 1");
    CheckNear(metrics.recall_at_k, 2.0 / 3.0, 1e-9,
              "retrieval recall should ignore queries with no class match");
    CheckNear(metrics.mean_reciprocal_rank, 1.0 / 3.0, 1e-9,
              "retrieval MRR should use first relevant rank when present");
}

void TestMetricValidation() {
    bool rejected_bad_pair_shape = false;
    try {
        (void)cyxwiz::ComputePairDistanceMetrics(
            FloatTensor({2, 2}, {0.0f, 0.0f, 1.0f, 1.0f}),
            FloatTensor({2, 3}, {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}),
            FloatTensor({2}, {0.0f, 1.0f}),
            cyxwiz::MetricLearningLabelConvention::
                ContrastiveZeroSimilarOneDissimilar,
            0.5);
    } catch (const std::invalid_argument&) {
        rejected_bad_pair_shape = true;
    }
    Check(rejected_bad_pair_shape,
          "pair metrics should reject mismatched embedding shapes");

    bool rejected_bad_label = false;
    try {
        (void)cyxwiz::ComputePairDistanceMetrics(
            FloatTensor({1, 2}, {0.0f, 0.0f}),
            FloatTensor({1, 2}, {0.0f, 0.0f}),
            FloatTensor({1}, {2.0f}),
            cyxwiz::MetricLearningLabelConvention::
                ContrastiveZeroSimilarOneDissimilar,
            0.5);
    } catch (const std::invalid_argument&) {
        rejected_bad_label = true;
    }
    Check(rejected_bad_label,
          "pair metrics should reject labels outside the convention");

    bool rejected_bad_retrieval = false;
    try {
        (void)cyxwiz::ComputeRetrievalMetrics(
            FloatTensor({1, 2}, {0.0f, 0.0f}),
            IntTensor({1}, {1}),
            1);
    } catch (const std::invalid_argument&) {
        rejected_bad_retrieval = true;
    }
    Check(rejected_bad_retrieval,
          "retrieval metrics should reject single-sample embeddings");
}

}  // namespace

int main() {
    TestContrastivePairDistanceMetrics();
    TestCosineConventionPairDistanceMetrics();
    TestRetrievalMetrics();
    TestRetrievalEffectiveKAndMRR();
    TestMetricValidation();
    std::cout << "Metric-learning metrics passed\n";
    return 0;
}
