#include "core/metric_learning_metrics.h"

#include "algorithms/arrayfire_backend_utils.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <arrayfire.h>

namespace {

#ifndef NDEBUG
std::vector<cyxwiz::ArrayFireHostSyncEvent>* g_host_sync_events = nullptr;

void CaptureHostSync(const cyxwiz::ArrayFireHostSyncEvent& event) {
    if (g_host_sync_events != nullptr) {
        g_host_sync_events->push_back(event);
    }
}

class ScopedEventCapture {
public:
    explicit ScopedEventCapture(
        std::vector<cyxwiz::ArrayFireHostSyncEvent>& host_sync_events) {
        g_host_sync_events = &host_sync_events;
    }

    ~ScopedEventCapture() { g_host_sync_events = nullptr; }
};
#endif

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

cyxwiz::Tensor DeviceFloatTensor(const std::vector<size_t>& shape,
                                 const std::vector<float>& values) {
    const auto host = FloatTensor(shape, values);
    return cyxwiz::Tensor::FromSemanticArray(
        host.GetSemanticArray(), host.Shape());
}

cyxwiz::Tensor DeviceIntTensor(const std::vector<size_t>& shape,
                               const std::vector<int64_t>& values) {
    const auto host = IntTensor(shape, values);
    return cyxwiz::Tensor::FromSemanticArray(
        host.GetSemanticArray(), host.Shape());
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

// Two same-class candidates at exactly the same distance from a query: the
// first relevant rank stays 1 whichever of them ArrayFire's min reports
// (the ArrayFire path counted the other one as preceding). The 12 embeddings
// are an untrained metric encoder's (TOFIX140 A5, metric_graph_pytorch.json
// test_step); row 7 has rows 1 and 4 tied. Expected values from torch.
void TestRetrievalTiedRelevantCandidates() {
    const auto tied = FloatTensor({4, 1}, {0.0f, 1.0f, -1.0f, 10.0f});
    const auto tied_ids = FloatTensor({4}, {1.0f, 1.0f, 1.0f, 2.0f});
    const auto simple = cyxwiz::ComputeRetrievalMetrics(tied, tied_ids, 1);
    // Rows 0-2: a same-class row first; row 3 has no same-class row.
    CheckNear(simple.nearest_neighbor_class_agreement, 0.75, 1e-9,
              "tied same-class candidates both count as nearest");
    CheckNear(simple.mean_reciprocal_rank, 0.75, 1e-9,
              "tied same-class candidates keep rank 1");

    const auto embeddings = FloatTensor({12, 2}, {
        -0.441503525f, -0.40581125f, 0.123165175f, 0.332084894f,
        -0.214860216f, -0.567039073f, -0.426469713f, -0.316539168f,
        -0.0192071665f, 0.250427395f, -0.19668746f, -0.568237543f,
        -0.500527382f, -0.613338113f, 0.0519790202f, 0.2912561f,
        -0.351335526f, -0.702403188f, -0.4854936f, -0.52406621f,
        0.0493288562f, 0.126683652f, -0.265223473f, -0.444493771f,
    });
    const auto class_ids = FloatTensor({12}, {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2});
    const auto metrics = cyxwiz::ComputeRetrievalMetrics(embeddings, class_ids, 2);
    CheckNear(metrics.recall_at_k, 1.0, 1e-9, "tied embeddings: Recall@2");
    CheckNear(metrics.mean_reciprocal_rank, 0.9583333333333334, 1e-6, "tied embeddings: MRR");
    CheckNear(metrics.nearest_neighbor_class_agreement, 0.9166666666666666, 1e-6,
              "tied embeddings: 1-NN agreement");
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

    bool rejected_non_finite_threshold = false;
    try {
        (void)cyxwiz::ComputePairDistanceMetrics(
            FloatTensor({1, 2}, {0.0f, 0.0f}),
            FloatTensor({1, 2}, {0.0f, 0.0f}),
            FloatTensor({1}, {0.0f}),
            cyxwiz::MetricLearningLabelConvention::
                ContrastiveZeroSimilarOneDissimilar,
            std::numeric_limits<double>::infinity());
    } catch (const std::invalid_argument&) {
        rejected_non_finite_threshold = true;
    }
    Check(rejected_non_finite_threshold,
          "pair metrics should reject non-finite thresholds");

    bool rejected_triplet_convention = false;
    try {
        (void)cyxwiz::ComputePairDistanceMetrics(
            FloatTensor({1, 2}, {0.0f, 0.0f}),
            FloatTensor({1, 2}, {0.0f, 0.0f}),
            FloatTensor({1}, {0.0f}),
            cyxwiz::MetricLearningLabelConvention::TripletNoLabels,
            0.5);
    } catch (const std::invalid_argument&) {
        rejected_triplet_convention = true;
    }
    Check(rejected_triplet_convention,
          "pair metrics should reject the triplet label convention");

    bool rejected_fractional_class_id = false;
    try {
        (void)cyxwiz::ComputeRetrievalMetrics(
            FloatTensor({2, 2}, {0.0f, 0.0f, 1.0f, 1.0f}),
            FloatTensor({2}, {1.5f, 1.0f}),
            1);
    } catch (const std::invalid_argument&) {
        rejected_fractional_class_id = true;
    }
    Check(rejected_fractional_class_id,
          "retrieval metrics should reject fractional Float32 class IDs");

    bool rejected_zero_k = false;
    try {
        (void)cyxwiz::ComputeRetrievalMetrics(
            FloatTensor({2, 2}, {0.0f, 0.0f, 1.0f, 1.0f}),
            IntTensor({2}, {1, 1}),
            0);
    } catch (const std::invalid_argument&) {
        rejected_zero_k = true;
    }
    Check(rejected_zero_k, "retrieval metrics should reject k=0");
}

// Embeddings of any rank run on the device (TOFIX140): a rank-5 batch scores
// like its [batch, features] view.
void TestHigherRankEmbeddings() {
    const std::vector<float> left_values = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f};
    const std::vector<float> right_values = {0.1f, 0.0f, 2.0f, 0.0f, 1.2f, 1.0f, 2.0f, 4.0f};
    const auto labels = FloatTensor({4}, {0.0f, 1.0f, 0.0f, 1.0f});
    const auto convention =
        cyxwiz::MetricLearningLabelConvention::ContrastiveZeroSimilarOneDissimilar;
    const auto flat = cyxwiz::ComputePairDistanceMetrics(
        FloatTensor({4, 2}, left_values), FloatTensor({4, 2}, right_values), labels, convention, 0.5);
    const auto rank5 = cyxwiz::ComputePairDistanceMetrics(
        FloatTensor({4, 1, 2, 1, 1}, left_values), FloatTensor({4, 1, 2, 1, 1}, right_values), labels,
        convention, 0.5);
    CheckNear(rank5.accuracy, flat.accuracy, 1e-9, "rank-5 pair accuracy");
    CheckNear(rank5.positive_distance_mean, flat.positive_distance_mean, 1e-6, "rank-5 positive mean");
    CheckNear(rank5.negative_distance_mean, flat.negative_distance_mean, 1e-6, "rank-5 negative mean");

    const std::vector<float> points = {0.0f, 0.0f, 0.1f, 0.0f, 5.0f, 5.0f, 5.1f, 5.0f, 0.0f, 5.0f};
    const auto ids = IntTensor({5}, {1, 1, 2, 2, 3});
    const auto retrieval = cyxwiz::ComputeRetrievalMetrics(FloatTensor({5, 1, 1, 2, 1}, points), ids, 1);
    CheckNear(retrieval.recall_at_k, 0.8, 1e-9, "rank-5 retrieval recall@1");
    CheckNear(retrieval.mean_reciprocal_rank, 0.8, 1e-9, "rank-5 retrieval MRR");
}

#ifndef NDEBUG
void TestDeviceResidencyTruth() {
    if (!cyxwiz::IsCurrentArrayFireBackendAvailable()) {
        std::cout << "SKIP: ArrayFire metric residency contract unavailable\n";
        return;
    }

    std::vector<cyxwiz::ArrayFireHostSyncEvent> host_sync_events;
    {
        const ScopedEventCapture capture(host_sync_events);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_sync_observer(
            &CaptureHostSync);
        (void)cyxwiz::ComputePairDistanceMetrics(
            DeviceFloatTensor({2, 2}, {0.0f, 0.0f, 1.0f, 1.0f}),
            DeviceFloatTensor({2, 2}, {0.1f, 0.0f, 3.0f, 1.0f}),
            DeviceFloatTensor({2}, {0.0f, 1.0f}),
            cyxwiz::MetricLearningLabelConvention::
                ContrastiveZeroSimilarOneDissimilar,
            0.5);
    }
    Check(host_sync_events.size() == 2,
          "pair metrics must read one validation scalar and one result vector");
    Check(host_sync_events[0].attribution_category ==
              "metric_input_validation" &&
              host_sync_events[0].bytes == sizeof(uint32_t),
          "pair metric label validation must read one attributed scalar");
    Check(host_sync_events[1].attribution_category ==
              "metric_scalar_readback" &&
              host_sync_events[1].bytes == 5 * sizeof(float),
          "pair metric result readback must contain five aggregates");

    host_sync_events.clear();
    {
        const ScopedEventCapture capture(host_sync_events);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_sync_observer(
            &CaptureHostSync);
        (void)cyxwiz::ComputeRetrievalMetrics(
            DeviceFloatTensor(
                {3, 2}, {0.0f, 0.0f, 0.1f, 0.0f, 5.0f, 5.0f}),
            DeviceIntTensor({3}, {1, 1, 2}),
            1);
    }
    Check(host_sync_events.size() == 1 &&
              host_sync_events.front().attribution_category ==
                  "metric_scalar_readback" &&
              host_sync_events.front().bytes == 3 * sizeof(float),
          "retrieval metrics must read exactly three attributed aggregates");
}
#endif

}  // namespace

int main() {
    TestContrastivePairDistanceMetrics();
    TestCosineConventionPairDistanceMetrics();
    TestRetrievalMetrics();
    TestRetrievalEffectiveKAndMRR();
    TestRetrievalTiedRelevantCandidates();
    TestMetricValidation();
    TestHigherRankEmbeddings();
#ifndef NDEBUG
    TestDeviceResidencyTruth();
#endif
    std::cout << "Metric-learning metrics passed\n";
    return 0;
}
