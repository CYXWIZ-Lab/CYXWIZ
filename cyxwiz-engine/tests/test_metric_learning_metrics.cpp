#include "core/metric_learning_metrics.h"

#include "algorithms/arrayfire_backend_utils.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace {

#if defined(CYXWIZ_HAS_ARRAYFIRE) && !defined(NDEBUG)
constexpr const char* kForceFallbackEnv =
    "CYXWIZ_TEST_FORCE_ARRAYFIRE_FALLBACK";

std::vector<cyxwiz::ArrayFireNativeCpuFallbackEvent>* g_fallback_events =
    nullptr;
std::vector<cyxwiz::ArrayFireHostSyncEvent>* g_host_sync_events = nullptr;

void CaptureFallback(
    const cyxwiz::ArrayFireNativeCpuFallbackEvent& event) {
    if (g_fallback_events != nullptr) {
        g_fallback_events->push_back(event);
    }
}

void CaptureHostSync(const cyxwiz::ArrayFireHostSyncEvent& event) {
    if (g_host_sync_events != nullptr) {
        g_host_sync_events->push_back(event);
    }
}

void SetFallbackEnv(const char* value) {
#ifdef _WIN32
    _putenv_s(kForceFallbackEnv, value);
#else
    setenv(kForceFallbackEnv, value, 1);
#endif
}

class ScopedFallbackEnv {
public:
    explicit ScopedFallbackEnv(const char* value) {
        const char* previous = std::getenv(kForceFallbackEnv);
        if (previous != nullptr) {
            had_previous_ = true;
            previous_ = previous;
        }
        SetFallbackEnv(value);
    }

    ~ScopedFallbackEnv() {
        SetFallbackEnv(had_previous_ ? previous_.c_str() : "");
    }

private:
    bool had_previous_ = false;
    std::string previous_;
};

class ScopedEventCapture {
public:
    ScopedEventCapture(
        std::vector<cyxwiz::ArrayFireNativeCpuFallbackEvent>& fallback_events,
        std::vector<cyxwiz::ArrayFireHostSyncEvent>& host_sync_events) {
        g_fallback_events = &fallback_events;
        g_host_sync_events = &host_sync_events;
    }

    ~ScopedEventCapture() {
        g_fallback_events = nullptr;
        g_host_sync_events = nullptr;
    }
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

#ifdef CYXWIZ_HAS_ARRAYFIRE
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
#endif

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

#if defined(CYXWIZ_HAS_ARRAYFIRE) && !defined(NDEBUG)
void TestStrictAndCompatibleFallbackTruth() {
    if (!cyxwiz::IsCurrentArrayFireBackendAvailable()) {
        std::cout << "SKIP: ArrayFire metric fallback contract unavailable\n";
        return;
    }

    const auto run_pair = [] {
        return cyxwiz::ComputePairDistanceMetrics(
            DeviceFloatTensor({2, 2}, {0.0f, 0.0f, 1.0f, 1.0f}),
            DeviceFloatTensor({2, 2}, {0.1f, 0.0f, 3.0f, 1.0f}),
            DeviceFloatTensor({2}, {0.0f, 1.0f}),
            cyxwiz::MetricLearningLabelConvention::
                ContrastiveZeroSimilarOneDissimilar,
            0.5);
    };
    const auto run_retrieval = [] {
        return cyxwiz::ComputeRetrievalMetrics(
            DeviceFloatTensor(
                {3, 2}, {0.0f, 0.0f, 0.1f, 0.0f, 5.0f, 5.0f}),
            DeviceIntTensor({3}, {1, 1, 2}),
            1);
    };

    const auto require_strict_rejection = [](const char* operation,
                                              const auto& invoke) {
        std::vector<cyxwiz::ArrayFireNativeCpuFallbackEvent> fallback_events;
        std::vector<cyxwiz::ArrayFireHostSyncEvent> host_sync_events;
        bool rejected = false;
        {
            const ScopedEventCapture capture(fallback_events, host_sync_events);
            const ScopedFallbackEnv forced(operation);
            const cyxwiz::ScopedArrayFireFallbackPolicy strict(
                cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
            const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver
                fallback_observer(&CaptureFallback);
            const cyxwiz::ScopedArrayFireHostSyncObserver host_sync_observer(
                &CaptureHostSync);
            try {
                (void)invoke();
            } catch (const std::runtime_error&) {
                rejected = true;
            }
        }
        Check(rejected, std::string(operation) +
                            " strict policy must reject forced fallback");
        Check(fallback_events.size() == 1,
              std::string(operation) + " must record one fallback attempt");
        Check(fallback_events.front().operation_name == operation &&
                  fallback_events.front().fallback_forbidden,
              std::string(operation) +
                  " must record the forbidden operation identity");
        Check(host_sync_events.empty(), std::string(operation) +
                                          " must reject before host access");
    };

    require_strict_rejection("MetricLearning::PairDistance", run_pair);
    require_strict_rejection("MetricLearning::Retrieval", run_retrieval);

    const auto expected_pair = run_pair();
    std::vector<cyxwiz::ArrayFireNativeCpuFallbackEvent> pair_fallback_events;
    std::vector<cyxwiz::ArrayFireHostSyncEvent> pair_host_sync_events;
    cyxwiz::PairMetricResult actual_pair;
    {
        const ScopedEventCapture capture(
            pair_fallback_events, pair_host_sync_events);
        const ScopedFallbackEnv forced("MetricLearning::PairDistance");
        const cyxwiz::ScopedArrayFireFallbackPolicy compatible(
            cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CaptureFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_sync_observer(
            &CaptureHostSync);
        actual_pair = run_pair();
    }
    Check(pair_fallback_events.size() == 1 &&
              !pair_fallback_events.front().fallback_forbidden,
          "pair metric compatible policy must record one allowed fallback");
    CheckNear(actual_pair.accuracy, expected_pair.accuracy, 1e-6,
              "pair metric fallback accuracy must match ArrayFire");
    CheckNear(actual_pair.positive_distance_mean,
              expected_pair.positive_distance_mean,
              1e-6,
              "pair metric fallback positive mean must match ArrayFire");
    CheckNear(actual_pair.negative_distance_mean,
              expected_pair.negative_distance_mean,
              1e-6,
              "pair metric fallback negative mean must match ArrayFire");
    Check(pair_host_sync_events.size() == 3,
          "pair metric fallback must materialize exactly three inputs");
    for (const auto& event : pair_host_sync_events) {
        Check(event.attribution_category == "metric_cpu_path" &&
                  event.attribution_operation ==
                      "MetricLearning::PairDistance",
              "pair metric fallback host access must be attributed");
    }

    const auto expected_retrieval = run_retrieval();
    std::vector<cyxwiz::ArrayFireNativeCpuFallbackEvent>
        retrieval_fallback_events;
    std::vector<cyxwiz::ArrayFireHostSyncEvent> retrieval_host_sync_events;
    cyxwiz::RetrievalMetricResult actual_retrieval;
    {
        const ScopedEventCapture capture(
            retrieval_fallback_events, retrieval_host_sync_events);
        const ScopedFallbackEnv forced("MetricLearning::Retrieval");
        const cyxwiz::ScopedArrayFireFallbackPolicy compatible(
            cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CaptureFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_sync_observer(
            &CaptureHostSync);
        actual_retrieval = run_retrieval();
    }
    Check(retrieval_fallback_events.size() == 1 &&
              !retrieval_fallback_events.front().fallback_forbidden,
          "retrieval compatible policy must record one allowed fallback");
    CheckNear(actual_retrieval.recall_at_k,
              expected_retrieval.recall_at_k,
              1e-6,
              "retrieval fallback recall must match ArrayFire");
    CheckNear(actual_retrieval.mean_reciprocal_rank,
              expected_retrieval.mean_reciprocal_rank,
              1e-6,
              "retrieval fallback MRR must match ArrayFire");
    Check(retrieval_host_sync_events.size() == 2,
          "retrieval fallback must materialize exactly two inputs");
    for (const auto& event : retrieval_host_sync_events) {
        Check(event.attribution_category == "metric_cpu_path" &&
                  event.attribution_operation == "MetricLearning::Retrieval",
              "retrieval fallback host access must be attributed");
    }
}

void TestDeviceResidencyTruth() {
    if (!cyxwiz::IsCurrentArrayFireBackendAvailable()) {
        std::cout << "SKIP: ArrayFire metric residency contract unavailable\n";
        return;
    }

    std::vector<cyxwiz::ArrayFireNativeCpuFallbackEvent> fallback_events;
    std::vector<cyxwiz::ArrayFireHostSyncEvent> host_sync_events;
    {
        const ScopedEventCapture capture(fallback_events, host_sync_events);
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CaptureFallback);
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
    Check(fallback_events.empty(),
          "device-current pair metrics must not use native fallback");
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

    fallback_events.clear();
    host_sync_events.clear();
    {
        const ScopedEventCapture capture(fallback_events, host_sync_events);
        const cyxwiz::ScopedArrayFireFallbackPolicy strict(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &CaptureFallback);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_sync_observer(
            &CaptureHostSync);
        (void)cyxwiz::ComputeRetrievalMetrics(
            DeviceFloatTensor(
                {3, 2}, {0.0f, 0.0f, 0.1f, 0.0f, 5.0f, 5.0f}),
            DeviceIntTensor({3}, {1, 1, 2}),
            1);
    }
    Check(fallback_events.empty(),
          "device-current retrieval metrics must not use native fallback");
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
    TestMetricValidation();
#if defined(CYXWIZ_HAS_ARRAYFIRE) && !defined(NDEBUG)
    TestStrictAndCompatibleFallbackTruth();
    TestDeviceResidencyTruth();
#endif
    std::cout << "Metric-learning metrics passed\n";
    return 0;
}
