#include "core/metric_learning_inference_outputs.h"

#include "algorithms/arrayfire_backend_utils.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace {

#ifdef CYXWIZ_HAS_ARRAYFIRE
std::vector<cyxwiz::ArrayFireHostSyncEvent>* g_host_sync_events = nullptr;

void CaptureHostSync(const cyxwiz::ArrayFireHostSyncEvent& event) {
    if (g_host_sync_events != nullptr) {
        g_host_sync_events->push_back(event);
    }
}
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

void TestEmbeddingOutputResponse() {
    const auto embeddings = FloatTensor({2, 2, 2}, {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
    });
    const auto sample_ids = IntTensor({2}, {10, 20});
    const auto class_ids = IntTensor({2, 1}, {3, 4});

    const auto response = cyxwiz::BuildEmbeddingOutputResponse(
        embeddings, sample_ids, class_ids);

    Check(response.embedding_shape == std::vector<size_t>({2, 2}),
          "embedding output should preserve per-sample shape");
    Check(response.records.size() == 2,
          "embedding output should emit one record per sample");
    Check(response.records[0].embedding == std::vector<float>({
              1.0f, 2.0f, 3.0f, 4.0f}),
          "embedding output should flatten the first sample embedding");
    Check(response.records[1].has_sample_id &&
              response.records[1].sample_id == 20,
          "embedding output should preserve sample IDs");
    Check(response.records[1].has_class_id &&
              response.records[1].class_id == 4,
          "embedding output should preserve class IDs");
}

void TestPairScoreDistanceResponse() {
    const auto left = FloatTensor({2, 2}, {
        0.0f, 0.0f,
        1.0f, 1.0f,
    });
    const auto right = FloatTensor({2, 2}, {
        3.0f, 4.0f,
        1.0f, 3.0f,
    });
    const auto sample_a = IntTensor({2}, {101, 102});
    const auto sample_b = IntTensor({2}, {201, 202});
    const auto class_a = IntTensor({2}, {1, 2});
    const auto class_b = IntTensor({2}, {1, 3});

    const auto response = cyxwiz::BuildPairScoreOutputResponse(
        left,
        right,
        cyxwiz::PairScoreMode::EuclideanDistance,
        sample_a,
        sample_b,
        class_a,
        class_b);

    Check(response.mode == cyxwiz::PairScoreMode::EuclideanDistance,
          "pair score response should record distance mode");
    Check(response.records.size() == 2,
          "pair score output should emit one record per pair");
    CheckNear(response.records[0].distance, 5.0, 1e-9,
              "pair score should compute Euclidean distance");
    CheckNear(response.records[0].score, 5.0, 1e-9,
              "distance mode score should equal distance");
    Check(response.records[1].has_sample_ids &&
              response.records[1].sample_id_a == 102 &&
              response.records[1].sample_id_b == 202,
          "pair score should preserve paired sample IDs");
    Check(response.records[1].has_class_ids &&
              response.records[1].class_id_a == 2 &&
              response.records[1].class_id_b == 3,
          "pair score should preserve paired class IDs");
}

void TestPairScoreSimilarityModes() {
    const auto left = FloatTensor({2, 2}, {
        1.0f, 0.0f,
        1.0f, 1.0f,
    });
    const auto right = FloatTensor({2, 2}, {
        0.0f, 1.0f,
        2.0f, 2.0f,
    });

    const auto negative_distance = cyxwiz::BuildPairScoreOutputResponse(
        left,
        right,
        cyxwiz::PairScoreMode::NegativeEuclideanDistance);
    CheckNear(negative_distance.records[0].score,
              -std::sqrt(2.0),
              1e-9,
              "negative-distance mode should negate Euclidean distance");

    const auto cosine = cyxwiz::BuildPairScoreOutputResponse(
        left,
        right,
        cyxwiz::PairScoreMode::CosineSimilarity);
    CheckNear(cosine.records[0].score, 0.0, 1e-9,
              "cosine mode should score orthogonal embeddings as zero");
    CheckNear(cosine.records[1].score, 1.0, 1e-9,
              "cosine mode should score same-direction embeddings as one");
}

void TestScoreModeParsing() {
    Check(cyxwiz::ParsePairScoreMode("") ==
              cyxwiz::PairScoreMode::EuclideanDistance,
          "empty score mode should default to Euclidean distance");
    Check(cyxwiz::ParsePairScoreMode("negative_distance") ==
              cyxwiz::PairScoreMode::NegativeEuclideanDistance,
          "negative_distance should parse as negative Euclidean distance");
    Check(cyxwiz::ParsePairScoreMode("cosine_similarity") ==
              cyxwiz::PairScoreMode::CosineSimilarity,
          "cosine_similarity should parse as cosine similarity");

    bool rejected_unknown_mode = false;
    try {
        (void)cyxwiz::ParsePairScoreMode("class_probability");
    } catch (const std::invalid_argument&) {
        rejected_unknown_mode = true;
    }
    Check(rejected_unknown_mode,
          "unknown pair score mode should be rejected");
}

void TestOutputJsonContracts() {
    const auto embedding_response = cyxwiz::BuildEmbeddingOutputResponse(
        FloatTensor({1, 2}, {0.25f, 0.75f}),
        IntTensor({1}, {123}),
        IntTensor({1}, {7}));
    const nlohmann::json embedding_json =
        cyxwiz::EmbeddingOutputResponseToJson(embedding_response);

    Check(embedding_json["output_type"] == "embedding",
          "embedding JSON should advertise output type");
    Check(embedding_json["embedding_shape"].size() == 1 &&
              embedding_json["embedding_shape"][0] == 2,
          "embedding JSON should preserve per-sample shape");
    Check(embedding_json["records"].size() == 1,
          "embedding JSON should emit records array");
    Check(embedding_json["records"][0]["sample_id"] == 123 &&
              embedding_json["records"][0]["class_id"] == 7,
          "embedding JSON should preserve metadata IDs");
    Check(embedding_json["records"][0]["embedding"][1] == 0.75f,
          "embedding JSON should serialize flattened vectors");

    const auto pair_response = cyxwiz::BuildPairScoreOutputResponse(
        FloatTensor({1, 2}, {1.0f, 0.0f}),
        FloatTensor({1, 2}, {0.0f, 1.0f}),
        cyxwiz::PairScoreMode::CosineSimilarity,
        IntTensor({1}, {11}),
        IntTensor({1}, {12}),
        IntTensor({1}, {3}),
        IntTensor({1}, {4}));
    const nlohmann::json pair_json =
        cyxwiz::PairScoreOutputResponseToJson(pair_response);

    Check(pair_json["output_type"] == "pair_score",
          "pair-score JSON should advertise output type");
    Check(pair_json["score_mode"] == "cosine_similarity",
          "pair-score JSON should serialize score mode");
    Check(pair_json["records"].size() == 1,
          "pair-score JSON should emit records array");
    Check(pair_json["records"][0]["sample_id_a"] == 11 &&
              pair_json["records"][0]["sample_id_b"] == 12,
          "pair-score JSON should preserve sample IDs");
    Check(pair_json["records"][0]["class_id_a"] == 3 &&
              pair_json["records"][0]["class_id_b"] == 4,
          "pair-score JSON should preserve class IDs");
    CheckNear(pair_json["records"][0]["score"].get<double>(),
              0.0,
              1e-9,
              "pair-score JSON should serialize score value");
}

void TestOutputValidation() {
    bool rejected_embedding_metadata = false;
    try {
        (void)cyxwiz::BuildEmbeddingOutputResponse(
            FloatTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}),
            IntTensor({1}, {10}));
    } catch (const std::invalid_argument&) {
        rejected_embedding_metadata = true;
    }
    Check(rejected_embedding_metadata,
          "embedding output should reject mismatched sample IDs");

    bool rejected_pair_shape = false;
    try {
        (void)cyxwiz::BuildPairScoreOutputResponse(
            FloatTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}),
            FloatTensor({2, 3}, {
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f}));
    } catch (const std::invalid_argument&) {
        rejected_pair_shape = true;
    }
    Check(rejected_pair_shape,
          "pair score output should reject mismatched embedding shapes");

    bool rejected_zero_cosine = false;
    try {
        (void)cyxwiz::BuildPairScoreOutputResponse(
            FloatTensor({1, 2}, {0.0f, 0.0f}),
            FloatTensor({1, 2}, {1.0f, 0.0f}),
            cyxwiz::PairScoreMode::CosineSimilarity);
    } catch (const std::invalid_argument&) {
        rejected_zero_cosine = true;
    }
    Check(rejected_zero_cosine,
          "cosine pair score should reject zero vectors");
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
void TestDeviceOutputMaterializationAttribution() {
    if (!cyxwiz::IsCurrentArrayFireBackendAvailable()) {
        std::cout << "SKIP: ArrayFire output attribution unavailable\n";
        return;
    }

    const auto embeddings =
        DeviceFloatTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const auto sample_ids = DeviceIntTensor({2}, {10, 20});
    const auto class_ids = DeviceIntTensor({2}, {1, 2});
    std::vector<cyxwiz::ArrayFireHostSyncEvent> events;
    g_host_sync_events = &events;
    {
        const cyxwiz::ScopedArrayFireHostSyncObserver observer(
            &CaptureHostSync);
        (void)cyxwiz::BuildEmbeddingOutputResponse(
            embeddings, sample_ids, class_ids);
    }
    g_host_sync_events = nullptr;
    Check(events.size() == 3,
          "embedding output should materialize exactly three tensors");
    for (const auto& event : events) {
        Check(event.attribution_category == "output_materialization" &&
                  event.attribution_operation ==
                      "MetricLearning::EmbeddingOutput",
              "embedding output reads must be attributed");
    }

    const auto left = DeviceFloatTensor({2, 2}, {
        0.0f, 0.0f,
        1.0f, 1.0f,
    });
    const auto right = DeviceFloatTensor({2, 2}, {
        3.0f, 4.0f,
        1.0f, 3.0f,
    });
    events.clear();
    g_host_sync_events = &events;
    {
        const cyxwiz::ScopedArrayFireHostSyncObserver observer(
            &CaptureHostSync);
        (void)cyxwiz::BuildPairScoreOutputResponse(
            left, right, cyxwiz::PairScoreMode::EuclideanDistance);
    }
    g_host_sync_events = nullptr;
    Check(events.size() == 2,
          "pair output should materialize exactly two embedding tensors");
    for (const auto& event : events) {
        Check(event.attribution_category == "output_materialization" &&
                  event.attribution_operation ==
                      "MetricLearning::PairScoreOutput",
              "pair output reads must be attributed");
    }
}
#endif

}  // namespace

int main() {
    TestEmbeddingOutputResponse();
    TestPairScoreDistanceResponse();
    TestPairScoreSimilarityModes();
    TestScoreModeParsing();
    TestOutputJsonContracts();
    TestOutputValidation();
#ifdef CYXWIZ_HAS_ARRAYFIRE
    TestDeviceOutputMaterializationAttribution();
#endif
    std::cout << "Metric-learning inference outputs passed\n";
    return 0;
}
