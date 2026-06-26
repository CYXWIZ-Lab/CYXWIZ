#include "inference/metric_learning_inference_input.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

template <typename Fn>
void CheckThrows(Fn&& fn, const std::string& message) {
    bool threw = false;
    try {
        fn();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Check(threw, message);
}

void TestEmbeddingInputParser() {
    const json request = {
        {"input", {{1.0, 2.0}, {3.0, 4.0}}},
        {"sample_ids", {101, 102}},
        {"class_ids", {7, 8}},
    };

    const auto parsed = cyxwiz::ParseMetricEmbeddingInferenceInput(request);
    Check(parsed.input.Shape() == std::vector<size_t>({2, 2}),
          "embedding parser should preserve 2D input shape");
    Check(parsed.input.At(0, 1) == 2.0f,
          "embedding parser should preserve numeric input values");
    Check(parsed.has_sample_ids && parsed.has_class_ids,
          "embedding parser should mark optional metadata as present");
    Check(parsed.sample_ids.Shape() == std::vector<size_t>({2}) &&
              parsed.sample_ids.AtAs<int64_t>(1) == 102,
          "embedding parser should parse sample IDs as int64 tensor");
    Check(parsed.class_ids.AtAs<int64_t>(0) == 7,
          "embedding parser should parse class IDs");
}

void TestEmbeddingOneDimensionalInput() {
    const json request = {
        {"input", {1.0, 2.0, 3.0}},
    };

    const auto parsed = cyxwiz::ParseMetricEmbeddingInferenceInput(request);
    Check(parsed.input.Shape() == std::vector<size_t>({1, 3}),
          "1D embedding input should become a single-row batch");
    Check(!parsed.has_sample_ids && !parsed.has_class_ids,
          "missing embedding metadata should stay absent");
}

void TestPairScoreInputParser() {
    const json request = {
        {"input_a", {{1.0, 0.0}, {0.0, 1.0}}},
        {"input_b", {{0.0, 1.0}, {1.0, 0.0}}},
        {"score_mode", "cosine_similarity"},
        {"sample_id_a", {11, 12}},
        {"sample_id_b", {21, 22}},
        {"class_id_a", {3, 4}},
        {"class_id_b", {5, 6}},
    };

    const auto parsed = cyxwiz::ParseMetricPairScoreInferenceInput(request);
    Check(parsed.input_a.Shape() == std::vector<size_t>({2, 2}) &&
              parsed.input_b.Shape() == std::vector<size_t>({2, 2}),
          "pair parser should preserve branch input shapes");
    Check(parsed.score_mode == cyxwiz::PairScoreMode::CosineSimilarity,
          "pair parser should parse score mode");
    Check(parsed.has_sample_ids && parsed.has_class_ids,
          "pair parser should mark paired metadata as present");
    Check(parsed.sample_id_a.AtAs<int64_t>(0) == 11 &&
              parsed.sample_id_b.AtAs<int64_t>(1) == 22,
          "pair parser should parse paired sample IDs");
    Check(parsed.class_id_a.AtAs<int64_t>(1) == 4 &&
              parsed.class_id_b.AtAs<int64_t>(0) == 5,
          "pair parser should parse paired class IDs");
}

void TestPairNestedInputParser() {
    const json request = {
        {"input", {
            {"a", {1.0, 2.0}},
            {"b", {3.0, 4.0}},
        }},
        {"score_mode", "negative_distance"},
    };

    const auto parsed = cyxwiz::ParseMetricPairScoreInferenceInput(request);
    Check(parsed.input_a.Shape() == std::vector<size_t>({1, 2}) &&
              parsed.input_b.Shape() == std::vector<size_t>({1, 2}),
          "pair parser should accept nested input.a/input.b");
    Check(parsed.score_mode ==
              cyxwiz::PairScoreMode::NegativeEuclideanDistance,
          "pair parser should accept distance aliases");
}

void TestInputValidation() {
    CheckThrows([] {
        (void)cyxwiz::ParseMetricEmbeddingInferenceInput(
            json{{"input", {{1.0, 2.0}, {3.0}}}});
    }, "embedding parser should reject ragged rows");

    CheckThrows([] {
        (void)cyxwiz::ParseMetricEmbeddingInferenceInput(
            json{{"input", {{1.0, 2.0}}},
                 {"sample_ids", {1, 2}}});
    }, "embedding parser should reject metadata length mismatch");

    CheckThrows([] {
        (void)cyxwiz::ParseMetricPairScoreInferenceInput(
            json{{"input_a", {{1.0, 2.0}}},
                 {"input_b", {{1.0, 2.0, 3.0}}}});
    }, "pair parser should reject mismatched branch shapes");

    CheckThrows([] {
        (void)cyxwiz::ParseMetricPairScoreInferenceInput(
            json{{"input_a", {{1.0, 2.0}}},
                 {"input_b", {{3.0, 4.0}}},
                 {"sample_id_a", {1}}});
    }, "pair parser should require paired sample metadata");

    CheckThrows([] {
        (void)cyxwiz::ParseMetricPairScoreInferenceInput(
            json{{"input_a", {{1.0, 2.0}}},
                 {"input_b", {{3.0, 4.0}}},
                 {"score_mode", "class_probability"}});
    }, "pair parser should reject unknown score modes");
}

}  // namespace

int main() {
    TestEmbeddingInputParser();
    TestEmbeddingOneDimensionalInput();
    TestPairScoreInputParser();
    TestPairNestedInputParser();
    TestInputValidation();
    std::cout << "Metric-learning inference input parsing passed\n";
    return 0;
}
