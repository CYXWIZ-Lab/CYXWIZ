// Sequence-tag metrics on the device (TOFIX140 F3): argmax over tag logits,
// token accuracy with ignore_index, BIO span precision / recall / F1 and the
// language-model next-token count. Hand-checked values; runs on the device
// CYXWIZ_TEST_ARRAYFIRE_BACKEND selects.
#include "core/sequence_tag_metrics.h"
#include "computation_truth/test_device_selection.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

bool Near(double a, double b) { return std::fabs(a - b) < 1e-9; }

template <typename Fn>
void CheckThrows(Fn&& fn, const std::string& fragment, const std::string& message) {
    try {
        fn();
    } catch (const std::exception& e) {
        Check(std::string(e.what()).find(fragment) != std::string::npos, message + ": " + e.what());
        return;
    }
    Check(false, message + ": no exception");
}

const std::vector<std::string> kLabels = {"O", "B-PER", "I-PER", "B-LOC", "I-LOC"};

void CheckArgmax() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    // [batch 2, seq 2, tags 3]
    const std::vector<float> logits = {
        0.5f, 0.5f, 0.1f,   // tie between 0 and 1 -> 0
        0.1f, 0.9f, 0.9f,   // tie between 1 and 2 -> 1
        nan, nan, nan,      // no maximum -> 0
        -1.0f, -2.0f, 3.0f, // -> 2
    };
    const cyxwiz::Tensor ids =
        cyxwiz::ArgmaxSequenceTagLogits(cyxwiz::Tensor({2, 2, 3}, logits.data(), cyxwiz::DataType::Float32));
    Check(ids.Shape() == std::vector<size_t>({2, 2}) && ids.GetDataType() == cyxwiz::DataType::Int64,
          "argmax gives Int64 [batch, seq]");
    const auto* host = ids.ReadData<int64_t>();
    Check(host[0] == 0 && host[1] == 1 && host[2] == 0 && host[3] == 2,
          "argmax picks the first maximum and maps an all-NaN row to 0");
}

void CheckSpanMetrics() {
    // Row 0 gold: PER[0,2) LOC[3,4), last position ignored. Row 1 gold: LOC[1,4).
    const std::vector<int64_t> gold = {1, 2, 0, 3, -100, 0, 3, 4, 4, 0};
    // Row 0 predicted: PER[0,2) PER[3,4) (the I-PER under the ignored position is
    // masked). Row 1 predicted: LOC[1,3).
    const std::vector<int64_t> predicted = {1, 2, 0, 1, 2, 0, 3, 4, 0, 0};
    const auto metrics = cyxwiz::ComputeSequenceTagMetrics(
        cyxwiz::Tensor({2, 5}, predicted.data(), cyxwiz::DataType::Int64),
        cyxwiz::Tensor({2, 5}, gold.data(), cyxwiz::DataType::Int64), kLabels);
    Check(metrics.total_tokens == 9 && metrics.correct_tokens == 7 && Near(metrics.token_accuracy, 7.0 / 9.0),
          "token accuracy skips ignored positions: " + std::to_string(metrics.correct_tokens) + "/" +
              std::to_string(metrics.total_tokens));
    Check(metrics.predicted_entities == 3 && metrics.gold_entities == 3 && metrics.matched_entities == 1,
          "span counts: " + std::to_string(metrics.predicted_entities) + " predicted, " +
              std::to_string(metrics.gold_entities) + " gold, " + std::to_string(metrics.matched_entities) + " matched");
    Check(Near(metrics.entity_precision, 1.0 / 3.0) && Near(metrics.entity_recall, 1.0 / 3.0) &&
              Near(metrics.entity_f1, 1.0 / 3.0),
          "span precision / recall / F1");

    // Int32 ids, from logits: the same result through the device argmax.
    std::vector<int32_t> gold32;
    for (const int64_t id : gold) gold32.push_back(static_cast<int32_t>(id));
    std::vector<float> logits(10 * 5, 0.0f);
    for (size_t i = 0; i < predicted.size(); ++i) logits[i * 5 + static_cast<size_t>(predicted[i])] = 1.0f;
    const auto from_logits = cyxwiz::ComputeSequenceTagMetricsFromLogits(
        cyxwiz::Tensor({2, 5, 5}, logits.data(), cyxwiz::DataType::Float32),
        cyxwiz::Tensor({2, 5}, gold32.data(), cyxwiz::DataType::Int32), kLabels);
    Check(from_logits.correct_tokens == 7 && from_logits.matched_entities == 1 && from_logits.predicted_entities == 3,
          "logits and Int32 gold ids give the same metrics");
}

void CheckNextTokenCount() {
    // [batch 1, seq 3, vocab 4]
    const std::vector<float> logits = {
        0.0f, 1.0f, 2.0f, 2.0f,  // argmax 2 (tie with 3), target 2: correct
        5.0f, 0.0f, 0.0f, 0.0f,  // target ignored
        0.0f, 0.0f, 0.0f, 1.0f,  // argmax 3, target 1: wrong
    };
    const std::vector<int64_t> targets = {2, -100, 1};
    const auto count = cyxwiz::CountNextTokenAccuracyFromLogits(
        cyxwiz::Tensor({1, 3, 4}, logits.data(), cyxwiz::DataType::Float32),
        cyxwiz::Tensor({1, 3}, targets.data(), cyxwiz::DataType::Int64), -100);
    Check(count.valid == 2 && count.correct == 1, "next-token count: " + std::to_string(count.correct) + "/" +
                                                      std::to_string(count.valid));

    const std::vector<int64_t> outside = {2, -100, 4};
    CheckThrows([&] {
        cyxwiz::CountNextTokenAccuracyFromLogits(
            cyxwiz::Tensor({1, 3, 4}, logits.data(), cyxwiz::DataType::Float32),
            cyxwiz::Tensor({1, 3}, outside.data(), cyxwiz::DataType::Int64), -100);
    }, "outside the vocabulary", "a target past the vocabulary is refused");

    std::vector<float> with_nan = logits;
    with_nan[5] = std::numeric_limits<float>::quiet_NaN();
    CheckThrows([&] {
        cyxwiz::CountNextTokenAccuracyFromLogits(
            cyxwiz::Tensor({1, 3, 4}, with_nan.data(), cyxwiz::DataType::Float32),
            cyxwiz::Tensor({1, 3}, targets.data(), cyxwiz::DataType::Int64), -100);
    }, "finite logits", "non-finite logits are refused");
}

}  // namespace

int main() {
    Check(cyxwiz::test::SelectTestDeviceFromEnvironment(), "requested test device");
    CheckArgmax();
    CheckSpanMetrics();
    CheckNextTokenCount();
    std::cout << "Sequence tag metrics passed\n";
    return 0;
}
