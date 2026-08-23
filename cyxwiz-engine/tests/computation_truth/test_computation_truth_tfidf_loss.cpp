#include "../../src/core/node_executors/tfidf_vectorizer_operator.h"
#include "../../src/core/materialization_memory_guard.h"

#include <cyxwiz/layers/linear.h>
#include <cyxwiz/loss.h>
#include <cyxwiz/optimizers/adam.h>
#include <cyxwiz/tensor.h>

#include <arrow/api.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

void CheckNear(float actual, float expected, float tolerance,
               const std::string& message) {
    if (std::fabs(actual - expected) > tolerance) {
        std::ostringstream ss;
        ss << message << " expected=" << expected << " actual=" << actual
           << " tolerance=" << tolerance;
        Check(false, ss.str());
    }
}

std::shared_ptr<arrow::Array> FinishStringArray(
    const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    for (const auto& value : values) {
        auto st = builder.Append(value);
        Check(st.ok(), st.ToString());
    }
    std::shared_ptr<arrow::Array> array;
    auto st = builder.Finish(&array);
    Check(st.ok(), st.ToString());
    return array;
}

float ReadFloatValue(const std::shared_ptr<arrow::Table>& table,
                     const std::string& column_name,
                     int64_t row) {
    auto column = table->GetColumnByName(column_name);
    Check(column != nullptr, "missing float column " + column_name);
    Check(column->num_chunks() == 1, "expected one chunk for " + column_name);
    auto array = std::static_pointer_cast<arrow::FloatArray>(column->chunk(0));
    return array->Value(row);
}

int32_t ReadIntValue(const std::shared_ptr<arrow::Table>& table,
                     const std::string& column_name,
                     int64_t row) {
    auto column = table->GetColumnByName(column_name);
    Check(column != nullptr, "missing int column " + column_name);
    Check(column->num_chunks() == 1, "expected one chunk for " + column_name);
    auto array = std::static_pointer_cast<arrow::Int32Array>(column->chunk(0));
    return array->Value(row);
}

float ReferenceCrossEntropyMean(const std::vector<float>& logits,
                                const std::vector<int64_t>& labels,
                                size_t batch,
                                size_t classes) {
    float total = 0.0f;
    for (size_t r = 0; r < batch; ++r) {
        const size_t base = r * classes;
        float max_logit = logits[base];
        for (size_t c = 1; c < classes; ++c) {
            max_logit = std::max(max_logit, logits[base + c]);
        }
        float sum_exp = 0.0f;
        for (size_t c = 0; c < classes; ++c) {
            sum_exp += std::exp(logits[base + c] - max_logit);
        }
        const float log_sum_exp = max_logit + std::log(sum_exp);
        total += -logits[base + static_cast<size_t>(labels[r])] + log_sum_exp;
    }
    return total / static_cast<float>(batch);
}

void TestMaterializationMemoryGuardThresholds() {
    const auto estimate = cyxwiz::EstimateDenseMaterializationMemory(
        10, 10, static_cast<uint64_t>(sizeof(float)));
    Check(estimate.raw_output_bytes == 400,
          "dense estimate should report raw float matrix bytes");
    Check(estimate.temporary_bytes == 800,
          "dense estimate should include conservative temporary bytes");
    Check(estimate.arrow_overhead_bytes == 50,
          "dense estimate should include Arrow overhead bytes");
    Check(estimate.estimated_peak_bytes == 1250,
          "dense estimate should report conservative peak bytes");

    cyxwiz::MaterializationMemorySnapshot snapshot;
    snapshot.detected = true;
    snapshot.available_bytes = 10000;
    auto decision = cyxwiz::EvaluateMaterializationMemory(estimate, snapshot);
    Check(decision.risk == cyxwiz::MaterializationMemoryRisk::Safe,
          "small estimate should be safe");
    Check(!decision.blocked, "safe estimate should not block");

    snapshot.available_bytes = 2000;
    decision = cyxwiz::EvaluateMaterializationMemory(estimate, snapshot);
    Check(decision.risk == cyxwiz::MaterializationMemoryRisk::Warning,
          "estimate above warning threshold should warn");
    Check(!decision.blocked, "warning estimate should not block");

    snapshot.available_bytes = 1600;
    decision = cyxwiz::EvaluateMaterializationMemory(estimate, snapshot);
    Check(decision.risk == cyxwiz::MaterializationMemoryRisk::Risky,
          "estimate above risky threshold should be risky");
    Check(!decision.blocked, "risky estimate is visible but not blocked yet");

    snapshot.available_bytes = 1300;
    decision = cyxwiz::EvaluateMaterializationMemory(estimate, snapshot);
    Check(decision.risk == cyxwiz::MaterializationMemoryRisk::Blocked,
          "estimate above blocked threshold should block");
    Check(decision.blocked, "blocked estimate should stop materialization");

    const auto overflow = cyxwiz::EstimateDenseMaterializationMemory(
        std::numeric_limits<uint64_t>::max(), 2, sizeof(float));
    Check(overflow.overflow, "overflowing estimate should be marked");
    decision = cyxwiz::EvaluateMaterializationMemory(overflow, snapshot);
    Check(decision.blocked, "overflowing estimate should block");

    cyxwiz::MaterializationMemoryPolicy hard_limit_policy;
    hard_limit_policy.hard_limit_bytes = 1000;
    snapshot.available_bytes = 10000;
    decision = cyxwiz::EvaluateMaterializationMemory(
        estimate, snapshot, hard_limit_policy);
    Check(decision.blocked,
          "configured hard limit should block independently of available RAM");
    Check(decision.reason.find("configured hard limit") != std::string::npos,
          "hard-limit decision should explain the configured limit");

    cyxwiz::MaterializationMemoryPolicy unknown_memory_policy;
    unknown_memory_policy.fallback_available_bytes = 0;
    snapshot = {};
    decision = cyxwiz::EvaluateMaterializationMemory(
        estimate, snapshot, unknown_memory_policy);
    Check(decision.blocked,
          "unknown memory without a conservative fallback should fail closed");
    Check(decision.reason.find("unknown") != std::string::npos,
          "unknown-memory decision should explain missing system memory");

    constexpr uint64_t kAcceptanceRows = 53043;
    const auto tfidf_5000 = cyxwiz::EstimateDenseMaterializationMemory(
        kAcceptanceRows, 5000, static_cast<uint64_t>(sizeof(float)));
    const auto tfidf_8000 = cyxwiz::EstimateDenseMaterializationMemory(
        kAcceptanceRows, 8000, static_cast<uint64_t>(sizeof(float)));
    Check(!tfidf_5000.overflow &&
              tfidf_5000.raw_output_bytes == 1060860000ULL &&
              tfidf_5000.estimated_peak_bytes == 3315187500ULL,
          "5,000-feature TF-IDF acceptance estimate should remain exact");
    Check(!tfidf_8000.overflow &&
              tfidf_8000.raw_output_bytes == 1697376000ULL &&
              tfidf_8000.estimated_peak_bytes == 5304300000ULL,
          "8,000-feature TF-IDF acceptance estimate should remain exact");
}

void TestTFIDFBlocksBeforeAllocationHeavyWork() {
    auto text = FinishStringArray({"alpha beta", "beta gamma"});
    auto input = arrow::Table::Make(
        arrow::schema({arrow::field("text", arrow::utf8())}), {text}, 2);

    cyxwiz::TFIDFVectorizerOperator op;
    std::string error;
    Check(op.Configure({
        {"text_col", "text"},
        {"max_features", "4"},
    }, error), error);

    cyxwiz::MaterializationMemoryContext memory_context;
    memory_context.policy.hard_limit_bytes = 1;
    memory_context.snapshot_override = cyxwiz::MaterializationMemorySnapshot{
        1024, 1024, true};
    op.SetMaterializationMemoryContext(memory_context);

    std::vector<cyxwiz::PipelineOperatorProgress> progress_events;
    op.SetProgressCallback(
        [&](const cyxwiz::PipelineOperatorProgress& event) {
            progress_events.push_back(event);
        });

    const auto result = op.Apply(input);
    Check(!result.ok(), "TF-IDF should reject a request above the hard limit");
    Check(result.status().IsCapacityError(),
          "blocked TF-IDF should return a structured capacity error");
    Check(progress_events.size() == 1,
          "blocked TF-IDF should stop after its preflight event");
    Check(progress_events.front().stage == "TF-IDF memory preflight",
          "blocked TF-IDF should identify the preflight stage");
    Check(progress_events.front().status == "blocked",
          "blocked TF-IDF should expose blocked status");
    Check(progress_events.front().memory_risk_level == "blocked",
          "blocked TF-IDF should expose blocked memory risk");
    Check(progress_events.front().message.find("configured hard limit") !=
              std::string::npos,
          "blocked TF-IDF should explain the configured hard limit");
}

void TestBoundedTFIDFMaterialization() {
    auto text = FinishStringArray({
        "apple banana apple",
        "banana carrot",
        "durian",
    });
    auto label = FinishStringArray({"pos", "neg", "pos"});

    auto schema = arrow::schema({
        arrow::field("text", arrow::utf8()),
        arrow::field("label", arrow::utf8()),
    });
    auto input = arrow::Table::Make(schema, {text, label}, 3);

    cyxwiz::TFIDFVectorizerOperator op;
    std::map<std::string, std::string> params = {
        {"text_col", "text"},
        {"label_col", "label"},
        {"max_features", "2"},
        {"use_idf", "true"},
        {"smooth_idf", "true"},
        {"norm", "none"},
    };

    std::vector<cyxwiz::PipelineOperatorProgress> progress_events;
    op.SetProgressCallback(
        [&](const cyxwiz::PipelineOperatorProgress& event) {
            progress_events.push_back(event);
        });

    std::string error;
    Check(op.Configure(params, error), error);
    auto result = op.Apply(input);
    Check(result.ok(), result.status().ToString());
    auto output = result.ValueOrDie();
    Check(!progress_events.empty(), "TF-IDF should emit materialization progress");
    Check(progress_events.front().stage == "TF-IDF memory preflight",
          "first TF-IDF progress event should be memory preflight");
    Check(progress_events.front().estimated_memory_bytes > 3 * 2 * sizeof(float),
          "preflight should report conservative peak memory, not raw bytes");
    Check(progress_events.front().message.find("risk=safe") != std::string::npos,
          "small TF-IDF preflight should report safe risk");
    Check(progress_events.front().memory_risk_level == "safe",
          "small TF-IDF preflight should expose structured safe risk");
    Check(progress_events.front().status == "running",
          "safe TF-IDF preflight should remain a running materialization event");
    Check(progress_events.front().message.find("Suggestion:") != std::string::npos,
          "TF-IDF preflight should include actionable suggestion");

    Check(output != nullptr, "TF-IDF output table is null");
    Check(output->num_rows() == 3, "TF-IDF output should preserve row count");
    Check(output->num_columns() == 3,
          "TF-IDF output should be max_features columns plus y");

    Check(output->GetColumnByName("tfidf_0") != nullptr,
          "missing tfidf_0");
    Check(output->GetColumnByName("tfidf_1") != nullptr,
          "missing tfidf_1");
    Check(output->GetColumnByName("tfidf_2") == nullptr,
          "bounded TF-IDF should not emit beyond max_features");

    const float idf_apple = std::log(4.0f / 2.0f) + 1.0f;
    const float idf_banana = std::log(4.0f / 3.0f) + 1.0f;
    CheckNear(ReadFloatValue(output, "tfidf_0", 0),
              (2.0f / 3.0f) * idf_apple,
              1e-5f,
              "row0 apple TF-IDF");
    CheckNear(ReadFloatValue(output, "tfidf_1", 0),
              (1.0f / 3.0f) * idf_banana,
              1e-5f,
              "row0 banana TF-IDF");
    CheckNear(ReadFloatValue(output, "tfidf_0", 1), 0.0f, 1e-6f,
              "row1 apple TF-IDF");
    CheckNear(ReadFloatValue(output, "tfidf_1", 1),
              0.5f * idf_banana,
              1e-5f,
              "row1 banana TF-IDF");
    CheckNear(ReadFloatValue(output, "tfidf_0", 2), 0.0f, 1e-6f,
              "row2 apple TF-IDF");
    CheckNear(ReadFloatValue(output, "tfidf_1", 2), 0.0f, 1e-6f,
              "row2 banana TF-IDF");

    Check(ReadIntValue(output, "y", 0) == 0, "first-seen label pos -> 0");
    Check(ReadIntValue(output, "y", 1) == 1, "second label neg -> 1");
    Check(ReadIntValue(output, "y", 2) == 0, "repeated label pos -> 0");
}

void TestTFIDFNGramMaterialization() {
    auto text = FinishStringArray({
        "not good",
        "not bad",
    });

    auto schema = arrow::schema({
        arrow::field("text", arrow::utf8()),
    });
    auto input = arrow::Table::Make(schema, {text}, 2);

    cyxwiz::TFIDFVectorizerOperator op;
    std::map<std::string, std::string> params = {
        {"text_col", "text"},
        {"max_features", "5"},
        {"use_idf", "false"},
        {"smooth_idf", "false"},
        {"norm", "none"},
        {"stop_words", "none"},
        {"ngram_range", "1,2"},
    };

    std::string error;
    Check(op.Configure(params, error), error);
    auto result = op.Apply(input);
    Check(result.ok(), result.status().ToString());
    auto output = result.ValueOrDie();
    Check(output != nullptr, "n-gram TF-IDF output table is null");
    Check(output->num_rows() == 2, "n-gram TF-IDF should preserve row count");
    Check(output->num_columns() == 5,
          "n-gram TF-IDF should emit unigram and bigram features");

    CheckNear(ReadFloatValue(output, "tfidf_0", 0), 0.0f, 1e-6f,
              "row0 bad unigram should be absent");
    CheckNear(ReadFloatValue(output, "tfidf_1", 0), 1.0f / 3.0f, 1e-6f,
              "row0 good unigram should be present");
    CheckNear(ReadFloatValue(output, "tfidf_2", 0), 1.0f / 3.0f, 1e-6f,
              "row0 not unigram should be present");
    CheckNear(ReadFloatValue(output, "tfidf_3", 0), 0.0f, 1e-6f,
              "row0 not bad bigram should be absent");
    CheckNear(ReadFloatValue(output, "tfidf_4", 0), 1.0f / 3.0f, 1e-6f,
              "row0 not good bigram should be present");
}

void TestCrossEntropyParity() {
    const std::vector<float> logits = {
        1.0f, 2.0f, 0.5f,
        0.1f, -0.2f, 0.0f,
    };
    const std::vector<int64_t> labels = {1, 0};

    cyxwiz::Tensor predictions({2, 3}, logits.data(), cyxwiz::DataType::Float32);
    cyxwiz::Tensor targets({2}, labels.data(), cyxwiz::DataType::Int64);
    cyxwiz::CrossEntropyLoss loss(cyxwiz::Reduction::Mean);
    cyxwiz::Tensor actual = loss.Forward(predictions, targets);

    const float expected =
        ReferenceCrossEntropyMean(logits, labels, 2, 3);
    CheckNear(actual.Data<float>()[0], expected, 1e-5f,
              "CrossEntropy mean should match reference/PyTorch semantics");
}

void TestLinearForwardBackwardParity() {
    const std::vector<float> input_values = {
        1.0f, 2.0f, -1.0f,
        0.0f, -3.0f, 4.0f,
    };
    const std::vector<float> weight_values = {
        0.5f, -1.0f, 2.0f,
        -0.25f, 0.75f, 1.5f,
    };
    const std::vector<float> bias_values = {0.1f, -0.2f};
    const std::vector<float> grad_output_values = {
        1.0f, -0.5f,
        0.25f, 2.0f,
    };

    cyxwiz::LinearLayer linear(3, 2, true);
    cyxwiz::Tensor weight({2, 3}, weight_values.data(),
                          cyxwiz::DataType::Float32);
    cyxwiz::Tensor bias({2}, bias_values.data(), cyxwiz::DataType::Float32);
    linear.SetParameters({{"weight", weight}, {"bias", bias}});

    cyxwiz::Tensor input({2, 3}, input_values.data(),
                         cyxwiz::DataType::Float32);
    cyxwiz::Tensor output = linear.Forward(input);
    const float* out = output.Data<float>();

    const std::vector<float> expected_output = {
        -3.4f, -0.45f,
        11.1f, 3.55f,
    };
    for (size_t i = 0; i < expected_output.size(); ++i) {
        CheckNear(out[i], expected_output[i], 1e-5f,
                  "Linear forward output[" + std::to_string(i) + "]");
    }

    cyxwiz::Tensor grad_output({2, 2}, grad_output_values.data(),
                               cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad_input = linear.Backward(grad_output);
    const float* grad_in = grad_input.Data<float>();

    const std::vector<float> expected_grad_input = {
        0.625f, -1.375f, 1.25f,
        -0.375f, 1.25f, 3.5f,
    };
    for (size_t i = 0; i < expected_grad_input.size(); ++i) {
        CheckNear(grad_in[i], expected_grad_input[i], 1e-5f,
                  "Linear backward grad_input[" + std::to_string(i) + "]");
    }

    const auto grads = linear.GetGradients();
    const float* grad_w = grads.at("weight").Data<float>();
    const float* grad_b = grads.at("bias").Data<float>();
    const std::vector<float> expected_grad_weight = {
        0.5f, 0.625f, 0.0f,
        -0.25f, -3.5f, 4.25f,
    };
    const std::vector<float> expected_grad_bias = {0.625f, 0.75f};
    for (size_t i = 0; i < expected_grad_weight.size(); ++i) {
        CheckNear(grad_w[i], expected_grad_weight[i], 1e-5f,
                  "Linear backward grad_weight[" + std::to_string(i) + "]");
    }
    for (size_t i = 0; i < expected_grad_bias.size(); ++i) {
        CheckNear(grad_b[i], expected_grad_bias[i], 1e-5f,
                  "Linear backward grad_bias[" + std::to_string(i) + "]");
    }
}

void TestAdamWOneStepParity() {
    const double lr = 0.01;
    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double eps = 1e-8;
    const double weight_decay = 0.1;
    const std::vector<float> initial = {1.0f, -2.0f, 0.5f};
    const std::vector<float> grad_values = {0.25f, -0.5f, 0.125f};

    std::map<std::string, cyxwiz::Tensor> parameters = {
        {"weight", cyxwiz::Tensor({3}, initial.data(), cyxwiz::DataType::Float32)},
    };
    std::map<std::string, cyxwiz::Tensor> gradients = {
        {"weight", cyxwiz::Tensor({3}, grad_values.data(), cyxwiz::DataType::Float32)},
    };

    cyxwiz::AdamWOptimizer optimizer(lr, beta1, beta2, eps, weight_decay);
    optimizer.Step(parameters, gradients);

    const float* actual = parameters.at("weight").Data<float>();
    for (size_t i = 0; i < initial.size(); ++i) {
        const double after_decay = static_cast<double>(initial[i]) *
            (1.0 - lr * weight_decay);
        const double g = static_cast<double>(grad_values[i]);
        const double m = (1.0 - beta1) * g;
        const double v = (1.0 - beta2) * g * g;
        const double m_hat = m / (1.0 - beta1);
        const double v_hat = v / (1.0 - beta2);
        const double expected =
            after_decay - lr * m_hat / (std::sqrt(v_hat) + eps);
        CheckNear(actual[i], static_cast<float>(expected), 1e-5f,
                  "AdamW first-step parameter[" + std::to_string(i) + "]");
    }
}

} // namespace

int main() {
    TestMaterializationMemoryGuardThresholds();
    TestTFIDFBlocksBeforeAllocationHeavyWork();
    TestBoundedTFIDFMaterialization();
    TestTFIDFNGramMaterialization();
    TestCrossEntropyParity();
    TestLinearForwardBackwardParity();
    TestAdamWOneStepParity();
    std::cout << "Computation truth TF-IDF + CrossEntropy + Linear + AdamW checks passed\n";
    return 0;
}
