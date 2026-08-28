#include "../../src/core/node_executors/tfidf_vectorizer_operator.h"
#include "../../src/core/materialization_memory_guard.h"
#include "../../src/core/execution_device_context.h"

#include <cyxwiz/cyxwiz.h>
#include <cyxwiz/layers/linear.h>
#include <cyxwiz/loss.h>
#include <cyxwiz/optimizers/adaptive.h>
#include <cyxwiz/optimizers/adam.h>
#include <cyxwiz/optimizers/lamb.h>
#include <cyxwiz/optimizers/sgd.h>
#include <cyxwiz/tensor.h>

#include "algorithms/arrayfire_backend_utils.h"

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;

thread_local std::vector<cyxwiz::ArrayFireNativeCpuFallbackEvent>*
    g_fallback_events = nullptr;
thread_local std::vector<cyxwiz::ArrayFireHostSyncEvent>* g_host_sync_events =
    nullptr;

void RecordFallbackEvent(
    const cyxwiz::ArrayFireNativeCpuFallbackEvent& event) {
    if (g_fallback_events != nullptr) {
        g_fallback_events->push_back(event);
    }
}

void RecordHostSyncEvent(const cyxwiz::ArrayFireHostSyncEvent& event) {
    if (g_host_sync_events != nullptr) {
        g_host_sync_events->push_back(event);
    }
}

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

void CheckNear(float actual, float expected, float tolerance,
               const std::string& message) {
    if (!std::isfinite(actual) || !std::isfinite(expected)) {
        Check(actual == expected,
              message + " encountered an unexpected non-finite value");
        return;
    }
    if (std::fabs(actual - expected) > tolerance) {
        std::ostringstream ss;
        ss << message << " expected=" << expected << " actual=" << actual
           << " tolerance=" << tolerance;
        Check(false, ss.str());
    }
}

struct Tolerance {
    float absolute = 0.0f;
    float relative = 0.0f;
};

Tolerance ReadTolerance(const json& test_case) {
    const auto& tolerance = test_case.at("tolerance");
    return {
        tolerance.at("atol").get<float>(),
        tolerance.at("rtol").get<float>(),
    };
}

void CheckNear(float actual, float expected, const Tolerance& tolerance,
               const std::string& message) {
    if (!std::isfinite(actual) || !std::isfinite(expected)) {
        Check(actual == expected,
              message + " encountered an unexpected non-finite value");
        return;
    }
    const float difference = std::fabs(actual - expected);
    const float allowed = tolerance.absolute +
        tolerance.relative * std::fabs(expected);
    if (difference > allowed) {
        std::ostringstream ss;
        ss << message << " expected=" << expected << " actual=" << actual
           << " absolute_error=" << difference
           << " allowed_error=" << allowed
           << " atol=" << tolerance.absolute
           << " rtol=" << tolerance.relative;
        Check(false, ss.str());
    }
}

std::vector<size_t> ReadShape(const json& tensor_fixture,
                              const std::string& context) {
    const auto shape = tensor_fixture.at("shape").get<std::vector<size_t>>();
    size_t element_count = 1;
    for (size_t dimension : shape) {
        Check(dimension == 0 ||
                  element_count <= std::numeric_limits<size_t>::max() / dimension,
              context + " shape product overflow");
        element_count *= dimension;
    }
    Check(tensor_fixture.at("values").size() == element_count,
          context + " value count does not match shape");
    return shape;
}

std::vector<float> ReadFloatValues(const json& tensor_fixture,
                                   const std::string& context) {
    ReadShape(tensor_fixture, context);
    return tensor_fixture.at("values").get<std::vector<float>>();
}

std::vector<int64_t> ReadInt64Values(const json& tensor_fixture,
                                     const std::string& context) {
    ReadShape(tensor_fixture, context);
    return tensor_fixture.at("values").get<std::vector<int64_t>>();
}

void CheckTensor(const cyxwiz::Tensor& actual,
                 const json& expected_fixture,
                 const Tolerance& tolerance,
                 const std::string& context) {
    const auto expected_shape = ReadShape(expected_fixture, context);
    Check(actual.Shape() == expected_shape, context + " shape mismatch");
    const auto expected = ReadFloatValues(expected_fixture, context);
    const cyxwiz::ScopedArrayFireHostSyncAttribution output_readback(
        cyxwiz::ArrayFireHostSyncCategory::DebugSampleDump, context);
    const float* actual_values = actual.ReadData<float>();
    for (size_t i = 0; i < expected.size(); ++i) {
        CheckNear(actual_values[i], expected[i], tolerance,
                  context + "[" + std::to_string(i) + "]");
    }
}

json LoadTrainingCoreFixture(const fs::path& executable_path,
                             const char* explicit_path) {
    fs::path fixture_path;
    if (explicit_path && *explicit_path) {
        fixture_path = explicit_path;
    } else {
        fixture_path = fs::absolute(executable_path).parent_path() /
            "computation_truth_fixtures" /
            "training_core_pytorch.json";
    }

    std::ifstream stream(fixture_path);
    Check(stream.is_open(), "unable to open PyTorch fixture: " +
          fixture_path.string());

    json fixture;
    try {
        stream >> fixture;
    } catch (const std::exception& error) {
        Check(false, "unable to parse PyTorch fixture " +
              fixture_path.string() + ": " + error.what());
    }

    Check(fixture.value("schema_version", 0) == 1,
          "unsupported PyTorch fixture schema version");
    Check(fixture.at("oracle").value("name", "") == "PyTorch",
          "training-core fixture must declare PyTorch as its oracle");
    Check(fixture.at("oracle").value("device", "") == "cpu",
          "training-core fixture must be generated on PyTorch CPU");
    Check(!fixture.at("oracle").value("version", "").empty(),
          "training-core fixture must record the PyTorch version");
    Check(fixture.contains("cases") && fixture.at("cases").is_object(),
          "training-core fixture must contain cases");
    return fixture;
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

void TestCrossEntropyParity(const json& cases) {
    const auto& test_case = cases.at("cross_entropy_index_mean_f32");
    Check(test_case.value("operation", "") ==
              "torch.nn.functional.cross_entropy",
          "CrossEntropy fixture operation mismatch");
    Check(test_case.value("dtype", "") == "float32" &&
              test_case.value("target_dtype", "") == "int64",
          "CrossEntropy fixture dtype mismatch");
    Check(test_case.value("reduction", "") == "mean",
          "CrossEntropy fixture reduction mismatch");
    const auto logits = ReadFloatValues(test_case.at("logits"), "CE logits");
    const auto labels = ReadInt64Values(test_case.at("targets"), "CE targets");

    cyxwiz::Tensor predictions(
        ReadShape(test_case.at("logits"), "CE logits"), logits.data(),
        cyxwiz::DataType::Float32);
    cyxwiz::Tensor targets(
        ReadShape(test_case.at("targets"), "CE targets"), labels.data(),
        cyxwiz::DataType::Int64);
    cyxwiz::CrossEntropyLoss loss(cyxwiz::Reduction::Mean);
    cyxwiz::Tensor actual = loss.Forward(predictions, targets);

    const float expected = test_case.at("expected").at("loss").get<float>();
    const cyxwiz::ScopedArrayFireHostSyncAttribution output_readback(
        cyxwiz::ArrayFireHostSyncCategory::DebugSampleDump,
        "CrossEntropy mean PyTorch parity");
    CheckNear(actual.ReadData<float>()[0], expected, ReadTolerance(test_case),
              "CrossEntropy mean PyTorch parity");
}

cyxwiz::Reduction ParseReduction(const std::string& reduction) {
    if (reduction == "none") {
        return cyxwiz::Reduction::None;
    }
    if (reduction == "mean") {
        return cyxwiz::Reduction::Mean;
    }
    Check(reduction == "sum", "unsupported CrossEntropy fixture reduction");
    return cyxwiz::Reduction::Sum;
}

void TestCrossEntropyMatrixParity(const json& cases) {
    const auto& matrix = cases.at("cross_entropy_matrix_f32");
    Check(matrix.is_array() && !matrix.empty(),
          "CrossEntropy matrix fixture must be non-empty");
    for (const auto& test_case : matrix) {
        const std::string name = test_case.at("name").get<std::string>();
        Check(test_case.value("operation", "") ==
                  "torch.nn.functional.cross_entropy",
              name + " operation mismatch");
        Check(test_case.value("dtype", "") == "float32",
              name + " dtype mismatch");
        const auto logits = ReadFloatValues(test_case.at("logits"), name);
        cyxwiz::Tensor predictions(
            ReadShape(test_case.at("logits"), name), logits.data(),
            cyxwiz::DataType::Float32);

        const std::string target_form =
            test_case.at("target_form").get<std::string>();
        cyxwiz::Tensor targets;
        if (target_form == "index") {
            const auto values = ReadInt64Values(test_case.at("targets"), name);
            targets = cyxwiz::Tensor(
                ReadShape(test_case.at("targets"), name), values.data(),
                cyxwiz::DataType::Int64);
        } else {
            Check(target_form == "soft", name + " target form mismatch");
            const auto values = ReadFloatValues(test_case.at("targets"), name);
            targets = cyxwiz::Tensor(
                ReadShape(test_case.at("targets"), name), values.data(),
                cyxwiz::DataType::Float32);
        }

        // Re-wrap through the semantic ArrayFire view so the public inputs are
        // device-current and host-stale at the compute boundary. A supported
        // CrossEntropy shape must neither materialize them nor fall back.
#ifdef CYXWIZ_HAS_ARRAYFIRE
        predictions = cyxwiz::Tensor::FromSemanticArray(
            predictions.GetSemanticArray(), predictions.Shape());
        targets = cyxwiz::Tensor::FromSemanticArray(
            targets.GetSemanticArray(), targets.Shape());
#endif

        cyxwiz::CrossEntropyLoss loss(
            ParseReduction(test_case.at("reduction").get<std::string>()),
            test_case.at("ignore_index").get<int>(),
            test_case.at("class_weights").get<std::vector<float>>(),
            test_case.at("label_smoothing").get<float>());
        const size_t fallback_count_before =
            g_fallback_events == nullptr ? 0 : g_fallback_events->size();
        const size_t host_sync_count_before =
            g_host_sync_events == nullptr ? 0 : g_host_sync_events->size();
        const auto actual_loss = loss.Forward(predictions, targets);
        const auto actual_gradient = loss.Backward(predictions, targets);
        Check(g_fallback_events == nullptr ||
                  g_fallback_events->size() == fallback_count_before,
              name + " attempted native CPU fallback");
        Check(g_host_sync_events == nullptr ||
                  g_host_sync_events->size() == host_sync_count_before,
              name + " materialized a tensor during compute");

        const auto& expected_loss = test_case.at("expected").at("loss");
        if (expected_loss.value("non_finite", "") == "nan") {
            Check(actual_loss.Shape() ==
                      expected_loss.at("shape").get<std::vector<size_t>>(),
                  name + " forward shape mismatch");
            const cyxwiz::ScopedArrayFireHostSyncAttribution output_readback(
                cyxwiz::ArrayFireHostSyncCategory::DebugSampleDump,
                name + " non-finite forward readback");
            Check(std::isnan(actual_loss.ReadData<float>()[0]),
                  name + " forward expected NaN");
        } else {
            CheckTensor(actual_loss, expected_loss,
                        ReadTolerance(test_case), name + " forward");
        }
        CheckTensor(actual_gradient,
                    test_case.at("expected").at("logit_gradient"),
                    ReadTolerance(test_case), name + " backward");
    }
}

void TestLinearForwardBackwardParity(const json& cases) {
    const auto& test_case = cases.at("linear_basic_f32");
    Check(test_case.value("operation", "") == "torch.nn.functional.linear",
          "Linear fixture operation mismatch");
    Check(test_case.value("dtype", "") == "float32",
          "Linear fixture dtype mismatch");
    Check(test_case.value("parameter_gradient_reduction", "") ==
              "sum_over_batch",
          "Linear fixture gradient reduction mismatch");
    const auto input_values = ReadFloatValues(test_case.at("input"), "Linear input");
    const auto weight_values = ReadFloatValues(test_case.at("weight"), "Linear weight");
    const auto bias_values = ReadFloatValues(test_case.at("bias"), "Linear bias");
    const auto grad_output_values =
        ReadFloatValues(test_case.at("grad_output"), "Linear grad_output");
    const auto weight_shape = ReadShape(test_case.at("weight"), "Linear weight");
    Check(weight_shape.size() == 2, "Linear weight fixture must be rank 2");
    const auto tolerance = ReadTolerance(test_case);

    cyxwiz::LinearLayer linear(weight_shape[1], weight_shape[0], true);
    cyxwiz::Tensor weight(weight_shape, weight_values.data(),
                          cyxwiz::DataType::Float32);
    cyxwiz::Tensor bias(ReadShape(test_case.at("bias"), "Linear bias"),
                        bias_values.data(), cyxwiz::DataType::Float32);
    linear.SetParameters({{"weight", weight}, {"bias", bias}});

    cyxwiz::Tensor input(ReadShape(test_case.at("input"), "Linear input"),
                         input_values.data(),
                         cyxwiz::DataType::Float32);
    cyxwiz::Tensor output = linear.Forward(input);
    CheckTensor(output, test_case.at("expected").at("output"), tolerance,
                "Linear forward output");

    cyxwiz::Tensor grad_output(
        ReadShape(test_case.at("grad_output"), "Linear grad_output"),
        grad_output_values.data(),
                               cyxwiz::DataType::Float32);
    cyxwiz::Tensor grad_input = linear.Backward(grad_output);
    CheckTensor(grad_input, test_case.at("expected").at("grad_input"),
                tolerance, "Linear backward grad_input");

    const auto grads = linear.GetGradients();
    CheckTensor(grads.at("weight"),
                test_case.at("expected").at("grad_weight"), tolerance,
                "Linear backward grad_weight");
    CheckTensor(grads.at("bias"),
                test_case.at("expected").at("grad_bias"), tolerance,
                "Linear backward grad_bias");
}

void CheckAdamFamilyState(
    const cyxwiz::OptimizerState& state,
    const std::string& parameter_name,
    const json& expected,
    const Tolerance& tolerance,
    const std::string& label,
    bool expect_mu_product) {
    CheckTensor(state.tensors.at("first_moment/" + parameter_name),
                expected.at("exp_avg"), tolerance,
                label + " first moment");
    CheckTensor(state.tensors.at("second_moment/" + parameter_name),
                expected.at("exp_avg_sq"), tolerance,
                label + " second moment");
    const auto& step_tensor =
        state.tensors.at("parameter_step/" + parameter_name);
    Check(step_tensor.GetDataType() == cyxwiz::DataType::Int32,
          label + " parameter-step dtype");
    Check(step_tensor.NumElements() == 1,
          label + " parameter-step scalar shape");
    Check(step_tensor.ReadData<int32_t>()[0] == expected.at("step").get<int>(),
          label + " per-parameter step parity");
    if (expect_mu_product) {
        const auto& mu_tensor =
            state.tensors.at("mu_product/" + parameter_name);
        Check(mu_tensor.GetDataType() == cyxwiz::DataType::Float32,
              label + " mu-product dtype");
        Check(std::abs(mu_tensor.ReadData<float>()[0] -
                       expected.at("mu_product").get<float>()) <= 2.0e-6f,
              label + " mu-product parity");
    }
}

template <typename OptimizerFactory>
void TestAdamFamilyCase(
    const json& test_case,
    const std::string& label,
    bool expect_mu_product,
    OptimizerFactory&& factory) {
    Check(test_case.value("dtype", "") == "float32",
          label + " fixture dtype mismatch");
    Check(test_case.value("zero_grad_contract", "") ==
              "clears gradients without resetting optimizer state",
          label + " zero_grad fixture contract mismatch");
    const auto initial = ReadFloatValues(
        test_case.at("initial_parameter"), label + " initial parameter");
    const auto shape = ReadShape(
        test_case.at("initial_parameter"), label + " initial parameter");
    const auto tolerance = ReadTolerance(test_case);
    std::map<std::string, cyxwiz::Tensor> parameters = {
        {"weight", cyxwiz::Tensor(
                       shape, initial.data(), cyxwiz::DataType::Float32)},
    };
    auto optimizer = factory();
    const auto& gradients = test_case.at("gradients");
    const auto& expected_steps = test_case.at("expected_steps");
    Check(gradients.size() == expected_steps.size(),
          label + " fixture gradient/step count mismatch");
    cyxwiz::OptimizerState checkpoint;
    std::map<std::string, cyxwiz::Tensor> checkpoint_parameters;

    for (size_t index = 0; index < gradients.size(); ++index) {
        const auto values = ReadFloatValues(
            gradients.at(index), label + " gradient");
        const std::map<std::string, cyxwiz::Tensor> step_gradients = {
            {"weight", cyxwiz::Tensor(
                           ReadShape(gradients.at(index), label + " gradient"),
                           values.data(), cyxwiz::DataType::Float32)},
        };
        optimizer->ZeroGrad();
        optimizer->Step(parameters, step_gradients);
        const auto& expected = expected_steps.at(index);
        CheckTensor(parameters.at("weight"), expected.at("parameter"),
                    tolerance, label + " multi-step parameter");
        cyxwiz::OptimizerState state;
        std::string error;
        Check(optimizer->ExportState(state, error),
              label + " state export failed: " + error);
        Check(state.step_count == static_cast<int>(index + 1),
              label + " global update-count parity");
        CheckAdamFamilyState(state, "weight", expected.at("state"),
                             tolerance, label, expect_mu_product);
        if (index + 2 == gradients.size()) {
            checkpoint = state;
            checkpoint_parameters = parameters;
        }
    }

    auto resumed = factory();
    resumed->SetLearningRate(0.123);
    std::string error;
    Check(resumed->ImportState(checkpoint, error),
          label + " state import failed: " + error);
    Check(resumed->GetLearningRate() == checkpoint.learning_rate,
          label + " state import did not restore learning rate");
    const size_t final_index = gradients.size() - 1;
    const auto final_values = ReadFloatValues(
        gradients.at(final_index), label + " resumed gradient");
    const std::map<std::string, cyxwiz::Tensor> final_gradients = {
        {"weight", cyxwiz::Tensor(
                       ReadShape(gradients.at(final_index),
                                 label + " resumed gradient"),
                       final_values.data(), cyxwiz::DataType::Float32)},
    };
    resumed->ZeroGrad();
    resumed->Step(checkpoint_parameters, final_gradients);
    CheckTensor(checkpoint_parameters.at("weight"),
                expected_steps.at(final_index).at("parameter"),
                tolerance, label + " resumed parameter");
    cyxwiz::OptimizerState resumed_state;
    Check(resumed->ExportState(resumed_state, error),
          label + " resumed state export failed: " + error);
    CheckAdamFamilyState(
        resumed_state, "weight",
        expected_steps.at(final_index).at("state"), tolerance,
        label + " resumed", expect_mu_product);

    const auto& sparse = test_case.at("missing_gradient_case");
    const auto initial_b = ReadFloatValues(
        sparse.at("initial_b"), label + " sparse initial b");
    std::map<std::string, cyxwiz::Tensor> sparse_parameters = {
        {"a", cyxwiz::Tensor(
                  shape, initial.data(), cyxwiz::DataType::Float32)},
        {"b", cyxwiz::Tensor(
                  shape, initial_b.data(), cyxwiz::DataType::Float32)},
    };
    auto sparse_optimizer = factory();
    const auto first_gradient = ReadFloatValues(
        gradients.at(0), label + " sparse first gradient");
    sparse_optimizer->Step(
        sparse_parameters,
        {{"a", cyxwiz::Tensor(
                   shape, first_gradient.data(), cyxwiz::DataType::Float32)}});
    CheckTensor(sparse_parameters.at("b"),
                sparse.at("b_after_missing_gradient"), tolerance,
                label + " missing-gradient parameter preservation");

    const auto second_gradient = ReadFloatValues(
        gradients.at(1), label + " sparse second gradient");
    const auto third_gradient = ReadFloatValues(
        gradients.at(2), label + " sparse third gradient");
    sparse_optimizer->ZeroGrad();
    sparse_optimizer->Step(
        sparse_parameters,
        {{"a", cyxwiz::Tensor(
                   shape, second_gradient.data(), cyxwiz::DataType::Float32)},
         {"b", cyxwiz::Tensor(
                   shape, third_gradient.data(), cyxwiz::DataType::Float32)}});
    CheckTensor(sparse_parameters.at("a"), sparse.at("final_a"),
                tolerance, label + " sparse parameter a");
    CheckTensor(sparse_parameters.at("b"), sparse.at("final_b"),
                tolerance, label + " sparse parameter b");
    cyxwiz::OptimizerState sparse_state;
    Check(sparse_optimizer->ExportState(sparse_state, error),
          label + " sparse state export failed: " + error);
    CheckAdamFamilyState(sparse_state, "a", sparse.at("state_a"),
                         tolerance, label + " sparse a", expect_mu_product);
    CheckAdamFamilyState(sparse_state, "b", sparse.at("state_b"),
                         tolerance, label + " sparse b", expect_mu_product);
}

void TestAdamFamilyMultiStepParity(const json& cases) {
    const auto& family = cases.at("adam_family_multistep_f32");

    const auto& adam = family.at("adam");
    Check(adam.value("operation", "") == "torch.optim.Adam",
          "Adam fixture operation mismatch");
    const auto& adam_hyper = adam.at("hyperparameters");
    TestAdamFamilyCase(
        adam, "Adam", false,
        [&adam_hyper]() {
            return std::make_unique<cyxwiz::AdamOptimizer>(
                adam_hyper.at("learning_rate").get<double>(),
                adam_hyper.at("beta1").get<double>(),
                adam_hyper.at("beta2").get<double>(),
                adam_hyper.at("epsilon").get<double>());
        });

    const auto& adamw = family.at("adamw");
    Check(adamw.value("operation", "") == "torch.optim.AdamW",
          "AdamW fixture operation mismatch");
    const auto& adamw_hyper = adamw.at("hyperparameters");
    TestAdamFamilyCase(
        adamw, "AdamW", false,
        [&adamw_hyper]() {
            return std::make_unique<cyxwiz::AdamWOptimizer>(
                adamw_hyper.at("learning_rate").get<double>(),
                adamw_hyper.at("beta1").get<double>(),
                adamw_hyper.at("beta2").get<double>(),
                adamw_hyper.at("epsilon").get<double>(),
                adamw_hyper.at("weight_decay").get<double>());
        });

    const auto& nadam = family.at("nadam");
    Check(nadam.value("operation", "") == "torch.optim.NAdam",
          "NAdam fixture operation mismatch");
    Check(nadam.at("hyperparameters").at("momentum_decay").get<double>() ==
              0.004,
          "NAdam fixture momentum-decay contract mismatch");
    const auto& nadam_hyper = nadam.at("hyperparameters");
    TestAdamFamilyCase(
        nadam, "NAdam", true,
        [&nadam_hyper]() {
            return std::make_unique<cyxwiz::NAdamOptimizer>(
                nadam_hyper.at("learning_rate").get<double>(),
                nadam_hyper.at("beta1").get<double>(),
                nadam_hyper.at("beta2").get<double>(),
                nadam_hyper.at("epsilon").get<double>());
        });
}

void TestSGDMomentumMultiStepParity(const json& cases) {
    const auto& test_case = cases.at("sgd_momentum_multistep_f32");
    Check(test_case.value("operation", "") == "torch.optim.SGD",
          "SGD fixture operation mismatch");
    Check(test_case.value("dtype", "") == "float32",
          "SGD fixture dtype mismatch");
    const auto& hyperparameters = test_case.at("hyperparameters");
    const double lr = hyperparameters.at("learning_rate").get<double>();
    const double momentum = hyperparameters.at("momentum").get<double>();
    const auto initial = ReadFloatValues(
        test_case.at("initial_parameter"), "SGD initial parameter");
    const auto parameter_shape = ReadShape(
        test_case.at("initial_parameter"), "SGD initial parameter");
    const auto& gradients = test_case.at("gradients");
    const auto& expected_steps = test_case.at("expected_steps");
    Check(gradients.size() == expected_steps.size(),
          "SGD fixture gradient/step count mismatch");

    std::map<std::string, cyxwiz::Tensor> parameters = {
        {"weight", cyxwiz::Tensor(parameter_shape, initial.data(),
                                  cyxwiz::DataType::Float32)},
    };
    cyxwiz::SGDOptimizer optimizer(lr, momentum);
    for (size_t index = 0; index < gradients.size(); ++index) {
        const auto gradient_values = ReadFloatValues(
            gradients.at(index), "SGD gradient");
        const std::map<std::string, cyxwiz::Tensor> step_gradients = {
            {"weight", cyxwiz::Tensor(
                           ReadShape(gradients.at(index), "SGD gradient"),
                           gradient_values.data(), cyxwiz::DataType::Float32)},
        };
        optimizer.Step(parameters, step_gradients);

        const auto& expected = expected_steps.at(index);
        Check(optimizer.GetStepCount() ==
                  expected.at("step_count").get<int>(),
              "SGD PyTorch step-count parity");
        CheckTensor(parameters.at("weight"), expected.at("parameter"),
                    ReadTolerance(test_case),
                    "SGD multi-step parameter");

        cyxwiz::OptimizerState state;
        std::string error;
        Check(optimizer.ExportState(state, error),
              "SGD state export failed: " + error);
        Check(error.empty(), "SGD state export returned an error");
        Check(state.step_count == optimizer.GetStepCount(),
              "SGD exported step count mismatch");
        Check(state.tensors.count("velocity/weight") == 1,
              "SGD momentum state is missing velocity/weight");
        CheckTensor(state.tensors.at("velocity/weight"),
                    expected.at("momentum_buffer"),
                    ReadTolerance(test_case),
                    "SGD PyTorch momentum-buffer parity");
    }
}

template <typename OptimizerFactory>
void TestAdaptiveOptimizerCase(
    const json& test_case,
    const std::string& label,
    OptimizerFactory&& factory,
    const std::vector<std::pair<std::string, std::string>>& state_mapping) {
    Check(test_case.value("dtype", "") == "float32",
          label + " fixture dtype mismatch");
    Check(test_case.value("zero_grad_contract", "") ==
              "clears gradients without resetting optimizer state",
          label + " zero_grad fixture contract mismatch");
    const auto initial = ReadFloatValues(
        test_case.at("initial_parameter"), label + " initial parameter");
    const auto parameter_shape = ReadShape(
        test_case.at("initial_parameter"), label + " initial parameter");
    const auto& gradients = test_case.at("gradients");
    const auto& expected_steps = test_case.at("expected_steps");
    Check(gradients.size() == expected_steps.size(),
          label + " fixture gradient/step count mismatch");

    std::map<std::string, cyxwiz::Tensor> parameters = {
        {"weight", cyxwiz::Tensor(parameter_shape, initial.data(),
                                  cyxwiz::DataType::Float32)},
    };
    auto optimizer = factory();
    cyxwiz::OptimizerState checkpoint;
    std::map<std::string, cyxwiz::Tensor> checkpoint_parameters;

    for (size_t index = 0; index < gradients.size(); ++index) {
        const auto gradient_values = ReadFloatValues(
            gradients.at(index), label + " gradient");
        const std::map<std::string, cyxwiz::Tensor> step_gradients = {
            {"weight", cyxwiz::Tensor(
                           ReadShape(gradients.at(index), label + " gradient"),
                           gradient_values.data(), cyxwiz::DataType::Float32)},
        };
        optimizer->ZeroGrad();
        optimizer->Step(parameters, step_gradients);

        const auto& expected = expected_steps.at(index);
        Check(optimizer->GetStepCount() ==
                  expected.at("step_count").get<int>(),
              label + " PyTorch step-count parity");
        CheckTensor(parameters.at("weight"), expected.at("parameter"),
                    ReadTolerance(test_case),
                    label + " multi-step parameter");

        cyxwiz::OptimizerState state;
        std::string error;
        Check(optimizer->ExportState(state, error),
              label + " state export failed: " + error);
        Check(error.empty(), label + " state export returned an error");
        Check(state.step_count == optimizer->GetStepCount(),
              label + " exported step count mismatch");
        for (const auto& [cyxwiz_name, pytorch_name] : state_mapping) {
            Check(state.tensors.count(cyxwiz_name) == 1,
                  label + " state is missing " + cyxwiz_name);
            CheckTensor(state.tensors.at(cyxwiz_name),
                        expected.at("state").at(pytorch_name),
                        ReadTolerance(test_case),
                        label + " state parity for " + pytorch_name);
        }
        if (index + 2 == gradients.size()) {
            checkpoint = state;
            checkpoint_parameters = parameters;
        }
    }

    auto resumed = factory();
    std::string error;
    Check(resumed->ImportState(checkpoint, error),
          label + " state import failed: " + error);
    Check(error.empty(), label + " state import returned an error");
    const size_t final_index = gradients.size() - 1;
    const auto final_gradient_values = ReadFloatValues(
        gradients.at(final_index), label + " resumed gradient");
    const std::map<std::string, cyxwiz::Tensor> final_gradients = {
        {"weight", cyxwiz::Tensor(
                       ReadShape(
                           gradients.at(final_index), label + " resumed gradient"),
                       final_gradient_values.data(), cyxwiz::DataType::Float32)},
    };
    resumed->ZeroGrad();
    resumed->Step(checkpoint_parameters, final_gradients);
    CheckTensor(checkpoint_parameters.at("weight"),
                expected_steps.at(final_index).at("parameter"),
                ReadTolerance(test_case),
                label + " resumed parameter");
    Check(resumed->GetStepCount() == optimizer->GetStepCount(),
          label + " resumed step count mismatch");
    cyxwiz::OptimizerState resumed_state;
    Check(resumed->ExportState(resumed_state, error),
          label + " resumed state export failed: " + error);
    for (const auto& [cyxwiz_name, pytorch_name] : state_mapping) {
        Check(resumed_state.tensors.count(cyxwiz_name) == 1,
              label + " resumed state is missing " + cyxwiz_name);
        CheckTensor(resumed_state.tensors.at(cyxwiz_name),
                    expected_steps.at(final_index).at("state").at(pytorch_name),
                    ReadTolerance(test_case),
                    label + " resumed state parity for " + pytorch_name);
    }
}

void TestAdaptiveOptimizerMultiStepParity(const json& cases) {
    const auto& adaptive = cases.at("adaptive_optimizer_multistep_f32");

    const auto& rmsprop = adaptive.at("rmsprop");
    Check(rmsprop.value("operation", "") == "torch.optim.RMSprop",
          "RMSprop fixture operation mismatch");
    const auto& rmsprop_hyper = rmsprop.at("hyperparameters");
    TestAdaptiveOptimizerCase(
        rmsprop,
        "RMSprop",
        [&rmsprop_hyper]() {
            return std::make_unique<cyxwiz::RMSpropOptimizer>(
                rmsprop_hyper.at("learning_rate").get<double>(),
                rmsprop_hyper.at("alpha").get<double>(),
                rmsprop_hyper.at("epsilon").get<double>(),
                rmsprop_hyper.at("momentum").get<double>());
        },
        {{"square_average/weight", "square_avg"},
         {"momentum_buffer/weight", "momentum_buffer"}});

    const auto& adagrad = adaptive.at("adagrad");
    Check(adagrad.value("operation", "") == "torch.optim.Adagrad",
          "AdaGrad fixture operation mismatch");
    const auto& adagrad_hyper = adagrad.at("hyperparameters");
    TestAdaptiveOptimizerCase(
        adagrad,
        "AdaGrad",
        [&adagrad_hyper]() {
            return std::make_unique<cyxwiz::AdaGradOptimizer>(
                adagrad_hyper.at("learning_rate").get<double>(),
                adagrad_hyper.at("epsilon").get<double>());
        },
        {{"sum/weight", "sum"}});

    const auto& adadelta = adaptive.at("adadelta");
    Check(adadelta.value("operation", "") == "torch.optim.Adadelta",
          "Adadelta fixture operation mismatch");
    const auto& adadelta_hyper = adadelta.at("hyperparameters");
    TestAdaptiveOptimizerCase(
        adadelta,
        "Adadelta",
        [&adadelta_hyper]() {
            auto optimizer = std::make_unique<cyxwiz::AdadeltaOptimizer>(
                adadelta_hyper.at("rho").get<double>(),
                adadelta_hyper.at("epsilon").get<double>());
            optimizer->SetLearningRate(
                adadelta_hyper.at("learning_rate").get<double>());
            return optimizer;
        },
        {{"square_average/weight", "square_avg"},
         {"accumulated_delta/weight", "acc_delta"}});
}

void TestLambMultiStepParity(const json& cases) {
    const auto& test_case = cases.at("lamb_multistep_f32");
    Check(test_case.value("operation", "") ==
              "independent PyTorch tensor LAMB equation",
          "LAMB fixture operation mismatch");
    Check(test_case.value("dtype", "") == "float32",
          "LAMB fixture dtype mismatch");
    Check(test_case.value("zero_grad_contract", "") ==
              "clears gradients without resetting optimizer state",
          "LAMB zero_grad fixture contract mismatch");
    const auto& hyper = test_case.at("hyperparameters");
    const auto tolerance = ReadTolerance(test_case);
    const auto make_optimizer = [&hyper]() {
        return std::make_unique<cyxwiz::LAMBOptimizer>(
            hyper.at("learning_rate").get<double>(),
            hyper.at("beta1").get<double>(),
            hyper.at("beta2").get<double>(),
            hyper.at("epsilon").get<double>(),
            hyper.at("weight_decay").get<double>());
    };

    const auto initial = ReadFloatValues(
        test_case.at("initial_parameter"), "LAMB initial parameter");
    const auto shape = ReadShape(
        test_case.at("initial_parameter"), "LAMB initial parameter");
    std::map<std::string, cyxwiz::Tensor> parameters = {
        {"weight", cyxwiz::Tensor(
                       shape, initial.data(), cyxwiz::DataType::Float32)},
    };
    auto optimizer = make_optimizer();
    const auto& gradients = test_case.at("gradients");
    const auto& expected_steps = test_case.at("expected_steps");
    Check(gradients.size() == expected_steps.size(),
          "LAMB fixture gradient/step count mismatch");
    cyxwiz::OptimizerState checkpoint;
    std::map<std::string, cyxwiz::Tensor> checkpoint_parameters;

    for (size_t index = 0; index < gradients.size(); ++index) {
        const auto values = ReadFloatValues(
            gradients.at(index), "LAMB gradient");
        const std::map<std::string, cyxwiz::Tensor> step_gradients = {
            {"weight", cyxwiz::Tensor(
                           ReadShape(gradients.at(index), "LAMB gradient"),
                           values.data(), cyxwiz::DataType::Float32)},
        };
        optimizer->ZeroGrad();
        optimizer->Step(parameters, step_gradients);
        const auto& expected = expected_steps.at(index);
        Check(optimizer->GetStepCount() ==
                  expected.at("step_count").get<int>(),
              "LAMB step-count parity");
        CheckTensor(parameters.at("weight"), expected.at("parameter"),
                    tolerance, "LAMB multi-step parameter");

        cyxwiz::OptimizerState state;
        std::string error;
        Check(optimizer->ExportState(state, error),
              "LAMB state export failed: " + error);
        CheckTensor(state.tensors.at("first_moment/weight"),
                    expected.at("first_moment"), tolerance,
                    "LAMB first-moment parity");
        CheckTensor(state.tensors.at("second_moment/weight"),
                    expected.at("second_moment"), tolerance,
                    "LAMB second-moment parity");
        if (index + 2 == gradients.size()) {
            checkpoint = state;
            checkpoint_parameters = parameters;
        }
    }

    auto resumed = make_optimizer();
    resumed->SetLearningRate(0.123);
    std::string error;
    Check(resumed->ImportState(checkpoint, error),
          "LAMB state import failed: " + error);
    Check(resumed->GetLearningRate() == checkpoint.learning_rate,
          "LAMB state import did not restore learning rate");
    const size_t final_index = gradients.size() - 1;
    const auto final_values = ReadFloatValues(
        gradients.at(final_index), "LAMB resumed gradient");
    const std::map<std::string, cyxwiz::Tensor> final_gradients = {
        {"weight", cyxwiz::Tensor(
                       ReadShape(gradients.at(final_index),
                                 "LAMB resumed gradient"),
                       final_values.data(), cyxwiz::DataType::Float32)},
    };
    resumed->ZeroGrad();
    resumed->Step(checkpoint_parameters, final_gradients);
    CheckTensor(checkpoint_parameters.at("weight"),
                expected_steps.at(final_index).at("parameter"),
                tolerance, "LAMB resumed parameter");
    cyxwiz::OptimizerState resumed_state;
    Check(resumed->ExportState(resumed_state, error),
          "LAMB resumed state export failed: " + error);
    CheckTensor(resumed_state.tensors.at("first_moment/weight"),
                expected_steps.at(final_index).at("first_moment"),
                tolerance, "LAMB resumed first moment");
    CheckTensor(resumed_state.tensors.at("second_moment/weight"),
                expected_steps.at(final_index).at("second_moment"),
                tolerance, "LAMB resumed second moment");

    for (const auto& [edge_name, edge] :
         test_case.at("zero_norm_edges").items()) {
        Check(edge.at("expected_trust_ratio").get<double>() == 1.0,
              "LAMB " + edge_name + " fixture trust-ratio edge mismatch");
        const auto edge_initial = ReadFloatValues(
            edge.at("initial_parameter"), "LAMB " + edge_name + " initial");
        const auto edge_gradient = ReadFloatValues(
            edge.at("gradient"), "LAMB " + edge_name + " gradient");
        const auto edge_shape = ReadShape(
            edge.at("initial_parameter"), "LAMB " + edge_name + " initial");
        std::map<std::string, cyxwiz::Tensor> edge_parameters = {
            {"weight", cyxwiz::Tensor(
                           edge_shape, edge_initial.data(),
                           cyxwiz::DataType::Float32)},
        };
        const std::map<std::string, cyxwiz::Tensor> edge_gradients = {
            {"weight", cyxwiz::Tensor(
                           edge_shape, edge_gradient.data(),
                           cyxwiz::DataType::Float32)},
        };
        cyxwiz::LAMBOptimizer edge_optimizer(
            hyper.at("learning_rate").get<double>(),
            hyper.at("beta1").get<double>(),
            hyper.at("beta2").get<double>(),
            hyper.at("epsilon").get<double>(),
            edge.at("weight_decay").get<double>());
        edge_optimizer.Step(edge_parameters, edge_gradients);
        CheckTensor(edge_parameters.at("weight"),
                    edge.at("expected_parameter"), tolerance,
                    "LAMB " + edge_name + " parameter");
        cyxwiz::OptimizerState edge_state;
        Check(edge_optimizer.ExportState(edge_state, error),
              "LAMB " + edge_name + " state export failed: " + error);
        CheckTensor(edge_state.tensors.at("first_moment/weight"),
                    edge.at("expected_first_moment"), tolerance,
                    "LAMB " + edge_name + " first moment");
        CheckTensor(edge_state.tensors.at("second_moment/weight"),
                    edge.at("expected_second_moment"), tolerance,
                    "LAMB " + edge_name + " second moment");
    }
}

void TestArrayFireCpuTrainingCoreTruth(const json& cases) {
    const bool initialized_here = !cyxwiz::IsInitialized();
    if (initialized_here) {
        Check(cyxwiz::Initialize(), "CyxWiz backend initialization failed");
    }

    cyxwiz::Device cpu(cyxwiz::DeviceType::CPU, 0);
    const auto activation = cpu.ActivateExact(true);
    Check(activation.success,
          "ArrayFire CPU activation failed: " + activation.message);
    Check(activation.execution_validated,
          "ArrayFire CPU activation did not validate execution");
    Check(activation.requested_type == cyxwiz::DeviceType::CPU &&
              activation.effective_type == cyxwiz::DeviceType::CPU &&
              activation.requested_device_id == 0 &&
              activation.effective_device_id == 0,
          "ArrayFire CPU requested/effective identity mismatch");
    Check(cyxwiz::CurrentArrayFireBackendName() == "cpu",
          "training-core truth test must execute on ArrayFire CPU");

    auto context = cyxwiz::CaptureCurrentExecutionDeviceContext(
        cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
    context.execution_validated = activation.execution_validated;
    Check(context.valid && context.platform == "arrayfire",
          "execution context must report a valid ArrayFire platform");
    Check(context.requested_backend == "arrayfire_cpu" &&
              context.effective_backend == "arrayfire_cpu" &&
              context.requested_device_id == 0 &&
              context.effective_device_id == 0,
          "execution context must preserve exact ArrayFire CPU identity");
    Check(!context.selection_fallback_applied,
          "exact ArrayFire CPU activation must not report selection fallback");

    std::vector<cyxwiz::ArrayFireNativeCpuFallbackEvent> fallback_events;
    std::vector<cyxwiz::ArrayFireHostSyncEvent> host_sync_events;
    g_fallback_events = &fallback_events;
    g_host_sync_events = &host_sync_events;
    {
        const cyxwiz::ScopedArrayFireFallbackPolicy strict_policy(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        const cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
            &RecordFallbackEvent);
        const cyxwiz::ScopedArrayFireHostSyncObserver host_sync_observer(
            &RecordHostSyncEvent);
        const cyxwiz::ScopedActiveExecutionDeviceContext active_run;
        const cyxwiz::ScopedExecutionDeviceContext bound_context(context);

        Check(cyxwiz::CurrentExecutionDeviceContext() == &context,
              "one immutable execution context must be bound before compute");
        Check(cyxwiz::IsArrayFireNativeCpuFallbackForbidden(),
              "training-core truth test must forbid native CPU fallback");

        TestCrossEntropyParity(cases);
        TestCrossEntropyMatrixParity(cases);
        TestLinearForwardBackwardParity(cases);
        TestAdamFamilyMultiStepParity(cases);
        TestSGDMomentumMultiStepParity(cases);
        TestAdaptiveOptimizerMultiStepParity(cases);
        TestLambMultiStepParity(cases);
    }
    g_fallback_events = nullptr;
    g_host_sync_events = nullptr;

    Check(fallback_events.empty(),
          "training-core ArrayFire CPU path attempted native CPU fallback");
    for (const auto& event : host_sync_events) {
        Check(event.selected_backend == "cpu",
              "host-sync evidence must retain ArrayFire CPU identity");
        Check(event.attribution_category == "debug_sample_dump",
              "training-core compute performed an undeclared host sync: " +
                  event.attribution_category + " operation=" +
                  event.operation_name);
        Check(event.bytes <= 4096,
              "truth-test output readback must remain bounded");
    }

    if (initialized_here) {
        cyxwiz::Shutdown();
    }
}

} // namespace

int main(int argc, char** argv) {
    const auto fixture = LoadTrainingCoreFixture(
        argc > 0 ? fs::path(argv[0]) : fs::path{}, argc > 1 ? argv[1] : nullptr);
    const auto& cases = fixture.at("cases");
    TestMaterializationMemoryGuardThresholds();
    TestTFIDFBlocksBeforeAllocationHeavyWork();
    TestBoundedTFIDFMaterialization();
    TestTFIDFNGramMaterialization();
    TestArrayFireCpuTrainingCoreTruth(cases);
    std::cout << "Computation truth TF-IDF + CrossEntropy + Linear + optimizer checks passed\n";
    return 0;
}
