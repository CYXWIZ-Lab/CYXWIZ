#include <cyxwiz/statistical_tests.h>
#include "core/arrow_dataset.h"
#include "core/data_analyzer.h"
#include "core/data_registry.h"
#include "core/pipeline_executor.h"
#include "core/data_budget.h"
#include "core/node_executors/pipeline_operator_factory.h"
#include "core/process_memory_snapshot.h"
#include "core/training_input_preparation.h"
#include "core/parquet_backed_dataset.h"
#include "core/sparse_feature_dataset.h"
#include "core/node_executors/preprocessing_operators.h"
#include "core/node_executors/regression_model_artifact.h"
#include "core/node_executors/regression_model_predictor_operator.h"
#include "core/node_executors/regression_operators.h"
#include "core/node_executors/differencing_operator.h"
#include "core/node_executors/sentiment_analyzer_operator.h"
#include "core/node_executors/time_series_split_operator.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::string JsonEscapePath(std::string path) {
    std::string escaped;
    escaped.reserve(path.size());
    for (char c : path) {
        if (c == '\\') {
            escaped += "\\\\";
        } else {
            escaped += c;
        }
    }
    return escaped;
}

std::shared_ptr<arrow::Array> FinishDoubleArray(
    const std::vector<double>& values) {
    arrow::DoubleBuilder builder;
    for (double value : values) {
        auto status = builder.Append(value);
        Check(status.ok(), status.ToString());
    }
    std::shared_ptr<arrow::Array> array;
    auto status = builder.Finish(&array);
    Check(status.ok(), status.ToString());
    return array;
}

std::shared_ptr<arrow::Array> FinishStringArray(
    const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    for (const auto& value : values) {
        auto status = builder.Append(value);
        Check(status.ok(), status.ToString());
    }
    std::shared_ptr<arrow::Array> array;
    auto status = builder.Finish(&array);
    Check(status.ok(), status.ToString());
    return array;
}

std::shared_ptr<arrow::Table> MakePreprocessingTable() {
    auto schema = arrow::schema({
        arrow::field("x", arrow::float64()),
        arrow::field("y", arrow::float64()),
    });
    return arrow::Table::Make(schema, {
        FinishDoubleArray({1.0, 2.0, 3.0, 4.0}),
        FinishDoubleArray({10.0, 20.0, 30.0, 40.0}),
    });
}

std::shared_ptr<arrow::Table> MakeMultiTargetPreprocessingTable() {
    auto schema = arrow::schema({
        arrow::field("x_0", arrow::float64()),
        arrow::field("x_1", arrow::float64()),
        arrow::field("y", arrow::float64()),
        arrow::field("y_1", arrow::float64()),
    });
    return arrow::Table::Make(schema, {
        FinishDoubleArray({1.0, 2.0, 3.0, 4.0}),
        FinishDoubleArray({2.0, 4.0, 6.0, 8.0}),
        FinishDoubleArray({10.0, 20.0, 30.0, 40.0}),
        FinishDoubleArray({11.0, 21.0, 31.0, 41.0}),
    });
}

double ReadNumericValue(const std::shared_ptr<arrow::Table>& table,
                        const std::string& column_name,
                        int64_t row_index) {
    const int column_index = table->schema()->GetFieldIndex(column_name);
    Check(column_index >= 0, "missing numeric column " + column_name);
    auto scalar_result = table->column(column_index)->GetScalar(row_index);
    Check(scalar_result.ok(), "numeric scalar can be read");
    auto scalar = *scalar_result;
    Check(scalar && scalar->is_valid, "numeric scalar is valid");

    switch (scalar->type->id()) {
    case arrow::Type::INT8:
        return std::static_pointer_cast<arrow::Int8Scalar>(scalar)->value;
    case arrow::Type::INT16:
        return std::static_pointer_cast<arrow::Int16Scalar>(scalar)->value;
    case arrow::Type::INT32:
        return std::static_pointer_cast<arrow::Int32Scalar>(scalar)->value;
    case arrow::Type::INT64:
        return static_cast<double>(
            std::static_pointer_cast<arrow::Int64Scalar>(scalar)->value);
    case arrow::Type::UINT8:
        return std::static_pointer_cast<arrow::UInt8Scalar>(scalar)->value;
    case arrow::Type::UINT16:
        return std::static_pointer_cast<arrow::UInt16Scalar>(scalar)->value;
    case arrow::Type::UINT32:
        return std::static_pointer_cast<arrow::UInt32Scalar>(scalar)->value;
    case arrow::Type::UINT64:
        return static_cast<double>(
            std::static_pointer_cast<arrow::UInt64Scalar>(scalar)->value);
    case arrow::Type::FLOAT:
        return std::static_pointer_cast<arrow::FloatScalar>(scalar)->value;
    case arrow::Type::DOUBLE:
        return std::static_pointer_cast<arrow::DoubleScalar>(scalar)->value;
    default:
        Check(false, "numeric scalar type should be supported");
        return 0.0;
    }
}

gui::MLNode MakeDataInputNode(int id, const std::string& source_name) {
    gui::MLNode node;
    node.id = id;
    node.type = gui::NodeType::DataInput;
    node.category = gui::NodeCategory::DataPipeline;
    node.name = "Parity Input";
    node.parameters["dataset_name"] = source_name;
    return node;
}

gui::MLNode MakeOperatorNode(
    int id,
    gui::NodeType type,
    const std::string& name,
    const std::map<std::string, std::string>& parameters) {
    gui::MLNode node;
    node.id = id;
    node.type = type;
    node.category = gui::NodeCategory::Preprocessing;
    node.name = name;
    node.parameters = parameters;
    return node;
}

std::string ParamsJson(
    const std::map<std::string, std::string>& parameters) {
    std::string out;
    bool first = true;
    for (const auto& [key, value] : parameters) {
        if (!first) {
            out += ",";
        }
        first = false;
        out += "\"" + key + "\":\"" + value + "\"";
    }
    return out;
}

void AssertTableContractEqual(
    const std::shared_ptr<arrow::Table>& executor_table,
    const std::shared_ptr<arrow::Table>& reference_table,
    const std::string& label) {
    Check(executor_table != nullptr, label + " executor table exists");
    Check(reference_table != nullptr, label + " operator table exists");
    Check(executor_table->num_rows() == reference_table->num_rows(),
          label + " row counts match");
    Check(executor_table->num_columns() == reference_table->num_columns(),
          label + " column counts match");
    for (int i = 0; i < executor_table->num_columns(); ++i) {
        Check(executor_table->schema()->field(i)->name() ==
                  reference_table->schema()->field(i)->name(),
              label + " schema field names match");
    }
}

void AssertNumericColumnClose(
    const std::shared_ptr<arrow::Table>& executor_table,
    const std::shared_ptr<arrow::Table>& reference_table,
    const std::string& column,
    const std::string& label) {
    for (int64_t row = 0; row < executor_table->num_rows(); ++row) {
        const double left = ReadNumericValue(executor_table, column, row);
        const double right = ReadNumericValue(reference_table, column, row);
        Check(std::fabs(left - right) < 0.0001,
              label + " values match for " + column);
    }
}

// The operator itself, configured and applied on `input`: the reference the
// pipeline run's result must equal.
std::unique_ptr<cyxwiz::IPipelineOperator> ConfiguredOperator(
    gui::NodeType node_type,
    const std::string& type_name,
    const std::map<std::string, std::string>& parameters) {
    auto op = cyxwiz::PipelineOperatorFactory::Instance().Create(node_type);
    Check(op != nullptr, type_name + " has a factory operator");
    std::string error;
    Check(op->Configure(parameters, error), type_name + " operator configures: " + error);
    return op;
}

std::shared_ptr<arrow::Table> ApplyOperatorDirectly(
    gui::NodeType node_type,
    const std::string& type_name,
    const std::map<std::string, std::string>& parameters,
    const std::shared_ptr<arrow::Table>& input) {
    auto op = ConfiguredOperator(node_type, type_name, parameters);
    auto result = op->Apply(input);
    Check(result.ok(), type_name + " operator applies: " + result.status().ToString());
    return result.ValueOrDie();
}

void RunParityCase(
    const std::filesystem::path& csv_path,
    const std::string& type_name,
    gui::NodeType node_type,
    int input_id,
    int operator_id,
    const std::map<std::string, std::string>& parameters,
    const std::vector<std::string>& numeric_columns) {

    auto& registry = cyxwiz::DataRegistry::Instance();
    registry.UnloadDataset("ds_datainput_" + std::to_string(input_id));
    registry.UnloadDataset("ds_operator_" + type_name + "_" +
                           std::to_string(operator_id));

    const std::string pipeline_json =
        R"({"nodes":[)"
        R"({"id":)" + std::to_string(input_id) +
        R"(,"type":"DataInput","name":"ParityInput","parameters":{)"
        R"("source_type":"file","file_path":")" +
        JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":)" + std::to_string(operator_id) +
        R"(,"type":")" + type_name +
        R"(","name":"ParityOperator","parameters":{)" +
        ParamsJson(parameters) +
        R"(}}],"links":[{"start_node":)" + std::to_string(input_id) +
        R"(,"end_node":)" + std::to_string(operator_id) + R"(}]})";

    cyxwiz::PipelineExecutor executor;
    Check(executor.ExecutePipeline(pipeline_json),
          type_name + " executor path succeeds: " + executor.GetLastError());

    auto source_dataset = registry.GetArrowDataset(
        "ds_datainput_" + std::to_string(input_id));
    Check(source_dataset != nullptr, type_name + " source dataset exists");
    auto source_table = source_dataset->GetArrowTable();
    Check(source_table != nullptr, type_name + " source table exists");

    auto executor_dataset = registry.GetArrowDataset(
        "ds_operator_" + type_name + "_" + std::to_string(operator_id));
    Check(executor_dataset != nullptr,
          type_name + " executor output dataset exists");
    auto executor_table = executor_dataset->GetArrowTable();
    // The reference: the same operator applied to the run's source table.
    const auto reference = ApplyOperatorDirectly(node_type, type_name, parameters, source_table);
    AssertTableContractEqual(executor_table, reference, type_name);
    if (numeric_columns.empty()) {
        for (int i = 0; i < executor_table->num_columns(); ++i) {
            AssertNumericColumnClose(executor_table,
                                     reference,
                                     executor_table->schema()->field(i)->name(),
                                     type_name);
        }
    } else {
        for (const auto& column : numeric_columns) {
            AssertNumericColumnClose(executor_table, reference, column,
                                     type_name);
        }
    }
}

// Sparse output (TOFIX178 S0-f): the pipeline run's CSR features equal the
// operator's own ApplySparse, value for value, with the same labels.
void RunSparseParityCase(
    const std::filesystem::path& csv_path,
    const std::string& type_name,
    gui::NodeType node_type,
    int input_id,
    int operator_id,
    const std::map<std::string, std::string>& parameters) {
    auto& registry = cyxwiz::DataRegistry::Instance();
    const std::string output_name = "ds_operator_" + type_name + "_" + std::to_string(operator_id);
    registry.UnregisterTabularDataset(output_name);
    const std::string pipeline_json =
        R"({"nodes":[{"id":)" + std::to_string(input_id) +
        R"(,"type":"DataInput","name":"ParityInput","parameters":{"source_type":"file","file_path":")" +
        JsonEscapePath(csv_path.string()) + R"(","type":"csv","has_header":"true"}},{"id":)" +
        std::to_string(operator_id) + R"(,"type":")" + type_name + R"(","name":"ParityOperator","parameters":{)" +
        ParamsJson(parameters) + R"(}}],"links":[{"start_node":)" + std::to_string(input_id) +
        R"(,"end_node":)" + std::to_string(operator_id) + R"(}]})";
    cyxwiz::PipelineExecutor executor;
    Check(executor.ExecutePipeline(pipeline_json), type_name + " sparse executor run: " + executor.GetLastError());
    const auto ran = registry.GetSparseFeatureDataset(output_name);
    Check(ran != nullptr, type_name + " the pipeline run registers sparse features");

    const auto source_table =
        registry.GetArrowDataset("ds_datainput_" + std::to_string(input_id))->GetArrowTable();
    auto op = ConfiguredOperator(node_type, type_name, parameters);
    auto direct = op->ApplySparse(source_table, output_name);
    Check(direct.ok() && direct.ValueOrDie(),
          type_name + " the operator's own sparse output: " + direct.status().ToString());
    const auto& made = *direct.ValueOrDie();
    Check(ran->GetNumRows() == made.GetNumRows() && ran->GetNumFeatures() == made.GetNumFeatures(),
          type_name + " sparse shapes match");
    Check(ran->GetRowOffsets() == made.GetRowOffsets() && ran->GetColumnIndices() == made.GetColumnIndices(),
          type_name + " sparse structure matches");
    Check(ran->GetValues().size() == made.GetValues().size(), type_name + " sparse value counts match");
    for (size_t i = 0; i < made.GetValues().size(); ++i)
        Check(std::fabs(ran->GetValues()[i] - made.GetValues()[i]) < 1e-6f, type_name + " sparse values match");
    Check(ran->GetFeatureNames() == made.GetFeatureNames(), type_name + " feature names match");
    Check(ran->GetLabelName() == made.GetLabelName() && ran->GetClassNames() == made.GetClassNames(),
          type_name + " labels match");
    Check(ran->GetLabels() && made.GetLabels() && ran->GetLabels()->Equals(*made.GetLabels()),
          type_name + " label values match");
    registry.UnregisterTabularDataset(output_name);
}

void TestStandardScalerEmitsMemoryPreflight() {
    cyxwiz::StandardScalerOperator op;
    std::string error;
    Check(op.Configure({
        {"columns", "x,y"},
        {"with_mean", "true"},
        {"with_std", "true"},
    }, error), error);

    std::vector<cyxwiz::PipelineOperatorProgress> progress_events;
    op.SetProgressCallback(
        [&](const cyxwiz::PipelineOperatorProgress& event) {
            progress_events.push_back(event);
        });

    auto result = op.Apply(MakePreprocessingTable());
    Check(result.ok(), result.status().ToString());
    const cyxwiz::PipelineOperatorProgress* preflight = nullptr;
    for (const auto& event : progress_events) {
        if (event.stage == "StandardScaler memory preflight") {
            preflight = &event;
            break;
        }
    }
    Check(preflight != nullptr,
          "StandardScaler should emit memory preflight progress");
    Check(preflight->status == "running",
          "safe StandardScaler preflight should stay in running status");
    Check(preflight->memory_risk_level == "safe",
          "safe StandardScaler preflight should report safe risk");
    Check(preflight->estimated_memory_bytes >
              4ULL * 2ULL * static_cast<uint64_t>(sizeof(double)),
          "StandardScaler preflight should include peak allocation overhead");
    Check(preflight->total_items == 4ULL * 2ULL,
          "StandardScaler preflight should report planned preprocessing cells");
    Check(preflight->message.find("Suggestion:") != std::string::npos,
          "StandardScaler preflight should include mitigation guidance");
    auto output = result.ValueOrDie();
    Check(output->num_rows() == 4,
          "StandardScaler preflight test should preserve row count");
}

// The first step's own memory estimate before training starts
// (EstimateFirstStepMemory, the GUI pre-start check): it stops at the first
// decision, applies nothing and identifies the step.
void TestFirstStepMemoryEstimate() {
    const auto source_table = MakePreprocessingTable();
    const std::vector<gui::MLNode> nodes = {
        MakeDataInputNode(471, "preflight_only_source"),
        MakeOperatorNode(
            472,
            gui::NodeType::StandardScaler,
            "Preflight StandardScaler",
            {{"columns", "x,y"},
             {"with_mean", "true"},
             {"with_std", "true"}}),
    };
    const std::vector<gui::NodeLink> links = {
        {1, 471, 0, 472, 0, gui::LinkType::TensorFlow},
    };

    const auto safe = cyxwiz::EstimateFirstStepMemory(nodes, links, 472, source_table, {});
    Check(safe.available && safe.error.empty(),
          "the first step publishes its memory estimate: " + safe.error);
    Check(safe.evidence.memory_risk_level == "safe",
          "small deterministic estimate should be safe");
    Check(safe.evidence.node_id == 472 &&
              safe.evidence.node_name == "Preflight StandardScaler",
          "the estimate should identify the responsible node");
    Check(safe.evidence.available_memory_bytes > 0 && safe.evidence.safe_memory_budget_bytes > 0,
          "the estimate carries the memory it was judged against");
    Check(source_table->Equals(*MakePreprocessingTable()),
          "the estimate must leave the source table untouched");

    const uint64_t estimated_peak = safe.evidence.estimated_memory_bytes;
    Check(estimated_peak > 0, "the estimate contains the estimated peak");
    cyxwiz::MaterializationMemoryContext warning_context;
    warning_context.snapshot_override =
        cyxwiz::MaterializationMemorySnapshot{
            estimated_peak * 2ULL, estimated_peak * 2ULL, true};
    const auto warning = cyxwiz::EstimateFirstStepMemory(nodes, links, 472, source_table, warning_context);
    Check(warning.available && warning.evidence.memory_risk_level == "warning",
          "deterministic 50 percent estimate should require warning handling");
    Check(warning.evidence.available_memory_bytes == estimated_peak * 2ULL,
          "the estimate is judged against the given memory");

    cyxwiz::MaterializationMemoryContext blocked_context;
    blocked_context.policy.hard_limit_bytes = 1;
    blocked_context.snapshot_override =
        cyxwiz::MaterializationMemorySnapshot{1024, 1024, true};
    const auto blocked = cyxwiz::EstimateFirstStepMemory(nodes, links, 472, source_table, blocked_context);
    Check(blocked.available && blocked.evidence.status == "blocked" &&
              blocked.evidence.memory_risk_level == "blocked",
          "a blocked estimate returns the operator's decision without applying it");

    // Invalid settings are an error, not an estimate.
    auto invalid = nodes;
    invalid[1].parameters["operation_mode"] = "bogus";
    const auto refused = cyxwiz::EstimateFirstStepMemory(invalid, links, 472, source_table, {});
    Check(!refused.available && refused.error.find("Preflight StandardScaler") != std::string::npos,
          "invalid settings name the step: " + refused.error);

    // Transform Only with a fitted-state file that is not there: the estimate
    // must not crash the pre-start check (was the materializer's preflight).
    auto missing_state = nodes;
    missing_state[1].parameters["operation_mode"] = "transform_only";
    missing_state[1].parameters["state_path"] =
        (std::filesystem::temp_directory_path() / "cyxwiz_absent_parent" / "fitted.cyxstate.json").string();
    try {
        const auto missing = cyxwiz::EstimateFirstStepMemory(missing_state, links, 472, source_table, {});
        // The step fails before it estimates: no estimate, and its reason
        // (the missing state file) instead of "no estimate required".
        Check(!missing.available && !missing.error.empty(),
              "a step that cannot load its fitted state says why: " + missing.error);
    } catch (const std::exception& e) {
        Check(false, std::string("a missing fitted state must not throw out of the estimate: ") + e.what());
    }
}

void TestStandardScalerExcludesMultipleTargetColumns() {
    cyxwiz::StandardScalerOperator op;
    std::string error;
    Check(op.Configure({
        {"columns", ""},
        {"label_col", "y"},
        {"exclude_columns", "y_1"},
        {"with_mean", "true"},
        {"with_std", "true"},
    }, error), error);

    auto result = op.Apply(MakeMultiTargetPreprocessingTable());
    Check(result.ok(), result.status().ToString());
    auto output = result.ValueOrDie();
    Check(std::abs(ReadNumericValue(output, "x_0", 0) - 1.0) > 0.1,
          "StandardScaler should transform auto-detected feature x_0");
    Check(std::abs(ReadNumericValue(output, "x_1", 0) - 2.0) > 0.1,
          "StandardScaler should transform auto-detected feature x_1");
    Check(ReadNumericValue(output, "y", 0) == 10.0,
          "StandardScaler should preserve the primary target");
    Check(ReadNumericValue(output, "y_1", 0) == 11.0,
          "StandardScaler should preserve additional excluded targets");

    Check(op.Configure({
        {"columns", ""},
        {"exclude_columns", "missing_target"},
    }, error), error);
    auto invalid = op.Apply(MakeMultiTargetPreprocessingTable());
    Check(!invalid.ok() &&
              invalid.status().ToString().find(
                  "excluded column 'missing_target' not found") !=
                  std::string::npos,
          "StandardScaler should reject a misspelled excluded column");
}

void TestRegressionMemoryPreflight() {
    const auto input = MakePreprocessingTable();

    cyxwiz::LinearRegressionOperator linear;
    std::string error;
    Check(linear.Configure({
        {"feature_cols", "x"},
        {"target_col", "y"},
        {"fit_intercept", "true"},
    }, error), error);
    std::vector<cyxwiz::PipelineOperatorProgress> linear_progress;
    linear.SetProgressCallback(
        [&](const cyxwiz::PipelineOperatorProgress& event) {
            linear_progress.push_back(event);
        });
    auto linear_result = linear.Apply(input);
    Check(linear_result.ok(), linear_result.status().ToString());
    Check(!linear_progress.empty(),
          "LinearRegression should emit memory preflight");
    Check(linear_progress.front().stage ==
              "LinearRegression memory preflight",
          "LinearRegression should preflight before reading dense columns");
    Check(linear_progress.front().memory_risk_level == "safe",
          "small LinearRegression fixture should report safe memory risk");
    Check(linear_progress.front().estimated_memory_bytes >
              4ULL * 6ULL * static_cast<uint64_t>(sizeof(double)),
          "LinearRegression estimate should include regression workspace");

    cyxwiz::PolynomialRegressionOperator polynomial;
    Check(polynomial.Configure({
        {"feature_col", "x"},
        {"target_col", "y"},
        {"degree", "2"},
    }, error), error);
    std::vector<cyxwiz::PipelineOperatorProgress> polynomial_progress;
    polynomial.SetProgressCallback(
        [&](const cyxwiz::PipelineOperatorProgress& event) {
            polynomial_progress.push_back(event);
        });
    auto polynomial_result = polynomial.Apply(input);
    Check(polynomial_result.ok(), polynomial_result.status().ToString());
    Check(!polynomial_progress.empty(),
          "PolynomialRegression should emit memory preflight");
    Check(polynomial_progress.front().stage ==
              "PolynomialRegression memory preflight",
          "PolynomialRegression should preflight before reading dense columns");
    Check(polynomial_progress.front().memory_risk_level == "safe",
          "small PolynomialRegression fixture should report safe memory risk");
    Check(polynomial_progress.front().estimated_memory_bytes >
              4ULL * 7ULL * static_cast<uint64_t>(sizeof(double)),
          "PolynomialRegression estimate should include degree workspace");

    cyxwiz::MaterializationMemoryContext blocked_context;
    blocked_context.policy.hard_limit_bytes = 1;
    blocked_context.snapshot_override = cyxwiz::MaterializationMemorySnapshot{
        1024, 1024, true};
    linear.SetMaterializationMemoryContext(blocked_context);
    linear_progress.clear();
    const auto blocked_result = linear.Apply(input);
    Check(!blocked_result.ok() && blocked_result.status().IsCapacityError(),
          "LinearRegression should block before dense column reads");
    Check(linear_progress.size() == 1,
          "blocked LinearRegression should emit only its preflight event");
    Check(linear_progress.front().status == "blocked",
          "blocked LinearRegression should expose blocked status");
}

std::shared_ptr<arrow::Table> MakeNoisyRegressionTable() {
    auto schema = arrow::schema({
        arrow::field("x", arrow::float64()),
        arrow::field("y", arrow::float64()),
    });
    return arrow::Table::Make(schema, {
        FinishDoubleArray({1.0, 2.0, 3.0, 4.0, 5.0}),
        FinishDoubleArray({2.0, 4.0, 5.0, 9.0, 10.0}),
    });
}

void TestRegressionMetricDefinitions() {
    // The facts a model artifact keeps: MSE = SSE / n (RMSE its root) and the
    // residual variance = SSE / df (the residual SE its root), from the
    // solver the regression nodes use.
    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> y = {2.0, 4.0, 5.0, 9.0, 10.0};
    const auto fit = cyxwiz::StatisticalTests::FitLinear({x}, y, {"x"});
    Check(fit.ok && fit.residuals.size() == x.size(), "FitLinear should produce one residual per observation");
    double sse = 0.0;
    for (double residual : fit.residuals) sse += residual * residual;
    Check(std::abs(fit.rmse - std::sqrt(sse / static_cast<double>(fit.n))) < 1e-9,
          "RMSE should use mean prediction error SSE/n");
    Check(std::abs(fit.rse - std::sqrt(sse / static_cast<double>(fit.df_resid))) < 1e-9,
          "the residual SE should retain SSE/df_resid");
    Check(fit.rse > fit.rmse, "non-perfect fixture should distinguish residual variance from MSE");
}

void TestRegressionFittedModelArtifact() {
    const auto artifact_path = std::filesystem::temp_directory_path() /
        "cyxwiz_linear_regression_model_test.cyxregression.json";
    const auto legacy_artifact_path = std::filesystem::temp_directory_path() /
        "cyxwiz_linear_regression_model_legacy.cyxregression.json";
    std::filesystem::remove(artifact_path);
    std::filesystem::remove(legacy_artifact_path);

    cyxwiz::LinearRegressionOperator fit;
    std::string error;
    Check(fit.Configure({
        {"feature_cols", "x"},
        {"target_col", "y"},
        {"fit_intercept", "true"},
        {"model_path", artifact_path.string()},
    }, error), error);
    const auto fitted = fit.Apply(MakeNoisyRegressionTable());
    Check(fitted.ok(), fitted.status().ToString());
    Check(std::filesystem::is_regular_file(artifact_path),
          "LinearRegression should produce its fitted Model artifact");

    cyxwiz::RegressionModelArtifact artifact;
    Check(cyxwiz::LoadRegressionModelArtifact(
              artifact_path.string(), artifact, &error), error);
    Check(artifact.type == cyxwiz::RegressionModelType::Linear &&
              artifact.feature_names == std::vector<std::string>{"x"} &&
              artifact.target_name == "y" &&
              artifact.coefficients.size() == 2,
          "fitted artifact should preserve regression type, schema, and coefficients");

    cyxwiz::RegressionModelPredictorOperator predict;
    Check(predict.Configure({
        {"model_path", artifact_path.string()},
        {"prediction_col", "scored_y"},
    }, error), error);
    const auto predicted = predict.Apply(MakeNoisyRegressionTable());
    Check(predicted.ok(), predicted.status().ToString());
    const auto scored = predicted.ValueOrDie();
    const auto fitted_table = fitted.ValueOrDie();
    Check(std::abs(ReadNumericValue(scored, "scored_y", 0) -
                   ReadNumericValue(fitted_table, "prediction", 0)) < 1e-6 &&
              std::abs(ReadNumericValue(scored, "scored_y", 4) -
                       ReadNumericValue(fitted_table, "prediction", 4)) < 1e-6,
          "predictor should reproduce the fitted linear equation");
    double artifact_sse = 0.0;
    for (int64_t row = 0; row < fitted_table->num_rows(); ++row) {
        const double residual = ReadNumericValue(fitted_table, "residual", row);
        artifact_sse += residual * residual;
    }
    Check(std::abs(artifact.mse - artifact_sse / 5.0) < 1e-5 &&
              artifact.residual_variance > artifact.mse,
          "artifact should persist distinct prediction and residual metrics");

    const auto missing_feature = arrow::Table::Make(
        arrow::schema({arrow::field("z", arrow::float64())}),
        {FinishDoubleArray({1.0, 2.0})});
    const auto rejected = predict.Apply(missing_feature);
    Check(!rejected.ok() && rejected.status().IsKeyError(),
          "predictor should fail closed when an artifact feature is missing");

    {
        const nlohmann::json legacy = {
            {"format", "cyxwiz_regression_model"},
            {"version", 1},
            {"model_type", "LinearRegression"},
            {"model", {
                {"feature_names", {"x"}},
                {"target_name", "y"},
                {"fit_intercept", true},
                {"degree", 1},
                {"coefficients", {1.0, 2.0}},
                {"sample_count", 5},
                {"metrics", {
                    {"r_squared", 0.5},
                    {"adjusted_r_squared", 0.3},
                    {"mse", 4.0},
                    {"rmse", 2.0},
                    {"mae", 1.0},
                }},
            }},
        };
        std::ofstream output(legacy_artifact_path);
        output << legacy.dump(2);
    }
    cyxwiz::RegressionModelArtifact migrated;
    Check(cyxwiz::LoadRegressionModelArtifact(
              legacy_artifact_path.string(), migrated, &error), error);
    Check(std::abs(migrated.residual_variance - 4.0) < 1e-12 &&
              std::abs(migrated.residual_standard_error - 2.0) < 1e-12 &&
              std::abs(migrated.mse - 2.4) < 1e-12 &&
              std::abs(migrated.rmse - std::sqrt(2.4)) < 1e-12,
          "legacy version-1 artifacts should migrate old SSE/df labels safely");
    std::filesystem::remove(artifact_path);
    std::filesystem::remove(legacy_artifact_path);
}

void TestRemainingRowTransformMemoryPreflight() {
    const auto numeric_input = MakePreprocessingTable();
    const auto text_input = arrow::Table::Make(
        arrow::schema({
            arrow::field("text", arrow::utf8()),
            arrow::field("label", arrow::float64()),
        }),
        {
            FinishStringArray({
                "good bright day", "bad dull day",
                "good calm night", "bad storm night"}),
            FinishDoubleArray({1.0, 0.0, 1.0, 0.0}),
        });

    auto assert_safe_preflight = [&](cyxwiz::IPipelineOperator& op,
                                     const std::shared_ptr<arrow::Table>& input,
                                     const std::string& expected_stage,
                                     uint64_t raw_lower_bound) {
        std::vector<cyxwiz::PipelineOperatorProgress> events;
        op.SetProgressCallback(
            [&](const cyxwiz::PipelineOperatorProgress& event) {
                events.push_back(event);
            });
        const auto result = op.Apply(input);
        Check(result.ok(), result.status().ToString());
        Check(!events.empty() && events.front().stage == expected_stage,
              expected_stage + " should precede materializing progress");
        Check(events.front().memory_risk_level == "safe",
              expected_stage + " should report safe risk for the fixture");
        Check(events.front().estimated_memory_bytes > raw_lower_bound,
              expected_stage + " should include peak allocation overhead");
        Check(events.front().message.find("Suggestion:") != std::string::npos,
              expected_stage + " should include mitigation guidance");
    };

    std::string error;
    cyxwiz::LogTransformOperator log;
    Check(log.Configure({{"columns", "y"}}, error), error);
    assert_safe_preflight(
        log,
        numeric_input,
        "LogTransform memory preflight",
        4ULL * 2ULL * static_cast<uint64_t>(sizeof(float)));

    cyxwiz::DifferencingOperator differencing;
    Check(differencing.Configure({
        {"value_col", "y"}, {"lag", "1"}, {"order", "1"}}, error), error);
    assert_safe_preflight(
        differencing,
        numeric_input,
        "Differencing memory preflight",
        4ULL * 3ULL * static_cast<uint64_t>(sizeof(float)));

    cyxwiz::TimeSeriesSplitOperator split;
    Check(split.Configure({
        {"train_ratio", "0.5"},
        {"val_ratio", "0.25"},
        {"test_ratio", "0.25"},
    }, error), error);
    assert_safe_preflight(
        split,
        numeric_input,
        "TimeSeriesSplit memory preflight",
        4ULL * 3ULL * static_cast<uint64_t>(sizeof(int64_t)));

    cyxwiz::SentimentAnalyzerOperator sentiment;
    Check(sentiment.Configure({
        {"text_col", "text"},
        {"label_col", "label"},
        {"method", "simple"},
    }, error), error);
    assert_safe_preflight(
        sentiment,
        text_input,
        "SentimentAnalyzer memory preflight",
        4ULL * 5ULL * static_cast<uint64_t>(sizeof(float)));

    cyxwiz::MaterializationMemoryContext blocked_context;
    blocked_context.policy.hard_limit_bytes = 1;
    blocked_context.snapshot_override = cyxwiz::MaterializationMemorySnapshot{
        1024, 1024, true};
    log.SetMaterializationMemoryContext(blocked_context);
    std::vector<cyxwiz::PipelineOperatorProgress> blocked_events;
    log.SetProgressCallback(
        [&](const cyxwiz::PipelineOperatorProgress& event) {
            blocked_events.push_back(event);
        });
    const auto blocked_result = log.Apply(numeric_input);
    Check(!blocked_result.ok() && blocked_result.status().IsCapacityError(),
          "LogTransform should return CapacityError at its preflight boundary");
    Check(blocked_events.size() == 1 &&
              blocked_events.front().stage == "LogTransform memory preflight" &&
              blocked_events.front().status == "blocked",
          "blocked LogTransform should stop before reading its numeric column");
}

// The pipeline nodes as the canvas has them: one input and one output pin.
gui::MLNode WithPins(gui::MLNode node) {
    node.inputs = {{0, gui::PinType::Tensor, "in", true}};
    node.outputs = {{0, gui::PinType::Tensor, "out", false}};
    return node;
}

std::string RunJson(const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links) {
    nlohmann::json pipeline;
    std::string error;
    Check(cyxwiz::PipelineJsonFromGraph(nodes, links, pipeline, error), "pipeline JSON: " + error);
    return pipeline.dump();
}

// A pipeline run stops cleanly: a cancel before or during a long step, and
// the runtime memory guard, publish no result for the step.
void TestPipelineRunStructuredFailures() {
    namespace fs = std::filesystem;
    auto& registry = cyxwiz::DataRegistry::Instance();
    const fs::path csv_path = fs::temp_directory_path() / "cyxwiz_pipeline_run_failures.csv";
    {
        std::ofstream csv(csv_path);
        csv << "x,y\n1,10\n2,20\n3,30\n4,40\n";
    }
    auto input = WithPins(MakeDataInputNode(901, "pipeline_failure_source"));
    input.parameters = {{"source_type", "file"}, {"file_path", csv_path.string()},
                        {"type", "csv"}, {"has_header", "true"}};
    const std::vector<gui::MLNode> nodes = {
        input,
        WithPins(MakeOperatorNode(902, gui::NodeType::LogTransform, "Failure LogTransform", {{"columns", "y"}})),
    };
    const std::vector<gui::NodeLink> links = {{1, 901, 0, 902, 0, gui::LinkType::TensorFlow}};
    const std::string log_output = "ds_operator_LogTransform_902";

    // Cancelled before the steps finish: no result, the cancel is reported.
    const auto roots = cyxwiz::StepRunRoots{};
    const auto cancelled = cyxwiz::PrepareStepResult(nodes, links, 902, registry, roots, {}, [] { return true; });
    Check(!cancelled.ok && cancelled.cancelled && cancelled.ran && cancelled.dataset.empty(),
          "a cancelled preparation is a cancel, not a result");
    Check(registry.GetArrowDataset(log_output) == nullptr, "a cancelled preparation publishes no result");

    // Cancelled while a long row operator runs: it stops cooperatively.
    {
        auto text = input;
        text.id = 903;
        const fs::path text_csv = fs::temp_directory_path() / "cyxwiz_pipeline_run_cancel_text.csv";
        {
            std::ofstream csv(text_csv);
            csv << "text\ngood bright day\nbad dull day\ngood calm night\nbad storm night\n";
        }
        text.parameters["file_path"] = text_csv.string();
        const std::vector<gui::MLNode> sentiment_nodes = {
            text,
            WithPins(MakeOperatorNode(904, gui::NodeType::SentimentAnalyzer, "Cancellable Sentiment",
                                      {{"text_col", "text"}, {"method", "simple"}})),
        };
        const std::vector<gui::NodeLink> sentiment_links = {{2, 903, 0, 904, 0, gui::LinkType::TensorFlow}};
        cyxwiz::PipelineExecutor executor;
        bool cancel_sent = false;
        executor.SetProgressCallback([&executor, &cancel_sent](float, const std::string& message) {
            if (!cancel_sent && message.find("Cancellable Sentiment") != std::string::npos) {
                cancel_sent = true;
                executor.RequestCancel();
            }
        });
        const bool ran = executor.ExecutePipeline(RunJson(sentiment_nodes, sentiment_links));
        Check(cancel_sent, "the sentiment step reports progress while it runs");
        Check(!ran && registry.GetArrowDataset("ds_operator_SentimentAnalyzer_904") == nullptr,
              "a cancel during a long row operator stops it and publishes no result: " + executor.GetLastError());
        executor.ReleaseAll();
        fs::remove(text_csv);
    }

    // The runtime memory guard: the process grows past the safe budget while
    // the step runs; the step stops, named, with the numbers.
    {
        cyxwiz::PipelineExecutor executor;
        auto samples = std::make_shared<int>(0);
        executor.SetProcessMemoryProbe([samples]() {
            cyxwiz::ProcessMemorySnapshot snapshot;
            snapshot.detected = true;
            snapshot.resident_bytes = (*samples)++ == 0 ? 100 : 100 + (1ULL << 62);
            snapshot.private_bytes = snapshot.resident_bytes;
            snapshot.source = "deterministic test";
            return snapshot;
        });
        const bool ran = executor.ExecutePipeline(RunJson(nodes, links));
        Check(!ran && executor.GetLastError().find("the memory guard stopped 'Failure LogTransform'") !=
                          std::string::npos,
              "actual process growth trips the runtime memory guard: " + executor.GetLastError());
        Check(registry.GetArrowDataset(log_output) == nullptr,
              "the guarded step publishes no partial result");
        executor.ReleaseAll();
    }

    // A clean run gives the result.
    const auto done = cyxwiz::PrepareStepResult(nodes, links, 902, registry, roots, {});
    Check(done.ok && done.ran && !done.from_disk && done.dataset == log_output && registry.GetArrowDataset(log_output),
          "an uncancelled preparation gives the step's result: " + done.error);
    registry.UnregisterTabularDataset(log_output);
    registry.UnregisterTabularDataset("ds_datainput_901");
    fs::remove(csv_path);
}

} // namespace

// A training set kept on disk streams through the pipeline run's steps, a
// row group at a time (TOFIX154 step 6b): the result on disk holds what the
// operators compute in memory, and a step that needs every row stops with
// the reason when the table cannot be read whole.
void TestStreamedRunMatchesMemory() {
    constexpr int kRows = 10000;
    std::vector<double> x, y;
    for (int i = 0; i < kRows; ++i) {
        x.push_back(std::sin(i * 0.37) * 50.0 + i * 0.01);
        y.push_back(1.0 + (i % 97) * 3.5);
    }
    auto table = arrow::Table::Make(
        arrow::schema({arrow::field("x", arrow::float64()), arrow::field("y", arrow::float64())}),
        {FinishDoubleArray(x), FinishDoubleArray(y)});

    const auto path = std::filesystem::temp_directory_path() / "cyxwiz_streamed_pipeline_run.parquet";
    {
        auto file = arrow::io::FileOutputStream::Open(path.string());
        Check(file.ok(), "streamed fixture file opens");
        auto status = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *file, 1500);
        Check(status.ok(), status.ToString());
        Check((*file)->Close().ok(), "streamed fixture closes");
    }
    const auto cache_root = std::filesystem::temp_directory_path() / "cyxwiz_streamed_pipeline_run_cache";
    std::filesystem::remove_all(cache_root);

    const std::map<std::string, std::string> scale = {{"columns", "x"}, {"with_mean", "true"}, {"with_std", "true"}};
    const std::map<std::string, std::string> range = {{"columns", "y"}, {"min", "0"}, {"max", "1"}};
    const std::map<std::string, std::string> log = {{"columns", "y"}};
    auto input = WithPins(MakeDataInputNode(901, "streamed_source"));
    input.parameters = {{"source_type", "file"}, {"file_path", path.string()},
                        {"type", "parquet"}, {"force_disk_backed", "true"}};
    std::vector<gui::MLNode> nodes = {
        input,
        WithPins(MakeOperatorNode(902, gui::NodeType::StandardScaler, "Scale x", scale)),
        WithPins(MakeOperatorNode(903, gui::NodeType::MinMaxScaler, "Range y", range)),
        WithPins(MakeOperatorNode(904, gui::NodeType::LogTransform, "Log y", log)),
    };
    std::vector<gui::NodeLink> links = {
        {1, 901, 0, 902, 0, gui::LinkType::TensorFlow},
        {2, 902, 0, 903, 0, gui::LinkType::TensorFlow},
        {3, 903, 0, 904, 0, gui::LinkType::TensorFlow},
    };

    // The reference: the three operators applied in memory, one after another.
    auto memory = ApplyOperatorDirectly(gui::NodeType::StandardScaler, "StandardScaler", scale, table);
    memory = ApplyOperatorDirectly(gui::NodeType::MinMaxScaler, "MinMaxScaler", range, memory);
    memory = ApplyOperatorDirectly(gui::NodeType::LogTransform, "LogTransform", log, memory);

    // Under a 1-byte data budget nothing is read whole: each step streams and
    // its result stays on disk. (The memory guard is out of the way: its
    // limit is the budget.)
    auto& registry = cyxwiz::DataRegistry::Instance();
    cyxwiz::SetDataBudgetBytes(1);
    const auto no_probe = [] { return cyxwiz::ProcessMemorySnapshot{}; };
    {
        cyxwiz::PipelineExecutor executor;
        executor.SetIngestionCacheRoot(cache_root.string());
        executor.SetProcessMemoryProbe(no_probe);
        Check(executor.ExecutePipeline(RunJson(nodes, links)), "the disk run succeeds: " + executor.GetLastError());
        Check(registry.IsParquetBackedDataset("ds_datainput_901") && !registry.GetArrowDataset("ds_datainput_901_in_memory"),
              "the input stays on disk (not read whole)");
        auto prepared = registry.GetParquetBackedDataset("ds_operator_LogTransform_904");
        Check(prepared != nullptr, "the last step's result is kept on disk");
        std::vector<int> groups;
        for (int i = 0; i < prepared->GetNumRowGroups(); ++i) groups.push_back(i);
        auto prepared_table = prepared->ReadRowGroups(groups);
        AssertTableContractEqual(memory, prepared_table, "streamed");
        for (const char* column : {"x", "y"}) {
            for (int64_t row = 0; row < kRows; ++row) {
                const double left = ReadNumericValue(memory, column, row);
                const double right = ReadNumericValue(prepared_table, column, row);
                Check(std::fabs(left - right) < 1e-9 * std::max(1.0, std::fabs(left)),
                      std::string("streamed values match for ") + column + " row " + std::to_string(row));
            }
        }
        prepared.reset();
        executor.ReleaseAll();
    }

    // RobustScaler needs medians over every row: on disk under the budget it
    // stops before loading, and says why.
    {
        std::vector<gui::MLNode> robust = {
            input,
            WithPins(MakeOperatorNode(902, gui::NodeType::RobustScaler, "Robust x",
                                      {{"columns", "x"}, {"with_centering", "true"}, {"with_scaling", "true"}})),
        };
        cyxwiz::PipelineExecutor executor;
        executor.SetIngestionCacheRoot(cache_root.string());
        executor.SetProcessMemoryProbe(no_probe);
        Check(!executor.ExecutePipeline(RunJson(robust, {links[0]})) &&
                  executor.GetLastError().find("over the data budget") != std::string::npos,
              "a whole-table step on disk stops with the reason: " + executor.GetLastError());
        Check(registry.GetArrowDataset("ds_operator_RobustScaler_902") == nullptr,
              "the refused step publishes nothing");
        executor.ReleaseAll();
    }
    cyxwiz::SetDataBudgetBytes(0);

    std::error_code error;
    std::filesystem::remove(path, error);
    std::filesystem::remove_all(cache_root, error);
}

int main() {
    TestStreamedRunMatchesMemory();
    TestStandardScalerEmitsMemoryPreflight();
    TestFirstStepMemoryEstimate();
    TestStandardScalerExcludesMultipleTargetColumns();
    TestRegressionMemoryPreflight();
    TestRegressionMetricDefinitions();
    TestRegressionFittedModelArtifact();
    TestRemainingRowTransformMemoryPreflight();
    TestPipelineRunStructuredFailures();

    const auto csv_path = std::filesystem::temp_directory_path() /
        "cyxwiz_operator_materializer_parity.csv";
    {
        std::ofstream csv(csv_path);
        csv << "x,y\n";
        csv << "1,10\n";
        csv << "2,20\n";
        csv << "3,30\n";
        csv << "4,40\n";
    }

    RunParityCase(csv_path, "StandardScaler", gui::NodeType::StandardScaler,
                  501, 502,
                  {{"columns", "x"}, {"with_mean", "true"},
                   {"with_std", "true"}},
                  {"x"});
    RunParityCase(csv_path, "MinMaxScaler", gui::NodeType::MinMaxScaler,
                  503, 504,
                  {{"columns", "x"}, {"min", "0"}, {"max", "1"}},
                  {"x"});
    RunParityCase(csv_path, "RobustScaler", gui::NodeType::RobustScaler,
                  505, 506,
                  {{"columns", "x"}, {"with_centering", "true"},
                   {"with_scaling", "true"}},
                  {"x"});
    RunParityCase(csv_path, "LogTransform", gui::NodeType::LogTransform,
                  507, 508, {{"columns", "y"}}, {"y"});
    RunParityCase(csv_path, "Differencing", gui::NodeType::Differencing,
                  509, 510,
                  {{"value_col", "y"}, {"lag", "1"}, {"order", "1"}},
                  {"y"});

    const auto categorical_csv_path = std::filesystem::temp_directory_path() /
        "cyxwiz_operator_materializer_parity_categorical.csv";
    {
        std::ofstream csv(categorical_csv_path);
        csv << "category,group,score,target\n";
        csv << "red,a,1,0\n";
        csv << "blue,b,2,1\n";
        csv << "red,a,3,0\n";
        csv << "green,b,50,1\n";
    }

    RunParityCase(categorical_csv_path, "LabelEncoder",
                  gui::NodeType::LabelEncoder, 601, 602,
                  {{"column", "category"}},
                  {"category"});
    RunParityCase(categorical_csv_path, "OrdinalEncoder",
                  gui::NodeType::OrdinalEncoder, 603, 604,
                  {{"columns", "category,group"}},
                  {"category", "group"});
    RunParityCase(categorical_csv_path, "TargetEncoder",
                  gui::NodeType::TargetEncoder, 605, 606,
                  {{"columns", "category"}, {"target_col", "target"},
                   {"smoothing", "1"}},
                  {"category"});
    RunParityCase(categorical_csv_path, "OutlierDetector",
                  gui::NodeType::OutlierDetector, 607, 608,
                  {{"columns", "score"}, {"method", "zscore"},
                   {"threshold", "1.0"}},
                  {"is_outlier"});

    const auto text_csv_path = std::filesystem::temp_directory_path() /
        "cyxwiz_operator_materializer_parity_text.csv";
    {
        std::ofstream csv(text_csv_path);
        csv << "text,label\n";
        csv << "good bright day,1\n";
        csv << "bad dull day,0\n";
        csv << "good calm night,1\n";
        csv << "bad storm night,0\n";
    }

    RunParityCase(text_csv_path, "TextTokenizer",
                  gui::NodeType::TextTokenizer, 701, 702,
                  {{"text_col", "text"}, {"label_col", "label"},
                   {"max_length", "4"}, {"tokenizer_type", "1"},
                   {"min_word_freq", "1"}, {"max_vocab_size", "20"},
                   {"lowercase", "true"}},
                  {"tok_0", "tok_1", "tok_2", "tok_3", "y"});
    RunParityCase(text_csv_path, "CountVectorizer",
                  gui::NodeType::CountVectorizer, 703, 704,
                  {{"text_col", "text"}, {"label_col", "label"},
                   {"max_features", "4"}, {"norm", "none"},
                   {"binary", "false"}, {"ngram_range", "1,1"},
                   {"stop_words", "none"}, {"output_format", "dense"}},
                  {});
    RunParityCase(text_csv_path, "TFIDFVectorizer",
                  gui::NodeType::TFIDFVectorizer, 705, 706,
                  {{"text_col", "text"}, {"label_col", "label"},
                   {"max_features", "4"}, {"min_df", "1"},
                   {"use_idf", "true"}, {"smooth_idf", "true"},
                   {"norm", "none"}, {"ngram_range", "1,1"},
                   {"stop_words", "none"}, {"output_format", "dense"}},
                  {});
    RunSparseParityCase(text_csv_path, "TFIDFVectorizer", gui::NodeType::TFIDFVectorizer, 711, 712,
                        {{"text_col", "text"}, {"label_col", "label"}, {"max_features", "4"}, {"min_df", "1"},
                         {"use_idf", "true"}, {"smooth_idf", "true"}, {"norm", "l2"}, {"ngram_range", "1,1"},
                         {"stop_words", "none"}, {"output_format", "sparse"}});
    RunSparseParityCase(text_csv_path, "CountVectorizer", gui::NodeType::CountVectorizer, 713, 714,
                        {{"text_col", "text"}, {"label_col", "label"}, {"max_features", "4"}, {"min_df", "1"},
                         {"binary", "false"}, {"ngram_range", "1,1"}, {"stop_words", "none"},
                         {"output_format", "sparse"}});
    RunParityCase(text_csv_path, "SentimentAnalyzer",
                  gui::NodeType::SentimentAnalyzer, 707, 708,
                  {{"text_col", "text"}, {"label_col", "label"},
                   {"method", "simple"}},
                  {});

    const auto pca_csv_path = std::filesystem::temp_directory_path() /
        "cyxwiz_operator_materializer_parity_pca.csv";
    {
        std::ofstream csv(pca_csv_path);
        csv << "a,b,c,label\n";
        csv << "1,2,3,0\n";
        csv << "2,3,4,1\n";
        csv << "3,5,7,0\n";
        csv << "4,7,10,1\n";
    }

    RunParityCase(pca_csv_path, "PCANode", gui::NodeType::PCANode, 709, 710,
                  {{"feature_cols", "a,b,c"}, {"label_col", "label"},
                   {"n_components", "2"}, {"center", "true"},
                   {"scale", "false"}},
                  {"pc_0", "pc_1", "y"});

    const auto time_series_csv_path = std::filesystem::temp_directory_path() /
        "cyxwiz_operator_materializer_parity_time_series.csv";
    {
        std::ofstream csv(time_series_csv_path);
        csv << "time,value,aux\n";
        csv << "1,10,100\n";
        csv << "2,20,101\n";
        csv << "3,30,102\n";
        csv << "4,40,103\n";
        csv << "5,50,104\n";
        csv << "6,60,105\n";
    }

    RunParityCase(time_series_csv_path, "TimeSeriesWindow",
                  gui::NodeType::TimeSeriesWindow, 801, 802,
                  {{"value_col", "value"}, {"feature_cols", "aux"},
                   {"time_col", "time"}, {"input_width", "2"},
                   {"label_width", "1"}, {"shift", "1"}},
                  {});
    RunParityCase(time_series_csv_path, "TimeSeriesFeatures",
                  gui::NodeType::TimeSeriesFeatures, 803, 804,
                  {{"value_col", "value"}, {"lag_values", "1,2"},
                   {"rolling_windows", "2"},
                   {"rolling_aggregations", "mean,std"}},
                  {});
    RunParityCase(time_series_csv_path, "TimeSeriesSplit",
                  gui::NodeType::TimeSeriesSplit, 805, 806,
                  {{"train_ratio", "0.5"}, {"val_ratio", "0.25"},
                   {"test_ratio", "0.25"}},
                  {});

    std::filesystem::remove(csv_path);
    std::filesystem::remove(categorical_csv_path);
    std::filesystem::remove(text_csv_path);
    std::filesystem::remove(pca_csv_path);
    std::filesystem::remove(time_series_csv_path);
    std::cout << "Pipeline operator materializer parity passed\n";
    return 0;
}
