#include "core/arrow_dataset.h"
#include "core/data_registry.h"
#include "core/pipeline_executor.h"
#include "core/pipeline_materializer.h"
#include "core/node_executors/preprocessing_operators.h"

#include <arrow/api.h>

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
    const std::shared_ptr<arrow::Table>& materialized_table,
    const std::string& label) {
    Check(executor_table != nullptr, label + " executor table exists");
    Check(materialized_table != nullptr, label + " materializer table exists");
    Check(executor_table->num_rows() == materialized_table->num_rows(),
          label + " row counts match");
    Check(executor_table->num_columns() == materialized_table->num_columns(),
          label + " column counts match");
    for (int i = 0; i < executor_table->num_columns(); ++i) {
        Check(executor_table->schema()->field(i)->name() ==
                  materialized_table->schema()->field(i)->name(),
              label + " schema field names match");
    }
}

void AssertNumericColumnClose(
    const std::shared_ptr<arrow::Table>& executor_table,
    const std::shared_ptr<arrow::Table>& materialized_table,
    const std::string& column,
    const std::string& label) {
    for (int64_t row = 0; row < executor_table->num_rows(); ++row) {
        const double left = ReadNumericValue(executor_table, column, row);
        const double right = ReadNumericValue(materialized_table, column, row);
        Check(std::fabs(left - right) < 0.0001,
              label + " values match for " + column);
    }
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

    const std::string source_name =
        "parity_source_" + type_name + "_" + std::to_string(operator_id);
    std::vector<gui::MLNode> nodes = {
        MakeDataInputNode(input_id, source_name),
        MakeOperatorNode(operator_id, node_type, type_name, parameters),
    };
    std::vector<gui::NodeLink> links = {
        {1, input_id, 0, operator_id, 0, gui::LinkType::TensorFlow},
    };

    auto materialized = cyxwiz::PipelineMaterializer::MaterializeTable(
        nodes, links, source_table, source_name);
    Check(materialized.success,
          type_name + " materializer path succeeds: " +
              materialized.error_message);
    Check(materialized.operators_applied == 1,
          type_name + " materializer applies one operator");

    AssertTableContractEqual(executor_table, materialized.table, type_name);
    if (numeric_columns.empty()) {
        for (int i = 0; i < executor_table->num_columns(); ++i) {
            AssertNumericColumnClose(executor_table,
                                     materialized.table,
                                     executor_table->schema()->field(i)->name(),
                                     type_name);
        }
    } else {
        for (const auto& column : numeric_columns) {
            AssertNumericColumnClose(executor_table, materialized.table, column,
                                     type_name);
        }
    }
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

} // namespace

int main() {
    TestStandardScalerEmitsMemoryPreflight();
    TestStandardScalerExcludesMultipleTargetColumns();

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
                  507, 508, {{"value_col", "y"}}, {"y"});
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
