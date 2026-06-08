#include "core/arrow_dataset.h"
#include "core/data_registry.h"
#include "core/pipeline_executor.h"

#include <arrow/api.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

double ReadFirstFloatValue(const std::shared_ptr<arrow::Table>& table,
                           const std::string& column_name) {
    const int column_index = table->schema()->GetFieldIndex(column_name);
    Check(column_index >= 0, "scaled output keeps requested column");

    auto column = table->column(column_index);
    Check(column && column->num_chunks() > 0, "scaled column has chunks");
    auto chunk = column->chunk(0);

    if (chunk->type_id() == arrow::Type::FLOAT) {
        auto values = std::static_pointer_cast<arrow::FloatArray>(chunk);
        return static_cast<double>(values->Value(0));
    }
    if (chunk->type_id() == arrow::Type::DOUBLE) {
        auto values = std::static_pointer_cast<arrow::DoubleArray>(chunk);
        return values->Value(0);
    }

    Check(false, "scaled column is floating point");
    return 0.0;
}

double ReadNumericValue(const std::shared_ptr<arrow::Table>& table,
                        const std::string& column_name,
                        int64_t row_index) {
    const int column_index = table->schema()->GetFieldIndex(column_name);
    Check(column_index >= 0, "numeric output keeps requested column");

    auto column = table->column(column_index);
    Check(column && column->num_chunks() > 0, "numeric column has chunks");
    auto scalar_result = column->GetScalar(row_index);
    Check(scalar_result.ok(), "numeric scalar can be read");
    auto scalar = *scalar_result;
    Check(scalar && scalar->is_valid, "numeric scalar is not null");

    switch (scalar->type->id()) {
        case arrow::Type::INT32:
            return std::static_pointer_cast<arrow::Int32Scalar>(scalar)->value;
        case arrow::Type::INT64:
            return static_cast<double>(
                std::static_pointer_cast<arrow::Int64Scalar>(scalar)->value);
        case arrow::Type::FLOAT:
            return std::static_pointer_cast<arrow::FloatScalar>(scalar)->value;
        case arrow::Type::DOUBLE:
            return std::static_pointer_cast<arrow::DoubleScalar>(scalar)->value;
        default:
            Check(false, "numeric scalar has supported type");
            return 0.0;
    }
}

std::string ReadStringValue(const std::shared_ptr<arrow::Table>& table,
                            const std::string& column_name,
                            int64_t row_index) {
    const int column_index = table->schema()->GetFieldIndex(column_name);
    Check(column_index >= 0, "string output keeps requested column");

    auto column = table->column(column_index);
    Check(column && column->num_chunks() > 0, "string column has chunks");
    int64_t remaining = row_index;
    for (const auto& chunk : column->chunks()) {
        if (remaining >= chunk->length()) {
            remaining -= chunk->length();
            continue;
        }
        Check(!chunk->IsNull(remaining), "string scalar is not null");
        if (chunk->type_id() == arrow::Type::STRING) {
            return std::static_pointer_cast<arrow::StringArray>(chunk)->GetString(remaining);
        }
        if (chunk->type_id() == arrow::Type::LARGE_STRING) {
            return std::static_pointer_cast<arrow::LargeStringArray>(chunk)->GetString(remaining);
        }
        break;
    }

    Check(false, "string value has supported type");
    return {};
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

} // namespace

int main() {
    namespace fs = std::filesystem;

    auto& registry = cyxwiz::DataRegistry::Instance();
    registry.UnloadDataset("ds_datainput_1");
    registry.UnloadDataset("ds_operator_StandardScaler_2");
    registry.UnloadDataset("ds_operator_ACFNode_202");

    const fs::path csv_path =
        fs::temp_directory_path() / "cyxwiz_pipeline_executor_operator_routing.csv";
    const fs::path ts_analysis_csv_path =
        fs::temp_directory_path() / "cyxwiz_pipeline_executor_ts_analysis.csv";
    const fs::path export_csv_path =
        fs::temp_directory_path() / "cyxwiz_pipeline_executor_operator_export.csv";
    const fs::path export_csv_alias_path =
        fs::temp_directory_path() / "cyxwiz_pipeline_executor_operator_export_alias.csv";
    const fs::path data_output_mixed_case_csv_path =
        fs::temp_directory_path() / "cyxwiz_pipeline_executor_data_output_mixed_case.csv";
    const fs::path save_dataset_csv_path =
        fs::temp_directory_path() / "cyxwiz_pipeline_executor_save_dataset.csv";
    const fs::path missing_csv_path =
        fs::temp_directory_path() / "cyxwiz_pipeline_executor_missing_values.csv";
    const fs::path string_csv_path =
        fs::temp_directory_path() / "cyxwiz_pipeline_executor_strings.csv";
    const fs::path mixed_csv_path =
        fs::temp_directory_path() / "cyxwiz_pipeline_executor_mixed.csv";
    const fs::path missing_string_csv_path =
        fs::temp_directory_path() / "cyxwiz_pipeline_executor_missing_strings.csv";
    const fs::path duplicates_csv_path =
        fs::temp_directory_path() / "cyxwiz_pipeline_executor_duplicates.csv";
    fs::remove(ts_analysis_csv_path);
    fs::remove(export_csv_path);
    fs::remove(export_csv_alias_path);
    fs::remove(data_output_mixed_case_csv_path);
    fs::remove(save_dataset_csv_path);
    fs::remove(missing_csv_path);
    fs::remove(string_csv_path);
    fs::remove(mixed_csv_path);
    fs::remove(missing_string_csv_path);
    fs::remove(duplicates_csv_path);
    {
        std::ofstream csv(csv_path);
        csv << "x,y\n";
        csv << "1,10\n";
        csv << "2,20\n";
        csv << "3,30\n";
    }
    {
        std::ofstream csv(missing_csv_path);
        csv << "x,y\n";
        csv << "1,10\n";
        csv << ",20\n";
        csv << "3,\n";
    }
    {
        std::ofstream csv(string_csv_path);
        csv << "phrase\n";
        csv << "tea cup\n";
        csv << "blue mug\n";
    }
    {
        std::ofstream csv(mixed_csv_path);
        csv << "x,y,phrase\n";
        csv << "1,10,tea cup\n";
        csv << "2,20,blue mug\n";
        csv << "3,30,green pot\n";
    }
    {
        std::ofstream csv(missing_string_csv_path);
        csv << "phrase,marker\n";
        csv << "tea cup,a\n";
        csv << ",b\n";
        csv << "blue mug,c\n";
    }
    {
        std::ofstream csv(duplicates_csv_path);
        csv << "x,y\n";
        csv << "1,10\n";
        csv << "1,11\n";
        csv << "2,20\n";
    }
    {
        std::ofstream csv(ts_analysis_csv_path);
        csv << "signal\n";
        for (int i = 0; i < 32; ++i) {
            csv << (10 + i + (i % 4)) << "\n";
        }
    }

    const std::string pipeline_json =
        R"({"nodes":[)"
        R"({"id":1,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":2,"type":"StandardScaler","name":"Scale","parameters":{)"
        R"("columns":"x","with_mean":"true","with_std":"true"}})"
        R"(],"links":[{"start_node":1,"end_node":2}]})";

    cyxwiz::PipelineExecutor executor;
    Check(executor.ExecutePipeline(pipeline_json),
          "PipelineExecutor routes StandardScaler through PipelineOperatorFactory: " +
              executor.GetLastError());

    auto output = registry.GetArrowDataset("ds_operator_StandardScaler_2");
    Check(output != nullptr, "operator output dataset is registered");

    auto table = output->GetArrowTable();
    Check(table != nullptr, "operator output table exists");
    Check(table->num_rows() == 3, "operator output preserves row count");
    Check(std::fabs(ReadFirstFloatValue(table, "x") - 1.0) > 0.1,
          "operator output changed the scaled column");

    const std::string acf_json =
        R"({"nodes":[)"
        R"({"id":201,"type":"DataInput","name":"TSInput","parameters":{)"
        R"("source_type":"file","file_path":")" +
        JsonEscapePath(ts_analysis_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":202,"type":"ACFNode","name":"ACF","parameters":{)"
        R"("signal_col":"signal","max_lag":"3"}})"
        R"(],"links":[{"start_node":201,"end_node":202}]})";

    cyxwiz::PipelineExecutor acf_executor;
    Check(acf_executor.ExecutePipeline(acf_json),
          "PipelineExecutor routes ACFNode through PipelineOperatorFactory: " +
              acf_executor.GetLastError());

    auto acf_output = registry.GetArrowDataset("ds_operator_ACFNode_202");
    Check(acf_output != nullptr, "ACF operator output dataset is registered");
    auto acf_table = acf_output->GetArrowTable();
    Check(acf_table != nullptr, "ACF operator output table exists");
    Check(acf_table->num_rows() == 4,
          "ACF operator emits max_lag + 1 rows");
    Check(acf_table->schema()->GetFieldIndex("acf") >= 0,
          "ACF operator output has acf column");

    const std::string missing_acf_signal_json =
        R"({"nodes":[)"
        R"({"id":203,"type":"DataInput","name":"TSInput","parameters":{)"
        R"("source_type":"file","file_path":")" +
        JsonEscapePath(ts_analysis_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":204,"type":"ACFNode","name":"ACF","parameters":{}})"
        R"(],"links":[{"start_node":203,"end_node":204}]})";

    cyxwiz::PipelineExecutor missing_acf_signal_executor;
    Check(!missing_acf_signal_executor.ExecutePipeline(
              missing_acf_signal_json),
          "ACFNode missing signal_col should fail validation");
    Check(missing_acf_signal_executor.GetLastError().find(
              "missing required parameter 'signal_col'") !=
              std::string::npos,
          "ACFNode missing signal_col validation should be specific: " +
              missing_acf_signal_executor.GetLastError());

    const std::string bad_acf_max_lag_json =
        R"({"nodes":[)"
        R"({"id":381,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":382,"type":"ACFNode","name":"BadACFLag","parameters":{)"
        R"("signal_col":"signal","max_lag":"0"}})"
        R"(],"links":[{"start_node":381,"end_node":382}]})";

    cyxwiz::PipelineExecutor bad_acf_max_lag_executor;
    Check(!bad_acf_max_lag_executor.ExecutePipeline(bad_acf_max_lag_json),
          "ACFNode zero max_lag should fail validation");
    Check(bad_acf_max_lag_executor.GetLastError().find(
              "ACFNode max_lag must be an integer >= -1 except 0") !=
              std::string::npos,
          "ACFNode max_lag validation should be specific: " +
              bad_acf_max_lag_executor.GetLastError());

    const std::string bad_pacf_lags_json =
        R"({"nodes":[)"
        R"({"id":383,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":384,"type":"PACFNode","name":"BadPACFLags","parameters":{)"
        R"("signal_col":"signal","lags":"0"}})"
        R"(],"links":[{"start_node":383,"end_node":384}]})";

    cyxwiz::PipelineExecutor bad_pacf_lags_executor;
    Check(!bad_pacf_lags_executor.ExecutePipeline(bad_pacf_lags_json),
          "PACFNode zero lags should fail validation");
    Check(bad_pacf_lags_executor.GetLastError().find(
              "PACFNode lags must be an integer >= -1 except 0") !=
              std::string::npos,
          "PACFNode lags validation should be specific: " +
              bad_pacf_lags_executor.GetLastError());

    const std::string bad_time_series_split_ratio_json =
        R"({"nodes":[)"
        R"({"id":334,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":335,"type":"TimeSeriesSplit","name":"BadSplit","parameters":{)"
        R"("train_ratio":"wide"}})"
        R"(],"links":[{"start_node":334,"end_node":335}]})";

    cyxwiz::PipelineExecutor bad_time_series_split_ratio_executor;
    Check(!bad_time_series_split_ratio_executor.ExecutePipeline(
              bad_time_series_split_ratio_json),
          "TimeSeriesSplit malformed train_ratio should fail validation");
    Check(bad_time_series_split_ratio_executor.GetLastError().find(
              "TimeSeriesSplit train_ratio must be a number between") !=
              std::string::npos,
          "TimeSeriesSplit malformed train_ratio error should be specific: " +
              bad_time_series_split_ratio_executor.GetLastError());

    const std::string bad_time_series_split_sum_json =
        R"({"nodes":[)"
        R"({"id":501,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":502,"type":"TimeSeriesSplit","name":"BadSplitSum","parameters":{)"
        R"("train_ratio":"0.8","val_ratio":"0.3","test_ratio":"0.1"}})"
        R"(],"links":[{"start_node":501,"end_node":502}]})";

    cyxwiz::PipelineExecutor bad_time_series_split_sum_executor;
    Check(!bad_time_series_split_sum_executor.ExecutePipeline(
              bad_time_series_split_sum_json),
          "TimeSeriesSplit invalid ratio sum should fail validation");
    Check(bad_time_series_split_sum_executor.GetLastError().find(
              "TimeSeriesSplit ratios must sum to 1.0") !=
              std::string::npos,
          "TimeSeriesSplit ratio-sum validation should be specific: " +
              bad_time_series_split_sum_executor.GetLastError());

    const std::string bad_time_series_split_train_zero_json =
        R"({"nodes":[)"
        R"({"id":503,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":504,"type":"TimeSeriesSplit","name":"BadSplitTrainZero","parameters":{)"
        R"("train_ratio":"0","val_ratio":"0.5","test_ratio":"0.5"}})"
        R"(],"links":[{"start_node":503,"end_node":504}]})";

    cyxwiz::PipelineExecutor bad_time_series_split_train_zero_executor;
    Check(!bad_time_series_split_train_zero_executor.ExecutePipeline(
              bad_time_series_split_train_zero_json),
          "TimeSeriesSplit zero train_ratio should fail validation");
    Check(bad_time_series_split_train_zero_executor.GetLastError().find(
              "TimeSeriesSplit train_ratio must be > 0") !=
              std::string::npos,
          "TimeSeriesSplit train_ratio cross-field validation should be specific: " +
              bad_time_series_split_train_zero_executor.GetLastError());

    const std::string typed_file_input_json =
        R"({"nodes":[)"
        R"({"id":133,"type":"FileInput","name":"LegacyFile","parameters":{)"
        R"("path":")" + JsonEscapePath(csv_path.string()) +
        R"("}},)"
        R"({"id":134,"type":"SelectColumns","name":"Select","parameters":{)"
        R"("columns":"x"}})"
        R"(],"links":[{"start_node":133,"end_node":134}]})";

    cyxwiz::PipelineExecutor typed_file_input_executor;
    Check(typed_file_input_executor.ExecutePipeline(typed_file_input_json),
          "FileInput should execute through typed CSVFile dispatch: " +
              typed_file_input_executor.GetLastError());
    auto typed_file_input_result = registry.GetArrowDataset("ds_select_134");
    Check(typed_file_input_result != nullptr,
          "typed FileInput downstream output dataset is registered");
    auto typed_file_input_table = typed_file_input_result->GetArrowTable();
    Check(typed_file_input_table != nullptr,
          "typed FileInput downstream output table exists");
    Check(typed_file_input_table->schema()->GetFieldIndex("x") >= 0,
          "typed FileInput should feed SelectColumns");

    const std::string explicit_file_input_format_json =
        R"({"nodes":[)"
        R"({"id":423,"type":"FileInput","name":"ExplicitFile","parameters":{)"
        R"("path":")" + JsonEscapePath(csv_path.string()) +
        R"(","format":" CSV "}},)"
        R"({"id":424,"type":"SelectColumns","name":"Select","parameters":{)"
        R"("columns":"y"}})"
        R"(],"links":[{"start_node":423,"end_node":424}]})";

    cyxwiz::PipelineExecutor explicit_file_input_format_executor;
    Check(explicit_file_input_format_executor.ExecutePipeline(
              explicit_file_input_format_json),
          "FileInput explicit CSV format should execute through CSV loader: " +
              explicit_file_input_format_executor.GetLastError());
    auto explicit_file_input_format_result =
        registry.GetArrowDataset("ds_select_424");
    Check(explicit_file_input_format_result != nullptr,
          "FileInput explicit format downstream output dataset is registered");

    const std::string bad_file_input_format_json =
        R"({"nodes":[)"
        R"({"id":425,"type":"FileInput","name":"BadFileFormat","parameters":{)"
        R"("path":")" + JsonEscapePath(csv_path.string()) +
        R"(","format":"json"}},)"
        R"({"id":426,"type":"SelectColumns","name":"Select","parameters":{)"
        R"("columns":"x"}})"
        R"(],"links":[{"start_node":425,"end_node":426}]})";

    cyxwiz::PipelineExecutor bad_file_input_format_executor;
    Check(!bad_file_input_format_executor.ExecutePipeline(
              bad_file_input_format_json),
          "FileInput json format should fail validation until JSON loading is real");
    Check(bad_file_input_format_executor.GetLastError().find(
              "FileInput format 'json' is not supported") != std::string::npos,
          "FileInput json format validation should be specific: " +
              bad_file_input_format_executor.GetLastError());

    const std::string unsupported_json =
        R"({"nodes":[)"
        R"({"id":3,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":4,"type":"TSNENode","name":"TSNE","parameters":{}})"
        R"(],"links":[{"start_node":3,"end_node":4}]})";

    cyxwiz::PipelineExecutor unsupported_executor;
    Check(!unsupported_executor.ExecutePipeline(unsupported_json),
          "TSNENode should fail closed in PipelineExecutor");
    Check(unsupported_executor.GetLastError().find("legacy t-SNE execution") !=
              std::string::npos,
          "TSNENode fail-closed error should come from runtime capabilities: " +
              unsupported_executor.GetLastError());

    const std::string missing_parameter_json =
        R"({"nodes":[)"
        R"({"id":5,"type":"DataInput","name":"MissingPath","parameters":{)"
        R"("source_type":"FILE","type":"CSV"}},)"
        R"({"id":6,"type":"StandardScaler","name":"Scale","parameters":{)"
        R"("columns":"x"}})"
        R"(],"links":[{"start_node":5,"end_node":6}]})";

    cyxwiz::PipelineExecutor missing_parameter_executor;
    Check(!missing_parameter_executor.ExecutePipeline(missing_parameter_json),
          "DataInput missing file_path should fail validation");
    Check(missing_parameter_executor.GetLastError().find(
              "missing required parameter 'file_path'") != std::string::npos,
          "missing file_path validation should be specific: " +
              missing_parameter_executor.GetLastError());

    const std::string unsupported_source_json =
        R"({"nodes":[)"
        R"({"id":7,"type":"DataInput","name":"BadSource","parameters":{)"
        R"("source_type":"database","file_path":"ignored.csv"}},)"
        R"({"id":8,"type":"StandardScaler","name":"Scale","parameters":{)"
        R"("columns":"x"}})"
        R"(],"links":[{"start_node":7,"end_node":8}]})";

    cyxwiz::PipelineExecutor unsupported_source_executor;
    Check(!unsupported_source_executor.ExecutePipeline(unsupported_source_json),
          "DataInput unsupported source_type should fail validation");
    Check(unsupported_source_executor.GetLastError().find(
              "source_type 'database' is not supported") != std::string::npos,
          "unsupported source_type validation should be specific: " +
              unsupported_source_executor.GetLastError());

    const std::string bad_data_input_type_json =
        R"({"nodes":[)"
        R"({"id":196,"type":"DataInput","name":"BadType","parameters":{)"
        R"("source_type":"file","file_path":"ignored.h5","type":"hdf5"}})"
        R"(],"links":[]})";

    cyxwiz::PipelineExecutor bad_data_input_type_executor;
    Check(!bad_data_input_type_executor.ExecutePipeline(
              bad_data_input_type_json),
          "DataInput unsupported file type should fail validation");
    Check(bad_data_input_type_executor.GetLastError().find(
              "DataInput type 'hdf5' is not supported") !=
              std::string::npos,
          "DataInput unsupported file type error should be specific: " +
              bad_data_input_type_executor.GetLastError());

    const std::string unsupported_ml_dataset_source_json =
        R"({"nodes":[)"
        R"({"id":360,"type":"DataInput","name":"PlannedMLDataset","parameters":{)"
        R"("source_type":"ml_dataset","ml_dataset_type":"mnist"}})"
        R"(],"links":[]})";

    cyxwiz::PipelineExecutor unsupported_ml_dataset_source_executor;
    Check(!unsupported_ml_dataset_source_executor.ExecutePipeline(
              unsupported_ml_dataset_source_json),
          "DataInput ml_dataset source should fail validation");
    Check(unsupported_ml_dataset_source_executor.GetLastError().find(
              "DataInput source_type 'ml_dataset' is not supported") !=
              std::string::npos,
          "DataInput ml_dataset source error should be specific: " +
              unsupported_ml_dataset_source_executor.GetLastError());

    const std::string bad_skip_rows_json =
        R"({"nodes":[)"
        R"({"id":9,"type":"DataInput","name":"BadSkipRows","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv","skip_rows":"nope"}},)"
        R"({"id":10,"type":"StandardScaler","name":"Scale","parameters":{)"
        R"("columns":"x"}})"
        R"(],"links":[{"start_node":9,"end_node":10}]})";

    cyxwiz::PipelineExecutor bad_skip_rows_executor;
    Check(!bad_skip_rows_executor.ExecutePipeline(bad_skip_rows_json),
          "DataInput bad skip_rows should fail validation");
    Check(bad_skip_rows_executor.GetLastError().find(
              "skip_rows must be a non-negative integer") != std::string::npos,
          "bad skip_rows validation should be specific: " +
              bad_skip_rows_executor.GetLastError());

    const std::string uppercase_file_bad_skip_rows_json =
        R"({"nodes":[)"
        R"({"id":333,"type":"DataInput","name":"BadUpperFileSkipRows","parameters":{)"
        R"("source_type":"FILE","file_path":"ignored.csv","type":"csv","skip_rows":"nope"}})"
        R"(],"links":[]})";

    cyxwiz::PipelineExecutor uppercase_file_bad_skip_rows_executor;
    Check(!uppercase_file_bad_skip_rows_executor.ExecutePipeline(
              uppercase_file_bad_skip_rows_json),
          "DataInput uppercase FILE bad skip_rows should fail validation");
    Check(uppercase_file_bad_skip_rows_executor.GetLastError().find(
              "skip_rows must be a non-negative integer") != std::string::npos,
          "DataInput uppercase FILE bad skip_rows validation should be "
          "specific: " +
              uppercase_file_bad_skip_rows_executor.GetLastError());

    const std::string bad_sheet_idx_json =
        R"({"nodes":[)"
        R"({"id":11,"type":"DataInput","name":"BadSheet","parameters":{)"
        R"("source_type":"file","file_path":"ignored.xlsx","type":"excel","sheet_idx":"-1"}},)"
        R"({"id":12,"type":"StandardScaler","name":"Scale","parameters":{)"
        R"("columns":"x"}})"
        R"(],"links":[{"start_node":11,"end_node":12}]})";

    cyxwiz::PipelineExecutor bad_sheet_idx_executor;
    Check(!bad_sheet_idx_executor.ExecutePipeline(bad_sheet_idx_json),
          "DataInput bad sheet_idx should fail validation");
    Check(bad_sheet_idx_executor.GetLastError().find(
              "sheet_idx must be a non-negative integer") != std::string::npos,
          "bad sheet_idx validation should be specific: " +
              bad_sheet_idx_executor.GetLastError());

    const std::string bad_has_header_json =
        R"({"nodes":[)"
        R"({"id":327,"type":"DataInput","name":"BadHeaderFlag","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv","has_header":"maybe"}},)"
        R"({"id":328,"type":"SelectColumns","name":"Select","parameters":{)"
        R"("columns":"x"}})"
        R"(],"links":[{"start_node":327,"end_node":328}]})";

    cyxwiz::PipelineExecutor bad_has_header_executor;
    Check(!bad_has_header_executor.ExecutePipeline(bad_has_header_json),
          "DataInput malformed has_header should fail validation");
    Check(bad_has_header_executor.GetLastError().find(
              "DataInput: 'has_header' must be 'true' or 'false'") !=
              std::string::npos,
          "DataInput has_header validation should be specific: " +
              bad_has_header_executor.GetLastError());

    const std::string bad_window_json =
        R"({"nodes":[)"
        R"({"id":13,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":14,"type":"TSWindow","name":"BadWindow","parameters":{)"
        R"("target_column":"x","window_size":"0","stride":"1"}})"
        R"(],"links":[{"start_node":13,"end_node":14}]})";

    cyxwiz::PipelineExecutor bad_window_executor;
    Check(!bad_window_executor.ExecutePipeline(bad_window_json),
          "TSWindow bad window_size should fail validation");
    Check(bad_window_executor.GetLastError().find(
              "TSWindow window_size must be an integer >= 1") != std::string::npos,
          "bad TSWindow validation should be specific: " +
              bad_window_executor.GetLastError());

    const std::string unsupported_stride_json =
        R"({"nodes":[)"
        R"({"id":211,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":212,"type":"TSWindow","name":"UnsupportedStride","parameters":{)"
        R"("target_column":"x","window_size":"2","stride":"2"}})"
        R"(],"links":[{"start_node":211,"end_node":212}]})";

    cyxwiz::PipelineExecutor unsupported_stride_executor;
    Check(!unsupported_stride_executor.ExecutePipeline(unsupported_stride_json),
          "TSWindow unsupported stride should fail validation");
    Check(unsupported_stride_executor.GetLastError().find(
              "TSWindow stride values other than 1 are not supported") !=
              std::string::npos,
          "unsupported TSWindow stride validation should be specific: " +
              unsupported_stride_executor.GetLastError());

    const std::string bad_text_tokenizer_lowercase_json =
        R"({"nodes":[)"
        R"({"id":300,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":301,"type":"TextTokenizer","name":"BadTokenizer","parameters":{)"
        R"("text_col":"phrase","lowercase":"maybe"}})"
        R"(],"links":[{"start_node":300,"end_node":301}]})";

    cyxwiz::PipelineExecutor bad_text_tokenizer_lowercase_executor;
    Check(!bad_text_tokenizer_lowercase_executor.ExecutePipeline(
              bad_text_tokenizer_lowercase_json),
          "TextTokenizer malformed lowercase should fail validation");
    Check(bad_text_tokenizer_lowercase_executor.GetLastError().find(
              "TextTokenizer: 'lowercase' must be 'true' or 'false'") !=
              std::string::npos,
          "TextTokenizer lowercase validation should be specific: " +
              bad_text_tokenizer_lowercase_executor.GetLastError());

    const std::string bad_linear_regression_intercept_json =
        R"({"nodes":[)"
        R"({"id":302,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":303,"type":"LinearRegressionNode","name":"BadLinearRegression","parameters":{)"
        R"("feature_cols":"x","target_col":"y","fit_intercept":"maybe"}})"
        R"(],"links":[{"start_node":302,"end_node":303}]})";

    cyxwiz::PipelineExecutor bad_linear_regression_intercept_executor;
    Check(!bad_linear_regression_intercept_executor.ExecutePipeline(
              bad_linear_regression_intercept_json),
          "LinearRegression malformed fit_intercept should fail validation");
    Check(bad_linear_regression_intercept_executor.GetLastError().find(
              "LinearRegression: 'fit_intercept' must be 'true' or 'false'") !=
              std::string::npos,
          "LinearRegression fit_intercept validation should be specific: " +
              bad_linear_regression_intercept_executor.GetLastError());

    const std::string bad_linear_regression_feature_type_json =
        R"({"nodes":[)"
        R"({"id":405,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(mixed_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":406,"type":"LinearRegressionNode","name":"BadLinearFeature","parameters":{)"
        R"("feature_cols":"phrase","target_col":"y"}})"
        R"(],"links":[{"start_node":405,"end_node":406}]})";

    cyxwiz::PipelineExecutor bad_linear_regression_feature_type_executor;
    Check(!bad_linear_regression_feature_type_executor.ExecutePipeline(
              bad_linear_regression_feature_type_json),
          "LinearRegression string feature_cols should fail schema validation");
    Check(bad_linear_regression_feature_type_executor.GetLastError().find(
              "LinearRegressionNode: feature column 'phrase' must be numeric") !=
              std::string::npos,
          "LinearRegression feature_cols validation should be specific: " +
              bad_linear_regression_feature_type_executor.GetLastError());

    const std::string bad_exp_smoothing_damped_json =
        R"({"nodes":[)"
        R"({"id":304,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" +
        JsonEscapePath(ts_analysis_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":305,"type":"ExponentialSmoothing","name":"BadSmoothing","parameters":{)"
        R"("signal_col":"signal","method":"holt","damped":"maybe"}})"
        R"(],"links":[{"start_node":304,"end_node":305}]})";

    cyxwiz::PipelineExecutor bad_exp_smoothing_damped_executor;
    Check(!bad_exp_smoothing_damped_executor.ExecutePipeline(
              bad_exp_smoothing_damped_json),
          "ExponentialSmoothing malformed damped should fail validation");
    Check(bad_exp_smoothing_damped_executor.GetLastError().find(
              "ExponentialSmoothing: 'damped' must be 'true' or 'false'") !=
              std::string::npos,
          "ExponentialSmoothing damped validation should be specific: " +
              bad_exp_smoothing_damped_executor.GetLastError());

    const std::string bad_standard_scaler_mean_json =
        R"({"nodes":[)"
        R"({"id":306,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":307,"type":"StandardScaler","name":"BadScaler","parameters":{)"
        R"("columns":"x","with_mean":"maybe"}})"
        R"(],"links":[{"start_node":306,"end_node":307}]})";

    cyxwiz::PipelineExecutor bad_standard_scaler_mean_executor;
    Check(!bad_standard_scaler_mean_executor.ExecutePipeline(
              bad_standard_scaler_mean_json),
          "StandardScaler malformed with_mean should fail validation");
    Check(bad_standard_scaler_mean_executor.GetLastError().find(
              "StandardScaler: 'with_mean' must be 'true' or 'false'") !=
              std::string::npos,
          "StandardScaler with_mean validation should be specific: " +
              bad_standard_scaler_mean_executor.GetLastError());

    const std::string bad_standard_scaler_column_type_json =
        R"({"nodes":[)"
        R"({"id":407,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(mixed_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":408,"type":"StandardScaler","name":"BadScalerColumn","parameters":{)"
        R"("columns":"phrase","with_mean":"true"}})"
        R"(],"links":[{"start_node":407,"end_node":408}]})";

    cyxwiz::PipelineExecutor bad_standard_scaler_column_type_executor;
    Check(!bad_standard_scaler_column_type_executor.ExecutePipeline(
              bad_standard_scaler_column_type_json),
          "StandardScaler string columns should fail schema validation");
    Check(bad_standard_scaler_column_type_executor.GetLastError().find(
              "StandardScaler: column 'phrase' must be numeric") !=
              std::string::npos,
          "StandardScaler columns validation should be specific: " +
              bad_standard_scaler_column_type_executor.GetLastError());

    const std::string bad_standard_scaler_label_col_json =
        R"({"nodes":[)"
        R"({"id":411,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(mixed_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":412,"type":"StandardScaler","name":"BadScalerLabel","parameters":{)"
        R"("label_col":"missing","with_mean":"true"}})"
        R"(],"links":[{"start_node":411,"end_node":412}]})";

    cyxwiz::PipelineExecutor bad_standard_scaler_label_col_executor;
    Check(!bad_standard_scaler_label_col_executor.ExecutePipeline(
              bad_standard_scaler_label_col_json),
          "StandardScaler missing label_col should fail schema validation");
    Check(bad_standard_scaler_label_col_executor.GetLastError().find(
              "StandardScaler: label column 'missing' not found") !=
              std::string::npos,
          "StandardScaler label_col validation should be specific: " +
              bad_standard_scaler_label_col_executor.GetLastError());

    const std::string bad_robust_scaler_centering_json =
        R"({"nodes":[)"
        R"({"id":308,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":309,"type":"RobustScaler","name":"BadRobust","parameters":{)"
        R"("columns":"x","with_centering":"maybe"}})"
        R"(],"links":[{"start_node":308,"end_node":309}]})";

    cyxwiz::PipelineExecutor bad_robust_scaler_centering_executor;
    Check(!bad_robust_scaler_centering_executor.ExecutePipeline(
              bad_robust_scaler_centering_json),
          "RobustScaler malformed with_centering should fail validation");
    Check(bad_robust_scaler_centering_executor.GetLastError().find(
              "RobustScaler: 'with_centering' must be 'true' or 'false'") !=
              std::string::npos,
          "RobustScaler with_centering validation should be specific: " +
              bad_robust_scaler_centering_executor.GetLastError());

    const std::string bad_robust_scaler_quantile_json =
        R"({"nodes":[)"
        R"({"id":336,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":337,"type":"RobustScaler","name":"BadRobustQuantile","parameters":{)"
        R"("quantile_min":"wide"}})"
        R"(],"links":[{"start_node":336,"end_node":337}]})";

    cyxwiz::PipelineExecutor bad_robust_scaler_quantile_executor;
    Check(!bad_robust_scaler_quantile_executor.ExecutePipeline(
              bad_robust_scaler_quantile_json),
          "RobustScaler malformed quantile_min should fail validation");
    Check(bad_robust_scaler_quantile_executor.GetLastError().find(
              "RobustScaler quantile_min must be a number between") !=
              std::string::npos,
          "RobustScaler malformed quantile_min error should be specific: " +
              bad_robust_scaler_quantile_executor.GetLastError());

    const std::string bad_robust_scaler_quantile_order_json =
        R"({"nodes":[)"
        R"({"id":505,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":506,"type":"RobustScaler","name":"BadRobustQuantileOrder","parameters":{)"
        R"("quantile_min":"80","quantile_max":"20"}})"
        R"(],"links":[{"start_node":505,"end_node":506}]})";

    cyxwiz::PipelineExecutor bad_robust_scaler_quantile_order_executor;
    Check(!bad_robust_scaler_quantile_order_executor.ExecutePipeline(
              bad_robust_scaler_quantile_order_json),
          "RobustScaler inverted quantiles should fail validation");
    Check(bad_robust_scaler_quantile_order_executor.GetLastError().find(
              "RobustScaler quantile_min must be less than quantile_max") !=
              std::string::npos,
          "RobustScaler quantile ordering validation should be specific: " +
              bad_robust_scaler_quantile_order_executor.GetLastError());

    const std::string bad_target_encoder_smoothing_json =
        R"({"nodes":[)"
        R"({"id":373,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":374,"type":"TargetEncoder","name":"BadTargetSmoothing","parameters":{)"
        R"("columns":"category","target_col":"y","smoothing":"-1"}})"
        R"(],"links":[{"start_node":373,"end_node":374}]})";

    cyxwiz::PipelineExecutor bad_target_encoder_smoothing_executor;
    Check(!bad_target_encoder_smoothing_executor.ExecutePipeline(
              bad_target_encoder_smoothing_json),
          "TargetEncoder negative smoothing should fail validation");
    Check(bad_target_encoder_smoothing_executor.GetLastError().find(
              "TargetEncoder smoothing must be a number greater than or equal to") !=
              std::string::npos,
          "TargetEncoder smoothing validation should be specific: " +
              bad_target_encoder_smoothing_executor.GetLastError());

    const std::string bad_target_encoder_column_type_json =
        R"({"nodes":[)"
        R"({"id":409,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(mixed_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":410,"type":"TargetEncoder","name":"BadTargetColumn","parameters":{)"
        R"("columns":"x","target_col":"y"}})"
        R"(],"links":[{"start_node":409,"end_node":410}]})";

    cyxwiz::PipelineExecutor bad_target_encoder_column_type_executor;
    Check(!bad_target_encoder_column_type_executor.ExecutePipeline(
              bad_target_encoder_column_type_json),
          "TargetEncoder numeric categorical columns should fail schema validation");
    Check(bad_target_encoder_column_type_executor.GetLastError().find(
              "TargetEncoder: categorical column 'x' must be string/large_string") !=
              std::string::npos,
          "TargetEncoder columns validation should be specific: " +
              bad_target_encoder_column_type_executor.GetLastError());

    const std::string bad_outlier_threshold_json =
        R"({"nodes":[)"
        R"({"id":375,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":376,"type":"OutlierDetector","name":"BadOutlierThreshold","parameters":{)"
        R"("columns":"x","threshold":"0"}})"
        R"(],"links":[{"start_node":375,"end_node":376}]})";

    cyxwiz::PipelineExecutor bad_outlier_threshold_executor;
    Check(!bad_outlier_threshold_executor.ExecutePipeline(
              bad_outlier_threshold_json),
          "OutlierDetector zero threshold should fail validation");
    Check(bad_outlier_threshold_executor.GetLastError().find(
              "OutlierDetector threshold must be a number greater than") !=
              std::string::npos,
          "OutlierDetector threshold validation should be specific: " +
              bad_outlier_threshold_executor.GetLastError());

    const std::string uppercase_outlier_method_json =
        R"({"nodes":[)"
        R"({"id":338,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":339,"type":"OutlierDetector","name":"Outliers","parameters":{)"
        R"("method":"IQR","threshold":"1.5"}})"
        R"(],"links":[{"start_node":338,"end_node":339}]})";

    cyxwiz::PipelineExecutor uppercase_outlier_method_executor;
    Check(uppercase_outlier_method_executor.ExecutePipeline(
              uppercase_outlier_method_json),
          "OutlierDetector uppercase IQR method should execute after normalization: " +
              uppercase_outlier_method_executor.GetLastError());
    auto outliers = registry.GetArrowDataset("ds_operator_OutlierDetector_339");
    Check(outliers != nullptr, "OutlierDetector output dataset is registered");
    auto outliers_table = outliers->GetArrowTable();
    Check(outliers_table != nullptr, "OutlierDetector output table exists");
    Check(outliers_table->schema()->GetFieldIndex("is_outlier") >= 0,
          "OutlierDetector output has is_outlier column");

    const std::string outlier_all_columns_json =
        R"({"nodes":[)"
        R"({"id":413,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(mixed_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":414,"type":"OutlierDetector","name":"OutliersAll","parameters":{)"
        R"("columns":"ALL","label_col":"y","method":"iqr","threshold":"1.5"}})"
        R"(],"links":[{"start_node":413,"end_node":414}]})";

    cyxwiz::PipelineExecutor outlier_all_columns_executor;
    Check(outlier_all_columns_executor.ExecutePipeline(outlier_all_columns_json),
          "OutlierDetector columns=ALL should preserve auto-detect semantics: " +
              outlier_all_columns_executor.GetLastError());
    auto outliers_all = registry.GetArrowDataset("ds_operator_OutlierDetector_414");
    Check(outliers_all != nullptr,
          "OutlierDetector columns=ALL output dataset is registered");
    auto outliers_all_table = outliers_all->GetArrowTable();
    Check(outliers_all_table != nullptr,
          "OutlierDetector columns=ALL output table exists");
    Check(outliers_all_table->schema()->GetFieldIndex("is_outlier") >= 0,
          "OutlierDetector columns=ALL output has is_outlier column");

    const std::string bad_pca_center_json =
        R"({"nodes":[)"
        R"({"id":310,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":311,"type":"PCANode","name":"BadPCA","parameters":{)"
        R"("feature_cols":"x,y","center":"maybe"}})"
        R"(],"links":[{"start_node":310,"end_node":311}]})";

    cyxwiz::PipelineExecutor bad_pca_center_executor;
    Check(!bad_pca_center_executor.ExecutePipeline(bad_pca_center_json),
          "PCANode malformed center should fail validation");
    Check(bad_pca_center_executor.GetLastError().find(
              "PCA: 'center' must be 'true' or 'false'") !=
              std::string::npos,
          "PCANode center validation should be specific: " +
              bad_pca_center_executor.GetLastError());

    const std::string bad_lags_json =
        R"({"nodes":[)"
        R"({"id":15,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":16,"type":"TSLag","name":"BadLag","parameters":{)"
        R"("columns":"x","lag_periods":"1,nope,3"}})"
        R"(],"links":[{"start_node":15,"end_node":16}]})";

    cyxwiz::PipelineExecutor bad_lags_executor;
    Check(!bad_lags_executor.ExecutePipeline(bad_lags_json),
          "TSLag bad lag_periods should fail validation");
    Check(bad_lags_executor.GetLastError().find(
              "TSLag lag_periods must be a comma-separated list of integers >= 1") !=
              std::string::npos,
          "bad TSLag validation should be specific: " +
              bad_lags_executor.GetLastError());

    const std::string missing_text_clean_column_json =
        R"({"nodes":[)"
        R"({"id":166,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":167,"type":"TextClean","name":"MissingTextColumn","parameters":{)"
        R"("lowercase":"true"}})"
        R"(],"links":[{"start_node":166,"end_node":167}]})";

    cyxwiz::PipelineExecutor missing_text_clean_column_executor;
    Check(!missing_text_clean_column_executor.ExecutePipeline(
              missing_text_clean_column_json),
          "TextClean missing text_column should fail validation");
    Check(missing_text_clean_column_executor.GetLastError().find(
              "missing required parameter 'text_column'") != std::string::npos,
          "TextClean missing text_column validation should be specific: " +
              missing_text_clean_column_executor.GetLastError());

    const std::string missing_text_tokenize_column_json =
        R"({"nodes":[)"
        R"({"id":168,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":169,"type":"TextTokenize","name":"MissingTokenColumn","parameters":{)"
        R"("method":"word"}})"
        R"(],"links":[{"start_node":168,"end_node":169}]})";

    cyxwiz::PipelineExecutor missing_text_tokenize_column_executor;
    Check(!missing_text_tokenize_column_executor.ExecutePipeline(
              missing_text_tokenize_column_json),
          "TextTokenize missing text_column should fail validation");
    Check(missing_text_tokenize_column_executor.GetLastError().find(
              "missing required parameter 'text_column'") != std::string::npos,
          "TextTokenize missing text_column validation should be specific: " +
              missing_text_tokenize_column_executor.GetLastError());

    const std::string missing_text_vectorize_column_json =
        R"({"nodes":[)"
        R"({"id":170,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":171,"type":"TextVectorize","name":"MissingVectorColumn","parameters":{)"
        R"("method":"count"}})"
        R"(],"links":[{"start_node":170,"end_node":171}]})";

    cyxwiz::PipelineExecutor missing_text_vectorize_column_executor;
    Check(!missing_text_vectorize_column_executor.ExecutePipeline(
              missing_text_vectorize_column_json),
          "TextVectorize missing text_column should fail validation");
    Check(missing_text_vectorize_column_executor.GetLastError().find(
              "missing required parameter 'text_column'") != std::string::npos,
          "TextVectorize missing text_column validation should be specific: " +
              missing_text_vectorize_column_executor.GetLastError());

    const std::string missing_text_tokenizer_text_col_json =
        R"({"nodes":[)"
        R"({"id":312,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":313,"type":"TextTokenizer","name":"MissingTokenizerText","parameters":{)"
        R"("max_length":"8"}})"
        R"(],"links":[{"start_node":312,"end_node":313}]})";

    cyxwiz::PipelineExecutor missing_text_tokenizer_text_col_executor;
    Check(!missing_text_tokenizer_text_col_executor.ExecutePipeline(
              missing_text_tokenizer_text_col_json),
          "TextTokenizer missing text_col should fail validation");
    Check(missing_text_tokenizer_text_col_executor.GetLastError().find(
              "missing required parameter 'text_col'") != std::string::npos,
          "TextTokenizer missing text_col validation should be specific: " +
              missing_text_tokenizer_text_col_executor.GetLastError());

    const std::string missing_linear_regression_target_json =
        R"({"nodes":[)"
        R"({"id":314,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":315,"type":"LinearRegressionNode","name":"MissingTarget","parameters":{)"
        R"("feature_cols":"x"}})"
        R"(],"links":[{"start_node":314,"end_node":315}]})";

    cyxwiz::PipelineExecutor missing_linear_regression_target_executor;
    Check(!missing_linear_regression_target_executor.ExecutePipeline(
              missing_linear_regression_target_json),
          "LinearRegressionNode missing target_col should fail validation");
    Check(missing_linear_regression_target_executor.GetLastError().find(
              "missing required parameter 'target_col'") != std::string::npos,
          "LinearRegressionNode missing target_col validation should be specific: " +
              missing_linear_regression_target_executor.GetLastError());

    const std::string missing_convolution_kernel_json =
        R"({"nodes":[)"
        R"({"id":316,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":317,"type":"Convolution1D","name":"MissingKernel","parameters":{)"
        R"("signal_col":"x"}})"
        R"(],"links":[{"start_node":316,"end_node":317}]})";

    cyxwiz::PipelineExecutor missing_convolution_kernel_executor;
    Check(!missing_convolution_kernel_executor.ExecutePipeline(
              missing_convolution_kernel_json),
          "Convolution1D missing kernel should fail validation");
    Check(missing_convolution_kernel_executor.GetLastError().find(
              "missing required parameter 'kernel'") != std::string::npos,
          "Convolution1D missing kernel validation should be specific: " +
              missing_convolution_kernel_executor.GetLastError());

    const std::string bad_convolution_kernel_json =
        R"({"nodes":[)"
        R"({"id":507,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":508,"type":"Convolution1D","name":"BadKernel","parameters":{)"
        R"("signal_col":"x","kernel":"0.25,oops,0.25"}})"
        R"(],"links":[{"start_node":507,"end_node":508}]})";

    cyxwiz::PipelineExecutor bad_convolution_kernel_executor;
    Check(!bad_convolution_kernel_executor.ExecutePipeline(
              bad_convolution_kernel_json),
          "Convolution1D malformed kernel should fail validation");
    Check(bad_convolution_kernel_executor.GetLastError().find(
              "Convolution1D kernel must be a comma-separated list of finite numbers") !=
              std::string::npos,
          "Convolution1D kernel validation should be specific: " +
              bad_convolution_kernel_executor.GetLastError());

    const std::string missing_label_encoder_column_json =
        R"({"nodes":[)"
        R"({"id":318,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":319,"type":"LabelEncoder","name":"MissingLabelColumn","parameters":{}})"
        R"(],"links":[{"start_node":318,"end_node":319}]})";

    cyxwiz::PipelineExecutor missing_label_encoder_column_executor;
    Check(!missing_label_encoder_column_executor.ExecutePipeline(
              missing_label_encoder_column_json),
          "LabelEncoder missing column should fail validation");
    Check(missing_label_encoder_column_executor.GetLastError().find(
              "missing required parameter 'column'") != std::string::npos,
          "LabelEncoder missing column validation should be specific: " +
              missing_label_encoder_column_executor.GetLastError());

    const std::string bad_count_vectorizer_norm_json =
        R"({"nodes":[)"
        R"({"id":320,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":321,"type":"CountVectorizer","name":"BadNorm","parameters":{)"
        R"("text_col":"phrase","norm":"cosine"}})"
        R"(],"links":[{"start_node":320,"end_node":321}]})";

    cyxwiz::PipelineExecutor bad_count_vectorizer_norm_executor;
    Check(!bad_count_vectorizer_norm_executor.ExecutePipeline(
              bad_count_vectorizer_norm_json),
          "CountVectorizer bad norm should fail validation");
    Check(bad_count_vectorizer_norm_executor.GetLastError().find(
              "CountVectorizer norm 'cosine' is not supported") !=
              std::string::npos,
          "CountVectorizer bad norm validation should be specific: " +
              bad_count_vectorizer_norm_executor.GetLastError());

    auto check_operator_field = [&registry](const std::string& dataset_id,
                                            const std::string& field_name,
                                            const std::string& message) {
        auto output = registry.GetArrowDataset(dataset_id);
        Check(output != nullptr, message + " output dataset is registered");
        auto table = output->GetArrowTable();
        Check(table != nullptr, message + " output table exists");
        Check(table->schema()->GetFieldIndex(field_name) >= 0,
              message + " output has " + field_name + " column");
    };

    const std::string uppercase_count_vectorizer_norm_json =
        R"({"nodes":[)"
        R"({"id":348,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":349,"type":"CountVectorizer","name":"CountVectorizer","parameters":{)"
        R"("text_col":"phrase","max_features":"2","norm":" NONE "}})"
        R"(],"links":[{"start_node":348,"end_node":349}]})";

    cyxwiz::PipelineExecutor uppercase_count_vectorizer_norm_executor;
    Check(uppercase_count_vectorizer_norm_executor.ExecutePipeline(
              uppercase_count_vectorizer_norm_json),
          "CountVectorizer uppercase norm should execute after normalization: " +
              uppercase_count_vectorizer_norm_executor.GetLastError());
    check_operator_field("ds_operator_CountVectorizer_349", "count_0",
                         "CountVectorizer uppercase norm");

    const std::string uppercase_tfidf_vectorizer_norm_json =
        R"({"nodes":[)"
        R"({"id":350,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":351,"type":"TFIDFVectorizer","name":"TFIDFVectorizer","parameters":{)"
        R"("text_col":"phrase","max_features":"2","norm":"L1"}})"
        R"(],"links":[{"start_node":350,"end_node":351}]})";

    cyxwiz::PipelineExecutor uppercase_tfidf_vectorizer_norm_executor;
    Check(uppercase_tfidf_vectorizer_norm_executor.ExecutePipeline(
              uppercase_tfidf_vectorizer_norm_json),
          "TFIDFVectorizer uppercase norm should execute after normalization: " +
              uppercase_tfidf_vectorizer_norm_executor.GetLastError());
    check_operator_field("ds_operator_TFIDFVectorizer_351", "tfidf_0",
                         "TFIDFVectorizer uppercase norm");

    const std::string uppercase_sentiment_method_json =
        R"({"nodes":[)"
        R"({"id":352,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":353,"type":"SentimentAnalyzer","name":"Sentiment","parameters":{)"
        R"("text_col":"phrase","method":" SIMPLE "}})"
        R"(],"links":[{"start_node":352,"end_node":353}]})";

    cyxwiz::PipelineExecutor uppercase_sentiment_method_executor;
    Check(uppercase_sentiment_method_executor.ExecutePipeline(
              uppercase_sentiment_method_json),
          "SentimentAnalyzer uppercase method should execute after normalization: " +
              uppercase_sentiment_method_executor.GetLastError());
    check_operator_field("ds_operator_SentimentAnalyzer_353", "sentiment_label",
                         "SentimentAnalyzer uppercase method");

    const std::string bad_tokenizer_text_type_json =
        R"({"nodes":[)"
        R"({"id":391,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":392,"type":"TextTokenizer","name":"BadTokenizerText","parameters":{)"
        R"("text_col":"x","max_length":"4"}})"
        R"(],"links":[{"start_node":391,"end_node":392}]})";

    cyxwiz::PipelineExecutor bad_tokenizer_text_type_executor;
    Check(!bad_tokenizer_text_type_executor.ExecutePipeline(
              bad_tokenizer_text_type_json),
          "TextTokenizer numeric text_col should fail schema validation");
    Check(bad_tokenizer_text_type_executor.GetLastError().find(
              "TextTokenizer: text column 'x' must be string/large_string") !=
              std::string::npos,
          "TextTokenizer text_col type validation should be specific: " +
              bad_tokenizer_text_type_executor.GetLastError());

    const std::string bad_count_vectorizer_text_type_json =
        R"({"nodes":[)"
        R"({"id":393,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":394,"type":"CountVectorizer","name":"BadCountText","parameters":{)"
        R"("text_col":"x","max_features":"2"}})"
        R"(],"links":[{"start_node":393,"end_node":394}]})";

    cyxwiz::PipelineExecutor bad_count_vectorizer_text_type_executor;
    Check(!bad_count_vectorizer_text_type_executor.ExecutePipeline(
              bad_count_vectorizer_text_type_json),
          "CountVectorizer numeric text_col should fail schema validation");
    Check(bad_count_vectorizer_text_type_executor.GetLastError().find(
              "CountVectorizer: text column 'x' must be string/large_string") !=
              std::string::npos,
          "CountVectorizer text_col type validation should be specific: " +
              bad_count_vectorizer_text_type_executor.GetLastError());

    const std::string bad_tfidf_label_column_json =
        R"({"nodes":[)"
        R"({"id":395,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":396,"type":"TFIDFVectorizer","name":"BadTfidfLabel","parameters":{)"
        R"("text_col":"phrase","max_features":"2","label_col":"missing"}})"
        R"(],"links":[{"start_node":395,"end_node":396}]})";

    cyxwiz::PipelineExecutor bad_tfidf_label_column_executor;
    Check(!bad_tfidf_label_column_executor.ExecutePipeline(
              bad_tfidf_label_column_json),
          "TFIDFVectorizer missing label_col should fail schema validation");
    Check(bad_tfidf_label_column_executor.GetLastError().find(
              "TFIDFVectorizer: label column 'missing' not found") !=
              std::string::npos,
          "TFIDFVectorizer label_col validation should be specific: " +
              bad_tfidf_label_column_executor.GetLastError());

    const std::string bad_sentiment_text_type_json =
        R"({"nodes":[)"
        R"({"id":397,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":398,"type":"SentimentAnalyzer","name":"BadSentimentText","parameters":{)"
        R"("text_col":"x","method":"simple"}})"
        R"(],"links":[{"start_node":397,"end_node":398}]})";

    cyxwiz::PipelineExecutor bad_sentiment_text_type_executor;
    Check(!bad_sentiment_text_type_executor.ExecutePipeline(
              bad_sentiment_text_type_json),
          "SentimentAnalyzer numeric text_col should fail schema validation");
    Check(bad_sentiment_text_type_executor.GetLastError().find(
              "SentimentAnalyzer: text column 'x' must be string/large_string") !=
              std::string::npos,
          "SentimentAnalyzer text_col type validation should be specific: " +
              bad_sentiment_text_type_executor.GetLastError());

    const std::string bad_kmeans_init_json =
        R"({"nodes":[)"
        R"({"id":322,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":323,"type":"KMeansCluster","name":"BadInit","parameters":{)"
        R"("init":"forgy"}})"
        R"(],"links":[{"start_node":322,"end_node":323}]})";

    cyxwiz::PipelineExecutor bad_kmeans_init_executor;
    Check(!bad_kmeans_init_executor.ExecutePipeline(bad_kmeans_init_json),
          "KMeansCluster bad init should fail validation");
    Check(bad_kmeans_init_executor.GetLastError().find(
              "KMeansCluster init 'forgy' is not supported") !=
              std::string::npos,
          "KMeansCluster bad init validation should be specific: " +
              bad_kmeans_init_executor.GetLastError());

    auto check_cluster_output = [&registry](const std::string& dataset_id,
                                            const std::string& message) {
        auto output = registry.GetArrowDataset(dataset_id);
        Check(output != nullptr, message + " output dataset is registered");
        auto table = output->GetArrowTable();
        Check(table != nullptr, message + " output table exists");
        Check(table->schema()->GetFieldIndex("cluster_id") >= 0,
              message + " output has cluster_id column");
    };

    const std::string uppercase_kmeans_init_json =
        R"({"nodes":[)"
        R"({"id":340,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":341,"type":"KMeansCluster","name":"KMeans","parameters":{)"
        R"("feature_cols":"x,y","n_clusters":"2","max_iter":"10","n_init":"1","init":" RANDOM "}})"
        R"(],"links":[{"start_node":340,"end_node":341}]})";

    cyxwiz::PipelineExecutor uppercase_kmeans_init_executor;
    Check(uppercase_kmeans_init_executor.ExecutePipeline(uppercase_kmeans_init_json),
          "KMeansCluster uppercase init should execute after normalization: " +
              uppercase_kmeans_init_executor.GetLastError());
    check_cluster_output("ds_operator_KMeansCluster_341", "KMeansCluster uppercase init");

    const std::string bad_kmeans_label_col_json =
        R"({"nodes":[)"
        R"({"id":415,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(mixed_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":416,"type":"KMeansCluster","name":"BadKMeansLabel","parameters":{)"
        R"("label_col":"missing","n_clusters":"2","max_iter":"10","n_init":"1"}})"
        R"(],"links":[{"start_node":415,"end_node":416}]})";

    cyxwiz::PipelineExecutor bad_kmeans_label_col_executor;
    Check(!bad_kmeans_label_col_executor.ExecutePipeline(
              bad_kmeans_label_col_json),
          "KMeansCluster missing label_col should fail schema validation");
    Check(bad_kmeans_label_col_executor.GetLastError().find(
              "KMeansCluster: label column 'missing' not found") !=
              std::string::npos,
          "KMeansCluster label_col validation should be specific: " +
              bad_kmeans_label_col_executor.GetLastError());

    const std::string bad_dbscan_eps_json =
        R"({"nodes":[)"
        R"({"id":377,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":378,"type":"DBSCANCluster","name":"BadDBSCANEps","parameters":{)"
        R"("feature_cols":"x,y","eps":"0","min_samples":"1"}})"
        R"(],"links":[{"start_node":377,"end_node":378}]})";

    cyxwiz::PipelineExecutor bad_dbscan_eps_executor;
    Check(!bad_dbscan_eps_executor.ExecutePipeline(bad_dbscan_eps_json),
          "DBSCANCluster zero eps should fail validation");
    Check(bad_dbscan_eps_executor.GetLastError().find(
              "DBSCANCluster eps must be a number greater than") !=
              std::string::npos,
          "DBSCANCluster eps validation should be specific: " +
              bad_dbscan_eps_executor.GetLastError());

    const std::string uppercase_dbscan_metric_json =
        R"({"nodes":[)"
        R"({"id":342,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":343,"type":"DBSCANCluster","name":"DBSCAN","parameters":{)"
        R"("feature_cols":"x,y","eps":"50","min_samples":"1","metric":"MANHATTAN"}})"
        R"(],"links":[{"start_node":342,"end_node":343}]})";

    cyxwiz::PipelineExecutor uppercase_dbscan_metric_executor;
    Check(uppercase_dbscan_metric_executor.ExecutePipeline(
              uppercase_dbscan_metric_json),
          "DBSCANCluster uppercase metric should execute after normalization: " +
              uppercase_dbscan_metric_executor.GetLastError());
    check_cluster_output("ds_operator_DBSCANCluster_343", "DBSCANCluster uppercase metric");

    const std::string uppercase_hierarchical_choices_json =
        R"({"nodes":[)"
        R"({"id":344,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":345,"type":"HierarchicalCluster","name":"Hierarchical","parameters":{)"
        R"("feature_cols":"x,y","n_clusters":"2","linkage":" COMPLETE ","metric":"COSINE"}})"
        R"(],"links":[{"start_node":344,"end_node":345}]})";

    cyxwiz::PipelineExecutor uppercase_hierarchical_choices_executor;
    Check(uppercase_hierarchical_choices_executor.ExecutePipeline(
              uppercase_hierarchical_choices_json),
          "HierarchicalCluster uppercase choices should execute after normalization: " +
              uppercase_hierarchical_choices_executor.GetLastError());
    check_cluster_output("ds_operator_HierarchicalCluster_345",
                         "HierarchicalCluster uppercase choices");

    const std::string uppercase_gmm_covariance_json =
        R"({"nodes":[)"
        R"({"id":346,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":347,"type":"GMMCluster","name":"GMM","parameters":{)"
        R"("feature_cols":"x,y","n_components":"1","max_iter":"10","n_init":"1","covariance_type":"DIAG"}})"
        R"(],"links":[{"start_node":346,"end_node":347}]})";

    cyxwiz::PipelineExecutor uppercase_gmm_covariance_executor;
    Check(uppercase_gmm_covariance_executor.ExecutePipeline(
              uppercase_gmm_covariance_json),
          "GMMCluster uppercase covariance_type should execute after normalization: " +
              uppercase_gmm_covariance_executor.GetLastError());
    check_cluster_output("ds_operator_GMMCluster_347", "GMMCluster uppercase covariance_type");

    const std::string bad_exp_smoothing_method_json =
        R"({"nodes":[)"
        R"({"id":324,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":325,"type":"ExponentialSmoothing","name":"BadSmoothingMethod","parameters":{)"
        R"("signal_col":"signal","method":"ets"}})"
        R"(],"links":[{"start_node":324,"end_node":325}]})";

    cyxwiz::PipelineExecutor bad_exp_smoothing_method_executor;
    Check(!bad_exp_smoothing_method_executor.ExecutePipeline(
              bad_exp_smoothing_method_json),
          "ExponentialSmoothing bad method should fail validation");
    Check(bad_exp_smoothing_method_executor.GetLastError().find(
              "ExponentialSmoothing method 'ets' is not supported") !=
              std::string::npos,
          "ExponentialSmoothing bad method validation should be specific: " +
              bad_exp_smoothing_method_executor.GetLastError());

    const std::string bad_filter_sample_rate_json =
        R"({"nodes":[)"
        R"({"id":379,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":380,"type":"FilterDesigner","name":"BadFilterSampleRate","parameters":{)"
        R"("signal_col":"signal","cutoff":"0.1","sample_rate":"0","order":"2"}})"
        R"(],"links":[{"start_node":379,"end_node":380}]})";

    cyxwiz::PipelineExecutor bad_filter_sample_rate_executor;
    Check(!bad_filter_sample_rate_executor.ExecutePipeline(
              bad_filter_sample_rate_json),
          "FilterDesigner zero sample_rate should fail validation");
    Check(bad_filter_sample_rate_executor.GetLastError().find(
              "FilterDesigner sample_rate must be a number greater than") !=
              std::string::npos,
          "FilterDesigner sample_rate validation should be specific: " +
              bad_filter_sample_rate_executor.GetLastError());

    const std::string bad_filter_cutoff_high_json =
        R"({"nodes":[)"
        R"({"id":509,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":510,"type":"FilterDesigner","name":"BadFilterCutoffHigh","parameters":{)"
        R"("signal_col":"signal","filter_type":"bandpass","cutoff":"0.4","cutoff_high":"0.2","sample_rate":"1","order":"2"}})"
        R"(],"links":[{"start_node":509,"end_node":510}]})";

    cyxwiz::PipelineExecutor bad_filter_cutoff_high_executor;
    Check(!bad_filter_cutoff_high_executor.ExecutePipeline(
              bad_filter_cutoff_high_json),
          "FilterDesigner inverted band cutoff should fail validation");
    Check(bad_filter_cutoff_high_executor.GetLastError().find(
              "FilterDesigner bandpass requires cutoff_high > cutoff") !=
              std::string::npos,
          "FilterDesigner band cutoff validation should be specific: " +
              bad_filter_cutoff_high_executor.GetLastError());

    const std::string missing_filter_cutoff_high_json =
        R"({"nodes":[)"
        R"({"id":511,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":512,"type":"FilterDesigner","name":"MissingFilterCutoffHigh","parameters":{)"
        R"("signal_col":"signal","filter_type":"bandstop","cutoff":"0.4","sample_rate":"1","order":"2"}})"
        R"(],"links":[{"start_node":511,"end_node":512}]})";

    cyxwiz::PipelineExecutor missing_filter_cutoff_high_executor;
    Check(!missing_filter_cutoff_high_executor.ExecutePipeline(
              missing_filter_cutoff_high_json),
          "FilterDesigner band filter without cutoff_high should fail validation");
    Check(missing_filter_cutoff_high_executor.GetLastError().find(
              "FilterDesigner bandstop requires cutoff_high") !=
              std::string::npos,
          "FilterDesigner missing cutoff_high validation should be specific: " +
              missing_filter_cutoff_high_executor.GetLastError());

    const std::string bad_fft_sample_rate_json =
        R"({"nodes":[)"
        R"({"id":385,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":386,"type":"FFTNode","name":"BadFFTSampleRate","parameters":{)"
        R"("signal_col":"signal","sample_rate":"0"}})"
        R"(],"links":[{"start_node":385,"end_node":386}]})";

    cyxwiz::PipelineExecutor bad_fft_sample_rate_executor;
    Check(!bad_fft_sample_rate_executor.ExecutePipeline(
              bad_fft_sample_rate_json),
          "FFTNode zero sample_rate should fail validation");
    Check(bad_fft_sample_rate_executor.GetLastError().find(
              "FFTNode sample_rate must be a number greater than") !=
              std::string::npos,
          "FFTNode sample_rate validation should be specific: " +
              bad_fft_sample_rate_executor.GetLastError());

    const std::string bad_fft_signal_type_json =
        R"({"nodes":[)"
        R"({"id":399,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":400,"type":"FFTNode","name":"BadFFTSignal","parameters":{)"
        R"("signal_col":"phrase","sample_rate":"1"}})"
        R"(],"links":[{"start_node":399,"end_node":400}]})";

    cyxwiz::PipelineExecutor bad_fft_signal_type_executor;
    Check(!bad_fft_signal_type_executor.ExecutePipeline(
              bad_fft_signal_type_json),
          "FFTNode string signal_col should fail schema validation");
    Check(bad_fft_signal_type_executor.GetLastError().find(
              "FFTNode: signal column 'phrase' must be numeric") !=
              std::string::npos,
          "FFTNode signal_col validation should be specific: " +
              bad_fft_signal_type_executor.GetLastError());

    const std::string uppercase_filter_type_json =
        R"({"nodes":[)"
        R"({"id":354,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(ts_analysis_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":355,"type":"FilterDesigner","name":"Filter","parameters":{)"
        R"("signal_col":"signal","filter_type":" HIGHPASS ","cutoff":"0.1","sample_rate":"1.0","order":"2"}})"
        R"(],"links":[{"start_node":354,"end_node":355}]})";

    cyxwiz::PipelineExecutor uppercase_filter_type_executor;
    Check(uppercase_filter_type_executor.ExecutePipeline(uppercase_filter_type_json),
          "FilterDesigner uppercase filter_type should execute after normalization: " +
              uppercase_filter_type_executor.GetLastError());
    check_operator_field("ds_operator_FilterDesigner_355", "signal",
                         "FilterDesigner uppercase filter_type");

    const std::string uppercase_decomposition_choices_json =
        R"({"nodes":[)"
        R"({"id":356,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(ts_analysis_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":357,"type":"TimeSeriesDecomposition","name":"Decompose","parameters":{)"
        R"("signal_col":"signal","period":"4","method":" MULTIPLICATIVE ","algorithm":"CLASSICAL"}})"
        R"(],"links":[{"start_node":356,"end_node":357}]})";

    cyxwiz::PipelineExecutor uppercase_decomposition_choices_executor;
    Check(uppercase_decomposition_choices_executor.ExecutePipeline(
              uppercase_decomposition_choices_json),
          "TimeSeriesDecomposition uppercase choices should execute after normalization: " +
              uppercase_decomposition_choices_executor.GetLastError());
    check_operator_field("ds_operator_TimeSeriesDecomposition_357", "trend",
                         "TimeSeriesDecomposition uppercase choices");

    const std::string bad_decomposition_signal_type_json =
        R"({"nodes":[)"
        R"({"id":401,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":402,"type":"TimeSeriesDecomposition","name":"BadDecomposeSignal","parameters":{)"
        R"("signal_col":"phrase","period":"2"}})"
        R"(],"links":[{"start_node":401,"end_node":402}]})";

    cyxwiz::PipelineExecutor bad_decomposition_signal_type_executor;
    Check(!bad_decomposition_signal_type_executor.ExecutePipeline(
              bad_decomposition_signal_type_json),
          "TimeSeriesDecomposition string signal_col should fail schema validation");
    Check(bad_decomposition_signal_type_executor.GetLastError().find(
              "TimeSeriesDecomposition: signal column 'phrase' must be numeric") !=
              std::string::npos,
          "TimeSeriesDecomposition signal_col validation should be specific: " +
              bad_decomposition_signal_type_executor.GetLastError());

    const std::string bad_stationarity_max_lags_json =
        R"({"nodes":[)"
        R"({"id":387,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":388,"type":"StationarityTest","name":"BadStationarityMaxLags","parameters":{)"
        R"("signal_col":"signal","max_lags":"-2"}})"
        R"(],"links":[{"start_node":387,"end_node":388}]})";

    cyxwiz::PipelineExecutor bad_stationarity_max_lags_executor;
    Check(!bad_stationarity_max_lags_executor.ExecutePipeline(
              bad_stationarity_max_lags_json),
          "StationarityTest max_lags below sentinel should fail validation");
    Check(bad_stationarity_max_lags_executor.GetLastError().find(
              "StationarityTest max_lags must be an integer >= -1") !=
              std::string::npos,
          "StationarityTest max_lags validation should be specific: " +
              bad_stationarity_max_lags_executor.GetLastError());

    const std::string bad_seasonality_min_period_json =
        R"({"nodes":[)"
        R"({"id":389,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":390,"type":"SeasonalityDetector","name":"BadSeasonalityMinPeriod","parameters":{)"
        R"("signal_col":"signal","min_period":"1"}})"
        R"(],"links":[{"start_node":389,"end_node":390}]})";

    cyxwiz::PipelineExecutor bad_seasonality_min_period_executor;
    Check(!bad_seasonality_min_period_executor.ExecutePipeline(
              bad_seasonality_min_period_json),
          "SeasonalityDetector min_period below 2 should fail validation");
    Check(bad_seasonality_min_period_executor.GetLastError().find(
              "SeasonalityDetector min_period must be an integer >= 2") !=
              std::string::npos,
          "SeasonalityDetector min_period validation should be specific: " +
              bad_seasonality_min_period_executor.GetLastError());

    const std::string uppercase_exp_smoothing_method_json =
        R"({"nodes":[)"
        R"({"id":358,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(ts_analysis_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":359,"type":"ExponentialSmoothing","name":"Smoothing","parameters":{)"
        R"("signal_col":"signal","method":" HOLT "}})"
        R"(],"links":[{"start_node":358,"end_node":359}]})";

    cyxwiz::PipelineExecutor uppercase_exp_smoothing_method_executor;
    Check(uppercase_exp_smoothing_method_executor.ExecutePipeline(
              uppercase_exp_smoothing_method_json),
          "ExponentialSmoothing uppercase method should execute after normalization: " +
              uppercase_exp_smoothing_method_executor.GetLastError());
    check_operator_field("ds_operator_ExponentialSmoothing_359", "fitted",
                         "ExponentialSmoothing uppercase method");

    const std::string bad_time_series_features_value_type_json =
        R"({"nodes":[)"
        R"({"id":403,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":404,"type":"TimeSeriesFeatures","name":"BadTSFeaturesValue","parameters":{)"
        R"("value_col":"phrase","lag_values":"1","rolling_windows":"2"}})"
        R"(],"links":[{"start_node":403,"end_node":404}]})";

    cyxwiz::PipelineExecutor bad_time_series_features_value_type_executor;
    Check(!bad_time_series_features_value_type_executor.ExecutePipeline(
              bad_time_series_features_value_type_json),
          "TimeSeriesFeatures string value_col should fail schema validation");
    Check(bad_time_series_features_value_type_executor.GetLastError().find(
              "TimeSeriesFeatures: value column 'phrase' must be numeric") !=
              std::string::npos,
          "TimeSeriesFeatures value_col validation should be specific: " +
              bad_time_series_features_value_type_executor.GetLastError());

    const std::string missing_decomposition_period_json =
        R"({"nodes":[)"
        R"({"id":326,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":327,"type":"TimeSeriesDecomposition","name":"MissingPeriod","parameters":{)"
        R"("signal_col":"signal"}})"
        R"(],"links":[{"start_node":326,"end_node":327}]})";

    cyxwiz::PipelineExecutor missing_decomposition_period_executor;
    Check(!missing_decomposition_period_executor.ExecutePipeline(
              missing_decomposition_period_json),
          "TimeSeriesDecomposition missing period should fail validation");
    Check(missing_decomposition_period_executor.GetLastError().find(
              "missing required parameter 'period'") != std::string::npos,
          "TimeSeriesDecomposition missing period validation should be specific: " +
              missing_decomposition_period_executor.GetLastError());

    const std::string bad_ts_window_input_width_json =
        R"({"nodes":[)"
        R"({"id":328,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":329,"type":"TimeSeriesWindow","name":"BadInputWidth","parameters":{)"
        R"("value_col":"value","input_width":"0"}})"
        R"(],"links":[{"start_node":328,"end_node":329}]})";

    cyxwiz::PipelineExecutor bad_ts_window_input_width_executor;
    Check(!bad_ts_window_input_width_executor.ExecutePipeline(
              bad_ts_window_input_width_json),
          "TimeSeriesWindow input_width 0 should fail validation");
    Check(bad_ts_window_input_width_executor.GetLastError().find(
              "TimeSeriesWindow input_width must be an integer >= 1") !=
              std::string::npos,
          "TimeSeriesWindow input_width validation should be specific: " +
              bad_ts_window_input_width_executor.GetLastError());

    const std::string bad_ts_window_feature_type_json =
        R"({"nodes":[)"
        R"({"id":417,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(mixed_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":418,"type":"TimeSeriesWindow","name":"BadWindowFeature","parameters":{)"
        R"("value_col":"x","feature_cols":"phrase","input_width":"1","shift":"1"}})"
        R"(],"links":[{"start_node":417,"end_node":418}]})";

    cyxwiz::PipelineExecutor bad_ts_window_feature_type_executor;
    Check(!bad_ts_window_feature_type_executor.ExecutePipeline(
              bad_ts_window_feature_type_json),
          "TimeSeriesWindow string feature_cols should fail schema validation");
    Check(bad_ts_window_feature_type_executor.GetLastError().find(
              "TimeSeriesWindow: feature column 'phrase' must be numeric") !=
              std::string::npos,
          "TimeSeriesWindow feature_cols validation should be specific: " +
              bad_ts_window_feature_type_executor.GetLastError());

    const std::string bad_ts_window_time_type_json =
        R"({"nodes":[)"
        R"({"id":419,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(mixed_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":420,"type":"TimeSeriesWindow","name":"BadWindowTime","parameters":{)"
        R"("value_col":"x","time_col":"phrase","input_width":"1","shift":"1"}})"
        R"(],"links":[{"start_node":419,"end_node":420}]})";

    cyxwiz::PipelineExecutor bad_ts_window_time_type_executor;
    Check(!bad_ts_window_time_type_executor.ExecutePipeline(
              bad_ts_window_time_type_json),
          "TimeSeriesWindow string time_col should fail schema validation");
    Check(bad_ts_window_time_type_executor.GetLastError().find(
              "TimeSeriesWindow: time column 'phrase' must be numeric") !=
              std::string::npos,
          "TimeSeriesWindow time_col validation should be specific: " +
              bad_ts_window_time_type_executor.GetLastError());

    const std::string ts_window_multivariate_time_json =
        R"({"nodes":[)"
        R"({"id":421,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(mixed_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":422,"type":"TimeSeriesWindow","name":"WindowMultiTime","parameters":{)"
        R"("value_col":"x","feature_cols":"y","time_col":"x","input_width":"1","shift":"1"}})"
        R"(],"links":[{"start_node":421,"end_node":422}]})";

    cyxwiz::PipelineExecutor ts_window_multivariate_time_executor;
    Check(ts_window_multivariate_time_executor.ExecutePipeline(
              ts_window_multivariate_time_json),
          "TimeSeriesWindow numeric feature_cols/time_col should execute: " +
              ts_window_multivariate_time_executor.GetLastError());
    auto ts_window_multi = registry.GetArrowDataset("ds_operator_TimeSeriesWindow_422");
    Check(ts_window_multi != nullptr,
          "TimeSeriesWindow multivariate/time output dataset is registered");
    auto ts_window_multi_table = ts_window_multi->GetArrowTable();
    Check(ts_window_multi_table != nullptr,
          "TimeSeriesWindow multivariate/time output table exists");
    Check(ts_window_multi_table->schema()->GetFieldIndex("y_x_0") >= 0,
          "TimeSeriesWindow multivariate output has feature block");
    Check(ts_window_multi_table->schema()->GetFieldIndex("__window_start_time") >= 0,
          "TimeSeriesWindow multivariate output has time metadata");

    const std::string bad_ts_features_lags_json =
        R"({"nodes":[)"
        R"({"id":330,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":331,"type":"TimeSeriesFeatures","name":"BadLags","parameters":{)"
        R"("value_col":"value","lag_values":"1,0"}})"
        R"(],"links":[{"start_node":330,"end_node":331}]})";

    cyxwiz::PipelineExecutor bad_ts_features_lags_executor;
    Check(!bad_ts_features_lags_executor.ExecutePipeline(
              bad_ts_features_lags_json),
          "TimeSeriesFeatures lag_values containing 0 should fail validation");
    Check(bad_ts_features_lags_executor.GetLastError().find(
              "TimeSeriesFeatures lag_values must be a comma-separated list of integers >= 1") !=
              std::string::npos,
          "TimeSeriesFeatures lag_values validation should be specific: " +
              bad_ts_features_lags_executor.GetLastError());

    const std::string bad_pca_components_json =
        R"({"nodes":[)"
        R"({"id":332,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":333,"type":"PCANode","name":"BadComponents","parameters":{)"
        R"("n_components":"0"}})"
        R"(],"links":[{"start_node":332,"end_node":333}]})";

    cyxwiz::PipelineExecutor bad_pca_components_executor;
    Check(!bad_pca_components_executor.ExecutePipeline(bad_pca_components_json),
          "PCANode n_components 0 should fail validation");
    Check(bad_pca_components_executor.GetLastError().find(
              "PCANode n_components must be an integer >= 1") !=
              std::string::npos,
          "PCANode n_components validation should be specific: " +
              bad_pca_components_executor.GetLastError());

    const std::string bad_kmeans_max_iter_json =
        R"({"nodes":[)"
        R"({"id":334,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":335,"type":"KMeansCluster","name":"BadMaxIter","parameters":{)"
        R"("max_iter":"0"}})"
        R"(],"links":[{"start_node":334,"end_node":335}]})";

    cyxwiz::PipelineExecutor bad_kmeans_max_iter_executor;
    Check(!bad_kmeans_max_iter_executor.ExecutePipeline(bad_kmeans_max_iter_json),
          "KMeansCluster max_iter 0 should fail validation");
    Check(bad_kmeans_max_iter_executor.GetLastError().find(
              "KMeansCluster max_iter must be an integer >= 1") !=
              std::string::npos,
          "KMeansCluster max_iter validation should be specific: " +
              bad_kmeans_max_iter_executor.GetLastError());

    const std::string bad_decomposition_period_json =
        R"({"nodes":[)"
        R"({"id":336,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":337,"type":"TimeSeriesDecomposition","name":"BadPeriod","parameters":{)"
        R"("signal_col":"signal","period":"1"}})"
        R"(],"links":[{"start_node":336,"end_node":337}]})";

    cyxwiz::PipelineExecutor bad_decomposition_period_executor;
    Check(!bad_decomposition_period_executor.ExecutePipeline(
              bad_decomposition_period_json),
          "TimeSeriesDecomposition period 1 should fail validation");
    Check(bad_decomposition_period_executor.GetLastError().find(
              "TimeSeriesDecomposition period must be an integer >= 2") !=
              std::string::npos,
          "TimeSeriesDecomposition period validation should be specific: " +
              bad_decomposition_period_executor.GetLastError());

    const std::string missing_ts_window_target_json =
        R"({"nodes":[)"
        R"({"id":172,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":173,"type":"TSWindow","name":"MissingTarget","parameters":{)"
        R"("window_size":"2","stride":"1"}})"
        R"(],"links":[{"start_node":172,"end_node":173}]})";

    cyxwiz::PipelineExecutor missing_ts_window_target_executor;
    Check(!missing_ts_window_target_executor.ExecutePipeline(
              missing_ts_window_target_json),
          "TSWindow missing target_column should fail validation");
    Check(missing_ts_window_target_executor.GetLastError().find(
              "missing required parameter 'target_column'") != std::string::npos,
          "TSWindow missing target_column validation should be specific: " +
              missing_ts_window_target_executor.GetLastError());

    const std::string missing_ts_features_columns_json =
        R"({"nodes":[)"
        R"({"id":174,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":175,"type":"TSFeatures","name":"MissingColumns","parameters":{)"
        R"("rolling_window":"2"}})"
        R"(],"links":[{"start_node":174,"end_node":175}]})";

    cyxwiz::PipelineExecutor missing_ts_features_columns_executor;
    Check(!missing_ts_features_columns_executor.ExecutePipeline(
              missing_ts_features_columns_json),
          "TSFeatures missing columns should fail validation");
    Check(missing_ts_features_columns_executor.GetLastError().find(
              "missing required parameter 'columns'") != std::string::npos,
          "TSFeatures missing columns validation should be specific: " +
              missing_ts_features_columns_executor.GetLastError());

    const std::string missing_ts_lag_columns_json =
        R"({"nodes":[)"
        R"({"id":176,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":177,"type":"TSLag","name":"MissingColumns","parameters":{)"
        R"("lag_periods":"1"}})"
        R"(],"links":[{"start_node":176,"end_node":177}]})";

    cyxwiz::PipelineExecutor missing_ts_lag_columns_executor;
    Check(!missing_ts_lag_columns_executor.ExecutePipeline(
              missing_ts_lag_columns_json),
          "TSLag missing columns should fail validation");
    Check(missing_ts_lag_columns_executor.GetLastError().find(
              "missing required parameter 'columns'") != std::string::npos,
          "TSLag missing columns validation should be specific: " +
              missing_ts_lag_columns_executor.GetLastError());

    const std::string missing_ts_diff_columns_json =
        R"({"nodes":[)"
        R"({"id":178,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":179,"type":"TSDiff","name":"MissingColumns","parameters":{)"
        R"("order":"1"}})"
        R"(],"links":[{"start_node":178,"end_node":179}]})";

    cyxwiz::PipelineExecutor missing_ts_diff_columns_executor;
    Check(!missing_ts_diff_columns_executor.ExecutePipeline(
              missing_ts_diff_columns_json),
          "TSDiff missing columns should fail validation");
    Check(missing_ts_diff_columns_executor.GetLastError().find(
              "missing required parameter 'columns'") != std::string::npos,
          "TSDiff missing columns validation should be specific: " +
              missing_ts_diff_columns_executor.GetLastError());

    const std::string bad_crop_json =
        R"({"nodes":[)"
        R"({"id":17,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":18,"type":"TableCropper","name":"BadCrop","parameters":{)"
        R"("start_row":"-2","end_row":"10"}})"
        R"(],"links":[{"start_node":17,"end_node":18}]})";

    cyxwiz::PipelineExecutor bad_crop_executor;
    Check(!bad_crop_executor.ExecutePipeline(bad_crop_json),
          "TableCropper bad start_row should fail validation");
    Check(bad_crop_executor.GetLastError().find(
              "TableCropper start_row must be an integer >= 0") !=
              std::string::npos,
          "bad TableCropper validation should be specific: " +
              bad_crop_executor.GetLastError());

    const std::string missing_filter_condition_json =
        R"({"nodes":[)"
        R"({"id":19,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":20,"type":"FilterRows","name":"MissingCondition","parameters":{}})"
        R"(],"links":[{"start_node":19,"end_node":20}]})";

    cyxwiz::PipelineExecutor missing_filter_executor;
    Check(!missing_filter_executor.ExecutePipeline(missing_filter_condition_json),
          "FilterRows missing condition should fail validation");
    Check(missing_filter_executor.GetLastError().find(
              "missing required parameter 'condition'") != std::string::npos,
          "FilterRows missing condition validation should be specific: " +
              missing_filter_executor.GetLastError());

    const std::string filter_rows_json =
        R"({"nodes":[)"
        R"({"id":188,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":189,"type":"FilterRows","name":"Filter","parameters":{)"
        R"("condition":"x >= 2 AND y < 30"}})"
        R"(],"links":[{"start_node":188,"end_node":189}]})";

    cyxwiz::PipelineExecutor filter_rows_executor;
    Check(filter_rows_executor.ExecutePipeline(filter_rows_json),
          "FilterRows should parse, validate, and quote numeric conditions: " +
              filter_rows_executor.GetLastError());
    auto filtered = registry.GetArrowDataset("ds_filter_189");
    Check(filtered != nullptr, "FilterRows output dataset is registered");
    auto filtered_table = filtered->GetArrowTable();
    Check(filtered_table != nullptr, "FilterRows output table exists");
    Check(filtered_table->num_rows() == 1,
          "FilterRows numeric condition should keep one matching row");
    Check(ReadNumericValue(filtered_table, "x", 0) == 2.0,
          "FilterRows numeric condition should keep the expected row");

    const std::string filter_string_json =
        R"({"nodes":[)"
        R"({"id":190,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" +
        JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":191,"type":"FilterRows","name":"FilterString","parameters":{)"
        R"("condition":"phrase = 'tea cup'"}})"
        R"(],"links":[{"start_node":190,"end_node":191}]})";

    cyxwiz::PipelineExecutor filter_string_executor;
    Check(filter_string_executor.ExecutePipeline(filter_string_json),
          "FilterRows should parse and quote string conditions: " +
              filter_string_executor.GetLastError());
    auto string_filtered = registry.GetArrowDataset("ds_filter_191");
    Check(string_filtered != nullptr,
          "FilterRows string output dataset is registered");
    auto string_filtered_table = string_filtered->GetArrowTable();
    Check(string_filtered_table != nullptr,
          "FilterRows string output table exists");
    Check(string_filtered_table->num_rows() == 1,
          "FilterRows string condition should keep one matching row");
    Check(ReadStringValue(string_filtered_table, "phrase", 0) == "tea cup",
          "FilterRows string condition should keep the expected row");

    const std::string filter_missing_column_json =
        R"({"nodes":[)"
        R"({"id":192,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":193,"type":"FilterRows","name":"FilterMissing","parameters":{)"
        R"("condition":"missing = 1"}})"
        R"(],"links":[{"start_node":192,"end_node":193}]})";

    cyxwiz::PipelineExecutor filter_missing_column_executor;
    Check(!filter_missing_column_executor.ExecutePipeline(
              filter_missing_column_json),
          "FilterRows missing column should fail schema validation");
    Check(filter_missing_column_executor.GetLastError().find(
              "FilterRows: column 'missing' not found") != std::string::npos,
          "FilterRows missing column error should be specific: " +
              filter_missing_column_executor.GetLastError());

    const std::string filter_raw_sql_json =
        R"({"nodes":[)"
        R"({"id":194,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":195,"type":"FilterRows","name":"FilterRaw","parameters":{)"
        R"("condition":"x > 1; DROP TABLE temp"}})"
        R"(],"links":[{"start_node":194,"end_node":195}]})";

    cyxwiz::PipelineExecutor filter_raw_sql_executor;
    Check(!filter_raw_sql_executor.ExecutePipeline(filter_raw_sql_json),
          "FilterRows raw SQL tokens should fail before query construction");
    Check(filter_raw_sql_executor.GetLastError().find(
              "FilterRows: unsupported token ';'") != std::string::npos,
          "FilterRows raw SQL token error should be specific: " +
              filter_raw_sql_executor.GetLastError());

    const std::string missing_join_column_json =
        R"({"nodes":[)"
        R"({"id":21,"type":"DataInput","name":"Left","parameters":{)"
        R"("source_type":"file","file_path":"left.csv","type":"csv"}},)"
        R"({"id":22,"type":"DataInput","name":"Right","parameters":{)"
        R"("source_type":"file","file_path":"right.csv","type":"csv"}},)"
        R"({"id":23,"type":"Join","name":"Join","parameters":{}})"
        R"(],"links":[{"start_node":21,"end_node":23},{"start_node":22,"end_node":23}]})";

    cyxwiz::PipelineExecutor missing_join_executor;
    Check(!missing_join_executor.ExecutePipeline(missing_join_column_json),
          "Join missing on_column should fail validation");
    Check(missing_join_executor.GetLastError().find(
              "missing required parameter 'on_column'") != std::string::npos,
          "Join missing on_column validation should be specific: " +
              missing_join_executor.GetLastError());

    const std::string join_json =
        R"({"nodes":[)"
        R"({"id":101,"type":"DataInput","name":"Left","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":102,"type":"DataInput","name":"Right","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":103,"type":"Join","name":"Join","parameters":{)"
        R"("on_column":" x ","join_type":"outer"}})"
        R"(],"links":[{"start_node":101,"end_node":103},{"start_node":102,"end_node":103}]})";

    cyxwiz::PipelineExecutor join_executor;
    Check(join_executor.ExecutePipeline(join_json),
          "Join should validate and quote the join column on both inputs: " +
              join_executor.GetLastError());
    auto joined = registry.GetArrowDataset("ds_join_103");
    Check(joined != nullptr, "Join output dataset is registered");
    auto joined_table = joined->GetArrowTable();
    Check(joined_table != nullptr, "Join output table exists");
    Check(joined_table->num_rows() == 3,
          "Join on x should match all rows in the self-join fixture");

    const std::string bad_join_type_json =
        R"({"nodes":[)"
        R"({"id":111,"type":"DataInput","name":"Left","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":112,"type":"DataInput","name":"Right","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":113,"type":"Join","name":"JoinBadType","parameters":{)"
        R"("on_column":"x","join_type":"sideways"}})"
        R"(],"links":[{"start_node":111,"end_node":113},{"start_node":112,"end_node":113}]})";

    cyxwiz::PipelineExecutor bad_join_type_executor;
    Check(!bad_join_type_executor.ExecutePipeline(bad_join_type_json),
          "Join unsupported join_type should fail validation");
    Check(bad_join_type_executor.GetLastError().find(
              "Join join_type 'sideways' is not supported") != std::string::npos,
          "Join unsupported join_type error should be specific: " +
              bad_join_type_executor.GetLastError());

    const std::string missing_join_input_column_json =
        R"({"nodes":[)"
        R"({"id":104,"type":"DataInput","name":"Left","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":105,"type":"DataInput","name":"Right","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":106,"type":"Join","name":"JoinMissing","parameters":{)"
        R"("on_column":"missing","join_type":"INNER"}})"
        R"(],"links":[{"start_node":104,"end_node":106},{"start_node":105,"end_node":106}]})";

    cyxwiz::PipelineExecutor missing_join_input_column_executor;
    Check(!missing_join_input_column_executor.ExecutePipeline(
              missing_join_input_column_json),
          "Join missing input column should fail schema validation");
    Check(missing_join_input_column_executor.GetLastError().find(
              "Join: column 'missing' not found in left input") !=
              std::string::npos,
          "Join missing column error should be specific: " +
              missing_join_input_column_executor.GetLastError());

    const std::string missing_poly_columns_json =
        R"({"nodes":[)"
        R"({"id":24,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":25,"type":"PolynomialFeatures","name":"MissingPolyColumns","parameters":{)"
        R"("degree":"2"}})"
        R"(],"links":[{"start_node":24,"end_node":25}]})";

    cyxwiz::PipelineExecutor missing_poly_columns_executor;
    Check(!missing_poly_columns_executor.ExecutePipeline(missing_poly_columns_json),
          "PolynomialFeatures missing columns should fail validation");
    Check(missing_poly_columns_executor.GetLastError().find(
              "missing required parameter 'columns'") != std::string::npos,
          "PolynomialFeatures missing columns validation should be specific: " +
              missing_poly_columns_executor.GetLastError());

    const std::string unknown_node_json =
        R"({"nodes":[)"
        R"({"id":26,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":27,"type":"DefinitelyMissingNode","name":"Unknown","parameters":{}})"
        R"(],"links":[{"start_node":26,"end_node":27}]})";

    cyxwiz::PipelineExecutor unknown_node_executor;
    Check(!unknown_node_executor.ExecutePipeline(unknown_node_json),
          "unknown node type should fail validation");
    Check(unknown_node_executor.GetLastError().find(
              "unsupported node type 'DefinitelyMissingNode'") !=
              std::string::npos,
          "unknown node validation should be specific: " +
              unknown_node_executor.GetLastError());

    const std::string parquet_input_json =
        R"({"nodes":[)"
        R"({"id":28,"type":"ParquetInput","name":"Parquet","parameters":{)"
        R"("file_path":"ignored.parquet"}})"
        R"(],"links":[]})";

    cyxwiz::PipelineExecutor parquet_input_executor;
    Check(!parquet_input_executor.ExecutePipeline(parquet_input_json),
          "legacy ParquetInput source should fail closed");
    Check(parquet_input_executor.GetLastError().find(
              "legacy ParquetInput execution is not implemented") !=
              std::string::npos,
          "ParquetInput should use fail-closed runtime support: " +
              parquet_input_executor.GetLastError());

    const std::string bad_output_format_json =
        R"({"nodes":[)"
        R"({"id":29,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":30,"type":"DataOutput","name":"BadOutput","parameters":{)"
        R"("file_path":"ignored.xml","format":"xml"}})"
        R"(],"links":[{"start_node":29,"end_node":30}]})";

    cyxwiz::PipelineExecutor bad_output_format_executor;
    Check(!bad_output_format_executor.ExecutePipeline(bad_output_format_json),
          "DataOutput bad format should fail validation");
    Check(bad_output_format_executor.GetLastError().find(
              "DataOutput format 'xml' is not supported") != std::string::npos,
          "bad DataOutput format validation should be specific: " +
              bad_output_format_executor.GetLastError());

    const std::string json_output_format_json =
        R"({"nodes":[)"
        R"({"id":364,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":365,"type":"DataOutput","name":"JsonOutput","parameters":{)"
        R"("file_path":"ignored.json","format":"json"}})"
        R"(],"links":[{"start_node":364,"end_node":365}]})";

    cyxwiz::PipelineExecutor json_output_format_executor;
    Check(!json_output_format_executor.ExecutePipeline(json_output_format_json),
          "DataOutput json format should fail validation until JSON export is real");
    Check(json_output_format_executor.GetLastError().find(
              "DataOutput format 'json' is not supported") != std::string::npos,
          "DataOutput json format validation should be specific: " +
              json_output_format_executor.GetLastError());

    const std::string missing_output_path_json =
        R"({"nodes":[)"
        R"({"id":31,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":32,"type":"DataOutput","name":"MissingOutputPath","parameters":{)"
        R"("format":"csv"}})"
        R"(],"links":[{"start_node":31,"end_node":32}]})";

    cyxwiz::PipelineExecutor missing_output_path_executor;
    Check(!missing_output_path_executor.ExecutePipeline(missing_output_path_json),
          "DataOutput missing file_path should fail validation");
    Check(missing_output_path_executor.GetLastError().find(
              "missing required parameter 'file_path'") != std::string::npos,
          "DataOutput missing file_path validation should be specific: " +
              missing_output_path_executor.GetLastError());

    const std::string data_output_mixed_case_json =
        R"({"nodes":[)"
        R"({"id":325,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"FILE","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"CSV","has_header":"TRUE"}},)"
        R"({"id":326,"type":"DataOutput","name":"Output","parameters":{)"
        R"("file_path":")" +
        JsonEscapePath(data_output_mixed_case_csv_path.string()) +
        R"(","format":"CSV"}})"
        R"(],"links":[{"start_node":325,"end_node":326}]})";

    cyxwiz::PipelineExecutor data_output_mixed_case_executor;
    Check(data_output_mixed_case_executor.ExecutePipeline(
              data_output_mixed_case_json),
          "DataInput/DataOutput should normalize accepted enum values before execution: " +
              data_output_mixed_case_executor.GetLastError());
    Check(fs::exists(data_output_mixed_case_csv_path),
          "DataOutput mixed-case CSV format should create the output file");

    const std::string column_appender_json =
        R"({"nodes":[)"
        R"({"id":33,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":34,"type":"ColumnAppender","name":"AppendColumns","parameters":{}})"
        R"(],"links":[{"start_node":33,"end_node":34}]})";

    cyxwiz::PipelineExecutor column_appender_executor;
    Check(!column_appender_executor.ExecutePipeline(column_appender_json),
          "ColumnAppender placeholder should fail closed");
    Check(column_appender_executor.GetLastError().find(
              "legacy ColumnAppender execution is still a passthrough placeholder") !=
              std::string::npos,
          "ColumnAppender should use fail-closed runtime support: " +
              column_appender_executor.GetLastError());

    const std::string export_csv_json =
        R"({"nodes":[)"
        R"({"id":35,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":36,"type":"ExportCSV","name":"Export","parameters":{)"
        R"("file_path":")" + JsonEscapePath(export_csv_path.string()) + R"("}})"
        R"(],"links":[{"start_node":35,"end_node":36}]})";

    cyxwiz::PipelineExecutor export_csv_executor;
    Check(export_csv_executor.ExecutePipeline(export_csv_json),
          "ExportCSV should write through DataRegistry: " +
              export_csv_executor.GetLastError());
    Check(fs::exists(export_csv_path), "ExportCSV should create the output file");

    const std::string export_csv_path_alias_json =
        R"({"nodes":[)"
        R"({"id":139,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":140,"type":"ExportCSV","name":"Export","parameters":{)"
        R"("path":")" + JsonEscapePath(export_csv_alias_path.string()) + R"("}})"
        R"(],"links":[{"start_node":139,"end_node":140}]})";

    cyxwiz::PipelineExecutor export_csv_path_alias_executor;
    Check(export_csv_path_alias_executor.ExecutePipeline(
              export_csv_path_alias_json),
          "ExportCSV should accept Data Studio path parameter: " +
              export_csv_path_alias_executor.GetLastError());
    Check(fs::exists(export_csv_alias_path),
          "ExportCSV path alias should create the output file");

    const std::string save_dataset_json =
        R"({"nodes":[)"
        R"({"id":135,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":136,"type":"SaveDataset","name":"Save","parameters":{)"
        R"("path":")" + JsonEscapePath(save_dataset_csv_path.string()) +
        R"(","format":"csv","name":"saved_alias"}})"
        R"(],"links":[{"start_node":135,"end_node":136}]})";

    cyxwiz::PipelineExecutor save_dataset_executor;
    Check(save_dataset_executor.ExecutePipeline(save_dataset_json),
          "SaveDataset should export when path is supplied: " +
              save_dataset_executor.GetLastError());
    Check(fs::exists(save_dataset_csv_path),
          "SaveDataset should create the requested output file");
    Check(registry.GetArrowDataset("saved_alias") != nullptr,
          "SaveDataset should preserve legacy in-memory alias behavior");

    const std::string save_dataset_downstream_json =
        R"({"nodes":[)"
        R"({"id":361,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":362,"type":"SaveDataset","name":"Save","parameters":{)"
        R"("name":"saved_downstream"}},)"
        R"({"id":363,"type":"SelectColumns","name":"Select","parameters":{)"
        R"("columns":"x"}})"
        R"(],"links":[{"start_node":361,"end_node":362},{"start_node":362,"end_node":363}]})";

    cyxwiz::PipelineExecutor save_dataset_downstream_executor;
    Check(save_dataset_downstream_executor.ExecutePipeline(
              save_dataset_downstream_json),
          "SaveDataset should publish a dataset binding for downstream nodes: " +
              save_dataset_downstream_executor.GetLastError());
    auto saved_downstream = registry.GetArrowDataset("ds_select_363");
    Check(saved_downstream != nullptr,
          "SelectColumns should consume the SaveDataset output binding");
    auto saved_downstream_table = saved_downstream->GetArrowTable();
    Check(saved_downstream_table != nullptr,
          "downstream SelectColumns output table should exist");
    Check(saved_downstream_table->num_columns() == 1,
          "downstream SelectColumns should preserve selected schema");

    const std::string bad_save_dataset_format_json =
        R"({"nodes":[)"
        R"({"id":137,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":138,"type":"SaveDataset","name":"BadSave","parameters":{)"
        R"("path":"ignored.arrow","format":"arrow"}})"
        R"(],"links":[{"start_node":137,"end_node":138}]})";

    cyxwiz::PipelineExecutor bad_save_dataset_format_executor;
    Check(!bad_save_dataset_format_executor.ExecutePipeline(
              bad_save_dataset_format_json),
          "SaveDataset unsupported format should fail validation");
    Check(bad_save_dataset_format_executor.GetLastError().find(
              "SaveDataset format 'arrow' is not supported") !=
              std::string::npos,
          "SaveDataset bad format validation should be specific: " +
              bad_save_dataset_format_executor.GetLastError());

    const std::string json_save_dataset_format_json =
        R"({"nodes":[)"
        R"({"id":366,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":367,"type":"SaveDataset","name":"JsonSave","parameters":{)"
        R"("path":"ignored.json","format":"json"}})"
        R"(],"links":[{"start_node":366,"end_node":367}]})";

    cyxwiz::PipelineExecutor json_save_dataset_format_executor;
    Check(!json_save_dataset_format_executor.ExecutePipeline(
              json_save_dataset_format_json),
          "SaveDataset json format should fail validation until JSON export is real");
    Check(json_save_dataset_format_executor.GetLastError().find(
              "SaveDataset format 'json' is not supported") !=
              std::string::npos,
          "SaveDataset json format validation should be specific: " +
              json_save_dataset_format_executor.GetLastError());

    const std::string deploy_downstream_json =
        R"({"nodes":[)"
        R"({"id":368,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":369,"type":"DeployToNodeEditor","name":"Deploy","parameters":{)"
        R"("name":"deployed_downstream"}},)"
        R"({"id":370,"type":"SelectColumns","name":"Select","parameters":{)"
        R"("columns":"y"}})"
        R"(],"links":[{"start_node":368,"end_node":369},{"start_node":369,"end_node":370}]})";

    cyxwiz::PipelineExecutor deploy_downstream_executor;
    Check(deploy_downstream_executor.ExecutePipeline(deploy_downstream_json),
          "DeployToNodeEditor should publish a dataset binding for downstream nodes: " +
              deploy_downstream_executor.GetLastError());
    Check(deploy_downstream_executor.IsDeploymentReady(),
          "DeployToNodeEditor should still mark deployment ready");
    Check(deploy_downstream_executor.GetDeploymentDataset() == "deployed_downstream",
          "DeployToNodeEditor should preserve deployment dataset name");
    auto deployed_downstream = registry.GetArrowDataset("ds_select_370");
    Check(deployed_downstream != nullptr,
          "SelectColumns should consume the DeployToNodeEditor output binding");
    auto deployed_downstream_table = deployed_downstream->GetArrowTable();
    Check(deployed_downstream_table != nullptr,
          "downstream DeployToNodeEditor output table should exist");
    Check(deployed_downstream_table->schema()->GetFieldIndex("y") >= 0,
          "downstream SelectColumns should preserve selected deploy schema");

    const std::string export_json_json =
        R"({"nodes":[)"
        R"({"id":37,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":38,"type":"ExportJSON","name":"Export","parameters":{)"
        R"("file_path":"ignored.json"}})"
        R"(],"links":[{"start_node":37,"end_node":38}]})";

    cyxwiz::PipelineExecutor export_json_executor;
    Check(!export_json_executor.ExecutePipeline(export_json_json),
          "ExportJSON fake-success placeholder should fail closed");
    Check(export_json_executor.GetLastError().find(
              "legacy ExportJSON execution is still a fake-success placeholder") !=
              std::string::npos,
          "ExportJSON should use fail-closed runtime support: " +
              export_json_executor.GetLastError());

    const std::string dangling_link_json =
        R"({"nodes":[)"
        R"({"id":39,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":40,"type":"StandardScaler","name":"Scale","parameters":{)"
        R"("columns":"x"}})"
        R"(],"links":[{"start_node":999,"end_node":40}]})";

    cyxwiz::PipelineExecutor dangling_link_executor;
    Check(!dangling_link_executor.ExecutePipeline(dangling_link_json),
          "dangling link start endpoint should fail parsing");
    Check(dangling_link_executor.GetLastError().find(
              "Link references missing start node id: 999") != std::string::npos,
          "dangling link parse error should keep the missing endpoint: " +
              dangling_link_executor.GetLastError());

    const std::string disconnected_json =
        R"({"nodes":[)"
        R"({"id":41,"type":"DataInput","name":"Left","parameters":{)"
        R"("source_type":"file","file_path":"left.csv","type":"csv"}},)"
        R"({"id":42,"type":"DataInput","name":"Right","parameters":{)"
        R"("source_type":"file","file_path":"right.csv","type":"csv"}})"
        R"(],"links":[]})";

    cyxwiz::PipelineExecutor disconnected_executor;
    Check(!disconnected_executor.ExecutePipeline(disconnected_json),
          "disconnected graph should fail validation");
    Check(disconnected_executor.GetLastError().find(
              "Pipeline contains disconnected nodes") != std::string::npos,
          "disconnected graph validation should be specific: " +
              disconnected_executor.GetLastError());

    const std::string cyclic_json =
        R"({"nodes":[)"
        R"({"id":200,"type":"StandardScaler","name":"ScaleA","parameters":{)"
        R"("columns":"x"}},)"
        R"({"id":201,"type":"StandardScaler","name":"ScaleB","parameters":{)"
        R"("columns":"x"}})"
        R"(],"links":[)"
        R"({"start_node":200,"end_node":201},)"
        R"({"start_node":201,"end_node":200})"
        R"(]})";

    cyxwiz::PipelineExecutor cyclic_executor;
    Check(!cyclic_executor.ExecutePipeline(cyclic_json),
          "cyclic graph should fail validation");
    Check(cyclic_executor.GetLastError().find(
              "Pipeline contains a cycle") != std::string::npos,
          "cycle validation should be specific: " +
              cyclic_executor.GetLastError());

    const std::string rename_columns_json =
        R"({"nodes":[)"
        R"({"id":43,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":44,"type":"RenameColumns","name":"Rename","parameters":{)"
        R"("mapping":"x:feature_x, y:target_y"}})"
        R"(],"links":[{"start_node":43,"end_node":44}]})";

    cyxwiz::PipelineExecutor rename_columns_executor;
    Check(rename_columns_executor.ExecutePipeline(rename_columns_json),
          "RenameColumns should rename Arrow schema fields: " +
              rename_columns_executor.GetLastError());
    auto renamed = registry.GetArrowDataset("ds_renamed_44");
    Check(renamed != nullptr, "RenameColumns output dataset is registered");
    auto renamed_table = renamed->GetArrowTable();
    Check(renamed_table != nullptr, "RenameColumns output table exists");
    Check(renamed_table->schema()->GetFieldIndex("feature_x") >= 0,
          "RenameColumns should expose renamed feature_x field");
    Check(renamed_table->schema()->GetFieldIndex("target_y") >= 0,
          "RenameColumns should expose renamed target_y field");
    Check(renamed_table->schema()->GetFieldIndex("x") < 0,
          "RenameColumns should remove old x field name");

    const std::string missing_rename_mapping_json =
        R"({"nodes":[)"
        R"({"id":45,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":46,"type":"RenameColumns","name":"Rename","parameters":{}})"
        R"(],"links":[{"start_node":45,"end_node":46}]})";

    cyxwiz::PipelineExecutor missing_rename_mapping_executor;
    Check(!missing_rename_mapping_executor.ExecutePipeline(missing_rename_mapping_json),
          "RenameColumns missing mapping should fail validation");
    Check(missing_rename_mapping_executor.GetLastError().find(
              "missing required parameter 'mapping'") != std::string::npos,
          "RenameColumns missing mapping validation should be specific: " +
              missing_rename_mapping_executor.GetLastError());

    const std::string bad_rename_mapping_json =
        R"({"nodes":[)"
        R"({"id":47,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":48,"type":"RenameColumns","name":"Rename","parameters":{)"
        R"("mapping":"missing:renamed"}})"
        R"(],"links":[{"start_node":47,"end_node":48}]})";

    cyxwiz::PipelineExecutor bad_rename_mapping_executor;
    Check(!bad_rename_mapping_executor.ExecutePipeline(bad_rename_mapping_json),
          "RenameColumns unknown input column should fail execution");
    Check(bad_rename_mapping_executor.GetLastError().find(
              "input column 'missing' does not exist") != std::string::npos,
          "RenameColumns unknown input column error should be specific: " +
              bad_rename_mapping_executor.GetLastError());

    const std::string row_to_column_names_json =
        R"({"nodes":[)"
        R"({"id":49,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"false"}},)"
        R"({"id":50,"type":"RowToColumnNames","name":"Promote","parameters":{)"
        R"("row_index":"0"}})"
        R"(],"links":[{"start_node":49,"end_node":50}]})";

    cyxwiz::PipelineExecutor row_to_column_names_executor;
    Check(row_to_column_names_executor.ExecutePipeline(row_to_column_names_json),
          "RowToColumnNames should promote a data row to Arrow schema fields: " +
              row_to_column_names_executor.GetLastError());
    auto promoted = registry.GetArrowDataset("ds_newheaders_50");
    Check(promoted != nullptr, "RowToColumnNames output dataset is registered");
    auto promoted_table = promoted->GetArrowTable();
    Check(promoted_table != nullptr, "RowToColumnNames output table exists");
    Check(promoted_table->num_rows() == 3,
          "RowToColumnNames should remove the promoted header row");
    Check(promoted_table->schema()->GetFieldIndex("x") >= 0,
          "RowToColumnNames should expose promoted x field");
    Check(promoted_table->schema()->GetFieldIndex("y") >= 0,
          "RowToColumnNames should expose promoted y field");

    const std::string bad_row_index_json =
        R"({"nodes":[)"
        R"({"id":51,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":52,"type":"RowToColumnNames","name":"Promote","parameters":{)"
        R"("row_index":"-1"}})"
        R"(],"links":[{"start_node":51,"end_node":52}]})";

    cyxwiz::PipelineExecutor bad_row_index_executor;
    Check(!bad_row_index_executor.ExecutePipeline(bad_row_index_json),
          "RowToColumnNames bad row_index should fail validation");
    Check(bad_row_index_executor.GetLastError().find(
              "RowToColumnNames row_index must be an integer >= 0") !=
              std::string::npos,
          "RowToColumnNames bad row_index validation should be specific: " +
              bad_row_index_executor.GetLastError());

    const std::string table_cropper_json =
        R"({"nodes":[)"
        R"({"id":53,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":54,"type":"TableCropper","name":"Crop","parameters":{)"
        R"("start_row":"1","end_row":"3"}})"
        R"(],"links":[{"start_node":53,"end_node":54}]})";

    cyxwiz::PipelineExecutor table_cropper_executor;
    Check(table_cropper_executor.ExecutePipeline(table_cropper_json),
          "TableCropper should produce a bounded Arrow slice: " +
              table_cropper_executor.GetLastError());
    auto cropped = registry.GetArrowDataset("ds_cropped_54");
    Check(cropped != nullptr, "TableCropper output dataset is registered");
    auto cropped_table = cropped->GetArrowTable();
    Check(cropped_table != nullptr, "TableCropper output table exists");
    Check(cropped_table->num_rows() == 2,
          "TableCropper should crop to the requested row range");

    const std::string bad_crop_range_json =
        R"({"nodes":[)"
        R"({"id":55,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":56,"type":"TableCropper","name":"Crop","parameters":{)"
        R"("start_row":"2","end_row":"1"}})"
        R"(],"links":[{"start_node":55,"end_node":56}]})";

    cyxwiz::PipelineExecutor bad_crop_range_executor;
    Check(!bad_crop_range_executor.ExecutePipeline(bad_crop_range_json),
          "TableCropper invalid row range should fail execution");
    Check(bad_crop_range_executor.GetLastError().find(
              "end_row must be >= start_row") != std::string::npos,
          "TableCropper invalid row range error should be specific: " +
              bad_crop_range_executor.GetLastError());

    const std::string missing_math_formula_json =
        R"({"nodes":[)"
        R"({"id":57,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":58,"type":"MathFormula","name":"Formula","parameters":{)"
        R"("output_column":"sum_xy"}})"
        R"(],"links":[{"start_node":57,"end_node":58}]})";

    cyxwiz::PipelineExecutor missing_math_formula_executor;
    Check(!missing_math_formula_executor.ExecutePipeline(missing_math_formula_json),
          "MathFormula missing formula should fail validation");
    Check(missing_math_formula_executor.GetLastError().find(
              "missing required parameter 'formula'") != std::string::npos,
          "MathFormula missing formula validation should be specific: " +
              missing_math_formula_executor.GetLastError());

    const std::string math_formula_json =
        R"({"nodes":[)"
        R"({"id":59,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":60,"type":"MathFormula","name":"Formula","parameters":{)"
        R"("formula":"x + y","output_column":"sum_xy"}})"
        R"(],"links":[{"start_node":59,"end_node":60}]})";

    cyxwiz::PipelineExecutor math_formula_executor;
    Check(math_formula_executor.ExecutePipeline(math_formula_json),
          "MathFormula should execute when formula is supplied: " +
              math_formula_executor.GetLastError());
    auto math_result = registry.GetArrowDataset("ds_math_60");
    Check(math_result != nullptr, "MathFormula output dataset is registered");
    auto math_table = math_result->GetArrowTable();
    Check(math_table != nullptr, "MathFormula output table exists");
    Check(math_table->schema()->GetFieldIndex("sum_xy") >= 0,
          "MathFormula should expose the computed output column");

    const std::string quoted_math_formula_json =
        R"({"nodes":[)"
        R"({"id":85,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":86,"type":"MathFormula","name":"Formula","parameters":{)"
        R"("formula":"x + y","output_column":"sum\"xy"}})"
        R"(],"links":[{"start_node":85,"end_node":86}]})";

    cyxwiz::PipelineExecutor quoted_math_formula_executor;
    Check(quoted_math_formula_executor.ExecutePipeline(quoted_math_formula_json),
          "MathFormula should quote output column identifiers: " +
              quoted_math_formula_executor.GetLastError());
    auto quoted_math_result = registry.GetArrowDataset("ds_math_86");
    Check(quoted_math_result != nullptr,
          "MathFormula quoted output dataset is registered");
    auto quoted_math_table = quoted_math_result->GetArrowTable();
    Check(quoted_math_table != nullptr, "MathFormula quoted output table exists");
    Check(quoted_math_table->schema()->GetFieldIndex("sum\"xy") >= 0,
          "MathFormula should preserve quoted output column names");
    Check(ReadNumericValue(quoted_math_table, "sum\"xy", 0) == 11.0,
          "MathFormula quoted output column should contain computed values");

    const std::string unknown_formula_column_json =
        R"({"nodes":[)"
        R"({"id":127,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":128,"type":"MathFormula","name":"Formula","parameters":{)"
        R"("formula":"x + missing","output_column":"bad"}})"
        R"(],"links":[{"start_node":127,"end_node":128}]})";

    cyxwiz::PipelineExecutor unknown_formula_column_executor;
    Check(!unknown_formula_column_executor.ExecutePipeline(
              unknown_formula_column_json),
          "MathFormula should reject formulas with unknown columns");
    Check(unknown_formula_column_executor.GetLastError().find(
              "MathFormula: formula references unknown column 'missing'") !=
              std::string::npos,
          "MathFormula unknown column error should be specific: " +
              unknown_formula_column_executor.GetLastError());

    const std::string text_formula_column_json =
        R"({"nodes":[)"
        R"({"id":129,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":130,"type":"MathFormula","name":"Formula","parameters":{)"
        R"("formula":"phrase + 1","output_column":"bad"}})"
        R"(],"links":[{"start_node":129,"end_node":130}]})";

    cyxwiz::PipelineExecutor text_formula_column_executor;
    Check(!text_formula_column_executor.ExecutePipeline(text_formula_column_json),
          "MathFormula should reject formulas over text columns");
    Check(text_formula_column_executor.GetLastError().find(
              "MathFormula: formula column 'phrase' must be numeric") !=
              std::string::npos,
          "MathFormula text column error should be specific: " +
              text_formula_column_executor.GetLastError());

    const std::string raw_sql_formula_json =
        R"({"nodes":[)"
        R"({"id":131,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":132,"type":"MathFormula","name":"Formula","parameters":{)"
        R"("formula":"x + y; DROP TABLE temp","output_column":"bad"}})"
        R"(],"links":[{"start_node":131,"end_node":132}]})";

    cyxwiz::PipelineExecutor raw_sql_formula_executor;
    Check(!raw_sql_formula_executor.ExecutePipeline(raw_sql_formula_json),
          "MathFormula should reject raw SQL formula tokens");
    Check(raw_sql_formula_executor.GetLastError().find(
              "MathFormula: formula contains unsupported token ';'") !=
              std::string::npos,
          "MathFormula raw SQL token error should be specific: " +
              raw_sql_formula_executor.GetLastError());

    const std::string rule_engine_json =
        R"({"nodes":[)"
        R"({"id":61,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":62,"type":"RuleEngine","name":"Rules","parameters":{)"
        R"("rules":"x > 1 => 'high'","default_value":"'low'","output_column":"bucket"}})"
        R"(],"links":[{"start_node":61,"end_node":62}]})";

    cyxwiz::PipelineExecutor rule_engine_executor;
    Check(!rule_engine_executor.ExecutePipeline(rule_engine_json),
          "RuleEngine placeholder should fail closed");
    Check(rule_engine_executor.GetLastError().find(
              "legacy RuleEngine execution ignores rules") != std::string::npos,
          "RuleEngine should use fail-closed runtime support: " +
              rule_engine_executor.GetLastError());

    const std::string fill_missing_mean_json =
        R"({"nodes":[)"
        R"({"id":63,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(missing_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":64,"type":"FillMissing","name":"Fill","parameters":{)"
        R"("strategy":"MEAN"}})"
        R"(],"links":[{"start_node":63,"end_node":64}]})";

    cyxwiz::PipelineExecutor fill_missing_mean_executor;
    Check(fill_missing_mean_executor.ExecutePipeline(fill_missing_mean_json),
          "FillMissing mean should use column statistics: " +
              fill_missing_mean_executor.GetLastError());
    auto filled = registry.GetArrowDataset("ds_fillmissing_64");
    Check(filled != nullptr, "FillMissing output dataset is registered");
    auto filled_table = filled->GetArrowTable();
    Check(filled_table != nullptr, "FillMissing output table exists");
    Check(std::fabs(ReadNumericValue(filled_table, "x", 1) - 2.0) < 0.001,
          "FillMissing mean should fill x with column mean");
    Check(std::fabs(ReadNumericValue(filled_table, "y", 2) - 15.0) < 0.001,
          "FillMissing mean should fill y with column mean");

    const std::string fill_missing_mean_string_json =
        R"({"nodes":[)"
        R"({"id":371,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" +
        JsonEscapePath(missing_string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":372,"type":"FillMissing","name":"BadMeanFill","parameters":{)"
        R"("strategy":"mean"}})"
        R"(],"links":[{"start_node":371,"end_node":372}]})";

    cyxwiz::PipelineExecutor fill_missing_mean_string_executor;
    Check(!fill_missing_mean_string_executor.ExecutePipeline(
              fill_missing_mean_string_json),
          "FillMissing mean on string columns should fail before partial no-op");
    Check(fill_missing_mean_string_executor.GetLastError().find(
              "FillMissing: strategy 'mean' requires numeric column 'phrase'") !=
              std::string::npos,
          "FillMissing mean string column error should be specific: " +
              fill_missing_mean_string_executor.GetLastError());

    const std::string fill_missing_string_constant_json =
        R"({"nodes":[)"
        R"({"id":184,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" +
        JsonEscapePath(missing_string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":185,"type":"FillMissing","name":"FillString","parameters":{)"
        R"("strategy":"constant","value":"owner's choice"}})"
        R"(],"links":[{"start_node":184,"end_node":185}]})";

    cyxwiz::PipelineExecutor fill_missing_string_constant_executor;
    Check(fill_missing_string_constant_executor.ExecutePipeline(
              fill_missing_string_constant_json),
          "FillMissing constant should quote string fill values: " +
              fill_missing_string_constant_executor.GetLastError());
    auto string_filled = registry.GetArrowDataset("ds_fillmissing_185");
    Check(string_filled != nullptr,
          "FillMissing string constant output dataset is registered");
    auto string_filled_table = string_filled->GetArrowTable();
    Check(string_filled_table != nullptr,
          "FillMissing string constant output table exists");
    Check(ReadStringValue(string_filled_table, "phrase", 0) == "tea cup",
          "FillMissing string constant should preserve existing string values");

    const std::string fill_missing_bad_numeric_constant_json =
        R"({"nodes":[)"
        R"({"id":186,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" +
        JsonEscapePath(missing_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":187,"type":"FillMissing","name":"BadFill","parameters":{)"
        R"("strategy":"constant","value":"not-a-number"}})"
        R"(],"links":[{"start_node":186,"end_node":187}]})";

    cyxwiz::PipelineExecutor fill_missing_bad_numeric_constant_executor;
    Check(!fill_missing_bad_numeric_constant_executor.ExecutePipeline(
              fill_missing_bad_numeric_constant_json),
          "FillMissing bad numeric constant should fail before SQL execution");
    Check(fill_missing_bad_numeric_constant_executor.GetLastError().find(
              "FillMissing: constant value 'not-a-number' is not numeric for column 'x'") !=
              std::string::npos,
          "FillMissing bad numeric constant error should be specific: " +
              fill_missing_bad_numeric_constant_executor.GetLastError());

    const std::string string_replace_json =
        R"({"nodes":[)"
        R"({"id":65,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":66,"type":"StringManipulation","name":"Replace","parameters":{)"
        R"("column":"phrase","operation":"REPLACE","param1":"tea","param2":"coffee"}})"
        R"(],"links":[{"start_node":65,"end_node":66}]})";

    cyxwiz::PipelineExecutor string_replace_executor;
    Check(string_replace_executor.ExecutePipeline(string_replace_json),
          "StringManipulation replace should execute real replacement: " +
              string_replace_executor.GetLastError());
    auto replaced = registry.GetArrowDataset("ds_string_66");
    Check(replaced != nullptr, "StringManipulation replace output dataset is registered");
    auto replaced_table = replaced->GetArrowTable();
    Check(replaced_table != nullptr, "StringManipulation replace output table exists");
    Check(ReadStringValue(replaced_table, "phrase_modified", 0) == "coffee cup",
          "StringManipulation replace should change matching text");

    const std::string string_substring_json =
        R"({"nodes":[)"
        R"({"id":67,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":68,"type":"StringManipulation","name":"Substring","parameters":{)"
        R"("column":"phrase","operation":"substring","param1":"1","param2":"3"}})"
        R"(],"links":[{"start_node":67,"end_node":68}]})";

    cyxwiz::PipelineExecutor string_substring_executor;
    Check(string_substring_executor.ExecutePipeline(string_substring_json),
          "StringManipulation substring should execute real substring: " +
              string_substring_executor.GetLastError());
    auto substring = registry.GetArrowDataset("ds_string_68");
    Check(substring != nullptr, "StringManipulation substring output dataset is registered");
    auto substring_table = substring->GetArrowTable();
    Check(substring_table != nullptr, "StringManipulation substring output table exists");
    Check(ReadStringValue(substring_table, "phrase_modified", 0) == "tea",
          "StringManipulation substring should use param1 start and param2 length");

    const std::string bad_string_operation_json =
        R"({"nodes":[)"
        R"({"id":69,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":70,"type":"StringManipulation","name":"BadStringOp","parameters":{)"
        R"("column":"phrase","operation":"reverse"}})"
        R"(],"links":[{"start_node":69,"end_node":70}]})";

    cyxwiz::PipelineExecutor bad_string_operation_executor;
    Check(!bad_string_operation_executor.ExecutePipeline(bad_string_operation_json),
          "StringManipulation unknown operation should fail validation");
    Check(bad_string_operation_executor.GetLastError().find(
              "StringManipulation operation 'reverse' is not supported") !=
              std::string::npos,
          "StringManipulation unknown operation error should be specific: " +
              bad_string_operation_executor.GetLastError());

    const std::string uppercase_replace_missing_param_json =
        R"({"nodes":[)"
        R"({"id":331,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":332,"type":"StringManipulation","name":"BadReplace","parameters":{)"
        R"("column":"phrase","operation":"REPLACE"}})"
        R"(],"links":[{"start_node":331,"end_node":332}]})";

    cyxwiz::PipelineExecutor uppercase_replace_missing_param_executor;
    Check(!uppercase_replace_missing_param_executor.ExecutePipeline(
              uppercase_replace_missing_param_json),
          "StringManipulation uppercase replace without param1 should fail validation");
    Check(uppercase_replace_missing_param_executor.GetLastError().find(
              "StringManipulation replace requires param1") != std::string::npos,
          "StringManipulation uppercase replace missing param1 error should be "
          "specific: " +
              uppercase_replace_missing_param_executor.GetLastError());

    const std::string numeric_string_manipulation_json =
        R"({"nodes":[)"
        R"({"id":87,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":88,"type":"StringManipulation","name":"BadStringColumn","parameters":{)"
        R"("column":"x","operation":"lower"}})"
        R"(],"links":[{"start_node":87,"end_node":88}]})";

    cyxwiz::PipelineExecutor numeric_string_manipulation_executor;
    Check(!numeric_string_manipulation_executor.ExecutePipeline(
              numeric_string_manipulation_json),
          "StringManipulation on numeric column should fail schema validation");
    Check(numeric_string_manipulation_executor.GetLastError().find(
              "StringManipulation: column 'x' must be string") !=
              std::string::npos,
          "StringManipulation numeric column error should be specific: " +
              numeric_string_manipulation_executor.GetLastError());

    const std::string text_clean_json =
        R"({"nodes":[)"
        R"({"id":152,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":153,"type":"TextClean","name":"Clean","parameters":{)"
        R"("text_column":"phrase","lowercase":"TRUE"}})"
        R"(],"links":[{"start_node":152,"end_node":153}]})";

    cyxwiz::PipelineExecutor text_clean_executor;
    Check(text_clean_executor.ExecutePipeline(text_clean_json),
          "TextClean should validate and quote text column: " +
              text_clean_executor.GetLastError());
    auto text_clean = registry.GetArrowDataset("ds_textclean_153");
    Check(text_clean != nullptr, "TextClean output dataset is registered");
    auto text_clean_table = text_clean->GetArrowTable();
    Check(text_clean_table != nullptr, "TextClean output table exists");
    Check(ReadStringValue(text_clean_table, "phrase_cleaned", 0) == "tea cup",
          "TextClean should write cleaned output column");

    const std::string text_clean_stopwords_json =
        R"({"nodes":[)"
        R"({"id":166,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":167,"type":"TextClean","name":"UnsupportedStopwords","parameters":{)"
        R"("text_column":"phrase","remove_stopwords":"TRUE"}})"
        R"(],"links":[{"start_node":166,"end_node":167}]})";

    cyxwiz::PipelineExecutor text_clean_stopwords_executor;
    Check(!text_clean_stopwords_executor.ExecutePipeline(
              text_clean_stopwords_json),
          "TextClean remove_stopwords should fail closed");
    Check(text_clean_stopwords_executor.GetLastError().find(
              "TextClean remove_stopwords is not supported") !=
              std::string::npos,
          "TextClean remove_stopwords error should be specific: " +
              text_clean_stopwords_executor.GetLastError());

    const std::string text_clean_bad_boolean_json =
        R"({"nodes":[)"
        R"({"id":329,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":330,"type":"TextClean","name":"BadCleanFlag","parameters":{)"
        R"("text_column":"phrase","lowercase":"maybe"}})"
        R"(],"links":[{"start_node":329,"end_node":330}]})";

    cyxwiz::PipelineExecutor text_clean_bad_boolean_executor;
    Check(!text_clean_bad_boolean_executor.ExecutePipeline(
              text_clean_bad_boolean_json),
          "TextClean malformed lowercase should fail validation");
    Check(text_clean_bad_boolean_executor.GetLastError().find(
              "TextClean: 'lowercase' must be 'true' or 'false'") !=
              std::string::npos,
          "TextClean lowercase validation should be specific: " +
              text_clean_bad_boolean_executor.GetLastError());

    const std::string text_tokenize_json =
        R"({"nodes":[)"
        R"({"id":154,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":155,"type":"TextTokenize","name":"Tokenize","parameters":{)"
        R"("text_column":"phrase","method":"WORD"}})"
        R"(],"links":[{"start_node":154,"end_node":155}]})";

    cyxwiz::PipelineExecutor text_tokenize_executor;
    Check(text_tokenize_executor.ExecutePipeline(text_tokenize_json),
          "TextTokenize should validate and quote text column: " +
              text_tokenize_executor.GetLastError());
    auto text_tokenized = registry.GetArrowDataset("ds_texttokenize_155");
    Check(text_tokenized != nullptr, "TextTokenize output dataset is registered");
    auto text_tokenized_table = text_tokenized->GetArrowTable();
    Check(text_tokenized_table != nullptr, "TextTokenize output table exists");
    Check(text_tokenized_table->schema()->GetFieldIndex("phrase_tokens") >= 0,
          "TextTokenize should write token output column");

    const std::string text_vectorize_json =
        R"({"nodes":[)"
        R"({"id":156,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":157,"type":"TextVectorize","name":"Vectorize","parameters":{)"
        R"("text_column":"phrase","method":"COUNT"}})"
        R"(],"links":[{"start_node":156,"end_node":157}]})";

    cyxwiz::PipelineExecutor text_vectorize_executor;
    Check(text_vectorize_executor.ExecutePipeline(text_vectorize_json),
          "TextVectorize should validate and quote text column: " +
              text_vectorize_executor.GetLastError());
    auto text_vectorized = registry.GetArrowDataset("ds_textvectorize_157");
    Check(text_vectorized != nullptr, "TextVectorize output dataset is registered");
    auto text_vectorized_table = text_vectorized->GetArrowTable();
    Check(text_vectorized_table != nullptr, "TextVectorize output table exists");
    Check(ReadNumericValue(text_vectorized_table, "text_length", 0) == 7.0,
          "TextVectorize should compute text length");
    Check(ReadNumericValue(text_vectorized_table, "word_count", 0) == 2.0,
          "TextVectorize should compute word count");

    const std::string text_clean_numeric_column_json =
        R"({"nodes":[)"
        R"({"id":158,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":159,"type":"TextClean","name":"BadClean","parameters":{)"
        R"("text_column":"x","lowercase":"true"}})"
        R"(],"links":[{"start_node":158,"end_node":159}]})";

    cyxwiz::PipelineExecutor text_clean_numeric_column_executor;
    Check(!text_clean_numeric_column_executor.ExecutePipeline(
              text_clean_numeric_column_json),
          "TextClean on numeric column should fail schema validation");
    Check(text_clean_numeric_column_executor.GetLastError().find(
              "TextClean: column 'x' must be string") != std::string::npos,
          "TextClean numeric column error should be specific: " +
              text_clean_numeric_column_executor.GetLastError());

    const std::string text_vectorize_missing_column_json =
        R"({"nodes":[)"
        R"({"id":160,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":161,"type":"TextVectorize","name":"MissingVectorize","parameters":{)"
        R"("text_column":"missing","method":"count"}})"
        R"(],"links":[{"start_node":160,"end_node":161}]})";

    cyxwiz::PipelineExecutor text_vectorize_missing_column_executor;
    Check(!text_vectorize_missing_column_executor.ExecutePipeline(
              text_vectorize_missing_column_json),
          "TextVectorize missing text column should fail schema validation");
    Check(text_vectorize_missing_column_executor.GetLastError().find(
              "TextVectorize: column 'missing' not found") != std::string::npos,
          "TextVectorize missing column error should be specific: " +
              text_vectorize_missing_column_executor.GetLastError());

    const std::string bad_text_tokenize_method_json =
        R"({"nodes":[)"
        R"({"id":162,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":163,"type":"TextTokenize","name":"BadTokenizeMethod","parameters":{)"
        R"("text_column":"phrase","method":"ngram"}})"
        R"(],"links":[{"start_node":162,"end_node":163}]})";

    cyxwiz::PipelineExecutor bad_text_tokenize_method_executor;
    Check(!bad_text_tokenize_method_executor.ExecutePipeline(
              bad_text_tokenize_method_json),
          "TextTokenize unsupported method should fail validation");
    Check(bad_text_tokenize_method_executor.GetLastError().find(
              "TextTokenize method 'ngram' is not supported") !=
              std::string::npos,
          "TextTokenize unsupported method error should be specific: " +
              bad_text_tokenize_method_executor.GetLastError());

    const std::string bad_text_vectorize_method_json =
        R"({"nodes":[)"
        R"({"id":164,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":165,"type":"TextVectorize","name":"BadVectorizeMethod","parameters":{)"
        R"("text_column":"phrase","method":"tfidf"}})"
        R"(],"links":[{"start_node":164,"end_node":165}]})";

    cyxwiz::PipelineExecutor bad_text_vectorize_method_executor;
    Check(!bad_text_vectorize_method_executor.ExecutePipeline(
              bad_text_vectorize_method_json),
          "TextVectorize unsupported method should fail validation");
    Check(bad_text_vectorize_method_executor.GetLastError().find(
              "TextVectorize method 'tfidf' is not supported") !=
              std::string::npos,
          "TextVectorize unsupported method error should be specific: " +
              bad_text_vectorize_method_executor.GetLastError());

    const std::string bad_text_vectorize_max_features_json =
        R"({"nodes":[)"
        R"({"id":213,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":214,"type":"TextVectorize","name":"IgnoredMaxFeatures","parameters":{)"
        R"("text_column":"phrase","method":"count","max_features":"1000"}})"
        R"(],"links":[{"start_node":213,"end_node":214}]})";

    cyxwiz::PipelineExecutor bad_text_vectorize_max_features_executor;
    Check(!bad_text_vectorize_max_features_executor.ExecutePipeline(
              bad_text_vectorize_max_features_json),
          "TextVectorize max_features should fail closed on the legacy path");
    Check(bad_text_vectorize_max_features_executor.GetLastError().find(
              "TextVectorize max_features is not supported") !=
              std::string::npos,
          "TextVectorize max_features error should be specific: " +
              bad_text_vectorize_max_features_executor.GetLastError());

    const std::string count_vectorizer_binary_json =
        R"({"nodes":[)"
        R"({"id":215,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":216,"type":"CountVectorizer","name":"BadCountVectorizer","parameters":{)"
        R"("text_col":"phrase","binary":"true"}})"
        R"(],"links":[{"start_node":215,"end_node":216}]})";

    cyxwiz::PipelineExecutor count_vectorizer_binary_executor;
    Check(!count_vectorizer_binary_executor.ExecutePipeline(
              count_vectorizer_binary_json),
          "CountVectorizer binary=true should fail closed");
    Check(count_vectorizer_binary_executor.GetLastError().find(
              "CountVectorizer: binary counts are not supported") !=
              std::string::npos,
          "CountVectorizer binary error should be specific: " +
              count_vectorizer_binary_executor.GetLastError());

    const std::string count_vectorizer_bad_binary_json =
        R"({"nodes":[)"
        R"({"id":219,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":220,"type":"CountVectorizer","name":"MalformedBinary","parameters":{)"
        R"("text_col":"phrase","binary":"maybe"}})"
        R"(],"links":[{"start_node":219,"end_node":220}]})";

    cyxwiz::PipelineExecutor count_vectorizer_bad_binary_executor;
    Check(!count_vectorizer_bad_binary_executor.ExecutePipeline(
              count_vectorizer_bad_binary_json),
          "CountVectorizer malformed binary should fail validation");
    Check(count_vectorizer_bad_binary_executor.GetLastError().find(
              "CountVectorizer: 'binary' must be 'true' or 'false'") !=
              std::string::npos,
          "CountVectorizer malformed binary error should be specific: " +
              count_vectorizer_bad_binary_executor.GetLastError());

    const std::string tfidf_vectorizer_min_df_json =
        R"({"nodes":[)"
        R"({"id":217,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":218,"type":"TFIDFVectorizer","name":"BadTfidfVectorizer","parameters":{)"
        R"("text_col":"phrase","min_df":"2"}})"
        R"(],"links":[{"start_node":217,"end_node":218}]})";

    cyxwiz::PipelineExecutor tfidf_vectorizer_min_df_executor;
    Check(!tfidf_vectorizer_min_df_executor.ExecutePipeline(
              tfidf_vectorizer_min_df_json),
          "TFIDFVectorizer min_df should fail closed when unsupported");
    Check(tfidf_vectorizer_min_df_executor.GetLastError().find(
              "TFIDFVectorizer: min_df values other than 1 are not supported") !=
              std::string::npos,
          "TFIDFVectorizer min_df error should be specific: " +
              tfidf_vectorizer_min_df_executor.GetLastError());

    const std::string tfidf_vectorizer_bad_bool_json =
        R"({"nodes":[)"
        R"({"id":221,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":222,"type":"TFIDFVectorizer","name":"BadTfidfBool","parameters":{)"
        R"("text_col":"phrase","use_idf":"maybe"}})"
        R"(],"links":[{"start_node":221,"end_node":222}]})";

    cyxwiz::PipelineExecutor tfidf_vectorizer_bad_bool_executor;
    Check(!tfidf_vectorizer_bad_bool_executor.ExecutePipeline(
              tfidf_vectorizer_bad_bool_json),
          "TFIDFVectorizer malformed boolean should fail validation");
    Check(tfidf_vectorizer_bad_bool_executor.GetLastError().find(
              "TFIDFVectorizer: 'use_idf' must be 'true' or 'false'") !=
              std::string::npos,
          "TFIDFVectorizer malformed boolean error should be specific: " +
              tfidf_vectorizer_bad_bool_executor.GetLastError());

    const std::string binning_json =
        R"({"nodes":[)"
        R"({"id":71,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":72,"type":"Binning","name":"Bin","parameters":{)"
        R"("columns":"x","method":"equal_width","n_bins":"2"}})"
        R"(],"links":[{"start_node":71,"end_node":72}]})";

    cyxwiz::PipelineExecutor binning_executor;
    Check(binning_executor.ExecutePipeline(binning_json),
          "Binning equal_width should execute real bins: " +
              binning_executor.GetLastError());
    auto binned = registry.GetArrowDataset("ds_binning_72");
    Check(binned != nullptr, "Binning output dataset is registered");
    auto binned_table = binned->GetArrowTable();
    Check(binned_table != nullptr, "Binning output table exists");
    Check(binned_table->num_columns() == 3,
          "Binning output should only add the requested bin column");
    Check(ReadNumericValue(binned_table, "x_bin", 0) == 1.0,
          "Binning equal_width should place minimum in first bin");
    Check(ReadNumericValue(binned_table, "x_bin", 2) == 2.0,
          "Binning equal_width should place maximum in last bin");

    const std::string binning_equal_frequency_alias_json =
        R"({"nodes":[)"
        R"({"id":197,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":198,"type":"Binning","name":"BinAlias","parameters":{)"
        R"("columns":"x","method":"equal_frequency","n_bins":"2"}})"
        R"(],"links":[{"start_node":197,"end_node":198}]})";

    cyxwiz::PipelineExecutor binning_equal_frequency_alias_executor;
    Check(binning_equal_frequency_alias_executor.ExecutePipeline(
              binning_equal_frequency_alias_json),
          "Binning equal_frequency alias should execute equal-frequency bins: " +
              binning_equal_frequency_alias_executor.GetLastError());
    auto alias_binned = registry.GetArrowDataset("ds_binning_198");
    Check(alias_binned != nullptr,
          "Binning equal_frequency alias output dataset is registered");
    auto alias_binned_table = alias_binned->GetArrowTable();
    Check(alias_binned_table != nullptr,
          "Binning equal_frequency alias output table exists");
    Check(alias_binned_table->num_columns() == 3,
          "Binning equal_frequency alias should add one bin column");
    Check(ReadNumericValue(alias_binned_table, "x_bin", 0) == 1.0,
          "Binning equal_frequency alias should route to equal_freq bins");

    const std::string missing_binning_column_json =
        R"({"nodes":[)"
        R"({"id":73,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":74,"type":"Binning","name":"MissingBinColumn","parameters":{)"
        R"("method":"equal_width","n_bins":"2"}})"
        R"(],"links":[{"start_node":73,"end_node":74}]})";

    cyxwiz::PipelineExecutor missing_binning_column_executor;
    Check(!missing_binning_column_executor.ExecutePipeline(missing_binning_column_json),
          "Binning missing columns should fail validation");
    Check(missing_binning_column_executor.GetLastError().find(
              "missing required parameter 'columns'") != std::string::npos,
          "Binning missing columns validation should be specific: " +
              missing_binning_column_executor.GetLastError());

    const std::string bad_binning_method_json =
        R"({"nodes":[)"
        R"({"id":75,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":76,"type":"Binning","name":"BadBinMethod","parameters":{)"
        R"("columns":"x","method":"quantile","n_bins":"2"}})"
        R"(],"links":[{"start_node":75,"end_node":76}]})";

    cyxwiz::PipelineExecutor bad_binning_method_executor;
    Check(!bad_binning_method_executor.ExecutePipeline(bad_binning_method_json),
          "Binning unknown method should fail validation");
    Check(bad_binning_method_executor.GetLastError().find(
              "Binning method 'quantile' is not supported") != std::string::npos,
          "Binning unknown method validation should be specific: " +
              bad_binning_method_executor.GetLastError());

    const std::string text_binning_json =
        R"({"nodes":[)"
        R"({"id":89,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":90,"type":"Binning","name":"TextBin","parameters":{)"
        R"("columns":"phrase","method":"equal_width","n_bins":"2"}})"
        R"(],"links":[{"start_node":89,"end_node":90}]})";

    cyxwiz::PipelineExecutor text_binning_executor;
    Check(!text_binning_executor.ExecutePipeline(text_binning_json),
          "Binning on text column should fail schema validation");
    Check(text_binning_executor.GetLastError().find(
              "Binning: column 'phrase' must be numeric") != std::string::npos,
          "Binning text column error should be specific: " +
              text_binning_executor.GetLastError());

    const std::string polynomial_json =
        R"({"nodes":[)"
        R"({"id":77,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":78,"type":"PolynomialFeatures","name":"Poly","parameters":{)"
        R"("columns":"x","degree":"3"}})"
        R"(],"links":[{"start_node":77,"end_node":78}]})";

    cyxwiz::PipelineExecutor polynomial_executor;
    Check(polynomial_executor.ExecutePipeline(polynomial_json),
          "PolynomialFeatures should generate requested powers: " +
              polynomial_executor.GetLastError());
    auto polynomial = registry.GetArrowDataset("ds_poly_78");
    Check(polynomial != nullptr, "PolynomialFeatures output dataset is registered");
    auto polynomial_table = polynomial->GetArrowTable();
    Check(polynomial_table != nullptr, "PolynomialFeatures output table exists");
    Check(polynomial_table->num_columns() == 4,
          "PolynomialFeatures degree 3 should add squared and cubed columns");
    Check(ReadNumericValue(polynomial_table, "x_squared", 2) == 9.0,
          "PolynomialFeatures should compute squared values");
    Check(ReadNumericValue(polynomial_table, "x_cubed", 2) == 27.0,
          "PolynomialFeatures should compute cubed values");

    const std::string bad_polynomial_degree_json =
        R"({"nodes":[)"
        R"({"id":79,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":80,"type":"PolynomialFeatures","name":"BadPolyDegree","parameters":{)"
        R"("columns":"x","degree":"1"}})"
        R"(],"links":[{"start_node":79,"end_node":80}]})";

    cyxwiz::PipelineExecutor bad_polynomial_degree_executor;
    Check(!bad_polynomial_degree_executor.ExecutePipeline(bad_polynomial_degree_json),
          "PolynomialFeatures degree 1 should fail validation");
    Check(bad_polynomial_degree_executor.GetLastError().find(
              "PolynomialFeatures degree must be an integer >= 2") !=
              std::string::npos,
          "PolynomialFeatures degree validation should be specific: " +
              bad_polynomial_degree_executor.GetLastError());

    const std::string multi_column_polynomial_json =
        R"({"nodes":[)"
        R"({"id":81,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":82,"type":"PolynomialFeatures","name":"MultiPoly","parameters":{)"
        R"("columns":"x,y","degree":"2"}})"
        R"(],"links":[{"start_node":81,"end_node":82}]})";

    cyxwiz::PipelineExecutor multi_column_polynomial_executor;
    Check(!multi_column_polynomial_executor.ExecutePipeline(multi_column_polynomial_json),
          "PolynomialFeatures comma-separated columns should fail validation");
    Check(multi_column_polynomial_executor.GetLastError().find(
              "PolynomialFeatures columns supports exactly one column") !=
              std::string::npos,
          "PolynomialFeatures multi-column validation should be specific: " +
              multi_column_polynomial_executor.GetLastError());

    const std::string missing_polynomial_column_json =
        R"({"nodes":[)"
        R"({"id":91,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":92,"type":"PolynomialFeatures","name":"MissingPolyColumn","parameters":{)"
        R"("columns":"missing","degree":"2"}})"
        R"(],"links":[{"start_node":91,"end_node":92}]})";

    cyxwiz::PipelineExecutor missing_polynomial_column_executor;
    Check(!missing_polynomial_column_executor.ExecutePipeline(
              missing_polynomial_column_json),
          "PolynomialFeatures missing input column should fail schema validation");
    Check(missing_polynomial_column_executor.GetLastError().find(
              "PolynomialFeatures: column 'missing' not found") !=
              std::string::npos,
          "PolynomialFeatures missing column error should be specific: " +
              missing_polynomial_column_executor.GetLastError());

    const std::string remove_duplicates_columns_json =
        R"({"nodes":[)"
        R"({"id":180,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" +
        JsonEscapePath(duplicates_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":181,"type":"RemoveDuplicates","name":"Dedup","parameters":{)"
        R"("columns":" x "}})"
        R"(],"links":[{"start_node":180,"end_node":181}]})";

    cyxwiz::PipelineExecutor remove_duplicates_columns_executor;
    Check(remove_duplicates_columns_executor.ExecutePipeline(
              remove_duplicates_columns_json),
          "RemoveDuplicates should validate and quote selected dedupe columns: " +
              remove_duplicates_columns_executor.GetLastError());
    auto deduped = registry.GetArrowDataset("ds_dedup_181");
    Check(deduped != nullptr, "RemoveDuplicates column output dataset is registered");
    auto deduped_table = deduped->GetArrowTable();
    Check(deduped_table != nullptr, "RemoveDuplicates column output table exists");
    Check(deduped_table->num_rows() == 2,
          "RemoveDuplicates columns should keep one row per x value");
    Check(deduped_table->schema()->GetFieldIndex("x") >= 0 &&
              deduped_table->schema()->GetFieldIndex("y") >= 0,
          "RemoveDuplicates columns should preserve the original schema");

    const std::string missing_remove_duplicates_column_json =
        R"({"nodes":[)"
        R"({"id":182,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" +
        JsonEscapePath(duplicates_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":183,"type":"RemoveDuplicates","name":"DedupMissing","parameters":{)"
        R"("columns":"missing"}})"
        R"(],"links":[{"start_node":182,"end_node":183}]})";

    cyxwiz::PipelineExecutor missing_remove_duplicates_column_executor;
    Check(!missing_remove_duplicates_column_executor.ExecutePipeline(
              missing_remove_duplicates_column_json),
          "RemoveDuplicates missing input column should fail schema validation");
    Check(missing_remove_duplicates_column_executor.GetLastError().find(
              "RemoveDuplicates: column 'missing' not found") !=
              std::string::npos,
          "RemoveDuplicates missing column error should be specific: " +
              missing_remove_duplicates_column_executor.GetLastError());

    const std::string select_columns_json =
        R"({"nodes":[)"
        R"({"id":93,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":94,"type":"SelectColumns","name":"Select","parameters":{)"
        R"("columns":" y , x "}})"
        R"(],"links":[{"start_node":93,"end_node":94}]})";

    cyxwiz::PipelineExecutor select_columns_executor;
    Check(select_columns_executor.ExecutePipeline(select_columns_json),
          "SelectColumns should validate and quote requested columns: " +
              select_columns_executor.GetLastError());
    auto selected = registry.GetArrowDataset("ds_select_94");
    Check(selected != nullptr, "SelectColumns output dataset is registered");
    auto selected_table = selected->GetArrowTable();
    Check(selected_table != nullptr, "SelectColumns output table exists");
    Check(selected_table->num_columns() == 2,
          "SelectColumns should keep only requested columns");
    Check(selected_table->schema()->field(0)->name() == "y",
          "SelectColumns should preserve requested order");
    Check(selected_table->schema()->field(1)->name() == "x",
          "SelectColumns should trim requested column names");

    const std::string missing_select_column_json =
        R"({"nodes":[)"
        R"({"id":95,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":96,"type":"SelectColumns","name":"SelectMissing","parameters":{)"
        R"("columns":"x,missing"}})"
        R"(],"links":[{"start_node":95,"end_node":96}]})";

    cyxwiz::PipelineExecutor missing_select_column_executor;
    Check(!missing_select_column_executor.ExecutePipeline(
              missing_select_column_json),
          "SelectColumns missing input column should fail schema validation");
    Check(missing_select_column_executor.GetLastError().find(
              "SelectColumns: column 'missing' not found") !=
              std::string::npos,
          "SelectColumns missing column error should be specific: " +
              missing_select_column_executor.GetLastError());

    const std::string sort_rows_json =
        R"({"nodes":[)"
        R"({"id":97,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":98,"type":"SortRows","name":"Sort","parameters":{)"
        R"("columns":" y ","order":"DESC"}})"
        R"(],"links":[{"start_node":97,"end_node":98}]})";

    cyxwiz::PipelineExecutor sort_rows_executor;
    Check(sort_rows_executor.ExecutePipeline(sort_rows_json),
          "SortRows should validate and quote requested columns: " +
              sort_rows_executor.GetLastError());
    auto sorted = registry.GetArrowDataset("ds_sort_98");
    Check(sorted != nullptr, "SortRows output dataset is registered");
    auto sorted_table = sorted->GetArrowTable();
    Check(sorted_table != nullptr, "SortRows output table exists");
    Check(ReadNumericValue(sorted_table, "y", 0) == 30.0,
          "SortRows should apply requested descending order");

    const std::string missing_sort_column_json =
        R"({"nodes":[)"
        R"({"id":99,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":100,"type":"SortRows","name":"SortMissing","parameters":{)"
        R"("columns":"missing","order":"ASC"}})"
        R"(],"links":[{"start_node":99,"end_node":100}]})";

    cyxwiz::PipelineExecutor missing_sort_column_executor;
    Check(!missing_sort_column_executor.ExecutePipeline(
              missing_sort_column_json),
          "SortRows missing input column should fail schema validation");
    Check(missing_sort_column_executor.GetLastError().find(
              "SortRows: column 'missing' not found") != std::string::npos,
          "SortRows missing column error should be specific: " +
              missing_sort_column_executor.GetLastError());

    const std::string sort_ascending_false_json =
        R"({"nodes":[)"
        R"({"id":114,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":115,"type":"SortRows","name":"SortLegacyAscending","parameters":{)"
        R"("columns":" y ","ascending":"false"}})"
        R"(],"links":[{"start_node":114,"end_node":115}]})";

    cyxwiz::PipelineExecutor sort_ascending_false_executor;
    Check(sort_ascending_false_executor.ExecutePipeline(sort_ascending_false_json),
          "SortRows should honor legacy ascending=false: " +
              sort_ascending_false_executor.GetLastError());
    auto sorted_desc = registry.GetArrowDataset("ds_sort_115");
    Check(sorted_desc != nullptr, "SortRows ascending=false output dataset is registered");
    auto sorted_desc_table = sorted_desc->GetArrowTable();
    Check(sorted_desc_table != nullptr, "SortRows ascending=false output table exists");
    Check(ReadNumericValue(sorted_desc_table, "y", 0) == 30.0,
          "SortRows ascending=false should sort descending");

    const std::string bad_sort_order_json =
        R"({"nodes":[)"
        R"({"id":116,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":117,"type":"SortRows","name":"SortBadOrder","parameters":{)"
        R"("columns":"y","order":"SIDEWAYS"}})"
        R"(],"links":[{"start_node":116,"end_node":117}]})";

    cyxwiz::PipelineExecutor bad_sort_order_executor;
    Check(!bad_sort_order_executor.ExecutePipeline(bad_sort_order_json),
          "SortRows unsupported order should fail validation");
    Check(bad_sort_order_executor.GetLastError().find(
              "SortRows order 'SIDEWAYS' is not supported") != std::string::npos,
          "SortRows unsupported order error should be specific: " +
              bad_sort_order_executor.GetLastError());

    const std::string group_by_json =
        R"({"nodes":[)"
        R"({"id":107,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":108,"type":"GroupBy","name":"Group","parameters":{)"
        R"("group_columns":" x ","aggregations":"COUNT(*) AS count_rows"}})"
        R"(],"links":[{"start_node":107,"end_node":108}]})";

    cyxwiz::PipelineExecutor group_by_executor;
    Check(group_by_executor.ExecutePipeline(group_by_json),
          "GroupBy should validate and quote group columns: " +
              group_by_executor.GetLastError());
    auto grouped = registry.GetArrowDataset("ds_groupby_108");
    Check(grouped != nullptr, "GroupBy output dataset is registered");
    auto grouped_table = grouped->GetArrowTable();
    Check(grouped_table != nullptr, "GroupBy output table exists");
    Check(grouped_table->num_rows() == 3,
          "GroupBy x should produce one group per x fixture value");
    Check(grouped_table->schema()->GetFieldIndex("count_rows") >= 0,
          "GroupBy should preserve aggregation alias");

    const std::string group_by_multi_agg_json =
        R"({"nodes":[)"
        R"({"id":118,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":119,"type":"GroupBy","name":"GroupMulti","parameters":{)"
        R"("group_columns":"x","aggregations":"COUNT(*) AS count_rows, SUM(y) AS total_y"}})"
        R"(],"links":[{"start_node":118,"end_node":119}]})";

    cyxwiz::PipelineExecutor group_by_multi_agg_executor;
    Check(group_by_multi_agg_executor.ExecutePipeline(group_by_multi_agg_json),
          "GroupBy should accept schema-checked aggregate functions: " +
              group_by_multi_agg_executor.GetLastError());
    auto grouped_multi = registry.GetArrowDataset("ds_groupby_119");
    Check(grouped_multi != nullptr, "GroupBy multi-agg output dataset is registered");
    auto grouped_multi_table = grouped_multi->GetArrowTable();
    Check(grouped_multi_table != nullptr, "GroupBy multi-agg output table exists");
    Check(grouped_multi_table->schema()->GetFieldIndex("total_y") >= 0,
          "GroupBy should preserve SUM aggregation alias");

    const std::string missing_group_column_json =
        R"({"nodes":[)"
        R"({"id":109,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":110,"type":"GroupBy","name":"GroupMissing","parameters":{)"
        R"("group_columns":"missing","aggregations":"COUNT(*) AS count_rows"}})"
        R"(],"links":[{"start_node":109,"end_node":110}]})";

    cyxwiz::PipelineExecutor missing_group_column_executor;
    Check(!missing_group_column_executor.ExecutePipeline(
              missing_group_column_json),
          "GroupBy missing group column should fail schema validation");
    Check(missing_group_column_executor.GetLastError().find(
              "GroupBy: column 'missing' not found") != std::string::npos,
          "GroupBy missing column error should be specific: " +
              missing_group_column_executor.GetLastError());

    const std::string missing_group_aggregation_column_json =
        R"({"nodes":[)"
        R"({"id":120,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":121,"type":"GroupBy","name":"GroupMissingAgg","parameters":{)"
        R"("group_columns":"x","aggregations":"SUM(missing) AS total_missing"}})"
        R"(],"links":[{"start_node":120,"end_node":121}]})";

    cyxwiz::PipelineExecutor missing_group_aggregation_column_executor;
    Check(!missing_group_aggregation_column_executor.ExecutePipeline(
              missing_group_aggregation_column_json),
          "GroupBy missing aggregation column should fail schema validation");
    Check(missing_group_aggregation_column_executor.GetLastError().find(
              "GroupBy: aggregation column 'missing' not found") !=
              std::string::npos,
          "GroupBy missing aggregation column error should be specific: " +
              missing_group_aggregation_column_executor.GetLastError());

    const std::string text_sum_group_json =
        R"({"nodes":[)"
        R"({"id":122,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":123,"type":"GroupBy","name":"GroupTextSum","parameters":{)"
        R"("group_columns":"phrase","aggregations":"SUM(phrase) AS total_phrase"}})"
        R"(],"links":[{"start_node":122,"end_node":123}]})";

    cyxwiz::PipelineExecutor text_sum_group_executor;
    Check(!text_sum_group_executor.ExecutePipeline(text_sum_group_json),
          "GroupBy numeric aggregation on text should fail type validation");
    Check(text_sum_group_executor.GetLastError().find(
              "GroupBy: aggregation column 'phrase' must be numeric for SUM") !=
              std::string::npos,
          "GroupBy text SUM error should be specific: " +
              text_sum_group_executor.GetLastError());

    const std::string raw_group_aggregation_json =
        R"({"nodes":[)"
        R"({"id":124,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":125,"type":"GroupBy","name":"GroupRawAgg","parameters":{)"
        R"("group_columns":"x","aggregations":"COUNT(*) FILTER (WHERE y > 10) AS filtered_count"}})"
        R"(],"links":[{"start_node":124,"end_node":125}]})";

    cyxwiz::PipelineExecutor raw_group_aggregation_executor;
    Check(!raw_group_aggregation_executor.ExecutePipeline(
              raw_group_aggregation_json),
          "GroupBy raw SQL aggregation fragments should fail validation");
    Check(raw_group_aggregation_executor.GetLastError().find(
              "GroupBy: unsupported aggregation") != std::string::npos,
          "GroupBy raw SQL aggregation error should be specific: " +
              raw_group_aggregation_executor.GetLastError());

    const std::string ts_window_json =
        R"({"nodes":[)"
        R"({"id":140,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":141,"type":"TSWindow","name":"Window","parameters":{)"
        R"("target_column":"x","window_size":"2","stride":"1"}})"
        R"(],"links":[{"start_node":140,"end_node":141}]})";

    cyxwiz::PipelineExecutor ts_window_executor;
    Check(ts_window_executor.ExecutePipeline(ts_window_json),
          "TSWindow should validate and quote numeric target column: " +
              ts_window_executor.GetLastError());
    auto ts_window = registry.GetArrowDataset("ds_tswindow_141");
    Check(ts_window != nullptr, "TSWindow output dataset is registered");
    auto ts_window_table = ts_window->GetArrowTable();
    Check(ts_window_table != nullptr, "TSWindow output table exists");
    Check(ReadNumericValue(ts_window_table, "window_t0", 2) == 3.0,
          "TSWindow LAG offset 0 should preserve target values");

    const std::string ts_features_json =
        R"({"nodes":[)"
        R"({"id":142,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":143,"type":"TSFeatures","name":"Features","parameters":{)"
        R"("columns":"y","rolling_window":"2"}})"
        R"(],"links":[{"start_node":142,"end_node":143}]})";

    cyxwiz::PipelineExecutor ts_features_executor;
    Check(ts_features_executor.ExecutePipeline(ts_features_json),
          "TSFeatures should validate and quote numeric source column: " +
              ts_features_executor.GetLastError());
    auto ts_features = registry.GetArrowDataset("ds_tsfeatures_143");
    Check(ts_features != nullptr, "TSFeatures output dataset is registered");
    auto ts_features_table = ts_features->GetArrowTable();
    Check(ts_features_table != nullptr, "TSFeatures output table exists");
    Check(std::fabs(ReadNumericValue(ts_features_table, "y_rolling_mean", 1) - 15.0) <
              0.001,
          "TSFeatures rolling mean should use requested column");

    const std::string ts_lag_json =
        R"({"nodes":[)"
        R"({"id":144,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":145,"type":"TSLag","name":"Lag","parameters":{)"
        R"("columns":"y","lag_periods":"1"}})"
        R"(],"links":[{"start_node":144,"end_node":145}]})";

    cyxwiz::PipelineExecutor ts_lag_executor;
    Check(ts_lag_executor.ExecutePipeline(ts_lag_json),
          "TSLag should validate and quote numeric source column: " +
              ts_lag_executor.GetLastError());
    auto ts_lag = registry.GetArrowDataset("ds_tslag_145");
    Check(ts_lag != nullptr, "TSLag output dataset is registered");
    auto ts_lag_table = ts_lag->GetArrowTable();
    Check(ts_lag_table != nullptr, "TSLag output table exists");
    Check(ReadNumericValue(ts_lag_table, "y_lag1", 1) == 10.0,
          "TSLag should create requested lag column");

    const std::string ts_diff_json =
        R"({"nodes":[)"
        R"({"id":146,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":147,"type":"TSDiff","name":"Diff","parameters":{)"
        R"("columns":"y","order":"1"}})"
        R"(],"links":[{"start_node":146,"end_node":147}]})";

    cyxwiz::PipelineExecutor ts_diff_executor;
    Check(ts_diff_executor.ExecutePipeline(ts_diff_json),
          "TSDiff should validate and quote numeric source column: " +
              ts_diff_executor.GetLastError());
    auto ts_diff = registry.GetArrowDataset("ds_tsdiff_147");
    Check(ts_diff != nullptr, "TSDiff output dataset is registered");
    auto ts_diff_table = ts_diff->GetArrowTable();
    Check(ts_diff_table != nullptr, "TSDiff output table exists");
    Check(ReadNumericValue(ts_diff_table, "y_diff1", 1) == 10.0,
          "TSDiff should create requested difference column");

    const std::string ts_features_text_column_json =
        R"({"nodes":[)"
        R"({"id":148,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":149,"type":"TSFeatures","name":"BadFeatures","parameters":{)"
        R"("columns":"phrase","rolling_window":"2"}})"
        R"(],"links":[{"start_node":148,"end_node":149}]})";

    cyxwiz::PipelineExecutor ts_features_text_column_executor;
    Check(!ts_features_text_column_executor.ExecutePipeline(
              ts_features_text_column_json),
          "TSFeatures on text column should fail schema validation");
    Check(ts_features_text_column_executor.GetLastError().find(
              "TSFeatures: column 'phrase' must be numeric") !=
              std::string::npos,
          "TSFeatures text column error should be specific: " +
              ts_features_text_column_executor.GetLastError());

    const std::string ts_diff_missing_column_json =
        R"({"nodes":[)"
        R"({"id":150,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":151,"type":"TSDiff","name":"MissingDiff","parameters":{)"
        R"("columns":"missing","order":"1"}})"
        R"(],"links":[{"start_node":150,"end_node":151}]})";

    cyxwiz::PipelineExecutor ts_diff_missing_column_executor;
    Check(!ts_diff_missing_column_executor.ExecutePipeline(
              ts_diff_missing_column_json),
          "TSDiff missing input column should fail schema validation");
    Check(ts_diff_missing_column_executor.GetLastError().find(
              "TSDiff: column 'missing' not found") != std::string::npos,
          "TSDiff missing column error should be specific: " +
              ts_diff_missing_column_executor.GetLastError());

    const std::string table_splitter_json =
        R"({"nodes":[)"
        R"({"id":83,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":84,"type":"TableSplitter","name":"Split","parameters":{)"
        R"("split_row":"1"}})"
        R"(],"links":[{"start_node":83,"end_node":84}]})";

    cyxwiz::PipelineExecutor table_splitter_executor;
    Check(!table_splitter_executor.ExecutePipeline(table_splitter_json),
          "TableSplitter should fail closed until multi-output routing exists");
    Check(table_splitter_executor.GetLastError().find(
              "pin-aware multi-output routing") != std::string::npos,
          "TableSplitter fail-closed error should explain routing limitation: " +
              table_splitter_executor.GetLastError());

    registry.UnloadDataset("ds_datainput_1");
    registry.UnloadDataset("ds_operator_StandardScaler_2");
    registry.UnloadDataset("ds_operator_ACFNode_202");
    registry.UnloadDataset("ds_input_133");
    registry.UnloadDataset("ds_select_134");
    registry.UnloadDataset("ds_datainput_300");
    registry.UnloadDataset("ds_datainput_302");
    registry.UnloadDataset("ds_datainput_304");
    registry.UnloadDataset("ds_datainput_306");
    registry.UnloadDataset("ds_datainput_308");
    registry.UnloadDataset("ds_datainput_310");
    registry.UnloadDataset("ds_datainput_188");
    registry.UnloadDataset("ds_filter_189");
    registry.UnloadDataset("ds_datainput_190");
    registry.UnloadDataset("ds_filter_191");
    registry.UnloadDataset("ds_datainput_192");
    registry.UnloadDataset("ds_datainput_194");
    registry.UnloadDataset("ds_datainput_3");
    registry.UnloadDataset("ds_datainput_35");
    registry.UnloadDataset("ds_datainput_139");
    registry.UnloadDataset("ds_datainput_135");
    registry.UnloadDataset("saved_alias");
    registry.UnloadDataset("ds_datainput_37");
    registry.UnloadDataset("ds_datainput_43");
    registry.UnloadDataset("ds_renamed_44");
    registry.UnloadDataset("ds_datainput_47");
    registry.UnloadDataset("ds_datainput_101");
    registry.UnloadDataset("ds_datainput_102");
    registry.UnloadDataset("ds_join_103");
    registry.UnloadDataset("ds_datainput_104");
    registry.UnloadDataset("ds_datainput_105");
    registry.UnloadDataset("ds_datainput_49");
    registry.UnloadDataset("ds_newheaders_50");
    registry.UnloadDataset("ds_datainput_53");
    registry.UnloadDataset("ds_cropped_54");
    registry.UnloadDataset("ds_datainput_55");
    registry.UnloadDataset("ds_datainput_59");
    registry.UnloadDataset("ds_math_60");
    registry.UnloadDataset("ds_datainput_85");
    registry.UnloadDataset("ds_math_86");
    registry.UnloadDataset("ds_datainput_61");
    registry.UnloadDataset("ds_datainput_63");
    registry.UnloadDataset("ds_fillmissing_64");
    registry.UnloadDataset("ds_datainput_184");
    registry.UnloadDataset("ds_fillmissing_185");
    registry.UnloadDataset("ds_datainput_186");
    registry.UnloadDataset("ds_datainput_65");
    registry.UnloadDataset("ds_string_66");
    registry.UnloadDataset("ds_datainput_67");
    registry.UnloadDataset("ds_string_68");
    registry.UnloadDataset("ds_datainput_87");
    registry.UnloadDataset("ds_datainput_152");
    registry.UnloadDataset("ds_textclean_153");
    registry.UnloadDataset("ds_datainput_154");
    registry.UnloadDataset("ds_texttokenize_155");
    registry.UnloadDataset("ds_datainput_156");
    registry.UnloadDataset("ds_textvectorize_157");
    registry.UnloadDataset("ds_datainput_158");
    registry.UnloadDataset("ds_datainput_160");
    registry.UnloadDataset("ds_datainput_71");
    registry.UnloadDataset("ds_binning_72");
    registry.UnloadDataset("ds_datainput_197");
    registry.UnloadDataset("ds_binning_198");
    registry.UnloadDataset("ds_datainput_89");
    registry.UnloadDataset("ds_datainput_77");
    registry.UnloadDataset("ds_poly_78");
    registry.UnloadDataset("ds_datainput_91");
    registry.UnloadDataset("ds_datainput_93");
    registry.UnloadDataset("ds_select_94");
    registry.UnloadDataset("ds_datainput_95");
    registry.UnloadDataset("ds_datainput_97");
    registry.UnloadDataset("ds_sort_98");
    registry.UnloadDataset("ds_datainput_99");
    registry.UnloadDataset("ds_datainput_114");
    registry.UnloadDataset("ds_sort_115");
    registry.UnloadDataset("ds_datainput_107");
    registry.UnloadDataset("ds_groupby_108");
    registry.UnloadDataset("ds_datainput_118");
    registry.UnloadDataset("ds_groupby_119");
    registry.UnloadDataset("ds_datainput_109");
    registry.UnloadDataset("ds_datainput_120");
    registry.UnloadDataset("ds_datainput_122");
    registry.UnloadDataset("ds_datainput_124");
    registry.UnloadDataset("ds_datainput_140");
    registry.UnloadDataset("ds_tswindow_141");
    registry.UnloadDataset("ds_datainput_142");
    registry.UnloadDataset("ds_tsfeatures_143");
    registry.UnloadDataset("ds_datainput_144");
    registry.UnloadDataset("ds_tslag_145");
    registry.UnloadDataset("ds_datainput_146");
    registry.UnloadDataset("ds_tsdiff_147");
    registry.UnloadDataset("ds_datainput_148");
    registry.UnloadDataset("ds_datainput_150");
    registry.UnloadDataset("ds_datainput_83");
    fs::remove(csv_path);
    fs::remove(ts_analysis_csv_path);
    fs::remove(export_csv_path);
    fs::remove(export_csv_alias_path);
    fs::remove(save_dataset_csv_path);
    fs::remove(missing_csv_path);
    fs::remove(string_csv_path);
    fs::remove(missing_string_csv_path);

    return 0;
}
