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

    const fs::path csv_path =
        fs::temp_directory_path() / "cyxwiz_pipeline_executor_operator_routing.csv";
    const fs::path export_csv_path =
        fs::temp_directory_path() / "cyxwiz_pipeline_executor_operator_export.csv";
    const fs::path missing_csv_path =
        fs::temp_directory_path() / "cyxwiz_pipeline_executor_missing_values.csv";
    const fs::path string_csv_path =
        fs::temp_directory_path() / "cyxwiz_pipeline_executor_strings.csv";
    fs::remove(export_csv_path);
    fs::remove(missing_csv_path);
    fs::remove(string_csv_path);
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
        R"("source_type":"file","type":"csv"}},)"
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

    const std::string bad_window_json =
        R"({"nodes":[)"
        R"({"id":13,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":14,"type":"TSWindow","name":"BadWindow","parameters":{)"
        R"("window_size":"0","stride":"1"}})"
        R"(],"links":[{"start_node":13,"end_node":14}]})";

    cyxwiz::PipelineExecutor bad_window_executor;
    Check(!bad_window_executor.ExecutePipeline(bad_window_json),
          "TSWindow bad window_size should fail validation");
    Check(bad_window_executor.GetLastError().find(
              "TSWindow window_size must be an integer >= 1") != std::string::npos,
          "bad TSWindow validation should be specific: " +
              bad_window_executor.GetLastError());

    const std::string bad_lags_json =
        R"({"nodes":[)"
        R"({"id":15,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":16,"type":"TSLag","name":"BadLag","parameters":{)"
        R"("lag_periods":"1,nope,3"}})"
        R"(],"links":[{"start_node":15,"end_node":16}]})";

    cyxwiz::PipelineExecutor bad_lags_executor;
    Check(!bad_lags_executor.ExecutePipeline(bad_lags_json),
          "TSLag bad lag_periods should fail validation");
    Check(bad_lags_executor.GetLastError().find(
              "TSLag lag_periods must be a comma-separated list of integers >= 1") !=
              std::string::npos,
          "bad TSLag validation should be specific: " +
              bad_lags_executor.GetLastError());

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
        R"("strategy":"mean"}})"
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

    const std::string string_replace_json =
        R"({"nodes":[)"
        R"({"id":65,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(string_csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":66,"type":"StringManipulation","name":"Replace","parameters":{)"
        R"("column":"phrase","operation":"replace","param1":"tea","param2":"coffee"}})"
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
    registry.UnloadDataset("ds_datainput_3");
    registry.UnloadDataset("ds_datainput_35");
    registry.UnloadDataset("ds_datainput_37");
    registry.UnloadDataset("ds_datainput_43");
    registry.UnloadDataset("ds_renamed_44");
    registry.UnloadDataset("ds_datainput_47");
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
    registry.UnloadDataset("ds_datainput_65");
    registry.UnloadDataset("ds_string_66");
    registry.UnloadDataset("ds_datainput_67");
    registry.UnloadDataset("ds_string_68");
    registry.UnloadDataset("ds_datainput_87");
    registry.UnloadDataset("ds_datainput_71");
    registry.UnloadDataset("ds_binning_72");
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
    registry.UnloadDataset("ds_datainput_83");
    fs::remove(csv_path);
    fs::remove(export_csv_path);
    fs::remove(missing_csv_path);
    fs::remove(string_csv_path);

    return 0;
}
