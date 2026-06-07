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
    fs::remove(export_csv_path);
    {
        std::ofstream csv(csv_path);
        csv << "x,y\n";
        csv << "1,10\n";
        csv << "2,20\n";
        csv << "3,30\n";
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
    fs::remove(csv_path);
    fs::remove(export_csv_path);

    return 0;
}
