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

    const std::string bad_crop_json =
        R"({"nodes":[)"
        R"({"id":15,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":16,"type":"TableCropper","name":"BadCrop","parameters":{)"
        R"("start_row":"-2","end_row":"10"}})"
        R"(],"links":[{"start_node":15,"end_node":16}]})";

    cyxwiz::PipelineExecutor bad_crop_executor;
    Check(!bad_crop_executor.ExecutePipeline(bad_crop_json),
          "TableCropper bad start_row should fail validation");
    Check(bad_crop_executor.GetLastError().find(
              "TableCropper start_row must be an integer >= 0") !=
              std::string::npos,
          "bad TableCropper validation should be specific: " +
              bad_crop_executor.GetLastError());

    const std::string unknown_node_json =
        R"({"nodes":[)"
        R"({"id":17,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":"ignored.csv","type":"csv"}},)"
        R"({"id":18,"type":"DefinitelyMissingNode","name":"Unknown","parameters":{}})"
        R"(],"links":[{"start_node":17,"end_node":18}]})";

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
        R"({"id":19,"type":"ParquetInput","name":"Parquet","parameters":{)"
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

    registry.UnloadDataset("ds_datainput_1");
    registry.UnloadDataset("ds_operator_StandardScaler_2");
    registry.UnloadDataset("ds_datainput_3");
    fs::remove(csv_path);

    return 0;
}
