#include "../src/core/arrow_dataset.h"
#include "../src/core/data_preview_service.h"
#include "../src/core/data_registry.h"
#include "../src/core/parquet_backed_dataset.h"

#include <arrow/api.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::shared_ptr<arrow::Array> FinishInt64Array(
    const std::vector<int64_t>& values) {
    arrow::Int64Builder builder;
    for (int64_t value : values) {
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

std::shared_ptr<arrow::Table> MakePreviewTable() {
    auto schema = arrow::schema({
        arrow::field("id", arrow::int64()),
        arrow::field("value", arrow::int64()),
        arrow::field("label", arrow::utf8()),
    });
    return arrow::Table::Make(
        schema,
        {FinishInt64Array({1, 2, 3, 4, 5}),
         FinishInt64Array({10, 20, 30, 40, 50}),
         FinishStringArray({"a", "b", "c", "d", "e"})},
        5);
}

} // namespace

int main() {
    auto& registry = cyxwiz::DataRegistry::Instance();
    registry.UnregisterTabularDataset("preview_arrow");
    registry.UnregisterTabularDataset("preview_parquet");

    auto table = MakePreviewTable();
    Check(registry.RegisterArrowTable(table, "preview_arrow") != nullptr,
          "Arrow preview fixture should register");

    cyxwiz::DataPreviewRequest request;
    request.dataset_name = "preview_arrow";
    request.offset = 1;
    request.row_limit = 2;
    request.selected_columns = {"id", "label"};
    auto page = cyxwiz::DataPreviewService::PreviewRegisteredTabular(
        registry, request);
    Check(page.ok, page.reason);
    Check(page.backend == "Arrow", "Arrow page should report backend");
    Check(page.total_rows == 5, "Arrow total rows should be preserved");
    Check(page.offset == 1, "Arrow offset should echo request");
    Check(page.rows_returned == 2, "Arrow should return requested bounded rows");
    Check(page.has_next && page.next_offset == 3,
          "Arrow next cursor should advance by returned rows");
    Check(page.schema.size() == 2 && page.schema[0].name == "id" &&
              page.schema[1].name == "label",
          "Arrow selected schema should be ordered");
    Check(page.rows[0][0] == "2" && page.rows[0][1] == "b",
          "Arrow preview should start at offset");

    request.selected_columns = {"missing"};
    auto missing = cyxwiz::DataPreviewService::PreviewRegisteredTabular(
        registry, request);
    Check(!missing.ok &&
              missing.reason.find("missing") != std::string::npos,
          "missing preview column should produce typed failure");

    const fs::path parquet_path =
        fs::temp_directory_path() / "cyxwiz_data_preview_service.parquet";
    auto arrow_dataset = registry.GetArrowDataset("preview_arrow");
    Check(arrow_dataset && arrow_dataset->ExportParquet(parquet_path.string()),
          "preview fixture should export to Parquet");
    auto parquet_dataset = cyxwiz::ParquetBackedDataset::Open(
        parquet_path.string(), "preview_parquet");
    Check(parquet_dataset != nullptr, "Parquet preview fixture should open");
    registry.RegisterParquetBacked("preview_parquet", parquet_dataset);
    auto source_match = registry.FindTabularDatasetBySourcePath(parquet_path.string());
    Check(source_match && *source_match == "preview_parquet",
           "registered Parquet source path should resolve to dataset name");

    cyxwiz::DataPreviewRequest parquet_request;
    parquet_request.dataset_name = "preview_parquet";
    parquet_request.offset = 3;
    parquet_request.row_limit = 20;
    parquet_request.selected_columns = {"value"};
    auto parquet_page = cyxwiz::DataPreviewService::PreviewRegisteredTabular(
        registry, parquet_request);
    Check(parquet_page.ok, parquet_page.reason);
    Check(parquet_page.backend == "Parquet", "Parquet page should report backend");
    Check(parquet_page.rows_returned == 2,
          "Parquet preview should cap at available rows");
    Check(!parquet_page.has_next && parquet_page.next_offset == 5,
          "Parquet next cursor should terminate at row count");
    Check(parquet_page.rows[0][0] == "40" &&
              parquet_page.rows[1][0] == "50",
          "Parquet preview should read only requested tail rows");

    cyxwiz::DataPreviewRequest unknown_request;
    unknown_request.dataset_name = "not_registered";
    auto unknown = cyxwiz::DataPreviewService::PreviewRegisteredTabular(
        registry, unknown_request);
    Check(!unknown.ok &&
              unknown.reason.find("registered tabular") != std::string::npos,
          "unknown dataset should produce unsupported reason");

    registry.UnregisterTabularDataset("preview_arrow");
    registry.UnregisterTabularDataset("preview_parquet");
    std::error_code ec;
    fs::remove(parquet_path, ec);

    std::cout << "Data preview service test passed\n";
    return 0;
}
