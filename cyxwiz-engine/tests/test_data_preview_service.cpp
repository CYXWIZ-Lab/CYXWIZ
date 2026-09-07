#include "../src/core/arrow_dataset.h"
#include "../src/core/data_preview_service.h"
#include "../src/core/data_registry.h"
#include "../src/core/parquet_backed_dataset.h"
#include "../src/gui/data_input_preview.h"
#include "../src/gui/data_preview_page_cache.h"

#include <arrow/api.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
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

std::shared_ptr<arrow::Array> FinishNullableLabelArray() {
    arrow::StringBuilder builder;
    Check(builder.Append("a").ok(), "append label a");
    Check(builder.Append("b").ok(), "append label b");
    Check(builder.AppendNull().ok(), "append null label");
    Check(builder.Append("d").ok(), "append label d");
    Check(builder.Append("e").ok(), "append label e");
    std::shared_ptr<arrow::Array> array;
    const auto status = builder.Finish(&array);
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
         FinishNullableLabelArray()},
        5);
}

void TestPreviewSourceContract() {
    using gui::data_input::MatchesAppliedTabularPreview;
    using gui::data_input::TabularPreviewSource;
    for (const auto& [format, type] :
         std::vector<std::pair<std::string, int>>{
             {"csv", 1}, {"tsv", 2}, {"parquet", 4},
             {"feather", 7}, {"arrow", 8}, {"ipc", 8}}) {
        const TabularPreviewSource source{
            "preview." + format, type, true, ",", '.', "NA", 0, 12};
        const std::map<std::string, std::string> applied{
            {"file_path", source.path}, {"file_type", format},
            {"has_header", "true"}, {"delimiter", ","},
            {"decimal_point", "."}, {"missing_value_tokens", "NA"},
            {"skip_rows", "0"}, {"max_rows", "12"}};
        Check(MatchesAppliedTabularPreview(applied, source),
              format + ": canonical Apply settings must permit registered preview");
        auto restored = applied;
        Check(MatchesAppliedTabularPreview(restored, source),
              format + ": restored settings must retain registered preview eligibility");
        restored.erase("file_type");
        restored["type"] = format;
        Check(MatchesAppliedTabularPreview(restored, source),
              format + ": legacy format key must remain compatible");
        restored["file_type"] = "auto";
        Check(MatchesAppliedTabularPreview(restored, source),
              format + ": concrete legacy format must resolve alongside canonical auto");
        restored["type"] = "auto";
        Check(MatchesAppliedTabularPreview(restored, source),
              format + ": auto must resolve from the source extension");
        restored["file_type"] = format;
        Check(MatchesAppliedTabularPreview(restored, source),
              format + ": canonical concrete format must resolve alongside legacy auto");
        restored["file_type"] = " \t" + format + "\n";
        Check(MatchesAppliedTabularPreview(restored, source),
              format + ": normalized aliases must match effective format");
        restored["type"] = "json";
        Check(!MatchesAppliedTabularPreview(restored, source),
              format + ": conflicting concrete aliases must not reuse registered preview");
        auto changed = source;
        changed.path = "different." + format;
        Check(!MatchesAppliedTabularPreview(applied, changed), "changed path must invalidate preview");
        changed = source;
        changed.detected_type = 3;
        Check(!MatchesAppliedTabularPreview(applied, changed), "changed format must invalidate preview");
        changed = source;
        changed.has_header = false;
        Check(!MatchesAppliedTabularPreview(applied, changed), "changed header must invalidate preview");
        changed = source;
        changed.delimiter = ";";
        Check(!MatchesAppliedTabularPreview(applied, changed), "changed delimiter must invalidate preview");
        changed = source;
        changed.decimal_point = ',';
        Check(!MatchesAppliedTabularPreview(applied, changed), "changed decimal point must invalidate preview");
        changed = source;
        changed.missing_value_tokens = "null";
        Check(!MatchesAppliedTabularPreview(applied, changed), "changed null tokens must invalidate preview");
        changed = source;
        changed.skip_rows = 5;
        Check(!MatchesAppliedTabularPreview(applied, changed), "changed skip rows must invalidate preview");
        changed = source;
        changed.max_rows = 1;
        Check(!MatchesAppliedTabularPreview(applied, changed), "changed row limit must invalidate preview");
        restored = applied;
        restored["file_type"] = "unknown";
        Check(!MatchesAppliedTabularPreview(restored, source), "unknown format must fail closed");
    }
    const TabularPreviewSource parquet{"preview.parquet", 4, true, ",", '.', "", 0, 0};
    Check(!MatchesAppliedTabularPreview({}, parquet), "unapplied source cannot reuse registered preview");
    for (const auto& [extension, type] :
         std::vector<std::pair<std::string, int>>{
             {"parquet", 4}, {"feather", 7}, {"arrow", 8}, {"ipc", 8},
             {"xlsx", 5}, {"json", 3}, {"hdf5", 6}, {"unknown", 0}}) {
        for (int selected_type : {0, type}) {
            const auto rejected = gui::data_input::LoadDelimitedPreview(
                "not-opened." + extension, true, ',', selected_type);
            Check(rejected.error.find("Apply this source first") != std::string::npos &&
                      rejected.columns.empty() && rejected.rows.empty(),
                  extension + ": explicit/auto non-delimited preview must reject before file I/O");
        }
    }
}

} // namespace

int main() {
    TestPreviewSourceContract();
    gui::data_input::PreviewPageCache page_cache(2, 2);
    page_cache.PutPage(0, {{"r0"}, {"r1"}});
    page_cache.PutPage(2, {{"r2"}, {"r3"}});
    Check(page_cache.PageCount() == 2 && page_cache.RowCount() == 4,
          "preview cache should retain only configured bounded pages");
    Check(page_cache.FindRow(0) && (*page_cache.FindRow(0))[0] == "r0",
          "preview cache should resolve a row by virtual dataset index");
    page_cache.PutPage(4, {{"r4"}, {"r5"}});
    Check(page_cache.FindRow(0) != nullptr,
          "recently used preview page should survive LRU eviction");
    Check(page_cache.FindRow(2) == nullptr,
          "least-recently used preview page should be evicted");
    Check(page_cache.FindRow(5) && (*page_cache.FindRow(5))[0] == "r5",
          "new preview page should be available after eviction");
    Check(page_cache.AlignOffset(5) == 4,
          "preview row should align to its bounded page offset");

    const fs::path preambled_csv_path =
        fs::temp_directory_path() / "cyxwiz_preambled_preview.csv";
    {
        std::ofstream csv(preambled_csv_path, std::ios::binary | std::ios::trunc);
        csv << "Dataset information\n"
            << "Copyright notice\n"
            << "License terms,with,commas\n"
            << "----------------\n"
            << "class,feature_a,feature_b\n"
            << "neg,1,2\n"
            << "pos,3,4\n";
    }
    const auto preambled_preview = gui::data_input::LoadDelimitedPreview(
        preambled_csv_path.string(), true, ',', 1, 4);
    Check(preambled_preview.error.empty(),
          "preambled CSV preview should load after skipping source metadata");
    Check(preambled_preview.columns ==
              std::vector<std::string>({"class", "feature_a", "feature_b"}),
          "first row after skipped metadata should become the header");
    Check(preambled_preview.rows.size() == 2 &&
              preambled_preview.rows[0][0] == "neg" &&
              preambled_preview.rows[1][2] == "4",
          "preambled CSV preview should return data rows after the header");
    const auto over_skipped_preview = gui::data_input::LoadDelimitedPreview(
        preambled_csv_path.string(), true, ',', 1, 50);
    Check(over_skipped_preview.error.find("No tabular rows remain") !=
              std::string::npos,
          "preview should explain when skip_rows consumes the source");
    const auto auto_csv = gui::data_input::LoadDelimitedPreview(
        preambled_csv_path.string(), true, ',', 0, 4);
    Check(auto_csv.error.empty() && auto_csv.rows == preambled_preview.rows,
          "Auto CSV preview must preserve delimited source sampling");

    const fs::path tsv_path = fs::temp_directory_path() / "cyxwiz_auto_preview.tsv";
    {
        std::ofstream tsv(tsv_path);
        tsv << "first\tsecond\n1\t2\n";
    }
    const auto auto_tsv = gui::data_input::LoadDelimitedPreview(tsv_path.string(), true, ',', 0);
    Check(auto_tsv.error.empty() && auto_tsv.columns.size() == 2 &&
              auto_tsv.rows == std::vector<std::vector<std::string>>{{"1", "2"}},
          "Auto TSV preview must use tabs, not the default comma");
    fs::remove(tsv_path);

    const fs::path quoted_csv_path =
        fs::temp_directory_path() / "cyxwiz_quoted_preview.csv";
    {
        std::ofstream csv(quoted_csv_path, std::ios::binary | std::ios::trunc);
        csv << "statement,status,optional\r\n"
            << "\"Good, really \"\"good\"\"\",positive,\r\n"
            << "\"First line\nsecond line\",negative,001\r\n";
    }
    const auto quoted = gui::data_input::LoadDelimitedPreview(
        quoted_csv_path.string(), true, ',', 1);
    Check(quoted.error.empty() && quoted.rows.size() == 2,
          "quoted CSV preview should count logical records, not physical lines");
    Check(quoted.rows[0] == std::vector<std::string>({"Good, really \"good\"", "positive", ""}) &&
          quoted.rows[1] == std::vector<std::string>({"First line\nsecond line", "negative", "001"}),
          "preview must preserve quoted commas, escapes, multiline text, empty and string values");
    const auto labels = gui::data_input::ComputeLabelDistribution(
        quoted.columns, quoted.rows, "status");
    Check(labels.total == 2 && labels.values.size() == 2 &&
          labels.values[0].first == "negative" && labels.values[1].first == "positive",
          "sentiment distribution must count labels, not sentence fragments");
    const auto bounded = gui::data_input::LoadDelimitedPreview(
        quoted_csv_path.string(), true, ',', 1, 0, 2);
    Check(bounded.rows.size() == 1, "preview must honor its logical-record limit");
    {
        std::ofstream csv(quoted_csv_path, std::ios::trunc);
        csv << "statement,status\n\"unterminated,positive\n";
    }
    const auto malformed = gui::data_input::LoadDelimitedPreview(
        quoted_csv_path.string(), true, ',', 1);
    Check(malformed.rows.empty() && malformed.error.find("unterminated") != std::string::npos,
          "malformed quoted rows must report an error, not fabricated preview data");
    fs::remove(quoted_csv_path);

    const fs::path limited_parquet_path =
        fs::temp_directory_path() / "cyxwiz_limited_csv_cache.parquet";
    Check(cyxwiz::ParquetBackedDataset::ConvertCsvToParquet(
              preambled_csv_path.string(),
              limited_parquet_path.string(),
              true,
              ',',
              4,
              1),
          "disk-backed CSV conversion should accept a bounded row limit");
    auto limited_parquet = cyxwiz::ParquetBackedDataset::Open(
        limited_parquet_path.string(), "limited_parquet");
    Check(limited_parquet && limited_parquet->GetNumRows() == 1,
          "disk-backed CSV conversion should preserve exactly max_rows rows");

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
    Check(page.status == cyxwiz::DataPreviewStatus::Ready,
          "successful preview should have Ready status");
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
    Check(page.rows[1][1] == "<null>" &&
              page.schema[1].sampled_values == 2 &&
              page.schema[1].sampled_nulls == 1,
          "preview schema should report page-local null counts");

    cyxwiz::DataPreviewRequest cancelled_request;
    cancelled_request.dataset_name = "preview_arrow";
    cancelled_request.row_limit = 2;
    cancelled_request.cancel_requested = []() { return true; };
    const auto cancelled = cyxwiz::DataPreviewService::PreviewRegisteredTabular(
        registry, cancelled_request);
    Check(!cancelled.ok &&
              cancelled.status == cyxwiz::DataPreviewStatus::Cancelled,
          "cooperatively cancelled preview should return typed Cancelled status");

    request.selected_columns = {"missing"};
    auto missing = cyxwiz::DataPreviewService::PreviewRegisteredTabular(
        registry, request);
    Check(!missing.ok &&
              missing.status == cyxwiz::DataPreviewStatus::InvalidRequest &&
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
    const std::map<std::string, std::string> parquet_parameters{
        {"file_type", "PARQUET"}, {"file_path", parquet_path.string()},
        {"has_header", "true"}, {"delimiter", ","}, {"decimal_point", "."},
        {"missing_value_tokens", ""}, {"skip_rows", "0"}, {"max_rows", "0"}};
    Check(gui::data_input::MatchesAppliedTabularPreview(parquet_parameters,
              {parquet_path.string(), 4, true, ",", '.', "", 0, 0}),
          "applied canonical Parquet must reach registered paging, not CSV sampling");
    const auto binary_sample = gui::data_input::LoadDelimitedPreview(
        parquet_path.string(), true, ',', 4);
    Check(binary_sample.error.find("Apply this source first") != std::string::npos &&
              binary_sample.rows.empty(),
          "real Parquet bytes must never be parsed as a delimited preview");

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

    // Sorted labels deliberately place every second class beyond the old 24-row preview.
    arrow::StringBuilder text_builder;
    arrow::StringBuilder class_builder;
    for (int row = 0; row < 60; ++row) {
        Check(text_builder.Append("Document " + std::to_string(row)).ok(), "append raw text");
        Check(class_builder.Append(row < 40 ? "class-one" : "class-two").ok(), "append class");
    }
    std::shared_ptr<arrow::Array> texts, classes;
    Check(text_builder.Finish(&texts).ok() && class_builder.Finish(&classes).ok(), "finish text fixture");
    auto text_table = arrow::Table::Make(arrow::schema({
        arrow::field("body", arrow::utf8()), arrow::field("category", arrow::utf8())}),
        {texts, classes});
    Check(registry.RegisterArrowTable(text_table, "generic_text_preview") != nullptr, "register raw text");
    cyxwiz::DataPreviewRequest text_request;
    text_request.dataset_name = "generic_text_preview";
    text_request.row_limit = 10;
    text_request.summarize_label_column = "category";
    const auto text_page = cyxwiz::DataPreviewService::PreviewRegisteredTabular(registry, text_request);
    Check(text_page.ok && text_page.rows_returned == 10 && text_page.total_rows == 60 && text_page.has_next,
          "text page must retain actual dataset extent beyond its first rows");
    Check(text_page.label_summary_complete && text_page.label_counts ==
          std::vector<std::pair<std::string, size_t>>({{"class-one", 40}, {"class-two", 20}}),
          "full class distribution must include classes absent from the first page");
    text_request.offset = 50;
    text_request.summarize_label_column.clear();
    const auto text_tail = cyxwiz::DataPreviewService::PreviewRegisteredTabular(registry, text_request);
    Check(text_tail.ok && text_tail.rows.front()[0] == "Document 50" && !text_tail.has_next &&
          text_tail.label_counts.empty(), "later text pages must navigate without recomputing full class statistics");
    text_request.offset = 0;
    text_request.summarize_label_column = "missing";
    const auto missing_label = cyxwiz::DataPreviewService::PreviewRegisteredTabular(registry, text_request);
    Check(missing_label.ok && !missing_label.label_summary_complete && !missing_label.label_summary_error.empty(),
          "invalid label column must explain missing distribution without blocking row preview");
    text_request.summarize_label_column = "category";
    int cancellation_checks = 0;
    text_request.cancel_requested = [&] { return ++cancellation_checks > 2; };
    const auto cancelled_labels = cyxwiz::DataPreviewService::PreviewRegisteredTabular(registry, text_request);
    Check(cancelled_labels.status == cyxwiz::DataPreviewStatus::Cancelled,
          "cancelled text summary must not be published as complete");
    registry.UnregisterTabularDataset("generic_text_preview");

    cyxwiz::DataPreviewRequest unknown_request;
    unknown_request.dataset_name = "not_registered";
    auto unknown = cyxwiz::DataPreviewService::PreviewRegisteredTabular(
        registry, unknown_request);
    Check(!unknown.ok &&
              unknown.status == cyxwiz::DataPreviewStatus::Unsupported &&
              unknown.reason.find("registered tabular") != std::string::npos,
          "unknown dataset should produce unsupported reason");

    registry.UnregisterTabularDataset("preview_arrow");
    registry.UnregisterTabularDataset("preview_parquet");
    std::error_code ec;
    fs::remove(parquet_path, ec);
    limited_parquet.reset();
    fs::remove(limited_parquet_path, ec);
    fs::remove(preambled_csv_path, ec);

    std::cout << "Data preview service test passed: " << checks << " checks\n";
    return 0;
}
