#include "data_convert_service.h"

#include "arrow_dataset.h"

#include <arrow/csv/api.h>
#include <arrow/io/file.h>
#include <arrow/scalar.h>
#include <nlohmann/json.hpp>
#include <parquet/arrow/writer.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace cyxwiz {
namespace {

std::string LowerAscii(std::string value) {
    for (char& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

std::string NowIsoLikeUtc() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &time);
#else
    gmtime_r(&time, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

bool IsCsvPath(const std::filesystem::path& path) {
    const std::string ext = LowerAscii(path.extension().string());
    return ext == ".csv" || ext == ".tsv";
}

bool IsParquetPath(const std::filesystem::path& path) {
    const std::string ext = LowerAscii(path.extension().string());
    return ext == ".parquet" || ext == ".pq";
}

int CountDelimiterOutsideQuotes(const std::string& line, char delimiter) {
    bool in_quotes = false;
    int count = 0;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (!in_quotes && c == delimiter) {
            ++count;
        }
    }
    return count;
}

char DetectCsvDelimiter(const std::string& path) {
    const std::filesystem::path input_path(path);
    if (LowerAscii(input_path.extension().string()) == ".tsv") {
        return '\t';
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return ',';
    }

    const std::array<char, 4> candidates = {',', '\t', ';', '|'};
    std::array<int, 4> total_counts = {};
    std::array<int, 4> lines_with_delimiter = {};
    std::array<int, 4> mismatch_counts = {};
    std::array<int, 4> first_nonzero_counts = {};

    std::string line;
    int sampled_lines = 0;
    while (sampled_lines < 32 && std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        ++sampled_lines;
        for (size_t i = 0; i < candidates.size(); ++i) {
            const int count = CountDelimiterOutsideQuotes(line, candidates[i]);
            total_counts[i] += count;
            if (count > 0) {
                ++lines_with_delimiter[i];
                if (first_nonzero_counts[i] == 0) {
                    first_nonzero_counts[i] = count;
                } else if (count != first_nonzero_counts[i]) {
                    ++mismatch_counts[i];
                }
            }
        }
    }

    int best_index = 0;
    int best_score = -1;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const int score = total_counts[i] * 4 +
                          lines_with_delimiter[i] * 2 -
                          mismatch_counts[i] * 3;
        if (score > best_score) {
            best_score = score;
            best_index = static_cast<int>(i);
        }
    }

    return best_score > 0 ? candidates[static_cast<size_t>(best_index)] : ',';
}

char ResolveDelimiter(const DataConvertOptions& options) {
    return options.auto_detect_delimiter
        ? DetectCsvDelimiter(options.input_path)
        : options.delimiter;
}

arrow::csv::ReadOptions BuildReadOptions(const DataConvertOptions& options) {
    auto read_options = arrow::csv::ReadOptions::Defaults();
    read_options.skip_rows = std::max(0, options.skip_rows);
    read_options.autogenerate_column_names = !options.has_header;
    return read_options;
}

arrow::csv::ParseOptions BuildParseOptions(const DataConvertOptions& options) {
    auto parse_options = arrow::csv::ParseOptions::Defaults();
    parse_options.delimiter = ResolveDelimiter(options);
    parse_options.newlines_in_values = options.allow_newlines_in_values;
    return parse_options;
}

std::shared_ptr<ArrowDataset> LoadCsv(const DataConvertOptions& options,
                                      std::string& error) {
    auto dataset = ArrowDataset::FromCSV(
        options.input_path,
        "data_convert_preview",
        BuildReadOptions(options),
        BuildParseOptions(options),
        arrow::csv::ConvertOptions::Defaults());
    if (!dataset || !dataset->GetArrowTable()) {
        error = "CSV read failed. Check the file path, delimiter, header setting, and row consistency.";
        return nullptr;
    }
    return dataset;
}

parquet::Compression::type ResolveCompression(const std::string& name,
                                              std::string& error) {
    const std::string normalized = LowerAscii(name);
    if (normalized == "none" || normalized == "uncompressed") {
        return parquet::Compression::UNCOMPRESSED;
    }
    if (normalized == "snappy") {
        return parquet::Compression::SNAPPY;
    }
    if (normalized == "gzip") {
        return parquet::Compression::GZIP;
    }
    if (normalized == "zstd") {
        return parquet::Compression::ZSTD;
    }
    if (normalized == "brotli") {
        return parquet::Compression::BROTLI;
    }

    error = "Unsupported Parquet compression '" + name +
            "'. Choose none, snappy, gzip, zstd, or brotli.";
    return parquet::Compression::UNCOMPRESSED;
}

std::string BuildManifestPath(const std::string& output_path) {
    return output_path + ".manifest.json";
}

std::string BuildSettingsHashInput(const DataConvertOptions& options) {
    std::ostringstream out;
    out << options.input_path << "|"
        << options.output_path << "|"
        << options.delimiter << "|"
        << options.auto_detect_delimiter << "|"
        << options.has_header << "|"
        << options.allow_newlines_in_values << "|"
        << options.skip_rows << "|"
        << options.parquet_compression << "|"
        << options.row_group_size;
    return out.str();
}

std::string SimpleSettingsHash(const DataConvertOptions& options) {
    const auto value = std::hash<std::string>{}(BuildSettingsHashInput(options));
    std::ostringstream out;
    out << std::hex << value;
    return out.str();
}

bool WriteManifest(const DataConvertOptions& options,
                   const DataConvertResult& result,
                   std::string& error) {
    const std::filesystem::path input_path(options.input_path);
    const std::filesystem::path output_path(options.output_path);

    nlohmann::json manifest = {
        {"node", "DataConvert"},
        {"version", 1},
        {"input_path", options.input_path},
        {"input_format", "csv"},
        {"output_path", options.output_path},
        {"output_format", "parquet"},
        {"rows_read", result.rows_read},
        {"rows_written", result.rows_written},
        {"columns", result.columns},
        {"compression", options.parquet_compression},
        {"delimiter", std::string(1, result.detected_delimiter)},
        {"delimiter_mode", options.auto_detect_delimiter ? "auto" : "manual"},
        {"allow_newlines_in_values", options.allow_newlines_in_values},
        {"settings_hash", SimpleSettingsHash(options)},
        {"created_at", NowIsoLikeUtc()}
    };

    std::error_code ec;
    if (std::filesystem::exists(input_path, ec)) {
        manifest["input_size"] = static_cast<int64_t>(std::filesystem::file_size(input_path, ec));
        const auto write_time = std::filesystem::last_write_time(input_path, ec);
        if (!ec) {
            manifest["input_modified_time_native"] =
                static_cast<int64_t>(write_time.time_since_epoch().count());
        }
    }
    if (std::filesystem::exists(output_path, ec)) {
        manifest["output_size"] = static_cast<int64_t>(std::filesystem::file_size(output_path, ec));
    }

    const std::string manifest_path = BuildManifestPath(options.output_path);
    std::ofstream out(manifest_path, std::ios::binary);
    if (!out) {
        error = "Conversion succeeded, but manifest write failed for '" +
                manifest_path + "'. Check folder permissions.";
        return false;
    }
    out << manifest.dump(2);
    return true;
}

std::string ScalarToPreviewString(const std::shared_ptr<arrow::Scalar>& scalar) {
    if (!scalar || !scalar->is_valid) {
        return "null";
    }
    return scalar->ToString();
}

std::string CellToPreviewString(const std::shared_ptr<arrow::Table>& table,
                                int column_index,
                                int64_t row_index) {
    auto column = table->column(column_index);
    int64_t remaining = row_index;
    for (const auto& chunk : column->chunks()) {
        if (remaining < chunk->length()) {
            auto scalar = chunk->GetScalar(static_cast<int>(remaining));
            if (!scalar.ok()) {
                return "<error>";
            }
            return ScalarToPreviewString(scalar.ValueOrDie());
        }
        remaining -= chunk->length();
    }
    return "";
}

} // namespace

DataConvertPreview DataConvertService::PreviewCsv(const DataConvertOptions& options) {
    DataConvertPreview preview;
    const std::filesystem::path input_path(options.input_path);
    if (options.input_path.empty()) {
        preview.error = "Input file is empty. Select a CSV file first.";
        return preview;
    }
    if (!std::filesystem::exists(input_path)) {
        preview.error = "Input file does not exist: " + options.input_path;
        return preview;
    }
    if (!IsCsvPath(input_path)) {
        preview.error = "Phase 1 DataConvert preview supports CSV/TSV input only.";
        return preview;
    }

    std::string error;
    auto dataset = LoadCsv(options, error);
    if (!dataset) {
        preview.error = error;
        return preview;
    }

    auto table = dataset->GetArrowTable();
    preview.detected_delimiter = ResolveDelimiter(options);
    preview.rows = table->num_rows();
    preview.columns = table->num_columns();
    preview.schema.reserve(static_cast<size_t>(table->num_columns()));
    for (int i = 0; i < table->num_columns(); ++i) {
        auto field = table->schema()->field(i);
        preview.schema.push_back({
            field->name(),
            field->type()->ToString(),
            field->nullable()
        });
    }
    const int64_t sample_count =
        std::min<int64_t>(preview.rows, std::max(0, options.preview_rows));
    preview.sample_rows.reserve(static_cast<size_t>(sample_count));
    for (int64_t row = 0; row < sample_count; ++row) {
        std::vector<std::string> values;
        values.reserve(static_cast<size_t>(preview.columns));
        for (int column = 0; column < table->num_columns(); ++column) {
            values.push_back(CellToPreviewString(table, column, row));
        }
        preview.sample_rows.push_back(std::move(values));
    }
    preview.ok = true;
    return preview;
}

DataConvertResult DataConvertService::ConvertCsvToParquet(
    const DataConvertOptions& options) {
    DataConvertResult result;
    const std::filesystem::path input_path(options.input_path);
    const std::filesystem::path output_path(options.output_path);

    if (options.input_path.empty()) {
        result.error = "Input file is empty. Select a CSV file first.";
        return result;
    }
    if (!std::filesystem::exists(input_path)) {
        result.error = "Input file does not exist: " + options.input_path;
        return result;
    }
    if (!IsCsvPath(input_path)) {
        result.error = "Phase 1 DataConvert supports CSV/TSV input only. Input was '" +
                       input_path.extension().string() + "'.";
        return result;
    }
    if (options.output_path.empty()) {
        result.error = "Output path is empty. Choose where to write the Parquet file.";
        return result;
    }
    if (!IsParquetPath(output_path)) {
        result.error = "Phase 1 DataConvert writes Parquet only. Use a .parquet or .pq output path.";
        return result;
    }
    if (std::filesystem::exists(output_path) && !options.overwrite) {
        result.error = "Output file already exists and overwrite is disabled: " +
                       options.output_path;
        return result;
    }

    std::error_code ec;
    if (options.create_parent_dirs && output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path(), ec);
        if (ec) {
            result.error = "Could not create output folder '" +
                           output_path.parent_path().string() + "': " + ec.message();
            return result;
        }
    }

    std::string error;
    auto dataset = LoadCsv(options, error);
    if (!dataset) {
        result.error = error;
        return result;
    }
    auto table = dataset->GetArrowTable();
    result.detected_delimiter = ResolveDelimiter(options);

    auto maybe_output = arrow::io::FileOutputStream::Open(options.output_path);
    if (!maybe_output.ok()) {
        result.error = "Could not open Parquet output '" + options.output_path +
                       "': " + maybe_output.status().ToString();
        return result;
    }

    std::string compression_error;
    const auto compression = ResolveCompression(options.parquet_compression,
                                                compression_error);
    if (!compression_error.empty()) {
        result.error = compression_error;
        return result;
    }

    parquet::WriterProperties::Builder props_builder;
    props_builder.compression(compression);
    auto props = props_builder.build();
    const int64_t row_group_size =
        std::max<int64_t>(1, options.row_group_size);

    const auto status = parquet::arrow::WriteTable(
        *table,
        arrow::default_memory_pool(),
        maybe_output.ValueOrDie(),
        row_group_size,
        props);
    if (!status.ok()) {
        result.error = "Parquet write failed for '" + options.output_path +
                       "': " + status.ToString();
        return result;
    }

    result.rows_read = table->num_rows();
    result.rows_written = table->num_rows();
    result.columns = table->num_columns();
    if (std::filesystem::exists(output_path, ec)) {
        result.bytes_written = static_cast<int64_t>(
            std::filesystem::file_size(output_path, ec));
    }

    if (options.write_manifest) {
        std::string manifest_error;
        if (!WriteManifest(options, result, manifest_error)) {
            result.error = manifest_error;
            return result;
        }
        result.manifest_path = BuildManifestPath(options.output_path);
    }

    result.ok = true;
    spdlog::info("DataConvert: converted '{}' -> '{}' ({} rows, {} columns)",
                 options.input_path, options.output_path,
                 result.rows_written, result.columns);
    return result;
}

} // namespace cyxwiz
