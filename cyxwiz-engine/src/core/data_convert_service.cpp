#include "data_convert_service.h"

#include "arrow_dataset.h"

#include <arrow/csv/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/writer.h>
#include <arrow/scalar.h>
#include <nlohmann/json.hpp>
#include <parquet/arrow/writer.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>

namespace cyxwiz {
namespace {

enum class DataConvertFormat {
    Unknown,
    Csv,
    Tsv,
    Parquet,
    Feather,
    ArrowIpc
};

std::string LowerAscii(std::string value) {
    for (char& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

std::string TrimAscii(const std::string& value) {
    const auto begin = std::find_if_not(value.begin(), value.end(), [](char c) {
        return std::isspace(static_cast<unsigned char>(c)) != 0;
    });
    const auto end = std::find_if_not(value.rbegin(), value.rend(), [](char c) {
        return std::isspace(static_cast<unsigned char>(c)) != 0;
    }).base();
    if (begin >= end) {
        return {};
    }
    return std::string(begin, end);
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

DataConvertFormat FormatFromName(const std::string& raw_name) {
    const std::string name = LowerAscii(TrimAscii(raw_name));
    if (name == "csv") return DataConvertFormat::Csv;
    if (name == "tsv") return DataConvertFormat::Tsv;
    if (name == "parquet" || name == "pq") return DataConvertFormat::Parquet;
    if (name == "feather" || name == "fea") return DataConvertFormat::Feather;
    if (name == "arrow" || name == "ipc" || name == "arrowipc") {
        return DataConvertFormat::ArrowIpc;
    }
    return DataConvertFormat::Unknown;
}

DataConvertFormat FormatFromPath(const std::filesystem::path& path) {
    return FormatFromName(path.extension().string().empty()
                              ? std::string{}
                              : path.extension().string().substr(1));
}

std::string FormatName(DataConvertFormat format) {
    switch (format) {
        case DataConvertFormat::Csv: return "csv";
        case DataConvertFormat::Tsv: return "tsv";
        case DataConvertFormat::Parquet: return "parquet";
        case DataConvertFormat::Feather: return "feather";
        case DataConvertFormat::ArrowIpc: return "ipc";
        default: return "unknown";
    }
}

bool IsDelimitedFormat(DataConvertFormat format) {
    return format == DataConvertFormat::Csv ||
           format == DataConvertFormat::Tsv;
}

bool IsSupportedFormat(DataConvertFormat format) {
    return format != DataConvertFormat::Unknown;
}

bool IsAutoFormat(const std::string& value) {
    const std::string normalized = LowerAscii(TrimAscii(value));
    return normalized.empty() || normalized == "auto";
}

DataConvertFormat ResolveInputFormat(const DataConvertOptions& options) {
    if (!IsAutoFormat(options.input_format)) {
        return FormatFromName(options.input_format);
    }
    return FormatFromPath(options.input_path);
}

DataConvertFormat ResolveOutputFormat(const DataConvertOptions& options) {
    if (!IsAutoFormat(options.output_format)) {
        return FormatFromName(options.output_format);
    }
    return FormatFromPath(options.output_path);
}

std::string SupportedFormatList() {
    return "csv, tsv, parquet, feather, arrow, or ipc";
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

char DetectCsvDelimiter(const std::string& path, DataConvertFormat format) {
    if (format == DataConvertFormat::Tsv) {
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

char ResolveDelimiter(const DataConvertOptions& options,
                      DataConvertFormat format) {
    if (format == DataConvertFormat::Tsv) {
        return '\t';
    }
    return options.auto_detect_delimiter
        ? DetectCsvDelimiter(options.input_path, format)
        : options.delimiter;
}

arrow::csv::ReadOptions BuildReadOptions(const DataConvertOptions& options) {
    auto read_options = arrow::csv::ReadOptions::Defaults();
    read_options.skip_rows = std::max(0, options.skip_rows);
    read_options.autogenerate_column_names = !options.has_header;
    return read_options;
}

arrow::csv::ParseOptions BuildParseOptions(const DataConvertOptions& options,
                                           DataConvertFormat format) {
    auto parse_options = arrow::csv::ParseOptions::Defaults();
    parse_options.delimiter = ResolveDelimiter(options, format);
    parse_options.newlines_in_values = options.allow_newlines_in_values;
    return parse_options;
}

std::shared_ptr<ArrowDataset> LoadDelimited(
    const DataConvertOptions& options,
    DataConvertFormat format,
    std::string& error) {
    auto dataset = ArrowDataset::FromCSV(
        options.input_path,
        "data_convert_input",
        BuildReadOptions(options),
        BuildParseOptions(options, format),
        arrow::csv::ConvertOptions::Defaults());
    if (!dataset || !dataset->GetArrowTable()) {
        error = "Delimited file read failed. Check the path, delimiter, header setting, and row consistency.";
        return nullptr;
    }
    return dataset;
}

std::shared_ptr<ArrowDataset> LoadInputDataset(
    const DataConvertOptions& options,
    DataConvertFormat input_format,
    std::string& error) {
    if (options.input_table) {
        return std::make_shared<ArrowDataset>(options.input_table,
                                              "data_convert_input");
    }
    if (IsDelimitedFormat(input_format)) {
        return LoadDelimited(options, input_format, error);
    }

    auto dataset = ArrowDataset::FromFile(options.input_path,
                                          "data_convert_input");
    if (!dataset || !dataset->GetArrowTable()) {
        error = "Could not load " + FormatName(input_format) +
                " input file: " + options.input_path;
        return nullptr;
    }
    return dataset;
}

parquet::Compression::type ResolveCompression(const std::string& name,
                                              std::string& error) {
    const std::string normalized = LowerAscii(TrimAscii(name));
    if (normalized == "none" || normalized == "uncompressed") {
        return parquet::Compression::UNCOMPRESSED;
    }
    if (normalized == "snappy") return parquet::Compression::SNAPPY;
    if (normalized == "gzip") return parquet::Compression::GZIP;
    if (normalized == "zstd") return parquet::Compression::ZSTD;
    if (normalized == "brotli") return parquet::Compression::BROTLI;

    error = "Unsupported Parquet compression '" + name +
            "'. Choose none, snappy, gzip, zstd, or brotli.";
    return parquet::Compression::UNCOMPRESSED;
}

bool WriteParquet(const std::shared_ptr<arrow::Table>& table,
                  const DataConvertOptions& options,
                  std::string& error) {
    auto maybe_output = arrow::io::FileOutputStream::Open(options.output_path);
    if (!maybe_output.ok()) {
        error = "Could not open Parquet output '" + options.output_path +
                "': " + maybe_output.status().ToString();
        return false;
    }

    std::string compression_error;
    const auto compression = ResolveCompression(options.parquet_compression,
                                                compression_error);
    if (!compression_error.empty()) {
        error = compression_error;
        return false;
    }

    parquet::WriterProperties::Builder props_builder;
    props_builder.compression(compression);
    const auto props = props_builder.build();
    const int64_t row_group_size =
        std::max<int64_t>(1, options.row_group_size);

    const auto status = parquet::arrow::WriteTable(
        *table,
        arrow::default_memory_pool(),
        maybe_output.ValueOrDie(),
        row_group_size,
        props);
    if (!status.ok()) {
        error = "Parquet write failed for '" + options.output_path +
                "': " + status.ToString();
        return false;
    }
    return true;
}

bool WriteArrowIpc(const std::shared_ptr<arrow::Table>& table,
                   const DataConvertOptions& options,
                   std::string& error) {
    auto maybe_output = arrow::io::FileOutputStream::Open(options.output_path);
    if (!maybe_output.ok()) {
        error = "Could not open Arrow IPC output '" + options.output_path +
                "': " + maybe_output.status().ToString();
        return false;
    }

    auto maybe_writer = arrow::ipc::MakeFileWriter(maybe_output.ValueOrDie(),
                                                   table->schema());
    if (!maybe_writer.ok()) {
        error = "Could not create Arrow IPC writer: " +
                maybe_writer.status().ToString();
        return false;
    }
    auto writer = maybe_writer.ValueOrDie();

    arrow::TableBatchReader reader(*table);
    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
        const auto read_status = reader.ReadNext(&batch);
        if (!read_status.ok()) {
            error = "Arrow IPC batch read failed: " + read_status.ToString();
            return false;
        }
        if (!batch) {
            break;
        }
        const auto write_status = writer->WriteRecordBatch(*batch);
        if (!write_status.ok()) {
            error = "Arrow IPC write failed: " + write_status.ToString();
            return false;
        }
    }

    const auto close_status = writer->Close();
    if (!close_status.ok()) {
        error = "Arrow IPC close failed: " + close_status.ToString();
        return false;
    }
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

std::string CsvEscape(const std::string& value, char delimiter) {
    const bool quote = value.find(delimiter) != std::string::npos ||
                       value.find('"') != std::string::npos ||
                       value.find('\n') != std::string::npos ||
                       value.find('\r') != std::string::npos;
    if (!quote) {
        return value;
    }

    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (char c : value) {
        if (c == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(c);
    }
    escaped.push_back('"');
    return escaped;
}

std::string ScalarToDelimitedString(
    const std::shared_ptr<arrow::Scalar>& scalar,
    char delimiter) {
    if (!scalar || !scalar->is_valid) {
        return "";
    }
    return CsvEscape(scalar->ToString(), delimiter);
}

bool WriteDelimited(const std::shared_ptr<arrow::Table>& table,
                    const DataConvertOptions& options,
                    DataConvertFormat output_format,
                    std::string& error) {
    const char delimiter =
        output_format == DataConvertFormat::Tsv ? '\t' : ',';
    std::ofstream out(options.output_path, std::ios::binary);
    if (!out) {
        error = "Could not open delimited output '" + options.output_path + "'.";
        return false;
    }

    for (int column = 0; column < table->num_columns(); ++column) {
        if (column > 0) out << delimiter;
        out << CsvEscape(table->schema()->field(column)->name(), delimiter);
    }
    out << '\n';

    for (int64_t row = 0; row < table->num_rows(); ++row) {
        for (int column = 0; column < table->num_columns(); ++column) {
            if (column > 0) out << delimiter;
            auto chunked = table->column(column);
            int64_t remaining = row;
            std::shared_ptr<arrow::Scalar> scalar;
            for (const auto& chunk : chunked->chunks()) {
                if (remaining < chunk->length()) {
                    auto maybe_scalar =
                        chunk->GetScalar(static_cast<int>(remaining));
                    if (!maybe_scalar.ok()) {
                        error = "Could not read cell while writing delimited output: " +
                                maybe_scalar.status().ToString();
                        return false;
                    }
                    scalar = maybe_scalar.ValueOrDie();
                    break;
                }
                remaining -= chunk->length();
            }
            out << ScalarToDelimitedString(scalar, delimiter);
        }
        out << '\n';
    }

    if (!out) {
        error = "Delimited write failed for '" + options.output_path + "'.";
        return false;
    }
    return true;
}

bool WriteOutputDataset(const std::shared_ptr<arrow::Table>& table,
                        const DataConvertOptions& options,
                        DataConvertFormat output_format,
                        std::string& error) {
    switch (output_format) {
        case DataConvertFormat::Csv:
        case DataConvertFormat::Tsv:
            return WriteDelimited(table, options, output_format, error);
        case DataConvertFormat::Parquet:
            return WriteParquet(table, options, error);
        case DataConvertFormat::Feather:
        case DataConvertFormat::ArrowIpc:
            return WriteArrowIpc(table, options, error);
        default:
            error = "Unsupported output format. Choose " + SupportedFormatList() + ".";
            return false;
    }
}

std::string BuildManifestPath(const std::string& output_path) {
    return output_path + ".manifest.json";
}

std::string BuildSettingsHashInput(const DataConvertOptions& options,
                                   DataConvertFormat input_format,
                                   DataConvertFormat output_format) {
    std::ostringstream out;
    out << options.input_path << "|"
        << options.output_path << "|"
        << FormatName(input_format) << "|"
        << FormatName(output_format) << "|"
        << options.delimiter << "|"
        << options.auto_detect_delimiter << "|"
        << options.has_header << "|"
        << options.allow_newlines_in_values << "|"
        << options.skip_rows << "|"
        << options.parquet_compression << "|"
        << options.row_group_size;
    return out.str();
}

std::string SimpleSettingsHash(const DataConvertOptions& options,
                               DataConvertFormat input_format,
                               DataConvertFormat output_format) {
    const auto value = std::hash<std::string>{}(
        BuildSettingsHashInput(options, input_format, output_format));
    std::ostringstream out;
    out << std::hex << value;
    return out.str();
}

bool WriteManifest(const DataConvertOptions& options,
                   const DataConvertResult& result,
                   DataConvertFormat input_format,
                   DataConvertFormat output_format,
                   std::string& error) {
    const std::filesystem::path input_path(options.input_path);
    const std::filesystem::path output_path(options.output_path);

    nlohmann::json manifest = {
        {"node", "DataConvert"},
        {"version", 2},
        {"input_path", options.input_path},
        {"input_format", FormatName(input_format)},
        {"output_path", options.output_path},
        {"output_format", FormatName(output_format)},
        {"rows_read", result.rows_read},
        {"rows_written", result.rows_written},
        {"columns", result.columns},
        {"compression", options.parquet_compression},
        {"delimiter", std::string(1, result.detected_delimiter)},
        {"delimiter_mode", options.auto_detect_delimiter ? "auto" : "manual"},
        {"allow_newlines_in_values", options.allow_newlines_in_values},
        {"settings_hash", SimpleSettingsHash(options, input_format, output_format)},
        {"created_at", NowIsoLikeUtc()}
    };

    std::error_code ec;
    if (!options.input_path.empty() && std::filesystem::exists(input_path, ec)) {
        manifest["input_size"] =
            static_cast<int64_t>(std::filesystem::file_size(input_path, ec));
        const auto write_time = std::filesystem::last_write_time(input_path, ec);
        if (!ec) {
            manifest["input_modified_time_native"] =
                static_cast<int64_t>(write_time.time_since_epoch().count());
        }
    }
    if (std::filesystem::exists(output_path, ec)) {
        manifest["output_size"] =
            static_cast<int64_t>(std::filesystem::file_size(output_path, ec));
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

bool TryUseFreshOutput(const DataConvertOptions& options,
                       DataConvertFormat input_format,
                       DataConvertFormat output_format,
                       DataConvertResult& result) {
    if (!options.write_manifest || options.input_path.empty()) {
        return false;
    }

    const std::filesystem::path input_path(options.input_path);
    const std::filesystem::path output_path(options.output_path);
    const std::filesystem::path manifest_path(BuildManifestPath(options.output_path));
    std::error_code ec;
    if (!std::filesystem::exists(input_path, ec) ||
        !std::filesystem::exists(output_path, ec) ||
        !std::filesystem::exists(manifest_path, ec)) {
        return false;
    }

    std::ifstream in(manifest_path, std::ios::binary);
    if (!in) {
        return false;
    }

    nlohmann::json manifest;
    try {
        in >> manifest;
    } catch (...) {
        return false;
    }

    const int manifest_version = manifest.value("version", 0);
    if (manifest.value("node", "") != "DataConvert" ||
        manifest_version < 1 ||
        manifest.value("input_path", "") != options.input_path ||
        manifest.value("output_path", "") != options.output_path ||
        manifest.value("output_format", "") != FormatName(output_format)) {
        return false;
    }
    if (manifest_version >= 2 &&
        manifest.value("input_format", "") != FormatName(input_format)) {
        return false;
    }
    if (manifest.value("settings_hash", "") !=
        SimpleSettingsHash(options, input_format, output_format)) {
        return false;
    }

    const int64_t current_input_size = static_cast<int64_t>(
        std::filesystem::file_size(input_path, ec));
    if (ec || manifest.value("input_size", int64_t{-1}) != current_input_size) {
        return false;
    }

    const auto current_input_time = std::filesystem::last_write_time(input_path, ec);
    if (ec) {
        return false;
    }
    const int64_t current_input_time_native =
        static_cast<int64_t>(current_input_time.time_since_epoch().count());
    if (manifest.value("input_modified_time_native", int64_t{-1}) !=
        current_input_time_native) {
        return false;
    }

    const int64_t current_output_size = static_cast<int64_t>(
        std::filesystem::file_size(output_path, ec));
    if (ec || current_output_size <= 0 ||
        manifest.value("output_size", int64_t{-1}) != current_output_size) {
        return false;
    }

    result.ok = true;
    result.skipped_fresh_output = true;
    result.output_path = options.output_path;
    result.manifest_path = manifest_path.string();
    result.rows_read = manifest.value("rows_read", int64_t{0});
    result.rows_written = manifest.value("rows_written", int64_t{0});
    result.columns = manifest.value("columns", int64_t{0});
    result.bytes_written = current_output_size;
    const std::string delimiter = manifest.value("delimiter", ",");
    result.detected_delimiter = delimiter.empty() ? ',' : delimiter.front();
    return true;
}

void FillPreviewFromTable(const std::shared_ptr<arrow::Table>& table,
                          int preview_rows,
                          char detected_delimiter,
                          DataConvertPreview& preview) {
    preview.detected_delimiter = detected_delimiter;
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
        std::min<int64_t>(preview.rows, std::max(0, preview_rows));
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
}

} // namespace

DataConvertPreview DataConvertService::Preview(const DataConvertOptions& options) {
    DataConvertPreview preview;
    const DataConvertFormat input_format = ResolveInputFormat(options);
    if (!options.input_table && options.input_path.empty()) {
        preview.error = "Input file is empty. Select a supported data file first.";
        return preview;
    }
    if (!options.input_table && !std::filesystem::exists(options.input_path)) {
        preview.error = "Input file does not exist: " + options.input_path;
        return preview;
    }
    if (!IsSupportedFormat(input_format)) {
        preview.error = "Input format is not supported. Choose " +
                        SupportedFormatList() + ".";
        return preview;
    }

    std::string error;
    auto dataset = LoadInputDataset(options, input_format, error);
    if (!dataset) {
        preview.error = error;
        return preview;
    }

    const char detected_delimiter = IsDelimitedFormat(input_format)
        ? ResolveDelimiter(options, input_format)
        : ',';
    FillPreviewFromTable(dataset->GetArrowTable(), options.preview_rows,
                         detected_delimiter, preview);
    return preview;
}

DataConvertResult DataConvertService::Convert(const DataConvertOptions& options) {
    DataConvertResult result;
    const DataConvertFormat input_format = ResolveInputFormat(options);
    const DataConvertFormat output_format = ResolveOutputFormat(options);
    const std::filesystem::path output_path(options.output_path);

    if (!options.input_table && options.input_path.empty()) {
        result.error = "Input file is empty. Select a supported data file first.";
        return result;
    }
    if (!options.input_table && !std::filesystem::exists(options.input_path)) {
        result.error = "Input file does not exist: " + options.input_path;
        return result;
    }
    if (!IsSupportedFormat(input_format)) {
        result.error = "Input format is not supported. Choose " +
                       SupportedFormatList() + ".";
        return result;
    }
    if (options.output_path.empty()) {
        result.error = "Output path is empty. Choose where to write the converted file.";
        return result;
    }
    if (!IsSupportedFormat(output_format)) {
        result.error = "Output format is not supported. Choose " +
                       SupportedFormatList() + ".";
        return result;
    }

    if (TryUseFreshOutput(options, input_format, output_format, result)) {
        spdlog::info("DataConvert: skipped fresh output '{}'", options.output_path);
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
                           output_path.parent_path().string() + "': " +
                           ec.message();
            return result;
        }
    }

    std::string error;
    auto dataset = LoadInputDataset(options, input_format, error);
    if (!dataset) {
        result.error = error;
        return result;
    }
    auto table = dataset->GetArrowTable();
    result.detected_delimiter = IsDelimitedFormat(input_format)
        ? ResolveDelimiter(options, input_format)
        : ',';

    if (!WriteOutputDataset(table, options, output_format, error)) {
        result.error = error;
        return result;
    }

    result.rows_read = table->num_rows();
    result.rows_written = table->num_rows();
    result.columns = table->num_columns();
    result.output_path = options.output_path;
    if (std::filesystem::exists(output_path, ec)) {
        result.bytes_written = static_cast<int64_t>(
            std::filesystem::file_size(output_path, ec));
    }

    if (options.write_manifest) {
        std::string manifest_error;
        if (!WriteManifest(options, result, input_format, output_format,
                           manifest_error)) {
            result.error = manifest_error;
            return result;
        }
        result.manifest_path = BuildManifestPath(options.output_path);
    }

    result.ok = true;
    spdlog::info("DataConvert: converted '{}' ({}) -> '{}' ({}) ({} rows, {} columns)",
                 options.input_path, FormatName(input_format),
                 options.output_path, FormatName(output_format),
                 result.rows_written, result.columns);
    return result;
}

DataConvertPreview DataConvertService::PreviewCsv(const DataConvertOptions& options) {
    DataConvertOptions csv_options = options;
    if (IsAutoFormat(csv_options.input_format)) {
        csv_options.input_format = FormatName(ResolveInputFormat(options));
    }
    return Preview(csv_options);
}

DataConvertResult DataConvertService::ConvertCsvToParquet(
    const DataConvertOptions& options) {
    DataConvertOptions parquet_options = options;
    if (IsAutoFormat(parquet_options.input_format)) {
        parquet_options.input_format = FormatName(ResolveInputFormat(options));
    }
    parquet_options.output_format = "parquet";
    return Convert(parquet_options);
}

} // namespace cyxwiz
