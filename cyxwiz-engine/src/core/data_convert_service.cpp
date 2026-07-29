#include "data_convert_service.h"

#include "arrow_dataset.h"
#include "csv_ingestion_options.h"
#include "error_codes.h"

#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/writer.h>
#include <arrow/json/api.h>
#include <arrow/scalar.h>
#include <nlohmann/json.hpp>
#include <parquet/arrow/writer.h>
#include <spdlog/spdlog.h>

#ifdef CYXWIZ_HAS_HDF5
#include <highfive/highfive.hpp>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <vector>

namespace cyxwiz {
namespace {

enum class DataConvertFormat {
    Unknown,
    Csv,
    Tsv,
    JsonLines,
    Text,
    Arff,
    Numpy,
    Hdf5,
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
    if (name == "json" || name == "jsonl" || name == "ndjson") {
        return DataConvertFormat::JsonLines;
    }
    if (name == "txt" || name == "text") return DataConvertFormat::Text;
    if (name == "arff") return DataConvertFormat::Arff;
    if (name == "npy") return DataConvertFormat::Numpy;
    if (name == "h5" || name == "hdf5" || name == "hdf") {
        return DataConvertFormat::Hdf5;
    }
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
        case DataConvertFormat::JsonLines: return "jsonl";
        case DataConvertFormat::Text: return "txt";
        case DataConvertFormat::Arff: return "arff";
        case DataConvertFormat::Numpy: return "npy";
        case DataConvertFormat::Hdf5: return "hdf5";
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
    return "csv, tsv, json/jsonl, txt, arff, npy, hdf5, parquet, feather, arrow, or ipc";
}

bool OutputExtensionMatchesFormat(const std::filesystem::path& output_path,
                                  DataConvertFormat output_format) {
    std::string extension = output_path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    switch (output_format) {
        case DataConvertFormat::Csv:
            return extension == ".csv";
        case DataConvertFormat::Tsv:
            return extension == ".tsv";
        case DataConvertFormat::JsonLines:
            return extension == ".json" || extension == ".jsonl" ||
                   extension == ".ndjson";
        case DataConvertFormat::Text:
            return extension == ".txt" || extension == ".text";
        case DataConvertFormat::Arff:
            return extension == ".arff";
        case DataConvertFormat::Numpy:
            return extension == ".npy";
        case DataConvertFormat::Hdf5:
            return extension == ".h5" || extension == ".hdf5" ||
                   extension == ".hdf";
        case DataConvertFormat::Parquet:
            return extension == ".parquet" || extension == ".pq";
        case DataConvertFormat::Feather:
            return extension == ".feather" || extension == ".fea";
        case DataConvertFormat::ArrowIpc:
            return extension == ".arrow" || extension == ".ipc";
        case DataConvertFormat::Unknown:
            return true;
    }
    return true;
}

std::string ExpectedExtensionsForFormat(DataConvertFormat output_format) {
    switch (output_format) {
        case DataConvertFormat::Csv:
            return ".csv";
        case DataConvertFormat::Tsv:
            return ".tsv";
        case DataConvertFormat::JsonLines:
            return ".jsonl, .json, or .ndjson";
        case DataConvertFormat::Text:
            return ".txt";
        case DataConvertFormat::Arff:
            return ".arff";
        case DataConvertFormat::Numpy:
            return ".npy";
        case DataConvertFormat::Hdf5:
            return ".h5 or .hdf5";
        case DataConvertFormat::Parquet:
            return ".parquet or .pq";
        case DataConvertFormat::Feather:
            return ".feather or .fea";
        case DataConvertFormat::ArrowIpc:
            return ".arrow or .ipc";
        case DataConvertFormat::Unknown:
            return "";
    }
    return "";
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

std::shared_ptr<ArrowDataset> LoadDelimitedAttempt(
    const DataConvertOptions& options,
    DataConvertFormat format,
    bool promote_inferred_integers,
    std::string& error) {
    auto convert_options = MakeTabularCsvConvertOptions(
        DefaultTabularMissingValueTokens(), options.decimal_point);
    size_t promoted_integer_columns = 0;
    auto reader_result = MakeStableCsvStreamingReader(
        options.input_path,
        BuildReadOptions(options),
        BuildParseOptions(options, format),
        convert_options,
        &promoted_integer_columns,
        promote_inferred_integers);
    if (!reader_result.ok()) {
        error = "Delimited file reader creation failed: " +
            reader_result.status().ToString();
        return nullptr;
    }
    if (promoted_integer_columns > 0) {
        spdlog::info(
            "DataConvert: promoted {} inferred integer columns to float64 for stable full-source parsing",
            promoted_integer_columns);
    }

    auto reader = reader_result.ValueOrDie();
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    while (true) {
        std::shared_ptr<arrow::RecordBatch> batch;
        auto status = reader->ReadNext(&batch);
        if (!status.ok()) {
            error = "Delimited file read failed: " + status.ToString();
            return nullptr;
        }
        if (!batch) break;
        batches.push_back(std::move(batch));
    }
    auto table_result = arrow::Table::FromRecordBatches(
        reader->schema(), batches);
    if (!table_result.ok()) {
        error = "Delimited table assembly failed: " +
            table_result.status().ToString();
        return nullptr;
    }
    return std::make_shared<ArrowDataset>(
        table_result.ValueOrDie(), "data_convert_input");
}

std::shared_ptr<ArrowDataset> LoadDelimited(
    const DataConvertOptions& options, DataConvertFormat format,
    std::string& error) {
    std::string first_error;
    auto result = LoadDelimitedAttempt(options, format, false, first_error);
    if (result) return result;
    result = LoadDelimitedAttempt(options, format, true, error);
    if (!result) {
        error = first_error + "; numeric fallback failed: " + error;
    } else {
        spdlog::warn("DataConvert: numeric schema fallback succeeded after: {}",
                     first_error);
    }
    return result;
}

std::shared_ptr<ArrowDataset> LoadJsonLines(const DataConvertOptions& options,
                                            std::string& error) {
    auto maybe_input = arrow::io::ReadableFile::Open(options.input_path);
    if (!maybe_input.ok()) {
        error = "Could not open JSON Lines input '" + options.input_path +
                "': " + maybe_input.status().ToString();
        return nullptr;
    }

    auto read_options = arrow::json::ReadOptions::Defaults();
    auto parse_options = arrow::json::ParseOptions::Defaults();
    parse_options.newlines_in_values = options.allow_newlines_in_values;

    auto maybe_reader = arrow::json::TableReader::Make(
        arrow::default_memory_pool(),
        maybe_input.ValueOrDie(),
        read_options,
        parse_options);
    if (!maybe_reader.ok()) {
        error = "JSON Lines reader creation failed for '" + options.input_path +
                "': " + maybe_reader.status().ToString();
        return nullptr;
    }

    auto maybe_table = maybe_reader.ValueOrDie()->Read();
    if (!maybe_table.ok()) {
        error = "JSON Lines input read failed for '" + options.input_path +
                "'. Expected newline-delimited JSON objects. Error: " +
                maybe_table.status().ToString();
        return nullptr;
    }

    return std::make_shared<ArrowDataset>(maybe_table.ValueOrDie(),
                                          "data_convert_input");
}

std::shared_ptr<ArrowDataset> LoadText(const DataConvertOptions& options,
                                       std::string& error) {
    std::ifstream input(options.input_path, std::ios::binary);
    if (!input) {
        error = "Could not open text input '" + options.input_path + "'.";
        return nullptr;
    }

    arrow::StringBuilder builder;
    std::string line;
    int skipped = 0;
    const int skip_rows = std::max(0, options.skip_rows);
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (skipped < skip_rows) {
            ++skipped;
            continue;
        }
        auto status = builder.Append(line);
        if (!status.ok()) {
            error = "Text input append failed: " + status.ToString();
            return nullptr;
        }
    }

    std::shared_ptr<arrow::Array> text_array;
    auto finish_status = builder.Finish(&text_array);
    if (!finish_status.ok()) {
        error = "Text input build failed: " + finish_status.ToString();
        return nullptr;
    }

    auto table = arrow::Table::Make(
        arrow::schema({arrow::field("text", arrow::utf8())}),
        {text_array});
    return std::make_shared<ArrowDataset>(table, "data_convert_input");
}

struct ArffAttribute {
    std::string name;
    bool numeric = false;
};

bool StartsWithCaseInsensitive(const std::string& value,
                               const std::string& prefix) {
    if (value.size() < prefix.size()) return false;
    return LowerAscii(value.substr(0, prefix.size())) == LowerAscii(prefix);
}

std::string UnquoteArffToken(const std::string& value) {
    const std::string trimmed = TrimAscii(value);
    if (trimmed.size() >= 2 &&
        ((trimmed.front() == '\'' && trimmed.back() == '\'') ||
         (trimmed.front() == '"' && trimmed.back() == '"'))) {
        return trimmed.substr(1, trimmed.size() - 2);
    }
    return trimmed;
}

std::vector<std::string> SplitArffDataRow(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    bool in_single_quote = false;
    bool in_double_quote = false;
    for (char c : line) {
        if (c == '\'' && !in_double_quote) {
            in_single_quote = !in_single_quote;
            current.push_back(c);
        } else if (c == '"' && !in_single_quote) {
            in_double_quote = !in_double_quote;
            current.push_back(c);
        } else if (c == ',' && !in_single_quote && !in_double_quote) {
            fields.push_back(UnquoteArffToken(current));
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    fields.push_back(UnquoteArffToken(current));
    return fields;
}

bool ParseArffAttribute(const std::string& line,
                        ArffAttribute& attribute,
                        std::string& error) {
    std::string rest = TrimAscii(line.substr(std::string("@attribute").size()));
    if (rest.empty()) {
        error = "ARFF attribute line is missing a name.";
        return false;
    }

    std::string name;
    if (rest.front() == '\'' || rest.front() == '"') {
        const char quote = rest.front();
        const size_t end_quote = rest.find(quote, 1);
        if (end_quote == std::string::npos) {
            error = "ARFF attribute has an unterminated quoted name.";
            return false;
        }
        name = rest.substr(1, end_quote - 1);
        rest = TrimAscii(rest.substr(end_quote + 1));
    } else {
        const size_t split = rest.find_first_of(" \t");
        if (split == std::string::npos) {
            error = "ARFF attribute line is missing a type.";
            return false;
        }
        name = rest.substr(0, split);
        rest = TrimAscii(rest.substr(split + 1));
    }

    if (name.empty() || rest.empty()) {
        error = "ARFF attribute line is missing a name or type.";
        return false;
    }

    const std::string type = LowerAscii(rest);
    attribute.name = name;
    attribute.numeric =
        type == "numeric" || type == "real" || type == "integer";
    return true;
}

std::shared_ptr<ArrowDataset> LoadArff(const DataConvertOptions& options,
                                       std::string& error) {
    std::ifstream input(options.input_path, std::ios::binary);
    if (!input) {
        error = "Could not open ARFF input '" + options.input_path + "'.";
        return nullptr;
    }

    std::vector<ArffAttribute> attributes;
    std::vector<std::vector<std::string>> rows;
    bool in_data = false;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const std::string trimmed = TrimAscii(line);
        if (trimmed.empty() || trimmed.front() == '%') {
            continue;
        }

        if (!in_data) {
            if (StartsWithCaseInsensitive(trimmed, "@attribute")) {
                ArffAttribute attribute;
                if (!ParseArffAttribute(trimmed, attribute, error)) {
                    return nullptr;
                }
                attributes.push_back(attribute);
            } else if (StartsWithCaseInsensitive(trimmed, "@data")) {
                in_data = true;
            }
            continue;
        }

        auto fields = SplitArffDataRow(trimmed);
        if (fields.size() != attributes.size()) {
            error = "ARFF data row has " + std::to_string(fields.size()) +
                    " values but " + std::to_string(attributes.size()) +
                    " attributes were declared.";
            return nullptr;
        }
        rows.push_back(std::move(fields));
    }

    if (attributes.empty()) {
        error = "ARFF input has no @attribute declarations.";
        return nullptr;
    }
    if (!in_data) {
        error = "ARFF input is missing @data section.";
        return nullptr;
    }

    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    fields.reserve(attributes.size());
    arrays.reserve(attributes.size());

    for (size_t column = 0; column < attributes.size(); ++column) {
        const auto& attribute = attributes[column];
        if (attribute.numeric) {
            arrow::DoubleBuilder builder;
            for (const auto& row : rows) {
                const std::string value = TrimAscii(row[column]);
                if (value.empty() || value == "?") {
                    auto status = builder.AppendNull();
                    if (!status.ok()) {
                        error = "ARFF numeric null append failed: " +
                                status.ToString();
                        return nullptr;
                    }
                    continue;
                }
                try {
                    auto status = builder.Append(std::stod(value));
                    if (!status.ok()) {
                        error = "ARFF numeric append failed: " +
                                status.ToString();
                        return nullptr;
                    }
                } catch (const std::exception&) {
                    error = "ARFF numeric value could not be parsed: " + value;
                    return nullptr;
                }
            }
            std::shared_ptr<arrow::Array> array;
            auto status = builder.Finish(&array);
            if (!status.ok()) {
                error = "ARFF numeric column build failed: " +
                        status.ToString();
                return nullptr;
            }
            fields.push_back(arrow::field(attribute.name, arrow::float64()));
            arrays.push_back(array);
        } else {
            arrow::StringBuilder builder;
            for (const auto& row : rows) {
                const std::string value = row[column];
                auto status = value == "?" ? builder.AppendNull()
                                           : builder.Append(value);
                if (!status.ok()) {
                    error = "ARFF string append failed: " + status.ToString();
                    return nullptr;
                }
            }
            std::shared_ptr<arrow::Array> array;
            auto status = builder.Finish(&array);
            if (!status.ok()) {
                error = "ARFF string column build failed: " +
                        status.ToString();
                return nullptr;
            }
            fields.push_back(arrow::field(attribute.name, arrow::utf8()));
            arrays.push_back(array);
        }
    }

    auto table = arrow::Table::Make(arrow::schema(fields), arrays);
    return std::make_shared<ArrowDataset>(table, "data_convert_input");
}

uint16_t ReadLittleEndianU16(const unsigned char* data) {
    return static_cast<uint16_t>(data[0]) |
           static_cast<uint16_t>(data[1] << 8);
}

uint32_t ReadLittleEndianU32(const unsigned char* data) {
    return static_cast<uint32_t>(data[0]) |
           (static_cast<uint32_t>(data[1]) << 8) |
           (static_cast<uint32_t>(data[2]) << 16) |
           (static_cast<uint32_t>(data[3]) << 24);
}

template <typename T>
T ReadPlainValue(const char* data) {
    T value{};
    std::memcpy(&value, data, sizeof(T));
    return value;
}

bool ParseNpyHeader(const std::string& header,
                    std::string& descr,
                    bool& fortran_order,
                    std::vector<int64_t>& shape,
                    std::string& error) {
    const auto descr_pos = header.find("'descr'");
    const auto descr_colon = header.find(':', descr_pos);
    const auto descr_quote = header.find('\'', descr_colon);
    const auto descr_end = header.find('\'', descr_quote + 1);
    if (descr_pos == std::string::npos || descr_colon == std::string::npos ||
        descr_quote == std::string::npos || descr_end == std::string::npos) {
        error = "NumPy .npy header is missing 'descr'.";
        return false;
    }
    descr = header.substr(descr_quote + 1, descr_end - descr_quote - 1);

    const auto fortran_pos = header.find("'fortran_order'");
    const auto fortran_colon = header.find(':', fortran_pos);
    if (fortran_pos == std::string::npos ||
        fortran_colon == std::string::npos) {
        error = "NumPy .npy header is missing 'fortran_order'.";
        return false;
    }
    const std::string fortran_tail = header.substr(fortran_colon + 1, 16);
    fortran_order = fortran_tail.find("True") != std::string::npos;

    const auto shape_pos = header.find("'shape'");
    const auto shape_open = header.find('(', shape_pos);
    const auto shape_close = header.find(')', shape_open);
    if (shape_pos == std::string::npos || shape_open == std::string::npos ||
        shape_close == std::string::npos) {
        error = "NumPy .npy header is missing 'shape'.";
        return false;
    }
    std::string shape_text =
        header.substr(shape_open + 1, shape_close - shape_open - 1);
    std::stringstream shape_stream(shape_text);
    std::string token;
    while (std::getline(shape_stream, token, ',')) {
        token = TrimAscii(token);
        if (token.empty()) continue;
        try {
            shape.push_back(std::stoll(token));
        } catch (const std::exception&) {
            error = "NumPy .npy shape value could not be parsed: " + token;
            return false;
        }
    }
    if (shape.empty() || shape.size() > 2) {
        error = "NumPy .npy input supports only 1D or 2D arrays.";
        return false;
    }
    for (int64_t dim : shape) {
        if (dim < 0) {
            error = "NumPy .npy shape contains a negative dimension.";
            return false;
        }
    }
    return true;
}

double ReadNpyNumericValue(const std::string& descr,
                           const char* data,
                           std::string& error) {
    if (descr == "<f8" || descr == "|f8") {
        return ReadPlainValue<double>(data);
    }
    if (descr == "<f4" || descr == "|f4") {
        return static_cast<double>(ReadPlainValue<float>(data));
    }
    if (descr == "<i8" || descr == "|i8") {
        return static_cast<double>(ReadPlainValue<int64_t>(data));
    }
    if (descr == "<i4" || descr == "|i4") {
        return static_cast<double>(ReadPlainValue<int32_t>(data));
    }
    if (descr == "<u1" || descr == "|u1") {
        return static_cast<double>(ReadPlainValue<uint8_t>(data));
    }
    error = "NumPy .npy dtype is not supported: " + descr +
            ". Supported dtypes: <f8, <f4, <i8, <i4, <u1.";
    return 0.0;
}

size_t NpyElementSize(const std::string& descr) {
    if (descr == "<f8" || descr == "|f8" ||
        descr == "<i8" || descr == "|i8") {
        return 8;
    }
    if (descr == "<f4" || descr == "|f4" ||
        descr == "<i4" || descr == "|i4") {
        return 4;
    }
    if (descr == "<u1" || descr == "|u1") {
        return 1;
    }
    return 0;
}

std::shared_ptr<ArrowDataset> LoadNumpy(const DataConvertOptions& options,
                                        std::string& error) {
    std::ifstream input(options.input_path, std::ios::binary);
    if (!input) {
        error = "Could not open NumPy input '" + options.input_path + "'.";
        return nullptr;
    }

    unsigned char prefix[8] = {};
    input.read(reinterpret_cast<char*>(prefix), 8);
    if (input.gcount() != 8 ||
        std::string(reinterpret_cast<char*>(prefix), 6) != "\x93NUMPY") {
        error = "NumPy input is not a valid .npy file.";
        return nullptr;
    }

    const unsigned char major = prefix[6];
    uint32_t header_length = 0;
    if (major == 1) {
        unsigned char length_bytes[2] = {};
        input.read(reinterpret_cast<char*>(length_bytes), 2);
        if (input.gcount() != 2) {
            error = "NumPy .npy header length is truncated.";
            return nullptr;
        }
        header_length = ReadLittleEndianU16(length_bytes);
    } else if (major == 2 || major == 3) {
        unsigned char length_bytes[4] = {};
        input.read(reinterpret_cast<char*>(length_bytes), 4);
        if (input.gcount() != 4) {
            error = "NumPy .npy header length is truncated.";
            return nullptr;
        }
        header_length = ReadLittleEndianU32(length_bytes);
    } else {
        error = "Unsupported NumPy .npy version.";
        return nullptr;
    }

    std::string header(header_length, '\0');
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (static_cast<uint32_t>(input.gcount()) != header_length) {
        error = "NumPy .npy header is truncated.";
        return nullptr;
    }

    std::string descr;
    bool fortran_order = false;
    std::vector<int64_t> shape;
    if (!ParseNpyHeader(header, descr, fortran_order, shape, error)) {
        return nullptr;
    }
    if (fortran_order) {
        error = "NumPy .npy Fortran-order arrays are not supported yet.";
        return nullptr;
    }
    const size_t element_size = NpyElementSize(descr);
    if (element_size == 0) {
        error = "NumPy .npy dtype is not supported: " + descr;
        return nullptr;
    }

    const int64_t rows = shape[0];
    const int64_t columns = shape.size() == 1 ? 1 : shape[1];
    const int64_t total_values = rows * columns;
    std::vector<char> raw(static_cast<size_t>(total_values) * element_size);
    input.read(raw.data(), static_cast<std::streamsize>(raw.size()));
    if (static_cast<size_t>(input.gcount()) != raw.size()) {
        error = "NumPy .npy data payload is truncated.";
        return nullptr;
    }

    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    fields.reserve(static_cast<size_t>(columns));
    arrays.reserve(static_cast<size_t>(columns));
    for (int64_t column = 0; column < columns; ++column) {
        arrow::DoubleBuilder builder;
        for (int64_t row = 0; row < rows; ++row) {
            const int64_t value_index = row * columns + column;
            const char* value_data =
                raw.data() + static_cast<size_t>(value_index) * element_size;
            const double value = ReadNpyNumericValue(descr, value_data, error);
            if (!error.empty()) {
                return nullptr;
            }
            auto status = builder.Append(value);
            if (!status.ok()) {
                error = "NumPy column append failed: " + status.ToString();
                return nullptr;
            }
        }
        std::shared_ptr<arrow::Array> array;
        auto status = builder.Finish(&array);
        if (!status.ok()) {
            error = "NumPy column build failed: " + status.ToString();
            return nullptr;
        }
        fields.push_back(arrow::field(
            columns == 1 ? "value" : "col_" + std::to_string(column),
            arrow::float64()));
        arrays.push_back(array);
    }

    auto table = arrow::Table::Make(arrow::schema(fields), arrays);
    return std::make_shared<ArrowDataset>(table, "data_convert_input");
}

std::shared_ptr<ArrowDataset> LoadHdf5(const DataConvertOptions& options,
                                       std::string& error) {
#ifdef CYXWIZ_HAS_HDF5
    try {
        HighFive::File file(options.input_path, HighFive::File::ReadOnly);
        std::string dataset_name = "data";
        if (!file.exist(dataset_name)) {
            const std::vector<std::string> common_names = {
                "features", "X", "x", "inputs", "values"
            };
            bool found = false;
            for (const auto& candidate : common_names) {
                if (file.exist(candidate)) {
                    dataset_name = candidate;
                    found = true;
                    break;
                }
            }
            if (!found) {
                error = "HDF5 input has no supported table dataset. Expected '/data' or one of: features, X, x, inputs, values.";
                return nullptr;
            }
        }

        auto dataset = file.getDataSet(dataset_name);
        const auto dims = dataset.getDimensions();
        if (dims.empty() || dims.size() > 2) {
            error = "HDF5 DataConvert input supports only 1D or 2D numeric datasets.";
            return nullptr;
        }
        const size_t rows = dims[0];
        const size_t columns = dims.size() == 1 ? 1 : dims[1];

        std::vector<std::shared_ptr<arrow::Field>> fields;
        std::vector<std::shared_ptr<arrow::Array>> arrays;
        fields.reserve(columns);
        arrays.reserve(columns);

        if (dims.size() == 1) {
            std::vector<double> values;
            dataset.read(values);
            arrow::DoubleBuilder builder;
            for (double value : values) {
                auto status = builder.Append(value);
                if (!status.ok()) {
                    error = "HDF5 value append failed: " + status.ToString();
                    return nullptr;
                }
            }
            std::shared_ptr<arrow::Array> array;
            auto status = builder.Finish(&array);
            if (!status.ok()) {
                error = "HDF5 column build failed: " + status.ToString();
                return nullptr;
            }
            fields.push_back(arrow::field("value", arrow::float64()));
            arrays.push_back(array);
        } else {
            std::vector<std::vector<double>> values;
            dataset.read(values);
            for (size_t column = 0; column < columns; ++column) {
                arrow::DoubleBuilder builder;
                for (size_t row = 0; row < rows; ++row) {
                    auto status = builder.Append(values[row][column]);
                    if (!status.ok()) {
                        error = "HDF5 value append failed: " + status.ToString();
                        return nullptr;
                    }
                }
                std::shared_ptr<arrow::Array> array;
                auto status = builder.Finish(&array);
                if (!status.ok()) {
                    error = "HDF5 column build failed: " + status.ToString();
                    return nullptr;
                }
                fields.push_back(arrow::field("col_" + std::to_string(column),
                                              arrow::float64()));
                arrays.push_back(array);
            }
        }

        auto table = arrow::Table::Make(arrow::schema(fields), arrays);
        return std::make_shared<ArrowDataset>(table, "data_convert_input");
    } catch (const std::exception& e) {
        error = "HDF5 input read failed: " + std::string(e.what());
        return nullptr;
    }
#else
    (void)options;
    error = "HDF5 support is not compiled into this build.";
    return nullptr;
#endif
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
    if (input_format == DataConvertFormat::JsonLines) {
        return LoadJsonLines(options, error);
    }
    if (input_format == DataConvertFormat::Text) {
        return LoadText(options, error);
    }
    if (input_format == DataConvertFormat::Arff) {
        return LoadArff(options, error);
    }
    if (input_format == DataConvertFormat::Numpy) {
        return LoadNumpy(options, error);
    }
    if (input_format == DataConvertFormat::Hdf5) {
        return LoadHdf5(options, error);
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

nlohmann::json ScalarToJsonValue(const std::shared_ptr<arrow::Scalar>& scalar) {
    if (!scalar || !scalar->is_valid) {
        return nullptr;
    }

    switch (scalar->type->id()) {
        case arrow::Type::BOOL:
            return std::static_pointer_cast<arrow::BooleanScalar>(scalar)->value;
        case arrow::Type::INT8:
            return std::static_pointer_cast<arrow::Int8Scalar>(scalar)->value;
        case arrow::Type::INT16:
            return std::static_pointer_cast<arrow::Int16Scalar>(scalar)->value;
        case arrow::Type::INT32:
            return std::static_pointer_cast<arrow::Int32Scalar>(scalar)->value;
        case arrow::Type::INT64:
            return std::static_pointer_cast<arrow::Int64Scalar>(scalar)->value;
        case arrow::Type::UINT8:
            return std::static_pointer_cast<arrow::UInt8Scalar>(scalar)->value;
        case arrow::Type::UINT16:
            return std::static_pointer_cast<arrow::UInt16Scalar>(scalar)->value;
        case arrow::Type::UINT32:
            return std::static_pointer_cast<arrow::UInt32Scalar>(scalar)->value;
        case arrow::Type::UINT64:
            return std::static_pointer_cast<arrow::UInt64Scalar>(scalar)->value;
        case arrow::Type::FLOAT:
            return std::static_pointer_cast<arrow::FloatScalar>(scalar)->value;
        case arrow::Type::DOUBLE:
            return std::static_pointer_cast<arrow::DoubleScalar>(scalar)->value;
        case arrow::Type::STRING:
            return std::static_pointer_cast<arrow::StringScalar>(scalar)->ToString();
        case arrow::Type::LARGE_STRING:
            return std::static_pointer_cast<arrow::LargeStringScalar>(scalar)->ToString();
        default:
            return scalar->ToString();
    }
}

bool WriteDelimited(const std::shared_ptr<arrow::Table>& table,
                    const DataConvertOptions& options,
                    DataConvertFormat output_format,
                    std::string& error) {
    const char delimiter =
        output_format == DataConvertFormat::Tsv ? '\t' : ',';
    auto maybe_output = arrow::io::FileOutputStream::Open(options.output_path);
    if (!maybe_output.ok()) {
        error = "Could not open delimited output '" + options.output_path + "'.";
        return false;
    }

    auto output = maybe_output.ValueOrDie();
    auto write_options = arrow::csv::WriteOptions::Defaults();
    write_options.delimiter = delimiter;
    write_options.include_header = true;
    const auto write_status =
        arrow::csv::WriteCSV(*table, write_options, output.get());
    if (!write_status.ok()) {
        error = "Delimited write failed for '" + options.output_path + "': " +
                write_status.ToString();
        return false;
    }

    const auto close_status = output->Close();
    if (!close_status.ok()) {
        error = "Delimited output close failed for '" + options.output_path +
                "': " + close_status.ToString();
        return false;
    }
    return true;
}

bool WriteJsonLines(const std::shared_ptr<arrow::Table>& table,
                    const DataConvertOptions& options,
                    std::string& error) {
    std::ofstream out(options.output_path, std::ios::binary);
    if (!out) {
        error = "Could not open JSON Lines output '" + options.output_path + "'.";
        return false;
    }

    for (int64_t row = 0; row < table->num_rows(); ++row) {
        nlohmann::json object = nlohmann::json::object();
        for (int column_index = 0; column_index < table->num_columns();
             ++column_index) {
            const auto field = table->schema()->field(column_index);
            const auto column = table->column(column_index);
            std::shared_ptr<arrow::Scalar> scalar;
            int64_t remaining = row;
            for (const auto& chunk : column->chunks()) {
                if (remaining < chunk->length()) {
                    auto maybe_scalar =
                        chunk->GetScalar(static_cast<int>(remaining));
                    if (!maybe_scalar.ok()) {
                        error = "Could not read cell while writing JSON Lines output: " +
                                maybe_scalar.status().ToString();
                        return false;
                    }
                    scalar = maybe_scalar.ValueOrDie();
                    break;
                }
                remaining -= chunk->length();
            }
            object[field->name()] = ScalarToJsonValue(scalar);
        }
        out << object.dump() << '\n';
    }

    if (!out) {
        error = "JSON Lines write failed for '" + options.output_path + "'.";
        return false;
    }
    return true;
}

bool WriteText(const std::shared_ptr<arrow::Table>& table,
               const DataConvertOptions& options,
               std::string& error) {
    if (table->num_columns() != 1) {
        error = "Text output requires exactly one column. Use TSV or CSV for multi-column tables.";
        return false;
    }

    std::ofstream out(options.output_path, std::ios::binary);
    if (!out) {
        error = "Could not open text output '" + options.output_path + "'.";
        return false;
    }

    const auto column = table->column(0);
    for (int64_t row = 0; row < table->num_rows(); ++row) {
        int64_t remaining = row;
        std::shared_ptr<arrow::Scalar> scalar;
        for (const auto& chunk : column->chunks()) {
            if (remaining < chunk->length()) {
                auto maybe_scalar = chunk->GetScalar(static_cast<int>(remaining));
                if (!maybe_scalar.ok()) {
                    error = "Could not read cell while writing text output: " +
                            maybe_scalar.status().ToString();
                    return false;
                }
                scalar = maybe_scalar.ValueOrDie();
                break;
            }
            remaining -= chunk->length();
        }
        out << ScalarToPreviewString(scalar) << '\n';
    }

    if (!out) {
        error = "Text write failed for '" + options.output_path + "'.";
        return false;
    }
    return true;
}

bool IsNumericArrowType(arrow::Type::type type_id) {
    switch (type_id) {
        case arrow::Type::INT8:
        case arrow::Type::INT16:
        case arrow::Type::INT32:
        case arrow::Type::INT64:
        case arrow::Type::UINT8:
        case arrow::Type::UINT16:
        case arrow::Type::UINT32:
        case arrow::Type::UINT64:
        case arrow::Type::FLOAT:
        case arrow::Type::DOUBLE:
            return true;
        default:
            return false;
    }
}

std::string EscapeArffName(const std::string& value) {
    const bool needs_quotes =
        value.empty() ||
        value.find_first_of(" \t,{}%'\"") != std::string::npos;
    if (!needs_quotes) {
        return value;
    }

    std::string escaped = "'";
    for (char c : value) {
        if (c == '\'') {
            escaped += "\\'";
        } else {
            escaped.push_back(c);
        }
    }
    escaped.push_back('\'');
    return escaped;
}

std::string EscapeArffValue(const std::string& value) {
    const bool needs_quotes =
        value.empty() ||
        value.find_first_of(" \t,{}%'\"") != std::string::npos;
    if (!needs_quotes) {
        return value;
    }

    std::string escaped = "'";
    for (char c : value) {
        if (c == '\'') {
            escaped += "\\'";
        } else {
            escaped.push_back(c);
        }
    }
    escaped.push_back('\'');
    return escaped;
}

bool WriteArff(const std::shared_ptr<arrow::Table>& table,
               const DataConvertOptions& options,
               std::string& error) {
    std::ofstream out(options.output_path, std::ios::binary);
    if (!out) {
        error = "Could not open ARFF output '" + options.output_path + "'.";
        return false;
    }

    out << "% Generated by CyxWiz DataConvert\n";
    out << "@RELATION data_convert\n\n";
    for (int column_index = 0; column_index < table->num_columns();
         ++column_index) {
        const auto field = table->schema()->field(column_index);
        const char* type_name =
            IsNumericArrowType(field->type()->id()) ? "NUMERIC" : "STRING";
        out << "@ATTRIBUTE " << EscapeArffName(field->name()) << ' '
            << type_name << '\n';
    }
    out << "\n@DATA\n";

    for (int64_t row = 0; row < table->num_rows(); ++row) {
        for (int column_index = 0; column_index < table->num_columns();
             ++column_index) {
            if (column_index > 0) out << ',';
            const auto column = table->column(column_index);
            std::shared_ptr<arrow::Scalar> scalar;
            int64_t remaining = row;
            for (const auto& chunk : column->chunks()) {
                if (remaining < chunk->length()) {
                    auto maybe_scalar =
                        chunk->GetScalar(static_cast<int>(remaining));
                    if (!maybe_scalar.ok()) {
                        error = "Could not read cell while writing ARFF output: " +
                                maybe_scalar.status().ToString();
                        return false;
                    }
                    scalar = maybe_scalar.ValueOrDie();
                    break;
                }
                remaining -= chunk->length();
            }

            if (!scalar || !scalar->is_valid) {
                out << '?';
            } else if (IsNumericArrowType(scalar->type->id())) {
                out << scalar->ToString();
            } else {
                out << EscapeArffValue(scalar->ToString());
            }
        }
        out << '\n';
    }

    if (!out) {
        error = "ARFF write failed for '" + options.output_path + "'.";
        return false;
    }
    return true;
}

double NumericScalarToDouble(const std::shared_ptr<arrow::Scalar>& scalar,
                             std::string& error) {
    if (!scalar || !scalar->is_valid) {
        return 0.0;
    }

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
            error = "NumPy output supports numeric columns only.";
            return 0.0;
    }
}

bool WriteNumpy(const std::shared_ptr<arrow::Table>& table,
                const DataConvertOptions& options,
                std::string& error) {
    for (int column_index = 0; column_index < table->num_columns();
         ++column_index) {
        const auto field = table->schema()->field(column_index);
        if (!IsNumericArrowType(field->type()->id())) {
            error = "NumPy output supports numeric columns only. Column '" +
                    field->name() + "' is " + field->type()->ToString() + ".";
            return false;
        }
    }

    std::ofstream out(options.output_path, std::ios::binary);
    if (!out) {
        error = "Could not open NumPy output '" + options.output_path + "'.";
        return false;
    }

    std::ostringstream header;
    header << "{'descr': '<f8', 'fortran_order': False, 'shape': ("
           << table->num_rows() << ", " << table->num_columns()
           << "), }";
    std::string header_text = header.str();
    const size_t preamble_size = 10;
    size_t padded_size = header_text.size() + 1;
    while ((preamble_size + padded_size) % 16 != 0) {
        ++padded_size;
    }
    header_text.append(padded_size - header_text.size() - 1, ' ');
    header_text.push_back('\n');
    if (header_text.size() > 65535) {
        error = "NumPy v1 header is too large.";
        return false;
    }

    const char magic[] = "\x93NUMPY";
    out.write(magic, 6);
    const unsigned char version[2] = {1, 0};
    out.write(reinterpret_cast<const char*>(version), 2);
    const uint16_t header_length =
        static_cast<uint16_t>(header_text.size());
    const unsigned char length_bytes[2] = {
        static_cast<unsigned char>(header_length & 0xff),
        static_cast<unsigned char>((header_length >> 8) & 0xff)};
    out.write(reinterpret_cast<const char*>(length_bytes), 2);
    out.write(header_text.data(),
              static_cast<std::streamsize>(header_text.size()));

    for (int64_t row = 0; row < table->num_rows(); ++row) {
        for (int column_index = 0; column_index < table->num_columns();
             ++column_index) {
            const auto column = table->column(column_index);
            int64_t remaining = row;
            std::shared_ptr<arrow::Scalar> scalar;
            for (const auto& chunk : column->chunks()) {
                if (remaining < chunk->length()) {
                    auto maybe_scalar =
                        chunk->GetScalar(static_cast<int>(remaining));
                    if (!maybe_scalar.ok()) {
                        error = "Could not read cell while writing NumPy output: " +
                                maybe_scalar.status().ToString();
                        return false;
                    }
                    scalar = maybe_scalar.ValueOrDie();
                    break;
                }
                remaining -= chunk->length();
            }
            const double value = NumericScalarToDouble(scalar, error);
            if (!error.empty()) {
                return false;
            }
            out.write(reinterpret_cast<const char*>(&value), sizeof(double));
        }
    }

    if (!out) {
        error = "NumPy write failed for '" + options.output_path + "'.";
        return false;
    }
    return true;
}

bool WriteHdf5(const std::shared_ptr<arrow::Table>& table,
               const DataConvertOptions& options,
               std::string& error) {
#ifdef CYXWIZ_HAS_HDF5
    for (int column_index = 0; column_index < table->num_columns();
         ++column_index) {
        const auto field = table->schema()->field(column_index);
        if (!IsNumericArrowType(field->type()->id())) {
            error = "HDF5 output supports numeric columns only. Column '" +
                    field->name() + "' is " + field->type()->ToString() + ".";
            return false;
        }
    }

    std::vector<std::vector<double>> values(
        static_cast<size_t>(table->num_rows()),
        std::vector<double>(static_cast<size_t>(table->num_columns()), 0.0));
    for (int64_t row = 0; row < table->num_rows(); ++row) {
        for (int column_index = 0; column_index < table->num_columns();
             ++column_index) {
            const auto column = table->column(column_index);
            int64_t remaining = row;
            std::shared_ptr<arrow::Scalar> scalar;
            for (const auto& chunk : column->chunks()) {
                if (remaining < chunk->length()) {
                    auto maybe_scalar =
                        chunk->GetScalar(static_cast<int>(remaining));
                    if (!maybe_scalar.ok()) {
                        error = "Could not read cell while writing HDF5 output: " +
                                maybe_scalar.status().ToString();
                        return false;
                    }
                    scalar = maybe_scalar.ValueOrDie();
                    break;
                }
                remaining -= chunk->length();
            }
            values[static_cast<size_t>(row)]
                  [static_cast<size_t>(column_index)] =
                NumericScalarToDouble(scalar, error);
            if (!error.empty()) {
                return false;
            }
        }
    }

    try {
        HighFive::File file(options.output_path, HighFive::File::Overwrite);
        auto dataset = file.createDataSet<double>(
            "data", HighFive::DataSpace::From(values));
        dataset.write(values);
        return true;
    } catch (const std::exception& e) {
        error = "HDF5 output write failed: " + std::string(e.what());
        return false;
    }
#else
    (void)table;
    (void)options;
    error = "HDF5 support is not compiled into this build.";
    return false;
#endif
}

bool WriteOutputDataset(const std::shared_ptr<arrow::Table>& table,
                        const DataConvertOptions& options,
                        DataConvertFormat output_format,
                        std::string& error) {
    switch (output_format) {
        case DataConvertFormat::Csv:
        case DataConvertFormat::Tsv:
            return WriteDelimited(table, options, output_format, error);
        case DataConvertFormat::JsonLines:
            return WriteJsonLines(table, options, error);
        case DataConvertFormat::Text:
            return WriteText(table, options, error);
        case DataConvertFormat::Arff:
            return WriteArff(table, options, error);
        case DataConvertFormat::Numpy:
            return WriteNumpy(table, options, error);
        case DataConvertFormat::Hdf5:
            return WriteHdf5(table, options, error);
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
        << options.decimal_point << "|"
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
        {"input_decimal_point", std::string(1, options.decimal_point)},
        {"output_delimiter", output_format == DataConvertFormat::Tsv ? "tab" :
            (output_format == DataConvertFormat::Csv ? "," : "not_applicable")},
        {"output_decimal_point", IsDelimitedFormat(output_format) ? "." : "not_applicable"},
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

namespace {

void SetLoadTableError(std::string& error,
                       const char* code,
                       const std::string& message) {
    error = errors::FormatError(code, message);
}

DataConvertPreview FailPreview(const char* code,
                               const std::string& message) {
    DataConvertPreview preview;
    preview.error = errors::FormatError(code, message);
    return preview;
}

DataConvertResult FailConvert(const char* code,
                              const std::string& message) {
    DataConvertResult result;
    result.error = errors::FormatError(code, message);
    return result;
}

} // namespace

std::shared_ptr<arrow::Table> DataConvertService::LoadTable(
    const DataConvertOptions& options,
    std::string& error) {
    const DataConvertFormat input_format = ResolveInputFormat(options);
    if (!options.input_table && options.input_path.empty()) {
        SetLoadTableError(
            error,
            errors::File::PathMissing,
            "Input file is empty. Select a supported data file first.");
        return nullptr;
    }
    if (!options.input_table && !std::filesystem::exists(options.input_path)) {
        SetLoadTableError(
            error,
            errors::File::NotFound,
            "Input file does not exist: " + options.input_path);
        return nullptr;
    }
    if (!options.input_table && !IsSupportedFormat(input_format)) {
        SetLoadTableError(
            error,
            errors::File::UnsupportedFormat,
            "Input format is not supported. Choose " +
                SupportedFormatList() + ".");
        return nullptr;
    }

    auto dataset = LoadInputDataset(options, input_format, error);
    if (!dataset || !dataset->GetArrowTable()) {
        if (error.empty()) {
            error = "Could not load input table.";
        }
        return nullptr;
    }
    return dataset->GetArrowTable();
}

DataConvertPreview DataConvertService::Preview(const DataConvertOptions& options) {
    DataConvertPreview preview;
    const DataConvertFormat input_format = ResolveInputFormat(options);
    if (!options.input_table && options.input_path.empty()) {
        return FailPreview(
            errors::File::PathMissing,
            "Input file is empty. Select a supported data file first.");
    }
    if (!options.input_table && !std::filesystem::exists(options.input_path)) {
        return FailPreview(
            errors::File::NotFound,
            "Input file does not exist: " + options.input_path);
    }
    if (!IsSupportedFormat(input_format)) {
        return FailPreview(
            errors::File::UnsupportedFormat,
            "Input format is not supported. Choose " +
                SupportedFormatList() + ".");
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
        return FailConvert(
            errors::File::PathMissing,
            "Input file is empty. Select a supported data file first.");
    }
    if (!options.input_table && !std::filesystem::exists(options.input_path)) {
        return FailConvert(
            errors::File::NotFound,
            "Input file does not exist: " + options.input_path);
    }
    if (!options.input_table && !IsSupportedFormat(input_format)) {
        return FailConvert(
            errors::File::UnsupportedFormat,
            "Input format is not supported. Choose " +
                SupportedFormatList() + ".");
    }
    if (options.output_path.empty()) {
        return FailConvert(
            errors::File::PathMissing,
            "Output path is empty. Choose where to write the converted file.");
    }
    if (!IsSupportedFormat(output_format)) {
        return FailConvert(
            errors::File::UnsupportedFormat,
            "Output format is not supported. Choose " +
                SupportedFormatList() + ".");
    }
    if (!IsAutoFormat(options.output_format) &&
        !OutputExtensionMatchesFormat(output_path, output_format)) {
        return FailConvert(
            errors::File::InvalidOption,
            "Output path extension does not match output_format '" +
                options.output_format + "'. Expected " +
                ExpectedExtensionsForFormat(output_format) + ".");
    }

    if (TryUseFreshOutput(options, input_format, output_format, result)) {
        spdlog::info("DataConvert: skipped fresh output '{}'", options.output_path);
        return result;
    }
    if (std::filesystem::exists(output_path) && !options.overwrite) {
        return FailConvert(
            errors::File::InvalidOption,
            "Output file already exists and overwrite is disabled: " +
                options.output_path);
    }

    std::error_code ec;
    if (options.create_parent_dirs && output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path(), ec);
        if (ec) {
            return FailConvert(
                errors::File::PermissionDenied,
                "Could not create output folder '" +
                    output_path.parent_path().string() + "': " +
                    ec.message());
        }
    }

    std::string error;
    auto dataset = LoadInputDataset(options, input_format, error);
    if (!dataset) {
        result.error = errors::FormatError(errors::File::ReadFailed, error);
        return result;
    }
    auto table = dataset->GetArrowTable();
    result.detected_delimiter = IsDelimitedFormat(input_format)
        ? ResolveDelimiter(options, input_format)
        : ',';

    if (!WriteOutputDataset(table, options, output_format, error)) {
        result.error = errors::FormatError(errors::File::WriteFailed, error);
        return result;
    }

    result.rows_read = table->num_rows();
    result.rows_written = table->num_rows();
    result.columns = table->num_columns();
    if (options.retain_output_table) {
        result.output_table = table;
    }
    result.output_path = options.output_path;
    if (std::filesystem::exists(output_path, ec)) {
        result.bytes_written = static_cast<int64_t>(
            std::filesystem::file_size(output_path, ec));
    }

    if (options.write_manifest) {
        std::string manifest_error;
        if (!WriteManifest(options, result, input_format, output_format,
                           manifest_error)) {
            result.error =
                errors::FormatError(errors::File::WriteFailed, manifest_error);
            return result;
        }
        result.manifest_path = BuildManifestPath(options.output_path);
    }

    result.ok = true;
    const std::string input_label =
        options.input_table ? std::string("<in-memory Arrow table>")
                            : options.input_path;
    spdlog::info("DataConvert: converted '{}' ({}) -> '{}' ({}) ({} rows, {} columns)",
                 input_label, FormatName(input_format),
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
