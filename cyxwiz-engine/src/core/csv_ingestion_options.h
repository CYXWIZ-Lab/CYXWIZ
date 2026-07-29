#pragma once

#include <arrow/csv/options.h>
#include <arrow/csv/reader.h>
#include <arrow/io/api.h>
#include <arrow/type.h>

#include <algorithm>
#include <cctype>
#include <functional>
#include <fstream>
#include <string>
#include <vector>

namespace cyxwiz {

using CsvProgressCallback =
    std::function<bool(float, const std::string&)>;

inline const std::vector<std::string>& DefaultTabularMissingValueTokens() {
    // Arrow already recognizes common NULL spellings. Lowercase "na" is a
    // frequent numeric-table sentinel that is not included in every Arrow
    // release, so tabular ingestion adds it explicitly.
    static const std::vector<std::string> tokens = {"na"};
    return tokens;
}

inline std::vector<std::string> ParseMissingValueTokens(
    const std::string& comma_separated) {
    std::vector<std::string> tokens;
    size_t start = 0;
    while (start <= comma_separated.size()) {
        const size_t end = comma_separated.find(',', start);
        std::string token = comma_separated.substr(
            start, end == std::string::npos ? std::string::npos : end - start);
        const auto first = std::find_if_not(
            token.begin(), token.end(),
            [](unsigned char c) { return std::isspace(c) != 0; });
        const auto last = std::find_if_not(
            token.rbegin(), token.rend(),
            [](unsigned char c) { return std::isspace(c) != 0; }).base();
        if (first < last) {
            token = std::string(first, last);
            if (std::find(tokens.begin(), tokens.end(), token) == tokens.end()) {
                tokens.push_back(std::move(token));
            }
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return tokens;
}

inline arrow::csv::ConvertOptions MakeTabularCsvConvertOptions(
    const std::vector<std::string>& additional_null_values =
        DefaultTabularMissingValueTokens(),
    char decimal_point = '.') {
    auto options = arrow::csv::ConvertOptions::Defaults();
    options.decimal_point = decimal_point;
    for (const auto& value : additional_null_values) {
        if (!value.empty() &&
            std::find(options.null_values.begin(), options.null_values.end(), value) ==
                options.null_values.end()) {
            options.null_values.push_back(value);
        }
    }
    // If inference legitimately chooses string, configured missing tokens
    // still become null rather than surviving as literal sentinel values.
    options.strings_can_be_null = true;
    return options;
}

inline size_t PromoteInferredCsvIntegersToDouble(
    const std::shared_ptr<arrow::Schema>& inferred_schema,
    arrow::csv::ConvertOptions& options) {
    if (!inferred_schema) return 0;

    size_t promoted = 0;
    for (const auto& field : inferred_schema->fields()) {
        if (!field || !arrow::is_integer(field->type()->id())) continue;
        options.column_types[field->name()] = arrow::float64();
        ++promoted;
    }
    return promoted;
}

inline bool CsvSourceSamplesContainDecimalValues(
    const std::string& path,
    char decimal_point,
    int64_t minimum_source_bytes = 64LL * 1024 * 1024,
    size_t sample_bytes = 256 * 1024,
    int sample_count = 12) {
    if (path.empty() || decimal_point == '\0' || sample_bytes == 0 ||
        sample_count <= 0) {
        return false;
    }

    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input.is_open()) return false;
    const std::streamoff source_size = input.tellg();
    if (source_size <= minimum_source_bytes) return false;

    auto contains_decimal_number = [decimal_point](const std::string& bytes) {
        for (size_t i = 1; i + 1 < bytes.size(); ++i) {
            if (bytes[i] != decimal_point) continue;
            const auto before = static_cast<unsigned char>(bytes[i - 1]);
            const auto after = static_cast<unsigned char>(bytes[i + 1]);
            if (std::isdigit(before) != 0 && std::isdigit(after) != 0) {
                return true;
            }
        }
        return false;
    };

    std::string sample(sample_bytes, '\0');
    for (int index = 1; index <= sample_count; ++index) {
        const std::streamoff offset =
            (source_size * index) / (sample_count + 1);
        input.clear();
        input.seekg(offset);
        if (!input.good()) continue;

        // Discard the partial record at the seek position. CSV sources with
        // embedded newlines remain safe because this scan only answers the
        // conservative question "do later source regions contain decimals?"
        std::string partial;
        std::getline(input, partial);
        input.read(sample.data(), static_cast<std::streamsize>(sample.size()));
        sample.resize(static_cast<size_t>(input.gcount()));
        if (contains_decimal_number(sample)) return true;
        sample.resize(sample_bytes);
    }
    return false;
}

inline arrow::Result<std::shared_ptr<arrow::csv::StreamingReader>>
MakeStableCsvStreamingReader(
    const std::string& path,
    const arrow::csv::ReadOptions& read_options,
    const arrow::csv::ParseOptions& parse_options,
    arrow::csv::ConvertOptions convert_options,
    size_t* promoted_integer_columns = nullptr,
    bool promote_inferred_integers = false,
    std::shared_ptr<arrow::io::ReadableFile>* input_source = nullptr) {
    if (promoted_integer_columns) {
        *promoted_integer_columns = 0;
    }
    auto input_result = arrow::io::ReadableFile::Open(path);
    if (!input_result.ok()) return input_result.status();
    auto input = input_result.ValueOrDie();
    if (input_source) *input_source = input;

    auto reader_result = arrow::csv::StreamingReader::Make(
        arrow::io::default_io_context(),
        input,
        read_options,
        parse_options,
        convert_options);
    if (!reader_result.ok()) return reader_result.status();

    auto reader = reader_result.ValueOrDie();
    if (!promote_inferred_integers) return reader;

    const size_t promoted = PromoteInferredCsvIntegersToDouble(
        reader->schema(), convert_options);
    if (promoted_integer_columns) {
        *promoted_integer_columns = promoted;
    }
    if (promoted == 0) return reader;

    // Reopen after the bounded inference block and parse the complete source
    // against the stable numeric schema. This prevents a column that starts
    // with integral values from failing when decimals appear in later blocks.
    input_result = arrow::io::ReadableFile::Open(path);
    if (!input_result.ok()) return input_result.status();
    input = input_result.ValueOrDie();
    if (input_source) *input_source = input;
    return arrow::csv::StreamingReader::Make(
        arrow::io::default_io_context(),
        input,
        read_options,
        parse_options,
        convert_options);
}

inline std::string MissingValueTokensSignature(
    const std::vector<std::string>& tokens) {
    std::string signature;
    for (const auto& token : tokens) {
        signature += std::to_string(token.size());
        signature += ':';
        signature += token;
        signature += ';';
    }
    return signature;
}

} // namespace cyxwiz
