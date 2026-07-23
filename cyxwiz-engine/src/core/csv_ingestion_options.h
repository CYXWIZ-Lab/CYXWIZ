#pragma once

#include <arrow/csv/options.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace cyxwiz {

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
        DefaultTabularMissingValueTokens()) {
    auto options = arrow::csv::ConvertOptions::Defaults();
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
