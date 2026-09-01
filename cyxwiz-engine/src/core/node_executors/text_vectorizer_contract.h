#pragma once

#include <charconv>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace cyxwiz::text_vectorizer_contract {

inline bool IsAsciiWhitespace(char value) {
    return value == ' ' || value == '\t' || value == '\r' || value == '\n';
}

inline std::string_view TrimAsciiWhitespace(std::string_view value) {
    while (!value.empty() && IsAsciiWhitespace(value.front())) {
        value.remove_prefix(1);
    }
    while (!value.empty() && IsAsciiWhitespace(value.back())) {
        value.remove_suffix(1);
    }
    return value;
}

inline bool ParsePositiveInteger(std::string_view raw_value,
                                 const std::string& operator_name,
                                 const std::string& field_name,
                                 int& output,
                                 std::string& error) {
    const std::string_view value = TrimAsciiWhitespace(raw_value);
    int parsed = 0;
    const auto result = std::from_chars(
        value.data(), value.data() + value.size(), parsed);
    if (value.empty() || result.ec != std::errc{} ||
        result.ptr != value.data() + value.size()) {
        error = operator_name + ": '" + field_name +
                "' is not a valid integer: " + std::string(raw_value);
        return false;
    }
    if (parsed < 1) {
        error = operator_name + ": '" + field_name +
                "' must be >= 1 (got " + std::to_string(parsed) + ")";
        return false;
    }
    output = parsed;
    return true;
}

inline bool ParseNGramRange(
    const std::map<std::string, std::string>& parameters,
    const std::string& operator_name,
    int& ngram_min,
    int& ngram_max,
    std::string& error) {
    ngram_min = 1;
    ngram_max = 1;

    const auto range = parameters.find("ngram_range");
    const bool has_range =
        range != parameters.end() && !range->second.empty();
    if (has_range) {
        const std::string_view value = range->second;
        const size_t delimiter = value.find_first_of(",;");
        if (delimiter == std::string_view::npos ||
            value.find_first_of(",;", delimiter + 1) !=
                std::string_view::npos) {
            error = operator_name +
                    ": ngram_range must be formatted as 'min,max' (got '" +
                    range->second + "')";
            return false;
        }
        if (!ParsePositiveInteger(value.substr(0, delimiter), operator_name,
                                  "ngram_range minimum", ngram_min, error) ||
            !ParsePositiveInteger(value.substr(delimiter + 1), operator_name,
                                  "ngram_range maximum", ngram_max, error)) {
            return false;
        }
    } else {
        const auto parse_legacy = [&](const char* key, int& output) {
            const auto value = parameters.find(key);
            return value == parameters.end() || value->second.empty() ||
                   ParsePositiveInteger(
                       value->second, operator_name, key, output, error);
        };
        if (!parse_legacy("ngram_min", ngram_min) ||
            !parse_legacy("ngram_max", ngram_max)) {
            return false;
        }
    }

    if (ngram_min > ngram_max) {
        error = operator_name + ": ngram_min must be <= ngram_max (got " +
                std::to_string(ngram_min) + "," +
                std::to_string(ngram_max) + ")";
        return false;
    }
    if (ngram_max > 3) {
        error = operator_name + ": ngram_max > 3 is not supported yet (got " +
                std::to_string(ngram_max) + ")";
        return false;
    }
    return true;
}

inline std::string JoinNGram(const std::vector<std::string>& tokens,
                             size_t start,
                             size_t width) {
    std::string output;
    for (size_t offset = 0; offset < width; ++offset) {
        if (offset > 0) {
            output += ' ';
        }
        output += tokens[start + offset];
    }
    return output;
}

inline std::vector<std::string> BuildNGramFeatures(
    const std::vector<std::string>& tokens,
    int ngram_min,
    int ngram_max) {
    std::vector<std::string> features;
    if (tokens.empty() || ngram_min < 1 || ngram_max < ngram_min) {
        return features;
    }

    size_t total = 0;
    for (int n = ngram_min; n <= ngram_max; ++n) {
        const size_t width = static_cast<size_t>(n);
        if (tokens.size() < width) {
            continue;
        }
        const size_t count = tokens.size() - width + 1;
        if (total > std::numeric_limits<size_t>::max() - count) {
            throw std::length_error("text n-gram feature count overflow");
        }
        total += count;
    }
    features.reserve(total);

    for (int n = ngram_min; n <= ngram_max; ++n) {
        const size_t width = static_cast<size_t>(n);
        if (tokens.size() < width) {
            continue;
        }
        for (size_t start = 0; start + width <= tokens.size(); ++start) {
            features.push_back(width == 1
                ? tokens[start]
                : JoinNGram(tokens, start, width));
        }
    }
    return features;
}

}  // namespace cyxwiz::text_vectorizer_contract
