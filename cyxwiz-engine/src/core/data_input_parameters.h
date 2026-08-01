#pragma once

#include <algorithm>
#include <cctype>
#include <map>
#include <string>

namespace cyxwiz {

inline std::string NormalizeDataInputFormat(std::string value) {
    const auto begin = std::find_if_not(
        value.begin(), value.end(),
        [](unsigned char c) { return std::isspace(c); });
    const auto end = std::find_if_not(
        value.rbegin(), value.rend(),
        [](unsigned char c) { return std::isspace(c); }).base();
    value = begin < end ? std::string(begin, end) : std::string{};
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return value.empty() ? "auto" : value;
}

// Data Input originally persisted its format as `type`; current nodes use
// `file_type`. During migration a graph can legitimately contain `auto` in
// one alias and a detected concrete format in the other. A concrete value is
// authoritative over `auto`, while two different concrete formats remain a
// real configuration error.
inline bool ResolveDataInputFormatAliases(
    const std::map<std::string, std::string>& parameters,
    std::string& format,
    std::string& error) {
    const auto legacy = parameters.find("type");
    const auto canonical = parameters.find("file_type");
    const bool has_legacy =
        legacy != parameters.end() && !legacy->second.empty();
    const bool has_canonical =
        canonical != parameters.end() && !canonical->second.empty();

    const std::string legacy_value = has_legacy
        ? NormalizeDataInputFormat(legacy->second)
        : "auto";
    const std::string canonical_value = has_canonical
        ? NormalizeDataInputFormat(canonical->second)
        : "auto";

    if (legacy_value != "auto" && canonical_value != "auto" &&
        legacy_value != canonical_value) {
        error = "DataInput type and file_type disagree";
        return false;
    }

    format = canonical_value != "auto"
        ? canonical_value
        : legacy_value;
    error.clear();
    return true;
}

// Canonicalize only compatible aliases. Genuine concrete conflicts are left
// intact so runtime validation can fail closed with an actionable message.
inline bool MigrateDataInputFormatAliases(
    std::map<std::string, std::string>& parameters) {
    std::string format;
    std::string error;
    if (!ResolveDataInputFormatAliases(parameters, format, error)) {
        return false;
    }
    parameters["file_type"] = format;
    parameters.erase("type");
    return true;
}

}  // namespace cyxwiz
