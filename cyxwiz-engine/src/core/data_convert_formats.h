#pragma once

#include <array>
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

namespace cyxwiz::data_convert {

// Table-file ingress/export only. This catalog does not enable DataInput's
// legacy Excel loader, media tensors, or Excel export.
enum class Format { Unknown, Csv, Tsv, JsonLines, Text, Arff, Numpy,
                    Hdf5, Excel, Parquet, Feather, ArrowIpc };
enum class Direction { Input, Output };
struct Features { bool xlsx; bool hdf5; };

// Internal linkage: reduced test targets may have different optional features.
static constexpr Features kBuildFeatures{
#ifdef CYXWIZ_HAS_XLSX
    true,
#else
    false,
#endif
#ifdef CYXWIZ_HAS_HDF5
    true
#else
    false
#endif
};

struct FormatInfo {
    Format format;
    const char* name;
    const char* label;
    const char* default_extension;
    std::array<const char*, 4> aliases;
    std::array<const char*, 4> extensions;
};

inline constexpr std::array<FormatInfo, 11> kFormats{{
    {Format::Csv, "csv", "CSV", ".csv", {"csv"}, {"csv"}},
    {Format::Tsv, "tsv", "TSV", ".tsv", {"tsv"}, {"tsv"}},
    {Format::JsonLines, "jsonl", "JSON Lines", ".jsonl", {"jsonl", "json", "ndjson"}, {"jsonl", "json", "ndjson"}},
    {Format::Text, "txt", "Plain Text (one column)", ".txt", {"txt", "text"}, {"txt", "text"}},
    {Format::Arff, "arff", "ARFF", ".arff", {"arff"}, {"arff"}},
    {Format::Numpy, "npy", "NumPy", ".npy", {"npy"}, {"npy"}},
    {Format::Hdf5, "hdf5", "HDF5", ".h5", {"hdf5", "h5", "hdf"}, {"hdf5", "h5", "hdf"}},
    {Format::Excel, "xlsx", "Excel XLSX (values only)", ".xlsx", {"xlsx", "excel"}, {"xlsx"}},
    {Format::Parquet, "parquet", "Parquet", ".parquet", {"parquet", "pq"}, {"parquet", "pq"}},
    {Format::Feather, "feather", "Feather", ".feather", {"feather", "fea"}, {"feather", "fea"}},
    {Format::ArrowIpc, "ipc", "Arrow IPC", ".ipc", {"ipc", "arrow", "arrowipc"}, {"ipc", "arrow"}}
}};

inline std::string Normalize(std::string_view value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) value.remove_prefix(1);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.remove_suffix(1);
    std::string result(value);
    for (char& c : result) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return result;
}

inline const FormatInfo* Find(std::string_view name) {
    const auto normalized = Normalize(name);
    for (const auto& info : kFormats)
        for (const char* alias : info.aliases)
            if (alias && normalized == alias) return &info;
    return nullptr;
}

inline const FormatInfo* Find(Format format) {
    for (const auto& info : kFormats) if (info.format == format) return &info;
    return nullptr;
}

inline Format FromName(std::string_view name) {
    const auto* info = Find(name);
    return info ? info->format : Format::Unknown;
}

// Explicit format takes precedence even when it is unknown; never silently
// reinterpret an unsupported saved setting as extension-based Auto.
inline Format Resolve(std::string_view name, std::string_view path_extension) {
    const auto normalized = Normalize(name);
    if (!normalized.empty() && normalized != "auto") return FromName(normalized);
    if (path_extension.starts_with('.')) path_extension.remove_prefix(1);
    return FromName(path_extension);
}

inline bool Available(Format format, Direction direction, Features features) {
    if (format == Format::Unknown) return false;
    if (format == Format::Excel) return features.xlsx && direction == Direction::Input;
    if (format == Format::Hdf5) return features.hdf5;
    return Find(format) != nullptr;
}

inline std::vector<const char*> AllowedNames(Direction direction, Features features) {
    std::vector<const char*> names{"auto"};
    for (const auto& info : kFormats) {
        if (!Available(info.format, direction, features)) continue;
        for (const char* alias : info.aliases) if (alias) names.push_back(alias);
    }
    return names;
}

inline std::vector<std::string> PropertyChoices(Direction direction, Features features) {
    const auto names = AllowedNames(direction, features);
    return {names.begin(), names.end()};
}

inline std::string ExtensionFilter(Direction direction, Features features,
                                   std::string_view separator = ",", std::string_view prefix = "") {
    std::string result;
    for (const auto& info : kFormats) {
        if (!Available(info.format, direction, features)) continue;
        for (const char* extension : info.extensions) {
            if (!extension) continue;
            if (!result.empty()) result += separator;
            result += prefix;
            result += extension;
        }
    }
    return result;
}

inline bool ExtensionMatches(std::string_view extension, Format format) {
    if (extension.starts_with('.')) extension.remove_prefix(1);
    const auto normalized = Normalize(extension);
    const auto* info = Find(format);
    if (!info) return false;
    for (const char* candidate : info->extensions)
        if (candidate && normalized == candidate) return true;
    return false;
}

inline constexpr const char* kXlsxRestrictions =
    "XLSX reads one worksheet: blank sheet name selects the first sheet. "
    "Values only; formulas, merged cells, mixed column types and binary attachments "
    "(including images) reject. Dates remain numeric Excel serials. "
    "UTF-8 XML only; 64 MiB compressed/expanded archive, 16 MiB per part, "
    "512 parts, one million cells, 1 MiB headers and 64 MiB strings. "
    "Excel export, .xls and media tensors are not supported.";

} // namespace cyxwiz::data_convert
