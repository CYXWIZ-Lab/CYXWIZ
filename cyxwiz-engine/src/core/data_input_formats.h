#pragma once

#include "data_input_parameters.h"

#include <array>
#include <filesystem>
#include <vector>

namespace cyxwiz::data_input {

enum class SourcePreview { Unsupported, Delimited, RegisteredDataset };
enum class SourceDependency { None, Hdf5 };

// IDs preserve the dialog's existing saved/detected format mapping. Recognition
// does not imply that a source has a production Data Input adapter.
struct SourceFormatCapability {
    int id;
    const char* name;
    const char* label;
    std::array<const char*, 3> aliases;
    std::array<const char*, 3> extensions;
    bool executable;
    SourcePreview preview;
    SourceDependency dependency = SourceDependency::None;
    const char* unsupported_reason = "Tabular file type is not supported yet";
};

inline constexpr std::array<SourceFormatCapability, 12> kSourceFormats{{
    {0, "auto", "Auto", {"auto"}, {}, true, SourcePreview::Unsupported},
    {1, "csv", "CSV", {"csv"}, {"csv"}, true, SourcePreview::Delimited},
    {2, "tsv", "TSV", {"tsv"}, {"tsv", "tab"}, true, SourcePreview::Delimited},
    {3, "json", "JSON", {"json"}, {"json", "jsonl"}, false, SourcePreview::Unsupported,
     SourceDependency::None, "Tabular JSON loading is not supported yet"},
    {4, "parquet", "Parquet", {"parquet"}, {"parquet", "pq"}, true, SourcePreview::RegisteredDataset},
    {5, "excel", "Excel", {"excel"}, {"xlsx", "xls"}, false, SourcePreview::Unsupported,
     SourceDependency::None, "Tabular Excel loading is not supported yet"},
    {6, "hdf5", "HDF5", {"hdf5", "h5", "hdf"}, {"h5", "hdf5", "hdf"}, false, SourcePreview::Unsupported,
     SourceDependency::Hdf5, "Tabular HDF5 loading is not supported yet; conversion and hierarchy inspection are separate capabilities"},
    {7, "feather", "Feather", {"feather"}, {"feather", "fea"}, true, SourcePreview::RegisteredDataset},
    {8, "arrow", "Arrow / IPC", {"arrow", "ipc"}, {"arrow", "ipc"}, true, SourcePreview::RegisteredDataset},
    {9, "txt", "TXT", {"txt"}, {"txt"}, false, SourcePreview::Unsupported,
     SourceDependency::None, "Tabular TXT loading is not supported on this path; use Text source for text files"},
    {10, "arff", "ARFF", {"arff"}, {"arff"}, false, SourcePreview::Unsupported,
     SourceDependency::None, "Tabular ARFF loading is not supported yet"},
    {11, "zip_text", "ZIP text document", {"zip_text"}, {"zip"}, true, SourcePreview::RegisteredDataset},
}};

// Internal linkage accommodates reduced test targets with optional dependencies.
static constexpr bool kHdf5BuildAvailable =
#ifdef CYXWIZ_HAS_HDF5
    true;
#else
    false;
#endif

inline bool BuildAvailable(const SourceFormatCapability& format, bool hdf5_available) {
    return format.dependency != SourceDependency::Hdf5 || hdf5_available;
}

inline const SourceFormatCapability* FindFormat(int id) {
    for (const auto& format : kSourceFormats) if (format.id == id) return &format;
    return nullptr;
}

inline const SourceFormatCapability* FindFormat(const std::string& name) {
    const auto normalized = NormalizeDataInputFormat(name);
    for (const auto& format : kSourceFormats)
        for (const char* alias : format.aliases)
            if (alias && normalized == alias) return &format;
    return nullptr;
}

// Extension recognition is only a routing hint, not a signature probe.
inline const SourceFormatCapability* DetectFormat(const std::string& path) {
    auto extension = std::filesystem::path(path).extension().string();
    if (!extension.empty()) extension.erase(0, 1);
    extension = NormalizeDataInputFormat(extension);
    for (const auto& format : kSourceFormats)
        for (const char* candidate : format.extensions)
            if (candidate && extension == candidate) return &format;
    return nullptr;
}

inline std::string ResolveFormat(const std::string& name, const std::string& path) {
    const auto normalized = NormalizeDataInputFormat(name);
    if (normalized != "auto") return normalized;
    const auto* format = DetectFormat(path);
    return format ? format->name : "auto";
}

inline bool IsExecutable(const std::string& name) {
    const auto* format = FindFormat(name);
    return format && format->executable;
}

inline const char* UnsupportedReason(const std::string& name, bool hdf5_available) {
    const auto* format = FindFormat(name);
    if (!format) return "Tabular file type is not supported yet";
    if (!BuildAvailable(*format, hdf5_available))
        return "Tabular HDF5 loading is not supported: HDF5 support is not compiled into this build";
    return format->unsupported_reason;
}

inline std::vector<const char*> AllowedNames() {
    std::vector<const char*> names;
    for (const auto& format : kSourceFormats) {
        if (!format.executable) continue;
        for (const char* alias : format.aliases) if (alias) names.push_back(alias);
    }
    return names;
}

inline std::string ExtensionFilter(const SourceFormatCapability& format) {
    std::string result;
    for (const char* extension : format.extensions) {
        if (!extension) continue;
        if (!result.empty()) result += ',';
        result += extension;
    }
    return result;
}

inline std::string ExtensionFilter() {
    std::string result;
    for (const auto& format : kSourceFormats) {
        if (!format.executable) continue;
        const auto extensions = ExtensionFilter(format);
        if (extensions.empty()) continue;
        if (!result.empty()) result += ',';
        result += extensions;
    }
    return result;
}

} // namespace cyxwiz::data_input
