#pragma once

// What a file's name in CyxCloud says about its format, in one place for the
// job plan (core/graph_training_job.cpp), the Data Input dialog and the
// cloud browser (TOFIX164). Remote nodes train from Arrow IPC and Parquet.

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>

namespace cyxwiz {

inline std::string CloudFileExtensionLower(const std::string& file_name) {
    std::string extension = std::filesystem::path(file_name).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return extension;
}

// Arrow IPC (.arrow, .feather, .ipc) or Parquet (.parquet): the two formats a
// node trains from without the Engine.
inline bool CloudDatasetFileFormatKnown(const std::string& file_name) {
    const std::string extension = CloudFileExtensionLower(file_name);
    return extension == ".arrow" || extension == ".feather" || extension == ".ipc" || extension == ".parquet";
}

// The format word a screen shows for a cloud file: "Parquet", "Arrow IPC",
// "CSV", "Text", "ZIP", "JSON", else the upper-cased extension or "File".
inline std::string CloudFileFormatWord(const std::string& file_name) {
    const std::string extension = CloudFileExtensionLower(file_name);
    if (extension == ".parquet") return "Parquet";
    if (extension == ".arrow" || extension == ".feather" || extension == ".ipc") return "Arrow IPC";
    if (extension == ".csv" || extension == ".tsv") return "CSV";
    if (extension == ".txt") return "Text";
    if (extension == ".zip") return "ZIP";
    if (extension == ".json" || extension == ".jsonl") return "JSON";
    if (extension.size() > 1) {
        std::string word = extension.substr(1);
        std::transform(word.begin(), word.end(), word.begin(),
                       [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        return word;
    }
    return "File";
}

}  // namespace cyxwiz
