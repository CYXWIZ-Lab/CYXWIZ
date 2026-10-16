#include "data_input_capabilities.h"
#include "../core/data_input_formats.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>

namespace gui::data_input {

namespace {

namespace fs = std::filesystem;
using FileCategory = cyxwiz::loaders::FileCategory;

std::string Trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string LowerExtension(const std::string& path) {
    const fs::path fs_path(path);
    std::string ext = fs_path.extension().string();
    if (!ext.empty() && ext.front() == '.') {
        ext.erase(ext.begin());
    }
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

bool IsTabularCategory(FileCategory file_category) {
    return file_category == FileCategory::Tabular ||
           file_category == FileCategory::TimeSeries;
}

} // namespace

FileCategory FileCategoryFromParam(const std::string& value, FileCategory fallback) {
    if (value == "tabular") return FileCategory::Tabular;
    if (value == "image") return FileCategory::Image;
    if (value == "audio") return FileCategory::Audio;
    if (value == "text") return FileCategory::Text;
    if (value == "timeseries") return FileCategory::TimeSeries;
    return fallback;
}

const char* FileCategoryParam(FileCategory file_category) {
    switch (file_category) {
        case FileCategory::Tabular: return "tabular";
        case FileCategory::Image: return "image";
        case FileCategory::Audio: return "audio";
        case FileCategory::Text: return "text";
        case FileCategory::TimeSeries: return "timeseries";
    }
    return "tabular";
}

bool IsPreviewSupported(cyxwiz::loaders::FileCategory file_category) {
    return file_category == FileCategory::Tabular ||
           file_category == FileCategory::Text ||
           file_category == FileCategory::TimeSeries;
}

const char* PreviewUnavailableMessage(cyxwiz::loaders::FileCategory file_category) {
    if (file_category == FileCategory::Image) {
        return "Image preview is not wired yet. Apply can still scan supported image folders.";
    }
    return "Audio preview is not wired yet. Apply can still scan supported audio folders.";
}

const char* FileTypeParam(int detected_type) {
    const auto* format = cyxwiz::data_input::FindFormat(detected_type);
    return format ? format->name : "auto";
}

const char* FileTypeName(int detected_type) {
    const auto* format = cyxwiz::data_input::FindFormat(detected_type);
    return format ? format->label : "Unknown";
}

int FileTypeFromParam(const std::string& value, int fallback) {
    if (Trim(value).empty()) return fallback;
    const auto* format = cyxwiz::data_input::FindFormat(value);
    return format ? format->id : fallback;
}

int DetectFileTypeForPath(const std::string& path, std::size_t* file_size) {
    if (path.empty()) {
        if (file_size) {
            *file_size = 0;
        }
        return 0;
    }

    if (file_size) {
        try {
            *file_size = static_cast<std::size_t>(fs::file_size(path));
        } catch (...) {
            *file_size = 0;
        }
    }

    const auto* format = cyxwiz::data_input::DetectFormat(path);
    return format ? format->id : 0;
}

FileCategory DetectFileCategoryForPath(const std::string& path, FileCategory current_category) {
    if (path.empty()) {
        return current_category;
    }

    const std::string ext = LowerExtension(path);
    if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" ||
        ext == "gif" || ext == "tiff" || ext == "webp") {
        return FileCategory::Image;
    }
    if (ext == "wav" || ext == "mp3" || ext == "flac" || ext == "ogg" ||
        ext == "m4a" || ext == "aac") {
        return FileCategory::Audio;
    }
    if (current_category == FileCategory::Text || current_category == FileCategory::TimeSeries) {
        return current_category;
    }
    return FileCategory::Tabular;
}

bool UsesFolderPath(FileCategory file_category, TextLayout text_layout) {
    return file_category == FileCategory::Image ||
           file_category == FileCategory::Audio ||
           (file_category == FileCategory::Text && text_layout == TextLayout::CorpusSubdirs);
}

std::string CurrentSourceLabel(
    FileCategory file_category,
    TextLayout text_layout) {
    switch (file_category) {
        case FileCategory::Tabular: return "Tabular file";
        case FileCategory::Image: return "Image folder";
        case FileCategory::Audio: return "Audio folder";
        case FileCategory::Text:
            return text_layout == TextLayout::CorpusSubdirs
                ? "Text corpus folder"
                : "Text file";
        case FileCategory::TimeSeries: return "Time series file";
    }
    return "Unknown";
}

std::string CurrentApplySummary(
    FileCategory file_category,
    TextLayout text_layout,
    bool force_disk_backed,
    int max_rows,
    int skip_rows) {
    std::string summary;
    switch (file_category) {
        case FileCategory::Tabular:
            summary = "load the selected file into DataRegistry as tabular data";
            break;
        case FileCategory::TimeSeries:
            summary = "load the selected file through the tabular loader and mark it as time series";
            break;
        case FileCategory::Image:
            summary = "scan the selected folder and register a lazy image dataset";
            break;
        case FileCategory::Audio:
            summary = "scan the selected folder and register a lazy audio dataset";
            break;
        case FileCategory::Text:
            summary = text_layout == TextLayout::CorpusSubdirs
                ? "scan the corpus folder, tokenize text, and build vocabulary"
                : "load the selected text file, tokenize text, and build vocabulary";
            break;
    }

    if (force_disk_backed && IsTabularCategory(file_category)) {
        summary += " using the forced Parquet cache path";
    }
    if (max_rows > 0 && IsTabularCategory(file_category)) {
        summary += " with a row limit of " + std::to_string(max_rows);
    }
    if (skip_rows > 0 && IsTabularCategory(file_category)) {
        summary += " after skipping " + std::to_string(skip_rows) + " rows";
    }
    summary += ".";
    return summary;
}

const char* BackendSummary(int loaded_backend) {
    switch (loaded_backend) {
        case 1: return "Arrow in-memory";
        case 2: return "Parquet disk-backed";
        case 3: return "Image folder lazy loader";
        case 4: return "Audio folder lazy loader";
        case 5: return "Text lazy loader";
        default: return "Not loaded";
    }
}

std::string GenerateDatasetName(const std::string& file_path, const std::string& folder_path) {
    std::string name;
    if (!file_path.empty()) {
        name = fs::path(file_path).stem().string();
    } else if (!folder_path.empty()) {
        name = fs::path(folder_path).filename().string();
    }

    for (char& c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-') {
            c = '_';
        }
    }

    if (name.empty()) {
        name = "dataset";
    }
    return name;
}

std::string FormatBytes(std::size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_idx = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit_idx < 4) {
        size /= 1024.0;
        unit_idx++;
    }

    char buffer[32];
    if (unit_idx == 0) {
        std::snprintf(buffer, sizeof(buffer), "%zu %s", bytes, units[unit_idx]);
    } else {
        std::snprintf(buffer, sizeof(buffer), "%.1f %s", size, units[unit_idx]);
    }
    return std::string(buffer);
}

} // namespace gui::data_input
