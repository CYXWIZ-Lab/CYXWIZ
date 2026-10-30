#include "data_input_capabilities.h"
#include "../core/data_input_formats.h"
#include "../core/image_formats.h"
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

FolderScan ScanFolder(const std::string& path, int max_files) {
    FolderScan scan;
    std::error_code ec;
    if (path.empty() || !fs::is_directory(path, ec)) return scan;
    const auto kind_of = [](const fs::path& file) {
        const std::string ext = LowerExtension(file.string());
        if (cyxwiz::image_formats::IsImageExtension(ext)) return FolderKind::Images;
        if (ext == "wav" || ext == "flac" || ext == "ogg" || ext == "aiff" || ext == "aif" || ext == "mp3")
            return FolderKind::Audio;
        if (ext == "txt" || ext == "text" || ext == "md") return FolderKind::Text;
        return FolderKind::Unknown;
    };
    int counts[4] = {0, 0, 0, 0};
    std::vector<std::pair<std::string, FolderKind>> classes;
    int seen = 0;
    const auto count = [&](const fs::path& file) {
        if (seen >= max_files) {
            scan.truncated = true;
            return FolderKind::Unknown;
        }
        ++seen;
        const FolderKind kind = kind_of(file);
        ++counts[static_cast<int>(kind)];
        return kind;
    };
    for (const auto& entry : fs::directory_iterator(path, fs::directory_options::skip_permission_denied, ec)) {
        if (entry.is_regular_file(ec)) {
            count(entry.path());
        } else if (entry.is_directory(ec)) {
            FolderKind inner = FolderKind::Unknown;
            for (const auto& sub : fs::recursive_directory_iterator(
                     entry.path(), fs::directory_options::skip_permission_denied, ec)) {
                if (!sub.is_regular_file(ec)) continue;
                const FolderKind kind = count(sub.path());
                if (kind != FolderKind::Unknown) inner = kind;
                if (scan.truncated) break;
            }
            if (inner != FolderKind::Unknown) classes.emplace_back(entry.path().filename().string(), inner);
        }
        if (scan.truncated) break;
    }
    int best = 0;
    for (int k = 1; k < 4; ++k) {
        if (counts[k] > counts[best] || (best == 0 && counts[k] > 0)) best = k;
    }
    if (best == 0) return scan;
    scan.kind = static_cast<FolderKind>(best);
    scan.files = counts[best];
    for (const auto& [name, kind] : classes)
        if (kind == scan.kind) scan.class_names.push_back(name);
    std::sort(scan.class_names.begin(), scan.class_names.end());
    return scan;
}

bool IsPreviewSupported(cyxwiz::loaders::FileCategory file_category) {
    return file_category == FileCategory::Tabular ||
           file_category == FileCategory::Text ||
           file_category == FileCategory::TimeSeries;
}

bool HasRowsAndColumns(cyxwiz::loaders::FileCategory file_category) {
    return file_category != FileCategory::Image && file_category != FileCategory::Audio;
}

const char* PreviewUnavailableMessage(cyxwiz::loaders::FileCategory) {
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

std::string UniqueDatasetName(const std::string& base, const std::vector<std::string>& taken) {
    const auto used = [&taken](const std::string& name) {
        return std::find(taken.begin(), taken.end(), name) != taken.end();
    };
    if (!used(base)) return base;
    for (int n = 2;; ++n) {
        const std::string candidate = base + "_" + std::to_string(n);
        if (!used(candidate)) return candidate;
    }
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
