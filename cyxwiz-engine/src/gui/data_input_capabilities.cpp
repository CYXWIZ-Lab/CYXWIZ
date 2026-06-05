#include "data_input_capabilities.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>

namespace gui::data_input {

namespace {

namespace fs = std::filesystem;
using FileCategory = cyxwiz::loaders::FileCategory;

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

SourceType SourceTypeFromParam(const std::string& value, SourceType fallback) {
    if (value == "file") return SourceType::File;
    if (value == "ml_dataset") return SourceType::MLDataset;
    if (value == "database") return SourceType::Database;
    if (value == "cloud") return SourceType::Cloud;
    return fallback;
}

FileCategory FileCategoryFromParam(const std::string& value, FileCategory fallback) {
    if (value == "tabular") return FileCategory::Tabular;
    if (value == "image") return FileCategory::Image;
    if (value == "audio") return FileCategory::Audio;
    if (value == "video") return FileCategory::Video;
    if (value == "text") return FileCategory::Text;
    if (value == "timeseries") return FileCategory::TimeSeries;
    return fallback;
}

const char* SourceTypeParam(SourceType source_type) {
    switch (source_type) {
        case SourceType::File: return "file";
        case SourceType::MLDataset: return "ml_dataset";
        case SourceType::Database: return "database";
        case SourceType::Cloud: return "cloud";
    }
    return "file";
}

const char* FileCategoryParam(FileCategory file_category) {
    switch (file_category) {
        case FileCategory::Tabular: return "tabular";
        case FileCategory::Image: return "image";
        case FileCategory::Audio: return "audio";
        case FileCategory::Video: return "video";
        case FileCategory::Text: return "text";
        case FileCategory::TimeSeries: return "timeseries";
    }
    return "tabular";
}

const char* MLDatasetTypeParam(MLDatasetType dataset_type) {
    switch (dataset_type) {
        case MLDatasetType::MNIST: return "mnist";
        case MLDatasetType::CIFAR10: return "cifar10";
        case MLDatasetType::CIFAR100: return "cifar100";
        case MLDatasetType::FashionMNIST: return "fashion_mnist";
        case MLDatasetType::ImageNet: return "imagenet";
        case MLDatasetType::ImageFolder: return "image_folder";
        case MLDatasetType::HuggingFace: return "huggingface";
        case MLDatasetType::Kaggle: return "kaggle";
        case MLDatasetType::Custom: return "custom";
    }
    return "mnist";
}

const char* DatabaseTypeParam(DatabaseType database_type) {
    switch (database_type) {
        case DatabaseType::SQLite: return "sqlite";
        case DatabaseType::PostgreSQL: return "postgresql";
        case DatabaseType::MySQL: return "mysql";
        case DatabaseType::DuckDB: return "duckdb";
    }
    return "sqlite";
}

bool IsApplySupported(SourceType source_type, cyxwiz::loaders::FileCategory file_category) {
    if (source_type != SourceType::File) {
        return false;
    }
    return file_category != FileCategory::Video;
}

bool IsPreviewSupported(SourceType source_type, cyxwiz::loaders::FileCategory file_category) {
    return source_type == SourceType::File &&
           (file_category == FileCategory::Tabular ||
            file_category == FileCategory::Text ||
            file_category == FileCategory::TimeSeries);
}

const char* UnsupportedApplyMessage(SourceType source_type, cyxwiz::loaders::FileCategory file_category) {
    if (source_type == SourceType::File && file_category == FileCategory::Video) {
        return "Video loading is planned but not wired yet.";
    }
    if (source_type == SourceType::MLDataset) {
        return "ML dataset downloads are planned but not wired yet. Use File source for loadable datasets.";
    }
    if (source_type == SourceType::Database) {
        return "Database loading is planned but not wired yet.";
    }
    if (source_type == SourceType::Cloud) {
        return "Cloud storage loading is planned but not wired yet.";
    }
    return "This data source is not available yet.";
}

const char* PreviewUnavailableMessage(SourceType source_type, cyxwiz::loaders::FileCategory file_category) {
    if (source_type == SourceType::File) {
        if (file_category == FileCategory::Image) {
            return "Image preview is not wired yet. Apply can still scan supported image folders.";
        }
        if (file_category == FileCategory::Audio) {
            return "Audio preview is not wired yet. Apply can still scan supported audio folders.";
        }
        if (file_category == FileCategory::Video) {
            return "Video preview is not available because video loading is not wired yet.";
        }
    }
    return UnsupportedApplyMessage(source_type, file_category);
}

const char* FileTypeParam(int detected_type) {
    static constexpr const char* kTypes[] = {
        "auto", "csv", "tsv", "json", "parquet", "excel",
        "hdf5", "feather", "arrow", "txt", "arff",
    };
    constexpr int kTypeCount = static_cast<int>(sizeof(kTypes) / sizeof(kTypes[0]));
    if (detected_type >= 0 && detected_type < kTypeCount) {
        return kTypes[detected_type];
    }
    return "auto";
}

const char* FileTypeName(int detected_type) {
    static constexpr const char* kNames[] = {
        "Auto", "CSV", "TSV", "JSON", "Parquet", "Excel",
        "HDF5", "Feather", "Arrow", "TXT", "ARFF",
    };
    constexpr int kNameCount = static_cast<int>(sizeof(kNames) / sizeof(kNames[0]));
    if (detected_type >= 0 && detected_type < kNameCount) {
        return kNames[detected_type];
    }
    return "Unknown";
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

    const std::string ext = LowerExtension(path);
    if (ext == "csv") return 1;
    if (ext == "tsv" || ext == "tab") return 2;
    if (ext == "json" || ext == "jsonl") return 3;
    if (ext == "parquet" || ext == "pq") return 4;
    if (ext == "xlsx" || ext == "xls") return 5;
    if (ext == "h5" || ext == "hdf5" || ext == "hdf") return 6;
    if (ext == "feather" || ext == "fea") return 7;
    if (ext == "arrow" || ext == "ipc") return 8;
    if (ext == "txt") return 9;
    if (ext == "arff") return 10;
    return 0;
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
    if (ext == "mp4" || ext == "avi" || ext == "mov" || ext == "mkv" ||
        ext == "webm" || ext == "wmv") {
        return FileCategory::Video;
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
    SourceType source_type,
    FileCategory file_category,
    TextLayout text_layout) {
    if (source_type == SourceType::File) {
        switch (file_category) {
            case FileCategory::Tabular: return "Tabular file";
            case FileCategory::Image: return "Image folder";
            case FileCategory::Audio: return "Audio folder";
            case FileCategory::Video: return "Video file (planned)";
            case FileCategory::Text:
                return text_layout == TextLayout::CorpusSubdirs
                    ? "Text corpus folder"
                    : "Text file";
            case FileCategory::TimeSeries: return "Time series file";
        }
    }
    if (source_type == SourceType::MLDataset) return "ML dataset (planned)";
    if (source_type == SourceType::Database) return "Database (planned)";
    if (source_type == SourceType::Cloud) return "Cloud storage (planned)";
    return "Unknown";
}

std::string CurrentApplySummary(
    SourceType source_type,
    FileCategory file_category,
    TextLayout text_layout,
    bool force_disk_backed,
    int max_rows,
    int skip_rows) {
    if (!IsApplySupported(source_type, file_category)) {
        return UnsupportedApplyMessage(source_type, file_category);
    }

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
        case FileCategory::Video:
            summary = UnsupportedApplyMessage(source_type, file_category);
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

std::string GenerateDatasetName(
    SourceType source_type,
    const std::string& file_path,
    const std::string& folder_path,
    const std::string& dataset_name,
    const std::string& db_name,
    const std::string& cloud_bucket) {
    std::string name;

    if (source_type == SourceType::File) {
        if (!file_path.empty()) {
            name = fs::path(file_path).stem().string();
        } else if (!folder_path.empty()) {
            name = fs::path(folder_path).filename().string();
        }
    } else if (source_type == SourceType::MLDataset) {
        name = dataset_name;
    } else if (source_type == SourceType::Database) {
        name = std::string("db_") + db_name;
    } else if (source_type == SourceType::Cloud) {
        name = std::string("cloud_") + cloud_bucket;
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
