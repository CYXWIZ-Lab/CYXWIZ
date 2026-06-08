#pragma once

#include "loaders/data_loader.h"
#include <cstddef>
#include <string>

namespace gui::data_input {

enum class SourceType {
    File,
    MLDataset,
    Database,
    Cloud,
};

enum class MLDatasetType {
    MNIST,
    CIFAR10,
    CIFAR100,
    FashionMNIST,
    ImageNet,
    ImageFolder,
    HuggingFace,
    Kaggle,
    Custom,
};

enum class DatabaseType {
    SQLite,
    PostgreSQL,
    MySQL,
    DuckDB,
};

enum class ImageLayout {
    ClassSubdirs = 0,
    FlatWithCSV = 1,
};

enum class AudioLayout {
    ClassSubdirs = 0,
    FlatWithCSV = 1,
};

enum class TextLayout {
    SingleFile = 0,
    CorpusSubdirs = 1,
};

SourceType SourceTypeFromParam(const std::string& value, SourceType fallback = SourceType::File);
cyxwiz::loaders::FileCategory FileCategoryFromParam(
    const std::string& value,
    cyxwiz::loaders::FileCategory fallback = cyxwiz::loaders::FileCategory::Tabular);
const char* SourceTypeParam(SourceType source_type);
const char* FileCategoryParam(cyxwiz::loaders::FileCategory file_category);
const char* MLDatasetTypeParam(MLDatasetType dataset_type);
const char* DatabaseTypeParam(DatabaseType database_type);
bool IsApplySupported(SourceType source_type, cyxwiz::loaders::FileCategory file_category);
bool IsPreviewSupported(SourceType source_type, cyxwiz::loaders::FileCategory file_category);
const char* UnsupportedApplyMessage(SourceType source_type, cyxwiz::loaders::FileCategory file_category);
const char* PreviewUnavailableMessage(SourceType source_type, cyxwiz::loaders::FileCategory file_category);
const char* FileTypeParam(int detected_type);
const char* FileTypeName(int detected_type);
int FileTypeFromParam(const std::string& value, int fallback = 0);
int DetectFileTypeForPath(const std::string& path, std::size_t* file_size);
cyxwiz::loaders::FileCategory DetectFileCategoryForPath(
    const std::string& path,
    cyxwiz::loaders::FileCategory current_category);
bool UsesFolderPath(cyxwiz::loaders::FileCategory file_category, TextLayout text_layout);
std::string CurrentSourceLabel(
    SourceType source_type,
    cyxwiz::loaders::FileCategory file_category,
    TextLayout text_layout);
std::string CurrentApplySummary(
    SourceType source_type,
    cyxwiz::loaders::FileCategory file_category,
    TextLayout text_layout,
    bool force_disk_backed,
    int max_rows,
    int skip_rows);
const char* BackendSummary(int loaded_backend);
std::string GenerateDatasetName(
    SourceType source_type,
    const std::string& file_path,
    const std::string& folder_path,
    const std::string& dataset_name,
    const std::string& db_name,
    const std::string& cloud_bucket);
std::string FormatBytes(std::size_t bytes);

} // namespace gui::data_input
