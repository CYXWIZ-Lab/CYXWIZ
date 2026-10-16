#pragma once

#include "loaders/data_loader.h"
#include <cstddef>
#include <string>

namespace gui::data_input {

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

cyxwiz::loaders::FileCategory FileCategoryFromParam(
    const std::string& value,
    cyxwiz::loaders::FileCategory fallback = cyxwiz::loaders::FileCategory::Tabular);
const char* FileCategoryParam(cyxwiz::loaders::FileCategory file_category);
bool IsPreviewSupported(cyxwiz::loaders::FileCategory file_category);
const char* PreviewUnavailableMessage(cyxwiz::loaders::FileCategory file_category);
const char* FileTypeParam(int detected_type);
const char* FileTypeName(int detected_type);
int FileTypeFromParam(const std::string& value, int fallback = 0);
int DetectFileTypeForPath(const std::string& path, std::size_t* file_size);
cyxwiz::loaders::FileCategory DetectFileCategoryForPath(
    const std::string& path,
    cyxwiz::loaders::FileCategory current_category);
bool UsesFolderPath(cyxwiz::loaders::FileCategory file_category, TextLayout text_layout);
std::string CurrentSourceLabel(
    cyxwiz::loaders::FileCategory file_category,
    TextLayout text_layout);
std::string CurrentApplySummary(
    cyxwiz::loaders::FileCategory file_category,
    TextLayout text_layout,
    bool force_disk_backed,
    int max_rows,
    int skip_rows);
const char* BackendSummary(int loaded_backend);
std::string GenerateDatasetName(const std::string& file_path, const std::string& folder_path);
std::string FormatBytes(std::size_t bytes);

} // namespace gui::data_input
