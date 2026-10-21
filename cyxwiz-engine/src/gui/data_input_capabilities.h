#pragma once

#include "loaders/data_loader.h"
#include <cstddef>
#include <string>
#include <vector>

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
// What a chosen folder holds (Data Input's Source, TOFIX129 B-10): images,
// audio or text files, in class subfolders or flat.
enum class FolderKind { Unknown, Images, Audio, Text };
struct FolderScan {
    FolderKind kind = FolderKind::Unknown;
    int files = 0;                         // files of that kind
    std::vector<std::string> class_names;  // subfolders holding them (sorted)
    bool truncated = false;                // stopped at max_files
};
// Walks the folder (one level of class subfolders, their files recursively)
// up to max_files files and names the kind most of them are.
FolderScan ScanFolder(const std::string& path, int max_files = 200000);

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
const char* BackendSummary(int loaded_backend);
std::string GenerateDatasetName(const std::string& file_path, const std::string& folder_path);
// `base`, or base_2, base_3, ... when another Data Input in the graph already
// uses it (two files with the same name in different folders). Same-file
// inputs still share one table in the registry (TOFIX160 P2).
std::string UniqueDatasetName(const std::string& base, const std::vector<std::string>& taken);
std::string FormatBytes(std::size_t bytes);

} // namespace gui::data_input
