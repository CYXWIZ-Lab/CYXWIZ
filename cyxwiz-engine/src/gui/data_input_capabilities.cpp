#include "data_input_capabilities.h"
#include <cstdio>

namespace gui::data_input {

bool IsApplySupported(SourceType source_type, cyxwiz::loaders::FileCategory file_category) {
    if (source_type != SourceType::File) {
        return false;
    }
    return file_category != cyxwiz::loaders::FileCategory::Video;
}

bool IsPreviewSupported(SourceType source_type, cyxwiz::loaders::FileCategory file_category) {
    return source_type == SourceType::File &&
           (file_category == cyxwiz::loaders::FileCategory::Tabular ||
            file_category == cyxwiz::loaders::FileCategory::Text ||
            file_category == cyxwiz::loaders::FileCategory::TimeSeries);
}

const char* UnsupportedApplyMessage(SourceType source_type, cyxwiz::loaders::FileCategory file_category) {
    if (source_type == SourceType::File && file_category == cyxwiz::loaders::FileCategory::Video) {
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
        if (file_category == cyxwiz::loaders::FileCategory::Image) {
            return "Image preview is not wired yet. Apply can still scan supported image folders.";
        }
        if (file_category == cyxwiz::loaders::FileCategory::Audio) {
            return "Audio preview is not wired yet. Apply can still scan supported audio folders.";
        }
        if (file_category == cyxwiz::loaders::FileCategory::Video) {
            return "Video preview is not available because video loading is not wired yet.";
        }
    }
    return UnsupportedApplyMessage(source_type, file_category);
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
