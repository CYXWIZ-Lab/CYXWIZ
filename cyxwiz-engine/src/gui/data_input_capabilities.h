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

bool IsApplySupported(SourceType source_type, cyxwiz::loaders::FileCategory file_category);
bool IsPreviewSupported(SourceType source_type, cyxwiz::loaders::FileCategory file_category);
const char* UnsupportedApplyMessage(SourceType source_type, cyxwiz::loaders::FileCategory file_category);
const char* PreviewUnavailableMessage(SourceType source_type, cyxwiz::loaders::FileCategory file_category);
std::string FormatBytes(std::size_t bytes);

} // namespace gui::data_input
