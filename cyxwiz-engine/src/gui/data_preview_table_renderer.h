#pragma once

#include "../core/data_preview_service.h"

#include <imgui.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace gui {

using DataPreviewRow = std::vector<std::string>;
using DataPreviewRowLookup =
    std::function<const DataPreviewRow*(int64_t absolute_row)>;

struct DataPreviewTableRenderResult {
    int64_t first_missing_row = -1;
    int64_t last_visible_row = -1;
};

DataPreviewTableRenderResult RenderDataPreviewTable(
    const char* table_id,
    const std::vector<cyxwiz::DataPreviewColumn>& columns,
    int64_t first_row,
    int64_t row_count,
    const DataPreviewRowLookup& lookup,
    const ImVec2& size,
    bool show_row_numbers = true);

} // namespace gui
