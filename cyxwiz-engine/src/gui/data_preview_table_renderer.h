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

// Owned by the preview dialog, never shared between tables or datasets.
struct DataPreviewViewState {
    int rows_per_page = 10;
    int64_t offset = 0;
    std::string selected_cell;
    std::string selected_column;
    int64_t selected_row = -1;
    bool inspect_cell = false;
};

// Returns the bounded number of rows to render starting at state.offset.
int64_t RenderDataPreviewControls(DataPreviewViewState& state, int64_t total_rows);
void RenderDataPreviewCellDetails(DataPreviewViewState& state);

DataPreviewTableRenderResult RenderDataPreviewTable(
    const char* table_id,
    const std::vector<cyxwiz::DataPreviewColumn>& columns,
    int64_t first_row,
    int64_t row_count,
    const DataPreviewRowLookup& lookup,
    const ImVec2& size,
    bool show_row_numbers = true,
    const char* wide_column = nullptr,
    DataPreviewViewState* view_state = nullptr);

} // namespace gui
