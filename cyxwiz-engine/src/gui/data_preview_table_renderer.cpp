#include "data_preview_table_renderer.h"

#include <algorithm>
#include <limits>

namespace gui {

int64_t RenderDataPreviewControls(DataPreviewViewState& state, int64_t total_rows) {
    total_rows = std::max<int64_t>(0, total_rows);
    ImGui::SetNextItemWidth(75.0f);
    const std::string current = std::to_string(state.rows_per_page);
    if (ImGui::BeginCombo("Rows/page", current.c_str())) {
        for (int count : {5, 10, 25, 50}) {
            if (ImGui::Selectable(std::to_string(count).c_str(), count == state.rows_per_page)) {
                state.rows_per_page = count;
                state.offset = 0;
            }
        }
        ImGui::EndCombo();
    }
    state.rows_per_page = std::clamp(state.rows_per_page, 1, 50);
    const int64_t last_offset = total_rows > 0
        ? ((total_rows - 1) / state.rows_per_page) * state.rows_per_page : 0;
    state.offset = std::clamp<int64_t>(state.offset, 0, last_offset);
    ImGui::BeginDisabled(state.offset == 0);
    if (ImGui::Button("First")) state.offset = 0;
    ImGui::SameLine();
    if (ImGui::Button("Previous")) {
        state.offset = std::max<int64_t>(0, state.offset - state.rows_per_page);
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(state.offset >= last_offset);
    if (ImGui::Button("Next")) state.offset += state.rows_per_page;
    ImGui::SameLine();
    if (ImGui::Button("Last")) state.offset = last_offset;
    ImGui::EndDisabled();
    const auto count = std::min<int64_t>(state.rows_per_page, total_rows - state.offset);
    ImGui::TextDisabled("Rows %lld-%lld of %lld",
        static_cast<long long>(count ? state.offset + 1 : 0),
        static_cast<long long>(state.offset + count), static_cast<long long>(total_rows));
    return count;
}

void RenderDataPreviewCellDetails(DataPreviewViewState& state) {
    if (state.inspect_cell) ImGui::OpenPopup("Preview cell");
    bool open = true;
    const auto work_size = ImGui::GetMainViewport()->WorkSize;
    ImGui::SetNextWindowSize(ImVec2(std::min(640.0f, work_size.x * 0.9f),
                                  std::min(400.0f, work_size.y * 0.8f)), ImGuiCond_FirstUseEver);
    if (ImGui::BeginPopupModal("Preview cell", &open)) {
        ImGui::TextWrapped("Row %lld | %s", static_cast<long long>(state.selected_row + 1),
                           state.selected_column.c_str());
        if (ImGui::Button("Copy cell")) ImGui::SetClipboardText(state.selected_cell.c_str());
        ImGui::SameLine();
        if (ImGui::Button("Close") || ImGui::IsKeyPressed(ImGuiKey_Escape)) open = false;
        ImGui::InputTextMultiline("##cell_contents", state.selected_cell.data(),
            state.selected_cell.size() + 1, ImVec2(-1, -1),
            ImGuiInputTextFlags_ReadOnly | ImGuiInputTextFlags_NoHorizontalScroll);
        if (!open) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
    if (!open) {
        state.inspect_cell = false;
        state.selected_cell.clear();
        state.selected_column.clear();
        state.selected_row = -1;
    }
}

DataPreviewTableRenderResult RenderDataPreviewTable(
    const char* table_id,
    const std::vector<cyxwiz::DataPreviewColumn>& columns,
    int64_t first_row,
    int64_t row_count,
    const DataPreviewRowLookup& lookup,
    const ImVec2& size,
    bool show_row_numbers,
    const char* wide_column,
    DataPreviewViewState* view_state) {
    DataPreviewTableRenderResult result;
    if (columns.empty() || !lookup) {
        ImGui::TextDisabled("No data to preview");
        return result;
    }

    const int data_column_count = static_cast<int>(std::min<size_t>(
        columns.size(), static_cast<size_t>(std::numeric_limits<int>::max() - 1)));
    const int table_column_count = data_column_count + (show_row_numbers ? 1 : 0);
    const ImGuiTableFlags flags =
        ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX |
        ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg |
        ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingFixedFit;

    ImGui::PushID(table_id);
    if (ImGui::BeginTable("##bounded_data_preview", table_column_count, flags, size)) {
        if (show_row_numbers) {
            ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 52.0f);
        }
        for (int index = 0; index < data_column_count; ++index) {
            ImGui::TableSetupColumn(
                columns[static_cast<size_t>(index)].name.c_str(),
                ImGuiTableColumnFlags_WidthFixed,
                wide_column && columns[static_cast<size_t>(index)].name == wide_column
                    ? 360.0f : 120.0f);
        }
        ImGui::TableSetupScrollFreeze(show_row_numbers ? 1 : 0, 1);
        ImGui::TableHeadersRow();

        const int clipped_rows = static_cast<int>(std::min<int64_t>(
            std::max<int64_t>(0, row_count),
            std::numeric_limits<int>::max()));
        ImGuiListClipper clipper;
        clipper.Begin(clipped_rows);
        while (clipper.Step()) {
            for (int local_row = clipper.DisplayStart;
                 local_row < clipper.DisplayEnd;
                 ++local_row) {
                const int64_t absolute_row = first_row + local_row;
                result.last_visible_row = absolute_row;
                const auto* row = lookup(absolute_row);
                if (!row && result.first_missing_row < 0) {
                    result.first_missing_row = absolute_row;
                }

                ImGui::TableNextRow();
                int table_column = 0;
                if (show_row_numbers) {
                    ImGui::TableSetColumnIndex(table_column++);
                    ImGui::TextDisabled("%lld",
                        static_cast<long long>(absolute_row + 1));
                }
                for (int data_column = 0;
                     data_column < data_column_count;
                     ++data_column, ++table_column) {
                    if (!ImGui::TableSetColumnIndex(table_column)) continue;
                    if (!row) {
                        ImGui::TextDisabled("...");
                    } else if (data_column < static_cast<int>(row->size())) {
                        const auto& cell = (*row)[static_cast<size_t>(data_column)];
                        // Keep clipped rows one line high, including multiline CSV fields.
                        const auto line_end = cell.find_first_of("\r\n");
                        ImGui::TextUnformatted(cell.data(), cell.data() +
                            std::min(cell.size(), line_end));
                        if (ImGui::IsItemHovered()) {
                            if (view_state && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                                view_state->selected_cell = cell;
                                view_state->selected_column = columns[static_cast<size_t>(data_column)].name;
                                view_state->selected_row = absolute_row;
                                view_state->inspect_cell = true;
                            }
                            ImGui::BeginTooltip();
                            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 40.0f);
                            ImGui::TextUnformatted(cell.data(), cell.data() +
                                std::min<size_t>(cell.size(), 4096));
                            ImGui::PopTextWrapPos();
                            if (cell.size() > 4096) {
                                ImGui::TextDisabled("Preview truncated at 4096 bytes.");
                            }
                            ImGui::TextDisabled("Right-click to copy the full cell");
                            if (view_state) ImGui::TextDisabled("Click to open the cell reader");
                            ImGui::EndTooltip();
                            if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                                ImGui::SetClipboardText(cell.c_str());
                            }
                        }
                    }
                }
            }
        }
        ImGui::EndTable();
    }
    ImGui::PopID();

    int64_t sampled_values = 0;
    int64_t sampled_nulls = 0;
    for (const auto& column : columns) {
        sampled_values += column.sampled_values;
        sampled_nulls += column.sampled_nulls;
    }
    if (sampled_values > 0) {
        ImGui::TextDisabled("Page nulls: %lld of %lld sampled values",
            static_cast<long long>(sampled_nulls),
            static_cast<long long>(sampled_values));
    }

    return result;
}

} // namespace gui
