#include "data_preview_table_renderer.h"

#include <algorithm>
#include <limits>

namespace gui {

DataPreviewTableRenderResult RenderDataPreviewTable(
    const char* table_id,
    const std::vector<cyxwiz::DataPreviewColumn>& columns,
    int64_t first_row,
    int64_t row_count,
    const DataPreviewRowLookup& lookup,
    const ImVec2& size,
    bool show_row_numbers) {
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
                90.0f);
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
                        ImGui::TextUnformatted(
                            (*row)[static_cast<size_t>(data_column)].c_str());
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
