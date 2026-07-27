// DataInputDialog transformation, row filter, and encoding tabs.

#include "node_config_dialog.h"

#include <algorithm>
#include <cstring>

namespace gui {

void DataInputDialog::RenderTransformationTab() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "Column Transformations");
    ImGui::Separator();
    ImGui::Spacing();

    if (ImGui::Checkbox("Include all columns", &select_all_columns_)) {
        has_changes_ = true;
        if (select_all_columns_) {
            for (size_t i = 0; i < selected_columns_.size(); i++) {
                selected_columns_[i] = true;
            }
        }
    }

    if (!select_all_columns_ && !available_columns_.empty()) {
        ImGui::Spacing();
        ImGui::Text("Select columns to include:");
        ImGui::BeginChild("ColumnList", ImVec2(0, 200), true);
        for (size_t i = 0; i < available_columns_.size() && i < selected_columns_.size(); i++) {
            const bool is_label =
                label_column_idx_ >= 0 &&
                i == static_cast<std::size_t>(label_column_idx_);
            if (is_label) {
                selected_columns_[i] = true;
                ImGui::BeginDisabled();
            }
            bool selected = selected_columns_[i];
            if (ImGui::Checkbox(available_columns_[i].c_str(), &selected)) {
                selected_columns_[i] = selected;
                has_changes_ = true;
            }
            if (is_label) {
                ImGui::SameLine();
                ImGui::TextDisabled("(label)");
                ImGui::EndDisabled();
            }
        }
        ImGui::EndChild();
    }

    if (!available_columns_.empty()) {
        const std::size_t included = select_all_columns_
            ? available_columns_.size()
            : static_cast<std::size_t>(std::count(
                  selected_columns_.begin(), selected_columns_.end(), true));
        ImGui::TextDisabled(
            "%zu of %zu columns will be loaded. Selection is stored by name.",
            included, available_columns_.size());
    }

    if (available_columns_.empty() && strlen(file_path_) > 0) {
        ImGui::TextDisabled("Load preview to see column list");
    }
}

void DataInputDialog::RenderLimitRowsTab() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "Row Filtering Options");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Skip first N source rows:");
    ImGui::SameLine(180);
    ImGui::SetNextItemWidth(100);
    if (ImGui::InputInt("##skip", &skip_rows_)) {
        if (skip_rows_ < 0) skip_rows_ = 0;
        RefreshColumnList();
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(before parsing)");
    ImGui::TextDisabled(
        "With Has header enabled, the next source row becomes the header.");

    ImGui::Spacing();

    bool limit_enabled = max_rows_ > 0;
    if (ImGui::Checkbox("Limit number of rows", &limit_enabled)) {
        max_rows_ = limit_enabled
            ? (max_rows_ > 0 ? max_rows_ : 1000)
            : 0;
        ResetPreviewPaging();
        preview_loaded_ = false;
        has_changes_ = true;
    }
    if (limit_enabled) {
        ImGui::SameLine(180);
        ImGui::SetNextItemWidth(100);
        if (ImGui::InputInt("##maxrows", &max_rows_)) {
            if (max_rows_ < 1) max_rows_ = 1;
            ResetPreviewPaging();
            preview_loaded_ = false;
            has_changes_ = true;
        }
    }
    ImGui::TextDisabled(
        "Caps rows available to splitting/training after Apply; unchecked means all rows.");
    if (max_rows_ > 0) {
        ImGui::TextDisabled(
            "CSV ingestion stops after this many data rows for both memory and disk-backed modes.");
    }
    ImGui::TextDisabled(
        "To include or exclude columns by name, use the Transformation tab.");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Filter rows (SQL WHERE clause):");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputTextMultiline("##where", where_clause_, sizeof(where_clause_), ImVec2(0, 60))) {
        has_changes_ = true;
    }
    ImGui::TextDisabled("Example: age > 18 AND status = 'active'");
}

void DataInputDialog::RenderEncodingTab() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "Character Encoding");
    ImGui::Separator();
    ImGui::Spacing();

    const char* encodings[] = {"Auto-detect", "UTF-8", "UTF-16", "ASCII", "ISO-8859-1", "Windows-1252", "UTF-8 with BOM"};
    ImGui::Text("File encoding:");
    ImGui::SameLine(150);
    ImGui::SetNextItemWidth(150);
    if (ImGui::Combo("##encoding", &encoding_idx_, encodings, 7)) {
        has_changes_ = true;
    }

    ImGui::Spacing();
    ImGui::TextDisabled("Auto-detect will scan the first bytes of the file to determine encoding.");
}

} // namespace gui
