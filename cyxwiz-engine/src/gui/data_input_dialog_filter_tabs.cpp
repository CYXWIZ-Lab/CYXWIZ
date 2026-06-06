// DataInputDialog transformation, row filter, and encoding tabs.

#include "node_config_dialog.h"

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
        for (size_t i = 0; i < selected_columns_.size(); i++) {
            selected_columns_[i] = select_all_columns_;
        }
    }

    if (!select_all_columns_ && !available_columns_.empty()) {
        ImGui::Spacing();
        ImGui::Text("Select columns to include:");
        ImGui::BeginChild("ColumnList", ImVec2(0, 200), true);
        for (size_t i = 0; i < available_columns_.size() && i < selected_columns_.size(); i++) {
            bool selected = selected_columns_[i];
            if (ImGui::Checkbox(available_columns_[i].c_str(), &selected)) {
                selected_columns_[i] = selected;
                has_changes_ = true;
            }
        }
        ImGui::EndChild();
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

    ImGui::Text("Skip first N rows:");
    ImGui::SameLine(180);
    ImGui::SetNextItemWidth(100);
    if (ImGui::InputInt("##skip", &skip_rows_)) {
        if (skip_rows_ < 0) skip_rows_ = 0;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(excluding header)");

    ImGui::Spacing();

    static bool limit_enabled = false;
    ImGui::Checkbox("Limit number of rows", &limit_enabled);
    if (limit_enabled) {
        ImGui::SameLine(180);
        ImGui::SetNextItemWidth(100);
        if (ImGui::InputInt("##maxrows", &max_rows_)) {
            if (max_rows_ < 1) max_rows_ = 1;
            has_changes_ = true;
        }
    }

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
