// DataInputDialog transformation, row filter, and encoding tabs.

#include "node_config_dialog.h"
#include "ui_tokens.h"

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
    if (ImGui::Checkbox("Limit dataset rows", &limit_enabled)) {
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
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("Dataset loading limit, not a preview display limit.");
    if (file_category_ == FileCategory::Text) {
        ImGui::TextColored(cyxwiz::ui::CurrentTokens().warning,
            "The current Text loader does not apply this limit. "
            "It does not reduce the text data available for training.");
    } else {
        ImGui::TextDisabled(
            "For tabular CSV loading, Apply loads at most this many data rows. "
            "Only those loaded rows are available for splitting and training. "
            "Unchecked means no row limit.");
    }
    ImGui::TextDisabled("To change how many preview rows are displayed per page, use Preview > Rows/page.");
    ImGui::TextDisabled(
        "To include or exclude columns by name, use the Transformation tab.");
    ImGui::TextDisabled(
        "To keep only some rows, filter them in Data Studio's Prepare step or with a Filter or SQL node.");
    ImGui::PopTextWrapPos();
}

} // namespace gui
