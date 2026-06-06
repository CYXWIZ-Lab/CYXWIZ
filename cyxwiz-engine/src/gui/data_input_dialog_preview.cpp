// DataInputDialog preview rendering.

#include "node_config_dialog.h"
#include "data_input_preview.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace gui {

void DataInputDialog::RenderPreviewPanel() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "PREVIEW");
    ImGui::Separator();
    ImGui::Spacing();

    if (!preview_loaded_) {
        if (!IsPreviewSupported()) {
            ImGui::BeginDisabled();
            ImGui::Button("Load Preview", ImVec2(-1, 30));
            ImGui::EndDisabled();
            ImGui::Spacing();
            ImGui::TextWrapped("%s", PreviewUnavailableMessage());
            return;
        }
        if (ImGui::Button("Load Preview", ImVec2(-1, 30))) {
            LoadPreview();
        }
        ImGui::Spacing();
        ImGui::TextDisabled("Configure source and click\nLoad Preview to see data");
    } else if (!preview_error_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error: %s", preview_error_.c_str());
    } else {
        if (source_type_ == SourceType::File) {
            switch (file_category_) {
                case FileCategory::Tabular: RenderTabularPreview(); break;
                case FileCategory::Image:   RenderImagePreview(); break;
                case FileCategory::Audio:   RenderAudioPreview(); break;
                case FileCategory::Text:    RenderTextPreview(); break;
                default: RenderTabularPreview(); break;
            }
        } else {
            RenderTabularPreview();
        }
    }

    if (preview_loaded_ && estimated_ram_mb_ > 0) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Text("Est. RAM: %.1f MB", estimated_ram_mb_);
    }
}

void DataInputDialog::RenderTabularPreview() {
    if (preview_columns_.empty()) {
        ImGui::TextDisabled("No data to preview");
        return;
    }

    ImGui::Text("%zu columns, %zu rows", preview_columns_.size(), preview_data_.size());
    ImGui::Spacing();

    if (ImGui::BeginTable("Preview", static_cast<int>(preview_columns_.size()),
        ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
        ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingFixedFit,
        ImVec2(0, ImGui::GetContentRegionAvail().y - 30))) {

        for (const auto& col : preview_columns_) {
            ImGui::TableSetupColumn(col.c_str(), ImGuiTableColumnFlags_WidthFixed, 80.0f);
        }
        ImGui::TableHeadersRow();

        int row_idx = 0;
        for (const auto& row : preview_data_) {
            if (row_idx >= 20) break;
            ImGui::TableNextRow();
            int col_idx = 0;
            for (const auto& cell : row) {
                if (col_idx < static_cast<int>(preview_columns_.size())) {
                    ImGui::TableSetColumnIndex(col_idx);
                    ImGui::TextUnformatted(cell.c_str());
                }
                col_idx++;
            }
            row_idx++;
        }
        ImGui::EndTable();
    }
}

void DataInputDialog::RenderImagePreview() {
    if (preview_image_textures_.empty()) {
        ImGui::TextWrapped("%s", PreviewUnavailableMessage());
        return;
    }

    ImGui::Text("%zu images", preview_image_textures_.size());
    ImGui::Spacing();

    float thumb_size = 64.0f;
    float avail_width = ImGui::GetContentRegionAvail().x;
    int cols = std::max(1, static_cast<int>(avail_width / (thumb_size + 8)));

    int idx = 0;
    for (ImTextureID tex_id : preview_image_textures_) {
        if (idx > 0 && (idx % cols) != 0) {
            ImGui::SameLine();
        }

        ImGui::BeginGroup();
        ImGui::Image(tex_id, ImVec2(thumb_size, thumb_size));
        if (idx < static_cast<int>(preview_image_labels_.size())) {
            ImGui::TextDisabled("%s", preview_image_labels_[idx].c_str());
        }
        ImGui::EndGroup();

        idx++;
        if (idx >= 12) break;
    }
}

void DataInputDialog::RenderAudioPreview() {
    ImGui::TextWrapped("%s", PreviewUnavailableMessage());
}

void DataInputDialog::RenderTextPreview() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    if (text_layout_ == TextLayout::CorpusSubdirs) {
        if (strlen(folder_path_) == 0) {
            ImGui::TextDisabled("No folder selected");
            return;
        }
        if (!fs::exists(folder_path_)) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                               "Folder does not exist");
            return;
        }
        int subdir_count = 0, file_count = 0;
        try {
            for (const auto& entry : fs::directory_iterator(folder_path_)) {
                if (!entry.is_directory()) continue;
                subdir_count++;
                for (const auto& sub : fs::recursive_directory_iterator(entry.path())) {
                    if (!sub.is_regular_file()) continue;
                    std::string ext = sub.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".txt" || ext == ".text" || ext == ".md") file_count++;
                }
            }
        } catch (...) {}
        ImGui::TextColored(accent, "Corpus scan");
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Text("Folder:  %s", folder_path_);
        ImGui::Text("Classes: %d subdirectories", subdir_count);
        ImGui::Text("Samples: %d text files", file_count);
        ImGui::Spacing();
        ImGui::TextDisabled("Each subdirectory name becomes a class label. "
                            "Apply to tokenize + build vocab (runs on a worker thread).");
        return;
    }

    if (preview_columns_.empty()) {
        ImGui::TextDisabled("No data to preview - click Load Preview above.");
        return;
    }

    auto col_exists = [&](const std::string& name) {
        if (name.empty()) return false;
        for (const auto& c : preview_columns_) if (c == name) return true;
        return false;
    };

    ImGui::TextColored(accent, "Column mapping");
    ImGui::Separator();
    ImGui::Spacing();

    std::string text_col_str = text_column_;
    std::string label_col_str = text_label_column_;
    ImVec4 ok_color  = ImVec4(0.6f, 0.9f, 0.6f, 1.0f);
    ImVec4 err_color = ImVec4(1.0f, 0.4f, 0.4f, 1.0f);

    ImGui::Text("Text:  ");
    ImGui::SameLine();
    if (text_col_str.empty()) {
        ImGui::TextColored(err_color, "(not set)");
    } else if (col_exists(text_col_str)) {
        ImGui::TextColored(ok_color, "%s", text_col_str.c_str());
    } else {
        ImGui::TextColored(err_color, "%s (not in file)", text_col_str.c_str());
    }

    ImGui::Text("Label: ");
    ImGui::SameLine();
    if (label_col_str.empty()) {
        ImGui::TextDisabled("(unlabeled)");
    } else if (col_exists(label_col_str)) {
        ImGui::TextColored(ok_color, "%s", label_col_str.c_str());
    } else {
        ImGui::TextColored(err_color, "%s (not in file)", label_col_str.c_str());
    }

    ImGui::Spacing();
    ImGui::Text("%zu columns, %zu rows shown", preview_columns_.size(), preview_data_.size());
    ImGui::Spacing();

    if (col_exists(label_col_str)) {
        if (label_distribution_column_ != label_col_str) {
            UpdateTextLabelDistribution();
        }
        RenderTextLabelDistribution();
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
    }

    if (ImGui::BeginTable("TextPreview", static_cast<int>(preview_columns_.size()),
        ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
        ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingFixedFit,
        ImVec2(0, ImGui::GetContentRegionAvail().y - 30))) {

        for (const auto& col : preview_columns_) {
            ImGui::TableSetupColumn(col.c_str(), ImGuiTableColumnFlags_WidthFixed, 200.0f);
        }
        ImGui::TableHeadersRow();

        int row_idx = 0;
        for (const auto& row : preview_data_) {
            if (row_idx >= 15) break;
            ImGui::TableNextRow();
            int col_idx = 0;
            for (const auto& cell : row) {
                if (col_idx < static_cast<int>(preview_columns_.size())) {
                    ImGui::TableSetColumnIndex(col_idx);
                    ImGui::TextUnformatted(cell.c_str());
                }
                col_idx++;
            }
            row_idx++;
        }
        ImGui::EndTable();
    }
}

void DataInputDialog::UpdateTextLabelDistribution() {
    const auto distribution = data_input::ComputeLabelDistribution(
        preview_columns_,
        preview_data_,
        text_label_column_);
    label_distribution_ = distribution.values;
    label_distribution_column_ = distribution.column;
    label_distribution_total_ = distribution.total;
}

void DataInputDialog::RenderTextLabelDistribution() {
    if (label_distribution_.empty() || label_distribution_total_ == 0) {
        ImGui::TextDisabled("No label distribution available for preview rows.");
        return;
    }

    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];
    ImGui::TextColored(accent, "Class distribution");
    ImGui::Spacing();
    ImGui::Text("%zu classes in %zu preview rows",
                label_distribution_.size(), label_distribution_total_);

    size_t max_count = 1;
    for (const auto& [_, count] : label_distribution_) {
        max_count = std::max(max_count, count);
    }

    const size_t max_rows = std::min<size_t>(label_distribution_.size(), 8);
    for (size_t i = 0; i < max_rows; ++i) {
        const auto& [label, count] = label_distribution_[i];
        float fraction = static_cast<float>(count) / static_cast<float>(max_count);
        float pct = 100.0f * static_cast<float>(count) /
                    static_cast<float>(label_distribution_total_);

        ImGui::TextUnformatted(label.c_str());
        ImGui::SameLine(150.0f);
        ImGui::ProgressBar(fraction, ImVec2(-70.0f, 0.0f), "");
        ImGui::SameLine();
        ImGui::Text("%zu (%.1f%%)", count, pct);
    }

    if (label_distribution_.size() > max_rows) {
        ImGui::TextDisabled("+ %zu more classes", label_distribution_.size() - max_rows);
    }
}

} // namespace gui
