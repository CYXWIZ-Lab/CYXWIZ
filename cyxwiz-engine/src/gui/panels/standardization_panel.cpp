#include "standardization_panel.h"
#include "../icons.h"
#include "../../core/data_registry.h"
#include "../../data/data_table.h"
#include "../../core/data_analyzer.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>

namespace cyxwiz {

StandardizationPanel::StandardizationPanel() : Panel("Standardization", true) {}

void StandardizationPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin((std::string(ICON_FA_CHART_SIMPLE) + " Standardization (Z-Score)###StandardizationPanel").c_str(), &visible_)) {
        if (data_registry_) available_tables_ = data_registry_->GetTableNames();

        ImGui::BeginChild("ConfigPanel", ImVec2(280, 0), true);
        RenderDataSelector();
        ImGui::Separator();
        RenderSettings();
        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);
        if (transform_result_.success) {
            if (ImGui::BeginTabBar("StdTabs")) {
                if (ImGui::BeginTabItem(ICON_FA_CHART_BAR " Preview")) {
                    RenderPreview();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_TABLE " Results")) {
                    RenderResults();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        } else {
            ImGui::Text("Select data and configure settings, then click 'Apply'.");
            ImGui::Spacing();
            ImGui::TextWrapped("Z-Score standardization transforms data to have mean=0 and std=1.");
            ImGui::TextWrapped("Formula: z = (x - mean) / std");
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

void StandardizationPanel::RenderDataSelector() {
    ImGui::Text(ICON_FA_TABLE " Data Selection");
    ImGui::Spacing();

    if (ImGui::BeginCombo("Dataset", selected_table_idx_ >= 0 && selected_table_idx_ < static_cast<int>(available_tables_.size())
                                     ? available_tables_[selected_table_idx_].c_str() : "Select...")) {
        for (int i = 0; i < static_cast<int>(available_tables_.size()); ++i) {
            if (ImGui::Selectable(available_tables_[i].c_str(), i == selected_table_idx_)) {
                selected_table_idx_ = i;
                LoadSelectedData();
            }
        }
        ImGui::EndCombo();
    }

    if (!column_names_.empty()) {
        ImGui::Spacing();
        ImGui::Text("Select Columns:");
        ImGui::BeginChild("ColumnSelect", ImVec2(0, 150), true);
        for (size_t i = 0; i < column_names_.size(); ++i) {
            bool selected = selected_columns_[i];
            if (ImGui::Checkbox(column_names_[i].c_str(), &selected)) {
                selected_columns_[i] = selected;
            }
        }
        ImGui::EndChild();

        if (ImGui::Button("Select All", ImVec2(80, 0))) {
            for (size_t i = 0; i < selected_columns_.size(); ++i) selected_columns_[i] = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear All", ImVec2(80, 0))) {
            for (size_t i = 0; i < selected_columns_.size(); ++i) selected_columns_[i] = false;
        }
    }
}

void StandardizationPanel::RenderSettings() {
    ImGui::Text(ICON_FA_COG " Settings");
    ImGui::Spacing();

    ImGui::Checkbox("Use Robust Scaling", &use_robust_scaling_);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Use median and IQR instead of mean and std.\nMore robust to outliers.");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    int selected_count = 0;
    for (bool sel : selected_columns_) if (sel) selected_count++;
    bool can_apply = current_table_ != nullptr && selected_count > 0;

    if (!can_apply) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_PLAY " Apply Standardization", ImVec2(-1, 30))) {
        ApplyStandardization();
    }

    if (!can_apply) ImGui::EndDisabled();

    if (!status_message_.empty()) {
        ImGui::Spacing();
        ImGui::TextWrapped("%s", status_message_.c_str());
    }
}

void StandardizationPanel::RenderPreview() {
    if (!transform_result_.success || original_stats_.empty()) return;

    ImGui::Text("Before/After Distribution Comparison:");
    ImGui::Spacing();

    // Find first selected column for visualization
    int vis_idx = 0;
    for (size_t i = 0; i < selected_columns_.size(); ++i) {
        if (selected_columns_[i]) {
            vis_idx = static_cast<int>(i);
            break;
        }
    }

    if (vis_idx < static_cast<int>(original_stats_.size())) {
        ImGui::Text("Column: %s", column_names_[vis_idx].c_str());

        std::vector<double> original_data;
        std::vector<double> transformed_data;

        if (current_table_) {
            int n_rows = static_cast<int>(current_table_->GetRowCount());
            for (int row = 0; row < n_rows && row < 10000; ++row) {
                auto val = DataAnalyzer::ToDouble(current_table_->GetCell(row, vis_idx));
                if (val.has_value()) {
                    original_data.push_back(val.value());
                }
            }
        }

        if (vis_idx < static_cast<int>(transform_result_.transformed_data.size())) {
            transformed_data = transform_result_.transformed_data[vis_idx];
        }

        if (!original_data.empty() && !transformed_data.empty()) {
            if (ImPlot::BeginPlot("##OriginalHist", ImVec2(ImGui::GetContentRegionAvail().x * 0.48f, 200))) {
                ImPlot::SetupAxes("Value", "Count");
                ImPlot::PlotHistogram("Original", original_data.data(), static_cast<int>(original_data.size()), 30);
                ImPlot::EndPlot();
            }

            ImGui::SameLine();

            if (ImPlot::BeginPlot("##StandardizedHist", ImVec2(ImGui::GetContentRegionAvail().x, 200))) {
                ImPlot::SetupAxes("Z-Score", "Count");
                ImPlot::SetNextFillStyle(ImVec4(0.2f, 0.6f, 1.0f, 0.7f));
                ImPlot::PlotHistogram("Standardized", transformed_data.data(), static_cast<int>(transformed_data.size()), 30);
                ImPlot::EndPlot();
            }
        }
    }

    // Stats table
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Statistics Comparison:");

    if (ImGui::BeginTable("StatsCompare", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Column");
        ImGui::TableSetupColumn("Orig Mean");
        ImGui::TableSetupColumn("Orig Std");
        ImGui::TableSetupColumn("New Mean");
        ImGui::TableSetupColumn("New Std");
        ImGui::TableHeadersRow();

        int trans_idx = 0;
        for (size_t i = 0; i < selected_columns_.size(); ++i) {
            if (!selected_columns_[i]) continue;
            if (trans_idx >= static_cast<int>(transformed_stats_.size())) break;

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", column_names_[i].c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", original_stats_[trans_idx].mean);
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", original_stats_[trans_idx].std_dev);
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", transformed_stats_[trans_idx].mean);
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", transformed_stats_[trans_idx].std_dev);

            trans_idx++;
        }

        ImGui::EndTable();
    }

    // Outlier detection
    ImGui::Spacing();
    ImGui::Text("Outliers (|z| > 3):");
    int trans_idx = 0;
    for (size_t i = 0; i < selected_columns_.size(); ++i) {
        if (!selected_columns_[i]) continue;
        if (trans_idx >= static_cast<int>(transform_result_.transformed_data.size())) break;

        int outlier_count = 0;
        for (double z : transform_result_.transformed_data[trans_idx]) {
            if (std::abs(z) > 3.0) outlier_count++;
        }

        ImGui::BulletText("%s: %d outliers (%.2f%%)",
                         column_names_[i].c_str(),
                         outlier_count,
                         100.0 * outlier_count / transform_result_.transformed_data[trans_idx].size());
        trans_idx++;
    }
}

void StandardizationPanel::RenderResults() {
    if (!transform_result_.success) return;

    ImGui::Text(ICON_FA_TABLE " Transformed Data Preview");
    ImGui::Spacing();

    int n_cols = static_cast<int>(transform_result_.transformed_data.size());
    int n_rows = n_cols > 0 ? static_cast<int>(transform_result_.transformed_data[0].size()) : 0;
    int show_rows = std::min(n_rows, 20);

    if (n_cols > 0 && ImGui::BeginTable("TransData", n_cols + 1, ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX)) {
        ImGui::TableSetupColumn("Row");

        int col_idx = 0;
        for (size_t i = 0; i < selected_columns_.size(); ++i) {
            if (selected_columns_[i]) {
                ImGui::TableSetupColumn(column_names_[i].c_str());
                col_idx++;
            }
        }
        ImGui::TableHeadersRow();

        for (int row = 0; row < show_rows; ++row) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%d", row);

            for (int col = 0; col < n_cols; ++col) {
                ImGui::TableNextColumn();
                double z = transform_result_.transformed_data[col][row];
                if (std::abs(z) > 3.0) {
                    ImGui::TextColored(ImVec4(1, 0.3f, 0.3f, 1), "%.4f", z);
                } else {
                    ImGui::Text("%.4f", z);
                }
            }
        }

        ImGui::EndTable();
    }

    if (n_rows > 20) {
        ImGui::Text("Showing first 20 of %d rows", n_rows);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Transform Parameters:");
    ImGui::BulletText("Method: %s", use_robust_scaling_ ? "Robust (median/IQR)" : "Z-Score (mean/std)");

    int param_idx = 0;
    for (size_t i = 0; i < selected_columns_.size(); ++i) {
        if (!selected_columns_[i]) continue;

        std::string key_mean = "col" + std::to_string(param_idx) + (use_robust_scaling_ ? "_median" : "_mean");
        std::string key_scale = "col" + std::to_string(param_idx) + (use_robust_scaling_ ? "_iqr" : "_std");

        if (transform_result_.params.count(key_mean) && transform_result_.params.count(key_scale)) {
            ImGui::BulletText("%s: center=%.4f, scale=%.4f",
                             column_names_[i].c_str(),
                             transform_result_.params[key_mean],
                             transform_result_.params[key_scale]);
        }
        param_idx++;
    }
}

void StandardizationPanel::LoadSelectedData() {
    if (selected_table_idx_ < 0 || !data_registry_) return;

    current_table_ = data_registry_->GetTable(available_tables_[selected_table_idx_]);
    if (!current_table_) return;

    column_names_ = current_table_->GetHeaders();
    selected_columns_.assign(column_names_.size(), false);

    transform_result_ = TransformResult();
    original_stats_.clear();
    transformed_stats_.clear();
    status_message_ = "Data loaded. Select columns to standardize.";
}

void StandardizationPanel::ApplyStandardization() {
    if (!current_table_) return;

    std::vector<std::vector<double>> data;
    original_stats_.clear();

    for (size_t col = 0; col < selected_columns_.size(); ++col) {
        if (!selected_columns_[col]) continue;

        std::vector<double> col_data;
        int n_rows = static_cast<int>(current_table_->GetRowCount());
        for (int row = 0; row < n_rows; ++row) {
            auto val = DataAnalyzer::ToDouble(current_table_->GetCell(row, static_cast<int>(col)));
            if (val.has_value()) {
                col_data.push_back(val.value());
            }
        }

        if (!col_data.empty()) {
            original_stats_.push_back(DataTransform::ComputeColumnStats(col_data));
            data.push_back(col_data);
        }
    }

    if (data.empty()) {
        status_message_ = "No valid numeric data found.";
        return;
    }

    if (use_robust_scaling_) {
        transform_result_ = DataTransform::RobustScale(data);
    } else {
        transform_result_ = DataTransform::Standardize(data);
    }

    if (transform_result_.success) {
        transformed_stats_.clear();
        for (const auto& col : transform_result_.transformed_data) {
            transformed_stats_.push_back(DataTransform::ComputeColumnStats(col));
        }
        status_message_ = "Standardization applied successfully.";
    } else {
        status_message_ = "Error: " + transform_result_.error_message;
    }
}

} // namespace cyxwiz
