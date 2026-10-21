#include "standardization_panel.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"
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
        ui::BeginSettingsColumn("standardization", 300.0f);
        RenderDataSelector();
        RenderSettings();
        ui::BeginResultsArea();
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
            ui::EmptyState(ICON_FA_CHART_SIMPLE, "No standardization yet",
                           "Pick a dataset and the columns, then Apply Standardization.");
        }
        ui::EndResultsArea();
    }
    ImGui::End();
}

void StandardizationPanel::RenderDataSelector() {
    ui::SettingsSection("Data");
    source_.Render("standardization_dataset", [this](std::shared_ptr<DataTable> table) { UseTable(std::move(table)); });

    if (!column_names_.empty()) {
        const ui::Tokens& t = ui::CurrentTokens();
        ImGui::AlignTextToFramePadding();  // level with the links beside it
        ImGui::TextColored(t.text_dim, "Columns");
        ImGui::SameLine();
        if (ui::LinkButton("Select all")) {
            for (size_t i = 0; i < selected_columns_.size(); ++i) selected_columns_[i] = true;
        }
        ImGui::SameLine();
        if (ui::LinkButton("Clear all")) {
            for (size_t i = 0; i < selected_columns_.size(); ++i) selected_columns_[i] = false;
        }
        ui::BeginInsetList("##columns", 150.0f);
        for (size_t i = 0; i < column_names_.size(); ++i) {
            bool selected = selected_columns_[i];
            if (ImGui::Checkbox(column_names_[i].c_str(), &selected)) {
                selected_columns_[i] = selected;
            }
        }
        ui::EndInsetList();
    }
}

void StandardizationPanel::RenderSettings() {
    ui::SettingsSection("Method");
    ImGui::TextWrapped("Z-Score standardization transforms data to have mean=0 and std=1.");
    ImGui::TextWrapped("Formula: z = (x - mean) / std");

    ImGui::Checkbox("Use Robust Scaling", &use_robust_scaling_);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Use median and IQR instead of mean and std.\nMore robust to outliers.");
    }

    ImGui::Spacing();

    int selected_count = 0;
    for (bool sel : selected_columns_) if (sel) selected_count++;
    bool can_apply = current_table_ != nullptr && selected_count > 0;
    const char* disabled_reason = current_table_ == nullptr ? "Pick a dataset first" : "Select at least one column";

    if (ui::PrimaryButton("Apply Standardization", can_apply, disabled_reason, ui::ButtonSize::Regular, -FLT_MIN)) {
        ApplyStandardization();
    }

    if (!status_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ui::CurrentTokens().text_dim);
        ImGui::TextWrapped("%s", status_message_.c_str());
        ImGui::PopStyleColor();
    }
}

void StandardizationPanel::RenderPreview() {
    if (!transform_result_.success || original_stats_.empty()) return;

    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::Text("Before/After Distribution Comparison:");
    ImGui::Spacing();

    // The first transformed column (transformed_data follows transformed_columns_).
    if (!transformed_columns_.empty() && !transform_result_.transformed_data.empty()) {
        const int vis_idx = transformed_columns_[0];
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

        transformed_data = transform_result_.transformed_data[0];

        if (!original_data.empty() && !transformed_data.empty()) {
            if (ImPlot::BeginPlot("##OriginalHist", ImVec2(ImGui::GetContentRegionAvail().x * 0.48f, 200))) {
                ImPlot::SetupAxes("Value", "Count");
                ImPlot::PlotHistogram("Original", original_data.data(), static_cast<int>(original_data.size()), 30);
                ImPlot::EndPlot();
            }

            ImGui::SameLine();

            if (ImPlot::BeginPlot("##StandardizedHist", ImVec2(ImGui::GetContentRegionAvail().x, 200))) {
                ImPlot::SetupAxes("Z-Score", "Count");
                ImPlot::SetNextFillStyle(ui::WithAlpha(t.series[3], 0.7f));
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

        for (size_t trans_idx = 0; trans_idx < transformed_columns_.size() && trans_idx < transformed_stats_.size(); ++trans_idx) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", column_names_[transformed_columns_[trans_idx]].c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", original_stats_[trans_idx].mean);
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", original_stats_[trans_idx].std_dev);
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", transformed_stats_[trans_idx].mean);
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", transformed_stats_[trans_idx].std_dev);
        }

        ImGui::EndTable();
    }

    // Outlier detection
    ImGui::Spacing();
    ImGui::Text("Outliers (|z| > 3):");
    for (size_t trans_idx = 0;
         trans_idx < transformed_columns_.size() && trans_idx < transform_result_.transformed_data.size(); ++trans_idx) {
        int outlier_count = 0;
        for (double z : transform_result_.transformed_data[trans_idx]) {
            if (std::abs(z) > 3.0) outlier_count++;
        }

        ImGui::BulletText("%s: %d outliers (%.2f%%)",
                         column_names_[transformed_columns_[trans_idx]].c_str(),
                         outlier_count,
                         100.0 * outlier_count / transform_result_.transformed_data[trans_idx].size());
    }
}

void StandardizationPanel::RenderResults() {
    if (!transform_result_.success) return;

    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::Text(ICON_FA_TABLE " Transformed Data Preview");
    ImGui::Spacing();

    int n_cols = static_cast<int>(transform_result_.transformed_data.size());
    int n_rows = n_cols > 0 ? static_cast<int>(transform_result_.transformed_data[0].size()) : 0;
    int show_rows = std::min(n_rows, 20);

    if (n_cols > 0 && ImGui::BeginTable("TransData", n_cols + 1, ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX)) {
        ImGui::TableSetupColumn("Row");

        for (int col = 0; col < n_cols && col < static_cast<int>(transformed_columns_.size()); ++col) {
            ImGui::TableSetupColumn(column_names_[transformed_columns_[col]].c_str());
        }
        ImGui::TableHeadersRow();

        for (int row = 0; row < show_rows; ++row) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%d", row);

            for (int col = 0; col < n_cols; ++col) {
                ImGui::TableNextColumn();
                const auto& values = transform_result_.transformed_data[col];  // columns can differ in length
                if (row >= static_cast<int>(values.size())) continue;
                double z = values[row];
                if (std::abs(z) > 3.0) {
                    ImGui::TextColored(t.error, "%.4f", z);
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

void StandardizationPanel::UseTable(std::shared_ptr<DataTable> table) {
    current_table_ = std::move(table);
    column_names_ = current_table_->GetHeaders();
    selected_columns_.assign(column_names_.size(), false);

    transform_result_ = TransformResult();
    original_stats_.clear();
    transformed_stats_.clear();
    transformed_columns_.clear();
    status_message_ = "Data loaded. Select columns to standardize.";
}

void StandardizationPanel::ApplyStandardization() {
    if (!current_table_) return;

    std::vector<std::vector<double>> data;
    original_stats_.clear();
    transformed_columns_.clear();

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
            transformed_columns_.push_back(static_cast<int>(col));
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
