#include "feature_scaling_panel.h"
#include "../icons.h"
#include "../../core/data_registry.h"
#include "../../data/data_table.h"
#include "../../core/data_analyzer.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>

namespace cyxwiz {

FeatureScalingPanel::FeatureScalingPanel() : Panel("Feature Scaling", true) {}

void FeatureScalingPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(900, 700), ImGuiCond_FirstUseEver);

    if (ImGui::Begin((std::string(ICON_FA_LAYER_GROUP) + " Feature Scaling (All Methods)###FeatureScalingPanel").c_str(), &visible_)) {
        if (data_registry_) available_tables_ = data_registry_->GetTableNames();

        ImGui::BeginChild("ConfigPanel", ImVec2(300, 0), true);
        RenderDataSelector();
        ImGui::Separator();
        RenderMethodTabs();
        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);
        if (transform_result_.success) {
            if (ImGui::BeginTabBar("ScalingResultTabs")) {
                if (ImGui::BeginTabItem(ICON_FA_CHART_BAR " Preview")) {
                    RenderPreview();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_CHART_LINE " Comparison")) {
                    RenderComparison();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        } else {
            ImGui::Text("Select data, choose a method, and click 'Apply'.");
            ImGui::Spacing();
            ImGui::Text("Available Scaling Methods:");
            ImGui::BulletText("Min-Max: Scale to [0,1] or custom range");
            ImGui::BulletText("Z-Score: Mean=0, Std=1");
            ImGui::BulletText("Robust: Use median/IQR (outlier resistant)");
            ImGui::BulletText("Max-Abs: Scale by max absolute value");
            ImGui::BulletText("Quantile: Uniform or normal distribution");
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

void FeatureScalingPanel::RenderDataSelector() {
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
        ImGui::BeginChild("ColumnSelect", ImVec2(0, 120), true);
        for (size_t i = 0; i < column_names_.size(); ++i) {
            bool selected = selected_columns_[i];
            if (ImGui::Checkbox(column_names_[i].c_str(), &selected)) {
                selected_columns_[i] = selected;
            }
        }
        ImGui::EndChild();

        if (ImGui::Button("All", ImVec2(50, 0))) {
            for (size_t i = 0; i < selected_columns_.size(); ++i) selected_columns_[i] = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("None", ImVec2(50, 0))) {
            for (size_t i = 0; i < selected_columns_.size(); ++i) selected_columns_[i] = false;
        }
    }
}

void FeatureScalingPanel::RenderMethodTabs() {
    ImGui::Text(ICON_FA_COG " Scaling Method");
    ImGui::Spacing();

    if (ImGui::BeginTabBar("MethodTabs")) {
        if (ImGui::BeginTabItem("Min-Max")) {
            selected_method_ = ScalingMethod::MinMax;
            RenderMinMaxTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Z-Score")) {
            selected_method_ = ScalingMethod::ZScore;
            RenderZScoreTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Robust")) {
            selected_method_ = ScalingMethod::Robust;
            RenderRobustTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("MaxAbs")) {
            selected_method_ = ScalingMethod::MaxAbs;
            RenderMaxAbsTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Quantile")) {
            selected_method_ = ScalingMethod::Quantile;
            RenderQuantileTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    int selected_count = 0;
    for (bool sel : selected_columns_) if (sel) selected_count++;
    bool can_apply = current_table_ != nullptr && selected_count > 0;

    if (!can_apply) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_PLAY " Apply Scaling", ImVec2(-1, 30))) {
        ApplyScaling();
    }

    ImGui::Spacing();

    if (ImGui::Button(ICON_FA_CHART_SIMPLE " Compare All Methods", ImVec2(-1, 25))) {
        // Apply all methods for comparison
        show_comparison_ = true;

        std::vector<std::vector<double>> data;
        for (size_t col = 0; col < selected_columns_.size(); ++col) {
            if (!selected_columns_[col]) continue;
            std::vector<double> col_data;
            int n_rows = static_cast<int>(current_table_->GetRowCount());
            for (int row = 0; row < n_rows; ++row) {
                auto val = DataAnalyzer::ToDouble(current_table_->GetCell(row, static_cast<int>(col)));
                if (val.has_value()) col_data.push_back(val.value());
            }
            if (!col_data.empty()) data.push_back(col_data);
        }

        if (!data.empty()) {
            method_results_[ScalingMethod::MinMax] = DataTransform::Normalize(data, minmax_range_min_, minmax_range_max_);
            method_results_[ScalingMethod::ZScore] = DataTransform::Standardize(data);
            method_results_[ScalingMethod::Robust] = DataTransform::RobustScale(data);
            method_results_[ScalingMethod::MaxAbs] = DataTransform::MaxAbsScale(data);
            method_results_[ScalingMethod::Quantile] = DataTransform::QuantileTransform(data, quantile_normal_output_ ? "normal" : "uniform", quantile_n_quantiles_);
        }
    }

    if (!can_apply) ImGui::EndDisabled();

    if (!status_message_.empty()) {
        ImGui::Spacing();
        ImGui::TextWrapped("%s", status_message_.c_str());
    }
}

void FeatureScalingPanel::RenderMinMaxTab() {
    ImGui::TextWrapped("Scale features to a given range.");
    ImGui::Spacing();
    ImGui::Text("Output Range:");
    ImGui::SliderFloat("Min##mm", &minmax_range_min_, -10.0f, 10.0f, "%.2f");
    ImGui::SliderFloat("Max##mm", &minmax_range_max_, -10.0f, 10.0f, "%.2f");

    if (minmax_range_min_ >= minmax_range_max_) {
        ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "Min must be < Max");
    }
}

void FeatureScalingPanel::RenderZScoreTab() {
    ImGui::TextWrapped("Standardize features to mean=0, std=1.");
    ImGui::Spacing();
    ImGui::Text("Formula: z = (x - mean) / std");
    ImGui::Spacing();
    ImGui::TextWrapped("Use when features have different units/scales and you want to remove the mean and scale to unit variance.");
}

void FeatureScalingPanel::RenderRobustTab() {
    ImGui::TextWrapped("Scale using median and IQR.");
    ImGui::Spacing();
    ImGui::Text("Formula: x_scaled = (x - median) / IQR");
    ImGui::Spacing();
    ImGui::TextWrapped("More robust to outliers than Z-Score. Use when data contains outliers that would skew mean/std calculations.");
}

void FeatureScalingPanel::RenderMaxAbsTab() {
    ImGui::TextWrapped("Scale by maximum absolute value to [-1, 1].");
    ImGui::Spacing();
    ImGui::Text("Formula: x_scaled = x / max(|x|)");
    ImGui::Spacing();
    ImGui::TextWrapped("Useful for data that is already centered at zero or sparse data.");
}

void FeatureScalingPanel::RenderQuantileTab() {
    ImGui::TextWrapped("Transform to uniform or normal distribution.");
    ImGui::Spacing();

    ImGui::Text("Output Distribution:");
    bool is_normal = quantile_normal_output_;
    if (ImGui::RadioButton("Uniform [0, 1]", !is_normal)) quantile_normal_output_ = false;
    if (ImGui::RadioButton("Normal (Gaussian)", is_normal)) quantile_normal_output_ = true;

    ImGui::Spacing();
    ImGui::SliderInt("Quantiles", &quantile_n_quantiles_, 100, 10000);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Number of quantiles to compute. Higher = smoother transform.");
    }
}

void FeatureScalingPanel::RenderPreview() {
    if (!transform_result_.success || original_stats_.empty()) return;

    ImGui::Text("Scaling Results:");
    ImGui::Spacing();

    // Method name
    const char* method_names[] = {"Min-Max", "Z-Score", "Robust", "Max-Abs", "Quantile"};
    ImGui::Text("Method: %s", method_names[static_cast<int>(selected_method_)]);

    // Stats comparison
    if (ImGui::BeginTable("ScalingStats", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollX)) {
        ImGui::TableSetupColumn("Column");
        ImGui::TableSetupColumn("Orig Range");
        ImGui::TableSetupColumn("New Range");
        ImGui::TableSetupColumn("Orig Mean");
        ImGui::TableSetupColumn("New Mean");
        ImGui::TableHeadersRow();

        int trans_idx = 0;
        for (size_t i = 0; i < selected_columns_.size(); ++i) {
            if (!selected_columns_[i]) continue;
            if (trans_idx >= static_cast<int>(transformed_stats_.size())) break;

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", column_names_[i].c_str());
            ImGui::TableNextColumn();
            ImGui::Text("[%.2f, %.2f]", original_stats_[trans_idx].min, original_stats_[trans_idx].max);
            ImGui::TableNextColumn();
            ImGui::Text("[%.2f, %.2f]", transformed_stats_[trans_idx].min, transformed_stats_[trans_idx].max);
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", original_stats_[trans_idx].mean);
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", transformed_stats_[trans_idx].mean);

            trans_idx++;
        }

        ImGui::EndTable();
    }

    // Histogram comparison for first column
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Distribution (first selected column):");

    int vis_idx = 0;
    for (size_t i = 0; i < selected_columns_.size(); ++i) {
        if (selected_columns_[i]) {
            vis_idx = static_cast<int>(i);
            break;
        }
    }

    std::vector<double> original_data;
    if (current_table_) {
        int n_rows = static_cast<int>(current_table_->GetRowCount());
        for (int row = 0; row < n_rows && row < 10000; ++row) {
            auto val = DataAnalyzer::ToDouble(current_table_->GetCell(row, vis_idx));
            if (val.has_value()) original_data.push_back(val.value());
        }
    }

    if (!original_data.empty() && !transform_result_.transformed_data.empty()) {
        const auto& transformed_data = transform_result_.transformed_data[0];

        if (ImPlot::BeginPlot("##OrigDistrib", ImVec2(ImGui::GetContentRegionAvail().x * 0.48f, 180))) {
            ImPlot::SetupAxes("Value", "Count");
            ImPlot::PlotHistogram("Original", original_data.data(), static_cast<int>(original_data.size()), 30);
            ImPlot::EndPlot();
        }

        ImGui::SameLine();

        if (ImPlot::BeginPlot("##ScaledDistrib", ImVec2(ImGui::GetContentRegionAvail().x, 180))) {
            ImPlot::SetupAxes("Scaled Value", "Count");
            ImPlot::SetNextFillStyle(ImVec4(0.2f, 0.7f, 0.3f, 0.7f));
            ImPlot::PlotHistogram("Scaled", transformed_data.data(), static_cast<int>(transformed_data.size()), 30);
            ImPlot::EndPlot();
        }
    }
}

void FeatureScalingPanel::RenderComparison() {
    if (!show_comparison_ || method_results_.empty()) {
        ImGui::Text("Click 'Compare All Methods' to see comparison.");
        return;
    }

    ImGui::Text("Method Comparison (first column):");
    ImGui::Spacing();

    // Get first column data
    std::vector<double> original_data;
    int vis_idx = 0;
    for (size_t i = 0; i < selected_columns_.size(); ++i) {
        if (selected_columns_[i]) {
            vis_idx = static_cast<int>(i);
            break;
        }
    }

    if (current_table_) {
        int n_rows = static_cast<int>(current_table_->GetRowCount());
        for (int row = 0; row < n_rows && row < 10000; ++row) {
            auto val = DataAnalyzer::ToDouble(current_table_->GetCell(row, vis_idx));
            if (val.has_value()) original_data.push_back(val.value());
        }
    }

    // Plot all methods side by side
    const char* method_names[] = {"Min-Max", "Z-Score", "Robust", "Max-Abs", "Quantile"};
    ScalingMethod methods[] = {ScalingMethod::MinMax, ScalingMethod::ZScore, ScalingMethod::Robust, ScalingMethod::MaxAbs, ScalingMethod::Quantile};

    // Show original
    if (ImPlot::BeginPlot("##OrigComp", ImVec2(180, 150))) {
        ImPlot::SetupAxes("", "");
        if (!original_data.empty()) {
            ImPlot::PlotHistogram("Original", original_data.data(), static_cast<int>(original_data.size()), 20);
        }
        ImPlot::EndPlot();
    }

    for (int m = 0; m < 5; ++m) {
        ImGui::SameLine();
        if (ImPlot::BeginPlot(("##Method" + std::to_string(m)).c_str(), ImVec2(130, 150))) {
            ImPlot::SetupAxes("", "");
            auto it = method_results_.find(methods[m]);
            if (it != method_results_.end() && it->second.success && !it->second.transformed_data.empty()) {
                const auto& data = it->second.transformed_data[0];
                ImPlot::SetNextFillStyle(ImVec4(0.2f + m * 0.15f, 0.7f - m * 0.1f, 0.3f + m * 0.1f, 0.7f));
                ImPlot::PlotHistogram(method_names[m], data.data(), static_cast<int>(data.size()), 20);
            }
            ImPlot::EndPlot();
        }
    }

    // Summary table
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Summary Statistics:");

    if (ImGui::BeginTable("MethodCompare", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Method");
        ImGui::TableSetupColumn("Min");
        ImGui::TableSetupColumn("Max");
        ImGui::TableSetupColumn("Mean");
        ImGui::TableSetupColumn("Std");
        ImGui::TableSetupColumn("Range");
        ImGui::TableHeadersRow();

        for (int m = 0; m < 5; ++m) {
            auto it = method_results_.find(methods[m]);
            if (it == method_results_.end() || !it->second.success || it->second.transformed_data.empty()) continue;

            auto stats = DataTransform::ComputeColumnStats(it->second.transformed_data[0]);

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", method_names[m]);
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", stats.min);
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", stats.max);
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", stats.mean);
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", stats.std_dev);
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", stats.max - stats.min);
        }

        ImGui::EndTable();
    }

    // Recommendations
    ImGui::Spacing();
    ImGui::Text(ICON_FA_LIGHTBULB " Recommendations:");
    ImGui::BulletText("Neural networks: Min-Max [0,1] or Z-Score");
    ImGui::BulletText("Outlier-prone data: Robust scaling");
    ImGui::BulletText("Sparse data: Max-Abs scaling");
    ImGui::BulletText("Non-linear relationships: Quantile transform");
}

void FeatureScalingPanel::LoadSelectedData() {
    if (selected_table_idx_ < 0 || !data_registry_) return;

    current_table_ = data_registry_->GetTable(available_tables_[selected_table_idx_]);
    if (!current_table_) return;

    column_names_ = current_table_->GetHeaders();
    selected_columns_.assign(column_names_.size(), false);

    transform_result_ = TransformResult();
    original_stats_.clear();
    transformed_stats_.clear();
    method_results_.clear();
    show_comparison_ = false;
    status_message_ = "Data loaded. Select columns and method.";
}

void FeatureScalingPanel::ApplyScaling() {
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

    switch (selected_method_) {
        case ScalingMethod::MinMax:
            transform_result_ = DataTransform::Normalize(data, minmax_range_min_, minmax_range_max_);
            break;
        case ScalingMethod::ZScore:
            transform_result_ = DataTransform::Standardize(data);
            break;
        case ScalingMethod::Robust:
            transform_result_ = DataTransform::RobustScale(data);
            break;
        case ScalingMethod::MaxAbs:
            transform_result_ = DataTransform::MaxAbsScale(data);
            break;
        case ScalingMethod::Quantile:
            transform_result_ = DataTransform::QuantileTransform(data, quantile_normal_output_ ? "normal" : "uniform", quantile_n_quantiles_);
            break;
    }

    if (transform_result_.success) {
        transformed_stats_.clear();
        for (const auto& col : transform_result_.transformed_data) {
            transformed_stats_.push_back(DataTransform::ComputeColumnStats(col));
        }
        status_message_ = "Scaling applied successfully.";
    } else {
        status_message_ = "Error: " + transform_result_.error_message;
    }
}

void FeatureScalingPanel::ExportData() {
    // TODO: Implement export functionality
    status_message_ = "Export not yet implemented.";
}

} // namespace cyxwiz
