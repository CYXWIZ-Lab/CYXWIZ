#include "feature_scaling_panel.h"
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
#include <algorithm>

namespace cyxwiz {

namespace {
// Indexed by ScalingMethod.
constexpr const char* kMethodNames[] = {"Min-Max", "Z-Score", "Robust", "Max-Abs", "Quantile"};
constexpr int kMethodCount = static_cast<int>(sizeof(kMethodNames) / sizeof(kMethodNames[0]));
}  // namespace

FeatureScalingPanel::FeatureScalingPanel() : Panel("Feature Scaling", true) {}

void FeatureScalingPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(900, 700), ImGuiCond_FirstUseEver);

    if (ImGui::Begin((std::string(ICON_FA_LAYER_GROUP) + " Feature Scaling (All Methods)###FeatureScalingPanel").c_str(), &visible_)) {
        ui::BeginSettingsColumn("feature_scaling", 300.0f);
        RenderDataSelector();
        RenderMethodTabs();
        ui::BeginResultsArea();
        // Either action has results to show: Apply fills Preview, Compare All
        // fills Comparison on its own.
        if (transform_result_.success || show_comparison_) {
            if (ImGui::BeginTabBar("ScalingResultTabs")) {
                if (transform_result_.success && ImGui::BeginTabItem(ICON_FA_CHART_BAR " Preview")) {
                    RenderPreview();
                    ImGui::EndTabItem();
                }
                const ImGuiTabItemFlags comparison_flags =
                    select_comparison_tab_ ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;
                select_comparison_tab_ = false;
                if (ImGui::BeginTabItem(ICON_FA_CHART_LINE " Comparison", nullptr, comparison_flags)) {
                    RenderComparison();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        } else {
            ui::EmptyState(ICON_FA_LAYER_GROUP, "No scaling yet",
                           "Pick a dataset, the columns and a method, then Apply Scaling.");
        }
        ui::EndResultsArea();
    }
    ImGui::End();
}

void FeatureScalingPanel::RenderDataSelector() {
    ui::SettingsSection("Data");
    source_.Render("feature_scaling_dataset", [this](std::shared_ptr<DataTable> table) { UseTable(std::move(table)); });

    if (!column_names_.empty()) {
        const ui::Tokens& t = ui::CurrentTokens();
        ImGui::AlignTextToFramePadding();  // level with the links beside it
        ImGui::TextColored(t.text_dim, "Columns");
        ImGui::SameLine();
        if (ui::LinkButton("All")) {
            for (size_t i = 0; i < selected_columns_.size(); ++i) selected_columns_[i] = true;
        }
        ImGui::SameLine();
        if (ui::LinkButton("None")) {
            for (size_t i = 0; i < selected_columns_.size(); ++i) selected_columns_[i] = false;
        }
        ui::BeginInsetList("##columns", 120.0f);
        for (size_t i = 0; i < column_names_.size(); ++i) {
            bool selected = selected_columns_[i];
            if (ImGui::Checkbox(column_names_[i].c_str(), &selected)) {
                selected_columns_[i] = selected;
            }
        }
        ui::EndInsetList();
    }
}

void FeatureScalingPanel::RenderMethodTabs() {
    ui::SettingsSection("Scaling");

    // A combo, not tabs: five tabs in the settings column were clipped to
    // "Min-M...", "Z-Scor...".
    int method_index = static_cast<int>(selected_method_);
    // Label with the overview of all methods (the old empty-results list).
    ImGui::TextColored(ui::CurrentTokens().text_dim, "Method");
    ImGui::SameLine();
    ui::HelpMarker("Min-Max: scale to [0, 1] or a custom range\n"
                   "Z-Score: mean 0, std 1\n"
                   "Robust: median / IQR, resistant to outliers\n"
                   "Max-Abs: divide by the largest absolute value\n"
                   "Quantile: map to a uniform or normal distribution");
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::Combo("##method", &method_index, kMethodNames, kMethodCount)) {
        selected_method_ = static_cast<ScalingMethod>(method_index);
    }
    switch (selected_method_) {
        case ScalingMethod::MinMax: RenderMinMaxTab(); break;
        case ScalingMethod::ZScore: RenderZScoreTab(); break;
        case ScalingMethod::Robust: RenderRobustTab(); break;
        case ScalingMethod::MaxAbs: RenderMaxAbsTab(); break;
        case ScalingMethod::Quantile: RenderQuantileTab(); break;
    }

    ImGui::Spacing();

    int selected_count = 0;
    for (bool sel : selected_columns_) if (sel) selected_count++;
    bool can_apply = current_table_ != nullptr && selected_count > 0;
    const char* disabled_reason = current_table_ == nullptr ? "Pick a dataset first" : "Select at least one column";

    if (ui::PrimaryButton("Apply Scaling", can_apply, disabled_reason, ui::ButtonSize::Regular, -FLT_MIN)) {
        ApplyScaling();
    }

    if (ui::SecondaryButton("Compare All Methods", can_apply, disabled_reason, ui::ButtonSize::Regular, -FLT_MIN)) {
        // Apply all methods for comparison
        show_comparison_ = true;
        select_comparison_tab_ = true;

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
            method_stats_.clear();
            for (const auto& [method, result] : method_results_) {
                if (result.success && !result.transformed_data.empty()) {
                    method_stats_[method] = DataTransform::ComputeColumnStats(result.transformed_data[0]);
                }
            }
        }
    }

    if (!status_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ui::CurrentTokens().text_dim);
        ImGui::TextWrapped("%s", status_message_.c_str());
        ImGui::PopStyleColor();
    }
}

void FeatureScalingPanel::RenderMinMaxTab() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::TextWrapped("Scale features to a given range.");
    ImGui::Spacing();
    if (ImGui::BeginTable("##mm_range", 2, ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextColumn();
        ui::FieldLabel("Min");
        ImGui::SliderFloat("##mm_min", &minmax_range_min_, -10.0f, 10.0f, "%.2f");
        ImGui::TableNextColumn();
        ui::FieldLabel("Max");
        ImGui::SliderFloat("##mm_max", &minmax_range_max_, -10.0f, 10.0f, "%.2f");
        ImGui::EndTable();
    }

    if (minmax_range_min_ >= minmax_range_max_) {
        ImGui::TextColored(t.caution, "Min must be < Max");
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

    ui::FieldLabel("Output distribution");
    bool is_normal = quantile_normal_output_;
    if (ImGui::RadioButton("Uniform [0, 1]", !is_normal)) quantile_normal_output_ = false;
    if (ImGui::RadioButton("Normal (Gaussian)", is_normal)) quantile_normal_output_ = true;

    ui::FieldLabel("Quantiles");
    ImGui::SliderInt("##quantiles", &quantile_n_quantiles_, 100, 10000);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Number of quantiles to compute. Higher = smoother transform.");
    }
}

void FeatureScalingPanel::RenderPreview() {
    if (!transform_result_.success || original_stats_.empty()) return;

    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::Text("Scaling Results:");
    ImGui::Spacing();

    // Method name
    ImGui::Text("Method: %s", kMethodNames[static_cast<int>(selected_method_)]);

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
            ImPlot::SetNextFillStyle(ui::WithAlpha(t.series[3], 0.7f));
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

    const ui::Tokens& t = ui::CurrentTokens();
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
    ScalingMethod methods[] = {ScalingMethod::MinMax, ScalingMethod::ZScore, ScalingMethod::Robust, ScalingMethod::MaxAbs, ScalingMethod::Quantile};

    // The original plus one plot per method, in as many equal columns as fit
    // (fixed widths ran past the right edge). The original is drawn in the
    // neutral tone so the five methods keep the five series colours.
    const int plot_count = kMethodCount + 1;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float avail = ImGui::GetContentRegionAvail().x;
    const int per_row = std::clamp(static_cast<int>((avail + spacing) / (150.0f + spacing)), 1, plot_count);
    const float plot_width = (avail - spacing * (per_row - 1)) / per_row;
    for (int p = 0; p < plot_count; ++p) {
        if (p % per_row != 0) ImGui::SameLine();
        if (!ImPlot::BeginPlot(("##Method" + std::to_string(p)).c_str(), ImVec2(plot_width, 150))) continue;
        ImPlot::SetupAxes("", "");
        if (p == 0) {
            if (!original_data.empty()) {
                ImPlot::SetNextFillStyle(ui::WithAlpha(t.text_dim, 0.7f));
                ImPlot::PlotHistogram("Original", original_data.data(), static_cast<int>(original_data.size()), 20);
            }
        } else {
            const int m = p - 1;
            auto it = method_results_.find(methods[m]);
            if (it != method_results_.end() && it->second.success && !it->second.transformed_data.empty()) {
                const auto& data = it->second.transformed_data[0];
                ImPlot::SetNextFillStyle(ui::WithAlpha(ui::SeriesColour(t, static_cast<size_t>(m)), 0.7f));
                ImPlot::PlotHistogram(kMethodNames[m], data.data(), static_cast<int>(data.size()), 20);
            }
        }
        ImPlot::EndPlot();
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

        for (int m = 0; m < kMethodCount; ++m) {
            auto it = method_stats_.find(methods[m]);
            if (it == method_stats_.end()) continue;
            const ColumnStats& stats = it->second;

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", kMethodNames[m]);
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

void FeatureScalingPanel::UseTable(std::shared_ptr<DataTable> table) {
    current_table_ = std::move(table);
    column_names_ = current_table_->GetHeaders();
    selected_columns_.assign(column_names_.size(), false);

    transform_result_ = TransformResult();
    original_stats_.clear();
    transformed_stats_.clear();
    method_results_.clear();
    method_stats_.clear();
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
