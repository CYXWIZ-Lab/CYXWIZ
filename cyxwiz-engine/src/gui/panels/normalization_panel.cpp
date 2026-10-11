#include "normalization_panel.h"
#include "../icons.h"
#include "../../core/data_registry.h"
#include "../../data/data_table.h"
#include "../../core/data_analyzer.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>

namespace cyxwiz {

NormalizationPanel::NormalizationPanel() : Panel("Normalization", true) {}

void NormalizationPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin((std::string(ICON_FA_ARROWS_LEFT_RIGHT) + " Normalization (Min-Max)###NormalizationPanel").c_str(), &visible_)) {
        ImGui::BeginChild("ConfigPanel", ImVec2(280, 0), true);
        RenderDataSelector();
        ImGui::Separator();
        RenderSettings();
        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);
        if (transform_result_.success) {
            if (ImGui::BeginTabBar("NormTabs")) {
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
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

void NormalizationPanel::RenderDataSelector() {
    ImGui::Text(ICON_FA_TABLE " Data Selection");
    ImGui::Spacing();

    source_.Render("normalization_dataset", [this](std::shared_ptr<DataTable> table) { UseTable(std::move(table)); });

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

void NormalizationPanel::RenderSettings() {
    ImGui::Text(ICON_FA_COG " Settings");
    ImGui::Spacing();

    ImGui::Text("Output Range:");
    ImGui::SliderFloat("Min", &range_min_, -10.0f, 10.0f, "%.2f");
    ImGui::SliderFloat("Max", &range_max_, -10.0f, 10.0f, "%.2f");

    if (range_min_ >= range_max_) {
        ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "Warning: Min must be less than Max");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    bool can_apply = current_table_ != nullptr && range_min_ < range_max_;
    int selected_count = 0;
    for (bool sel : selected_columns_) if (sel) selected_count++;
    can_apply = can_apply && selected_count > 0;

    if (!can_apply) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_PLAY " Apply Normalization", ImVec2(-1, 30))) {
        ApplyNormalization();
    }

    if (!can_apply) ImGui::EndDisabled();

    if (!status_message_.empty()) {
        ImGui::Spacing();
        ImGui::TextWrapped("%s", status_message_.c_str());
    }
}

void NormalizationPanel::RenderPreview() {
    if (!transform_result_.success || original_stats_.empty()) return;

    ImGui::Text("Before/After Distribution Comparison:");
    ImGui::Spacing();

    // The first transformed column (transformed_data follows transformed_columns_).
    if (!transformed_columns_.empty() && !transform_result_.transformed_data.empty()) {
        const int vis_idx = transformed_columns_[0];
        ImGui::Text("Column: %s", column_names_[vis_idx].c_str());

        // Create histograms from data
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
            // Plot side by side
            if (ImPlot::BeginPlot("##OriginalHist", ImVec2(ImGui::GetContentRegionAvail().x * 0.48f, 200))) {
                ImPlot::SetupAxes("Value", "Count");
                ImPlot::PlotHistogram("Original", original_data.data(), static_cast<int>(original_data.size()), 30);
                ImPlot::EndPlot();
            }

            ImGui::SameLine();

            if (ImPlot::BeginPlot("##TransformedHist", ImVec2(ImGui::GetContentRegionAvail().x, 200))) {
                ImPlot::SetupAxes("Value", "Count");
                ImPlot::SetNextFillStyle(ImVec4(0.2f, 0.8f, 0.2f, 0.7f));
                ImPlot::PlotHistogram("Normalized", transformed_data.data(), static_cast<int>(transformed_data.size()), 30);
                ImPlot::EndPlot();
            }
        }
    }

    // Stats comparison table
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Statistics Comparison:");

    if (ImGui::BeginTable("StatsCompare", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Column");
        ImGui::TableSetupColumn("Orig Min");
        ImGui::TableSetupColumn("Orig Max");
        ImGui::TableSetupColumn("New Min");
        ImGui::TableSetupColumn("New Max");
        ImGui::TableHeadersRow();

        for (size_t trans_idx = 0; trans_idx < transformed_columns_.size() && trans_idx < transformed_stats_.size(); ++trans_idx) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", column_names_[transformed_columns_[trans_idx]].c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", original_stats_[trans_idx].min);
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", original_stats_[trans_idx].max);
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", transformed_stats_[trans_idx].min);
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", transformed_stats_[trans_idx].max);
        }

        ImGui::EndTable();
    }
}

void NormalizationPanel::RenderResults() {
    if (!transform_result_.success) return;

    ImGui::Text(ICON_FA_TABLE " Transformed Data Preview");
    ImGui::Spacing();

    // Show first few rows of transformed data
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
                if (row < static_cast<int>(values.size())) ImGui::Text("%.4f", values[row]);
            }
        }

        ImGui::EndTable();
    }

    if (n_rows > 20) {
        ImGui::Text("Showing first 20 of %d rows", n_rows);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Transform Parameters (for inverse transform):");
    ImGui::BulletText("Method: %s", transform_result_.method.c_str());
    ImGui::BulletText("Range: [%.2f, %.2f]", range_min_, range_max_);
}

void NormalizationPanel::UseTable(std::shared_ptr<DataTable> table) {
    current_table_ = std::move(table);
    column_names_ = current_table_->GetHeaders();
    selected_columns_.assign(column_names_.size(), false);

    transform_result_ = TransformResult();
    original_stats_.clear();
    transformed_stats_.clear();
    transformed_columns_.clear();
    status_message_ = "Data loaded. Select columns and configure range.";
}

void NormalizationPanel::ApplyNormalization() {
    if (!current_table_) return;

    // Collect selected column data
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

    transform_result_ = DataTransform::Normalize(data, static_cast<double>(range_min_), static_cast<double>(range_max_));

    if (transform_result_.success) {
        // Compute transformed stats
        transformed_stats_.clear();
        for (const auto& col : transform_result_.transformed_data) {
            transformed_stats_.push_back(DataTransform::ComputeColumnStats(col));
        }
        status_message_ = "Normalization applied successfully.";
    } else {
        status_message_ = "Error: " + transform_result_.error_message;
    }
}

} // namespace cyxwiz
