#include "normalization_panel.h"
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

NormalizationPanel::NormalizationPanel() : Panel("Normalization", true) {}

void NormalizationPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin((std::string(ICON_FA_ARROWS_LEFT_RIGHT) + " Normalization (Min-Max)###NormalizationPanel").c_str(), &visible_)) {
        ui::BeginSettingsColumn("normalization", 300.0f);
        RenderDataSelector();
        RenderSettings();
        ui::BeginResultsArea();
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
            ui::EmptyState(ICON_FA_ARROWS_LEFT_RIGHT, "No normalization yet",
                           "Pick a dataset and the columns, then Apply Normalization.");
        }
        ui::EndResultsArea();
    }
    ImGui::End();
}

void NormalizationPanel::RenderDataSelector() {
    ui::SettingsSection("Data");
    source_.Render("normalization_dataset", [this](std::shared_ptr<DataTable> table) { UseTable(std::move(table)); });

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

void NormalizationPanel::RenderSettings() {
    const ui::Tokens& t = ui::CurrentTokens();
    ui::SettingsSection("Output range");

    if (ImGui::BeginTable("##range", 2, ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextColumn();
        ui::FieldLabel("Min");
        ImGui::SliderFloat("##range_min", &range_min_, -10.0f, 10.0f, "%.2f");
        ImGui::TableNextColumn();
        ui::FieldLabel("Max");
        ImGui::SliderFloat("##range_max", &range_max_, -10.0f, 10.0f, "%.2f");
        ImGui::EndTable();
    }

    if (range_min_ >= range_max_) {
        ImGui::TextColored(t.caution, "Warning: Min must be less than Max");
    }

    ImGui::Spacing();

    bool can_apply = current_table_ != nullptr && range_min_ < range_max_;
    int selected_count = 0;
    for (bool sel : selected_columns_) if (sel) selected_count++;
    can_apply = can_apply && selected_count > 0;
    const char* disabled_reason = current_table_ == nullptr ? "Pick a dataset first"
                                : selected_count == 0       ? "Select at least one column"
                                                            : "Min must be less than Max";

    if (ui::PrimaryButton("Apply Normalization", can_apply, disabled_reason, ui::ButtonSize::Regular, -FLT_MIN)) {
        ApplyNormalization();
    }

    if (!status_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
        ImGui::TextWrapped("%s", status_message_.c_str());
        ImGui::PopStyleColor();
    }
}

void NormalizationPanel::RenderPreview() {
    if (!transform_result_.success || original_stats_.empty()) return;

    const ui::Tokens& t = ui::CurrentTokens();
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
                ImPlot::SetNextFillStyle(ui::WithAlpha(t.series[3], 0.7f));
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
