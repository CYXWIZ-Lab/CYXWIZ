#include "log_transform_panel.h"
#include "../icons.h"
#include "../../core/data_registry.h"
#include "../../data/data_table.h"
#include "../../core/data_analyzer.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>

namespace cyxwiz {

LogTransformPanel::LogTransformPanel() : Panel("Log Transform", true) {}

void LogTransformPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin((std::string(ICON_FA_CHART_LINE) + " Log Transform###LogTransformPanel").c_str(), &visible_)) {
        if (data_registry_) available_tables_ = data_registry_->GetTableNames();

        ImGui::BeginChild("ConfigPanel", ImVec2(280, 0), true);
        RenderDataSelector();
        ImGui::Separator();
        RenderSettings();
        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);
        if (transform_result_.success) {
            if (ImGui::BeginTabBar("LogTabs")) {
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
            ImGui::TextWrapped("Log transform is useful for:");
            ImGui::BulletText("Right-skewed distributions");
            ImGui::BulletText("Multiplicative relationships");
            ImGui::BulletText("Reducing variance heterogeneity");

            if (!can_transform_) {
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(1, 0.5f, 0, 1),
                    "Warning: Some columns have non-positive values.");
                ImGui::TextWrapped("Enable 'Use log1p' or check your data.");
            }
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

void LogTransformPanel::RenderDataSelector() {
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

void LogTransformPanel::RenderSettings() {
    ImGui::Text(ICON_FA_COG " Settings");
    ImGui::Spacing();

    ImGui::Text("Logarithm Base:");
    ImGui::RadioButton("Natural (ln)", &log_base_, 0);
    ImGui::RadioButton("Base 10 (log10)", &log_base_, 1);
    ImGui::RadioButton("Base 2 (log2)", &log_base_, 2);

    ImGui::Spacing();
    ImGui::Checkbox("Use log1p (log(1 + x))", &use_log1p_);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Safer for values close to zero.\nAllows x >= -1 instead of x > 0.");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    int selected_count = 0;
    for (bool sel : selected_columns_) if (sel) selected_count++;
    bool can_apply = current_table_ != nullptr && selected_count > 0 && can_transform_;

    if (!can_apply) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_PLAY " Apply Log Transform", ImVec2(-1, 30))) {
        ApplyLogTransform();
    }

    if (!can_apply) ImGui::EndDisabled();

    if (!status_message_.empty()) {
        ImGui::Spacing();
        ImGui::TextWrapped("%s", status_message_.c_str());
    }
}

void LogTransformPanel::RenderPreview() {
    if (!transform_result_.success || original_stats_.empty()) return;

    ImGui::Text("Before/After Distribution:");
    ImGui::Spacing();

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

            if (ImPlot::BeginPlot("##LogHist", ImVec2(ImGui::GetContentRegionAvail().x, 200))) {
                ImPlot::SetupAxes("Log Value", "Count");
                ImPlot::SetNextFillStyle(ImVec4(0.8f, 0.6f, 0.2f, 0.7f));
                ImPlot::PlotHistogram("Log Transform", transformed_data.data(), static_cast<int>(transformed_data.size()), 30);
                ImPlot::EndPlot();
            }
        }
    }

    // Skewness comparison
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Skewness Comparison:");

    if (ImGui::BeginTable("SkewCompare", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Column");
        ImGui::TableSetupColumn("Original Skew");
        ImGui::TableSetupColumn("Transformed Skew");
        ImGui::TableHeadersRow();

        int trans_idx = 0;
        for (size_t i = 0; i < selected_columns_.size(); ++i) {
            if (!selected_columns_[i]) continue;
            if (trans_idx >= static_cast<int>(transformed_stats_.size())) break;

            // Calculate skewness
            auto calc_skew = [](const std::vector<double>& data, double mean, double std) -> double {
                if (data.empty() || std < 1e-10) return 0;
                double skew = 0;
                for (const auto& x : data) {
                    skew += std::pow((x - mean) / std, 3);
                }
                return skew / data.size();
            };

            double orig_skew = 0, trans_skew = 0;

            if (trans_idx < static_cast<int>(transform_result_.transformed_data.size())) {
                const auto& orig = original_stats_[trans_idx];
                const auto& trans = transformed_stats_[trans_idx];

                // Approximate skew from stats
                orig_skew = (orig.mean - orig.median) / (orig.std_dev + 1e-10) * 3;
                trans_skew = (trans.mean - trans.median) / (trans.std_dev + 1e-10) * 3;
            }

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", column_names_[i].c_str());
            ImGui::TableNextColumn();
            ImVec4 color1 = std::abs(orig_skew) > 1 ? ImVec4(1, 0.5f, 0, 1) : ImVec4(0, 0.8f, 0, 1);
            ImGui::TextColored(color1, "%.3f", orig_skew);
            ImGui::TableNextColumn();
            ImVec4 color2 = std::abs(trans_skew) > 1 ? ImVec4(1, 0.5f, 0, 1) : ImVec4(0, 0.8f, 0, 1);
            ImGui::TextColored(color2, "%.3f", trans_skew);

            trans_idx++;
        }

        ImGui::EndTable();
    }
}

void LogTransformPanel::RenderResults() {
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
                ImGui::Text("%.4f", transform_result_.transformed_data[col][row]);
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
    const char* base_names[] = {"Natural (ln)", "Base 10", "Base 2"};
    ImGui::BulletText("Base: %s", base_names[log_base_]);
    ImGui::BulletText("Method: %s", use_log1p_ ? "log1p (log(1 + x))" : "log(x)");
}

void LogTransformPanel::LoadSelectedData() {
    if (selected_table_idx_ < 0 || !data_registry_) return;

    current_table_ = data_registry_->GetTable(available_tables_[selected_table_idx_]);
    if (!current_table_) return;

    column_names_ = current_table_->GetHeaders();
    selected_columns_.assign(column_names_.size(), false);

    transform_result_ = TransformResult();
    original_stats_.clear();
    transformed_stats_.clear();
    can_transform_ = true;
    status_message_ = "Data loaded. Select columns.";
}

void LogTransformPanel::ApplyLogTransform() {
    if (!current_table_) return;

    std::vector<std::vector<double>> data;
    original_stats_.clear();

    std::string base_str = log_base_ == 0 ? "natural" : (log_base_ == 1 ? "log10" : "log2");

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
            // Check if transform is possible
            if (!DataTransform::CanApplyLogTransform(col_data, use_log1p_)) {
                can_transform_ = false;
                status_message_ = "Column '" + column_names_[col] + "' has invalid values for log transform.";
                return;
            }

            original_stats_.push_back(DataTransform::ComputeColumnStats(col_data));
            data.push_back(col_data);
        }
    }

    if (data.empty()) {
        status_message_ = "No valid numeric data found.";
        return;
    }

    transform_result_ = DataTransform::LogTransform(data, base_str, use_log1p_);

    if (transform_result_.success) {
        transformed_stats_.clear();
        for (const auto& col : transform_result_.transformed_data) {
            transformed_stats_.push_back(DataTransform::ComputeColumnStats(col));
        }
        can_transform_ = true;
        status_message_ = "Log transform applied successfully.";
    } else {
        can_transform_ = false;
        status_message_ = "Error: " + transform_result_.error_message;
    }
}

} // namespace cyxwiz
