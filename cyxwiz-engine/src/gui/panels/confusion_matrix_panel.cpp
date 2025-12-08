#include "confusion_matrix_panel.h"
#include "../icons.h"
#include "../../core/data_registry.h"
#include "../../data/data_table.h"
#include "../../core/data_analyzer.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace cyxwiz {

const ImU32 ConfusionMatrixPanel::HEATMAP_COLORS[5] = {
    IM_COL32(255, 255, 255, 255),   // 0% - White
    IM_COL32(198, 219, 239, 255),   // 25% - Light blue
    IM_COL32(107, 174, 214, 255),   // 50% - Medium blue
    IM_COL32(33, 113, 181, 255),    // 75% - Dark blue
    IM_COL32(8, 48, 107, 255),      // 100% - Very dark blue
};

ConfusionMatrixPanel::ConfusionMatrixPanel() : Panel("Confusion Matrix", true) {}

void ConfusionMatrixPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin((std::string(ICON_FA_TABLE_CELLS) + " Confusion Matrix###ConfusionMatrixPanel").c_str(), &visible_)) {
        if (data_registry_) available_tables_ = data_registry_->GetTableNames();

        ImGui::BeginChild("ConfigPanel", ImVec2(250, 0), true);
        RenderDataSelector();
        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);
        if (result_.success) {
            if (ImGui::BeginTabBar("MatrixTabs")) {
                if (ImGui::BeginTabItem(ICON_FA_TABLE " Matrix")) {
                    RenderMatrixHeatmap();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_PERCENT " Normalized")) {
                    RenderNormalizedMatrix();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_CHART_BAR " Metrics")) {
                    RenderMetricsTable();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        } else {
            ImGui::Text("Select true and predicted label columns, then click Compute.");
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

void ConfusionMatrixPanel::RenderDataSelector() {
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
        ImGui::Separator();
        ImGui::Text(ICON_FA_TAGS " Label Columns");
        ImGui::Spacing();

        std::vector<const char*> col_names_c;
        for (const auto& name : column_names_) col_names_c.push_back(name.c_str());

        ImGui::Combo("True Labels", &true_label_column_, col_names_c.data(), static_cast<int>(col_names_c.size()));
        ImGui::Combo("Predicted Labels", &pred_label_column_, col_names_c.data(), static_cast<int>(col_names_c.size()));

        ImGui::Spacing();
        ImGui::Separator();

        if (ImGui::Button(ICON_FA_CALCULATOR " Compute Matrix", ImVec2(-1, 30))) {
            ComputeMatrix();
        }
    }

    if (!status_message_.empty()) {
        ImGui::Spacing();
        ImGui::TextWrapped("%s", status_message_.c_str());
    }
}

void ConfusionMatrixPanel::RenderMatrixHeatmap() {
    if (!result_.success || result_.matrix.empty()) return;

    int n = result_.n_classes;

    // Find max value for color scaling
    int max_val = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (result_.matrix[i][j] > max_val) max_val = result_.matrix[i][j];
        }
    }

    ImGui::Text("Confusion Matrix (Rows: True, Columns: Predicted)");
    ImGui::Spacing();

    float cell_size = 50.0f;
    float label_width = 80.0f;

    // Draw header row
    ImGui::Dummy(ImVec2(label_width, 0));
    ImGui::SameLine();
    for (int j = 0; j < n; ++j) {
        ImGui::BeginGroup();
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(2, 2));
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (cell_size - ImGui::CalcTextSize(result_.class_names[j].c_str()).x) / 2);
        ImGui::Text("%s", result_.class_names[j].c_str());
        ImGui::PopStyleVar();
        ImGui::EndGroup();
        if (j < n - 1) ImGui::SameLine(0, 2);
    }

    // Draw matrix cells
    for (int i = 0; i < n; ++i) {
        // Row label
        ImGui::Text("%-10s", result_.class_names[i].c_str());
        ImGui::SameLine(label_width);

        for (int j = 0; j < n; ++j) {
            int val = result_.matrix[i][j];
            float ratio = (max_val > 0) ? static_cast<float>(val) / max_val : 0.0f;

            // Interpolate color
            int color_idx = static_cast<int>(ratio * 4);
            color_idx = (color_idx > 4) ? 4 : color_idx;
            ImU32 bg_color = HEATMAP_COLORS[color_idx];

            // Use dark text for light backgrounds, light text for dark backgrounds
            ImU32 text_color = (color_idx < 2) ? IM_COL32(0, 0, 0, 255) : IM_COL32(255, 255, 255, 255);

            ImVec2 pos = ImGui::GetCursorScreenPos();
            ImDrawList* draw_list = ImGui::GetWindowDrawList();

            // Draw cell background
            draw_list->AddRectFilled(pos, ImVec2(pos.x + cell_size, pos.y + cell_size), bg_color);
            draw_list->AddRect(pos, ImVec2(pos.x + cell_size, pos.y + cell_size), IM_COL32(100, 100, 100, 255));

            // Draw cell value
            char buf[32];
            snprintf(buf, sizeof(buf), "%d", val);
            ImVec2 text_size = ImGui::CalcTextSize(buf);
            ImVec2 text_pos = ImVec2(pos.x + (cell_size - text_size.x) / 2, pos.y + (cell_size - text_size.y) / 2);
            draw_list->AddText(text_pos, text_color, buf);

            ImGui::Dummy(ImVec2(cell_size, cell_size));
            if (j < n - 1) ImGui::SameLine(0, 2);
        }
    }

    ImGui::Spacing();
    ImGui::Text("Total Samples: %d | Accuracy: %.2f%%", result_.total_samples, result_.accuracy * 100);
}

void ConfusionMatrixPanel::RenderNormalizedMatrix() {
    if (!result_.success || result_.matrix.empty()) return;

    int n = result_.n_classes;

    ImGui::Text("Normalized Confusion Matrix (Row-wise %)");
    ImGui::Spacing();

    float cell_size = 60.0f;
    float label_width = 80.0f;

    // Draw header row
    ImGui::Dummy(ImVec2(label_width, 0));
    ImGui::SameLine();
    for (int j = 0; j < n; ++j) {
        ImGui::BeginGroup();
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (cell_size - ImGui::CalcTextSize(result_.class_names[j].c_str()).x) / 2);
        ImGui::Text("%s", result_.class_names[j].c_str());
        ImGui::EndGroup();
        if (j < n - 1) ImGui::SameLine(0, 2);
    }

    // Draw matrix cells
    for (int i = 0; i < n; ++i) {
        int row_sum = result_.support[i];

        ImGui::Text("%-10s", result_.class_names[i].c_str());
        ImGui::SameLine(label_width);

        for (int j = 0; j < n; ++j) {
            float ratio = (row_sum > 0) ? static_cast<float>(result_.matrix[i][j]) / row_sum : 0.0f;

            int color_idx = static_cast<int>(ratio * 4);
            color_idx = (color_idx > 4) ? 4 : color_idx;
            ImU32 bg_color = HEATMAP_COLORS[color_idx];
            ImU32 text_color = (color_idx < 2) ? IM_COL32(0, 0, 0, 255) : IM_COL32(255, 255, 255, 255);

            ImVec2 pos = ImGui::GetCursorScreenPos();
            ImDrawList* draw_list = ImGui::GetWindowDrawList();

            draw_list->AddRectFilled(pos, ImVec2(pos.x + cell_size, pos.y + cell_size), bg_color);
            draw_list->AddRect(pos, ImVec2(pos.x + cell_size, pos.y + cell_size), IM_COL32(100, 100, 100, 255));

            char buf[32];
            snprintf(buf, sizeof(buf), "%.1f%%", ratio * 100);
            ImVec2 text_size = ImGui::CalcTextSize(buf);
            ImVec2 text_pos = ImVec2(pos.x + (cell_size - text_size.x) / 2, pos.y + (cell_size - text_size.y) / 2);
            draw_list->AddText(text_pos, text_color, buf);

            ImGui::Dummy(ImVec2(cell_size, cell_size));
            if (j < n - 1) ImGui::SameLine(0, 2);
        }
    }
}

void ConfusionMatrixPanel::RenderMetricsTable() {
    if (!result_.success) return;

    ImGui::Text(ICON_FA_CHART_BAR " Classification Metrics");
    ImGui::Separator();

    // Overall metrics
    ImGui::Text("Overall Metrics:");
    ImGui::BulletText("Accuracy: %.4f (%.2f%%)", result_.accuracy, result_.accuracy * 100);
    ImGui::BulletText("Macro Precision: %.4f", result_.macro_precision);
    ImGui::BulletText("Macro Recall: %.4f", result_.macro_recall);
    ImGui::BulletText("Macro F1: %.4f", result_.macro_f1);
    ImGui::BulletText("Weighted F1: %.4f", result_.weighted_f1);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Per-class metrics table
    ImGui::Text("Per-Class Metrics:");
    if (ImGui::BeginTable("MetricsTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableSetupColumn("Class", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableSetupColumn("Precision", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Recall", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("F1-Score", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Support", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableHeadersRow();

        for (int i = 0; i < result_.n_classes; ++i) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", result_.class_names[i].c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", result_.precision[i]);
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", result_.recall[i]);
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", result_.f1_scores[i]);
            ImGui::TableNextColumn();
            ImGui::Text("%d", result_.support[i]);
        }

        ImGui::EndTable();
    }

    // Bar chart of F1 scores
    ImGui::Spacing();
    if (ImPlot::BeginPlot("##F1Scores", ImVec2(-1, 200))) {
        ImPlot::SetupAxes("Class", "F1 Score");
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1.1);

        std::vector<double> x_pos(result_.n_classes);
        for (int i = 0; i < result_.n_classes; ++i) x_pos[i] = i;

        ImPlot::PlotBars("F1", x_pos.data(), result_.f1_scores.data(), result_.n_classes, 0.6);
        ImPlot::EndPlot();
    }
}

void ConfusionMatrixPanel::LoadSelectedData() {
    if (selected_table_idx_ < 0 || !data_registry_) return;

    current_table_ = data_registry_->GetTable(available_tables_[selected_table_idx_]);
    if (!current_table_) return;

    column_names_ = current_table_->GetHeaders();
    true_label_column_ = 0;
    int max_col = static_cast<int>(column_names_.size()) - 1;
    pred_label_column_ = (1 < max_col) ? 1 : max_col;

    result_ = ConfusionMatrixData();
    status_message_ = "Data loaded. Select label columns.";
}

void ConfusionMatrixPanel::ComputeMatrix() {
    if (!current_table_) {
        status_message_ = "No data loaded.";
        return;
    }

    std::vector<int> y_true, y_pred;
    int n_rows = static_cast<int>(current_table_->GetRowCount());

    for (int row = 0; row < n_rows; ++row) {
        auto true_val = DataAnalyzer::ToDouble(current_table_->GetCell(row, true_label_column_));
        auto pred_val = DataAnalyzer::ToDouble(current_table_->GetCell(row, pred_label_column_));

        if (true_val.has_value() && pred_val.has_value()) {
            y_true.push_back(static_cast<int>(true_val.value()));
            y_pred.push_back(static_cast<int>(pred_val.value()));
        }
    }

    if (y_true.empty()) {
        status_message_ = "No valid label data found.";
        return;
    }

    result_ = ModelEvaluation::ComputeConfusionMatrix(y_true, y_pred);

    if (result_.success) {
        status_message_ = "Computed: " + std::to_string(result_.n_classes) + " classes, " +
                          std::to_string(result_.total_samples) + " samples.";
    } else {
        status_message_ = "Error: " + result_.error_message;
    }
}

} // namespace cyxwiz
