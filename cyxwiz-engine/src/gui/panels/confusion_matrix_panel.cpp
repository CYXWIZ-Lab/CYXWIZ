#include "confusion_matrix_panel.h"
#include "../icons.h"
#include "../ui_tokens.h"
#include "../../core/data_registry.h"
#include "../../data/data_table.h"
#include "../../core/table_class_labels.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <functional>
#include <set>
#include <utility>

namespace cyxwiz {

ConfusionMatrixPanel::ConfusionMatrixPanel() : Panel("Confusion Matrix", true) {}

void ConfusionMatrixPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin((std::string(ICON_FA_TABLE_CELLS) + " Confusion Matrix###ConfusionMatrixPanel").c_str(), &visible_)) {
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

    source_.Render("confusion_matrix_dataset", [this](std::shared_ptr<DataTable> table) { UseTable(std::move(table)); });

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

    // Colour by count relative to the largest cell.
    int max_val = 0;
    for (const auto& row : result_.matrix) {
        for (int value : row) max_val = std::max(max_val, value);
    }

    ImGui::Text("Confusion Matrix (Rows: True, Columns: Predicted)");
    ImGui::Spacing();
    RenderGrid([&](int i, int j) {
        const int value = result_.matrix[i][j];
        return std::make_pair(max_val > 0 ? static_cast<float>(value) / max_val : 0.0f, std::to_string(value));
    });

    ImGui::Spacing();
    ImGui::Text("Total Samples: %d | Accuracy: %.2f%%", result_.total_samples, result_.accuracy * 100);
}

void ConfusionMatrixPanel::RenderNormalizedMatrix() {
    if (!result_.success || result_.matrix.empty()) return;

    ImGui::TextUnformatted("Normalized Confusion Matrix (Row-wise %)");
    ImGui::Spacing();
    RenderGrid([&](int i, int j) {
        const int row_sum = result_.support[i];
        const float ratio = row_sum > 0 ? static_cast<float>(result_.matrix[i][j]) / row_sum : 0.0f;
        char buf[32];
        snprintf(buf, sizeof(buf), "%.1f%%", ratio * 100);
        return std::make_pair(ratio, std::string(buf));
    });
}

void ConfusionMatrixPanel::RenderGrid(const std::function<std::pair<float, std::string>(int, int)>& cell_at) {
    const int n = result_.n_classes;
    const ui::Tokens& t = ui::CurrentTokens();
    const ImGuiStyle& style = ImGui::GetStyle();

    // Sized from the text it holds (fixed 50 px cells and an 80 px label
    // column clipped "tested_negative"): cells fit their widest value, the
    // label column the longest class name, and column names alternate between
    // two lines so each has two cells of width.
    const float gap = 2.0f;
    const float line = ImGui::GetTextLineHeight();
    float value_width = 0.0f;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) value_width = std::max(value_width, ImGui::CalcTextSize(cell_at(i, j).second.c_str()).x);
    }
    const float cell = std::max(50.0f, value_width + 2.0f * style.FramePadding.x + gap);
    float label_width = 0.0f;
    for (const std::string& name : result_.class_names) label_width = std::max(label_width, ImGui::CalcTextSize(name.c_str()).x);
    label_width += style.ItemSpacing.x * 2.0f;
    const float header_height = (n > 1 ? 2.0f : 1.0f) * line + style.ItemSpacing.y * 2.0f;

    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const ImU32 label_colour = ui::ToU32(t.text_dim);
    const float grid_x = origin.x + label_width;
    const float grid_y = origin.y + header_height;

    for (int j = 0; j < n; ++j) {
        const char* name = result_.class_names[j].c_str();
        const float width = ImGui::CalcTextSize(name).x;
        const float room = (n > 1 ? 2.0f * cell : cell) - style.ItemSpacing.x;
        // Centred over the column when it fits, otherwise left-aligned and clipped to its room.
        const float x = width <= cell - gap ? grid_x + j * cell + (cell - gap - width) / 2 : grid_x + j * cell;
        const ImVec2 pos(x, origin.y + (n > 1 ? (j % 2) * (line + style.ItemSpacing.y) : 0.0f));
        const ImVec4 clip(pos.x, pos.y, grid_x + j * cell + room, pos.y + line);
        draw_list->AddText(nullptr, 0.0f, pos, label_colour, name, nullptr, 0.0f, &clip);
    }

    for (int i = 0; i < n; ++i) {
        draw_list->AddText(ImVec2(origin.x, grid_y + i * cell + (cell - gap - line) / 2), label_colour,
                           result_.class_names[i].c_str());
        for (int j = 0; j < n; ++j) {
            const auto [ratio, text] = cell_at(i, j);
            const ImVec4 fill = ui::SequentialAt(t, ratio);
            const ImVec2 min(grid_x + j * cell, grid_y + i * cell);
            const ImVec2 max(min.x + cell - gap, min.y + cell - gap);
            draw_list->AddRectFilled(min, max, ui::ToU32(fill));
            const ImVec2 text_size = ImGui::CalcTextSize(text.c_str());
            draw_list->AddText(ImVec2(min.x + (cell - gap - text_size.x) / 2, min.y + (cell - gap - text_size.y) / 2),
                               ui::ToU32(ui::ReadableOn(t, fill)), text.c_str());
        }
    }

    ImGui::Dummy(ImVec2(label_width + cell * n, header_height + cell * n));
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

        // One tick per class, named (a numeric axis put ticks at -0.2, 0.2).
        std::vector<double> x_pos(result_.n_classes);
        std::vector<const char*> tick_labels(result_.n_classes);
        for (int i = 0; i < result_.n_classes; ++i) {
            x_pos[i] = i;
            tick_labels[i] = result_.class_names[i].c_str();
        }
        ImPlot::SetupAxisTicks(ImAxis_X1, x_pos.data(), result_.n_classes, tick_labels.data());

        ImPlot::PlotBars("F1", x_pos.data(), result_.f1_scores.data(), result_.n_classes, 0.6);
        ImPlot::EndPlot();
    }
}

void ConfusionMatrixPanel::UseTable(std::shared_ptr<DataTable> table) {
    current_table_ = std::move(table);
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

    // True and predicted labels share one class set; text labels
    // ("tested_negative") keep their names.
    const TableClassLabels labels = ReadTableClassLabels(*current_table_, {true_label_column_, pred_label_column_});
    std::vector<int> y_true, y_pred;
    for (size_t row = 0; row < labels.columns[0].size(); ++row) {
        if (labels.columns[0][row] && labels.columns[1][row]) {
            y_true.push_back(*labels.columns[0][row]);
            y_pred.push_back(*labels.columns[1][row]);
        }
    }

    if (y_true.empty()) {
        status_message_ = "No valid label data found.";
        return;
    }

    // Names line up with the sorted class ids present in these rows.
    std::vector<std::string> class_names;
    if (labels.IsText()) {
        std::set<int> present(y_true.begin(), y_true.end());
        present.insert(y_pred.begin(), y_pred.end());
        for (int id : present) class_names.push_back(labels.Name(id));
    }
    result_ = ModelEvaluation::ComputeConfusionMatrix(y_true, y_pred, class_names);

    if (result_.success) {
        status_message_ = "Computed: " + std::to_string(result_.n_classes) + " classes, " +
                          std::to_string(result_.total_samples) + " samples.";
    } else {
        status_message_ = "Error: " + result_.error_message;
    }
}

} // namespace cyxwiz
