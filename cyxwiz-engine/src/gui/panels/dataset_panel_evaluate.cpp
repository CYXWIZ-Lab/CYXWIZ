/**
 * Dataset Panel - Evaluate Tab
 * Post-training evaluation dashboard embedding existing evaluation panels inline.
 * Supports classification metrics (confusion matrix, ROC, PR, F1) and
 * regression metrics (MSE, RMSE, MAE, R²).
 */

#include "dataset_panel.h"
#include "../icons.h"
#include "../../core/test_executor.h"
#include <imgui.h>
#include <implot.h>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace gui {

void TrainingEvaluationPanel::RenderEvaluateContent() {
    if (!IsDatasetLoaded()) {
        ImGui::TextDisabled("No dataset loaded. Load a dataset first.");
        return;
    }

    // Check if we have evaluation results
    if (!has_eval_results_) {
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
        ImGui::TextWrapped(
            ICON_FA_CHART_BAR "  No evaluation results yet.\n\n"
            "Train a model in the Training tab first, then results will appear here automatically.\n\n"
            "You can also load results from a previous evaluation."
        );
        ImGui::PopStyleColor();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Option to load previous results
        if (ImGui::Button(ICON_FA_FOLDER_OPEN " Load Results from File...")) {
            // TODO: Load saved evaluation results (JSON/CSV)
        }
        return;
    }

    // Task type selector
    ImGui::Text("Task Type:");
    ImGui::SameLine();
    ImGui::RadioButton("Classification", &eval_task_type_, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Regression", &eval_task_type_, 1);
    ImGui::SameLine();
    ImGui::Dummy(ImVec2(20, 0));
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_FLOPPY_DISK " Export Results")) {
        // TODO: Export to CSV/JSON
    }

    ImGui::Separator();
    ImGui::Spacing();

    if (eval_task_type_ == 0) {
        RenderClassificationEvaluation();
    } else {
        RenderRegressionEvaluation();
    }
}

void TrainingEvaluationPanel::RenderClassificationEvaluation() {
    // Overview metrics bar
    if (ImGui::CollapsingHeader(ICON_FA_GAUGE_HIGH " Overview Metrics", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent(8);

        // Show key metrics in a row
        float col_width = ImGui::GetContentRegionAvail().x / 5.0f;

        auto MetricCard = [&](const char* label, float value, ImVec4 color) {
            ImGui::BeginGroup();
            ImGui::PushStyleColor(ImGuiCol_Text, color);
            ImGui::Text("%.1f%%", value * 100.0f);
            ImGui::PopStyleColor();
            ImGui::TextDisabled("%s", label);
            ImGui::EndGroup();
            ImGui::SameLine(0, col_width - ImGui::GetItemRectSize().x);
        };

        MetricCard("Accuracy", eval_classification_.accuracy,
                   ImVec4(0.3f, 0.9f, 0.3f, 1.0f));
        MetricCard("Precision", eval_classification_.macro_precision,
                   ImVec4(0.3f, 0.7f, 1.0f, 1.0f));
        MetricCard("Recall", eval_classification_.macro_recall,
                   ImVec4(1.0f, 0.7f, 0.3f, 1.0f));
        MetricCard("F1 Score", eval_classification_.macro_f1,
                   ImVec4(0.9f, 0.3f, 0.9f, 1.0f));
        MetricCard("Weighted F1", eval_classification_.weighted_f1,
                   ImVec4(0.9f, 0.9f, 0.3f, 1.0f));
        ImGui::NewLine();

        ImGui::Unindent(8);
    }

    ImGui::Spacing();

    // Confusion Matrix
    if (ImGui::CollapsingHeader(ICON_FA_TABLE_CELLS " Confusion Matrix", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent(8);
        RenderInlineConfusionMatrix();
        ImGui::Unindent(8);
    }

    ImGui::Spacing();

    // Per-Class Metrics Table
    if (ImGui::CollapsingHeader(ICON_FA_LIST " Per-Class Metrics")) {
        ImGui::Indent(8);
        RenderPerClassMetrics();
        ImGui::Unindent(8);
    }

    ImGui::Spacing();

    // ROC Curve
    if (eval_has_roc_data_ && ImGui::CollapsingHeader(ICON_FA_CHART_LINE " ROC Curve")) {
        ImGui::Indent(8);
        RenderInlineROCCurve();
        ImGui::Unindent(8);
    }

    ImGui::Spacing();

    // PR Curve
    if (eval_has_pr_data_ && ImGui::CollapsingHeader(ICON_FA_CHART_AREA " Precision-Recall Curve")) {
        ImGui::Indent(8);
        RenderInlinePRCurve();
        ImGui::Unindent(8);
    }

    ImGui::Spacing();

    // Learning Curves
    if (eval_has_learning_curves_ && ImGui::CollapsingHeader(ICON_FA_GRADUATION_CAP " Learning Curves")) {
        ImGui::Indent(8);
        RenderInlineLearningCurves();
        ImGui::Unindent(8);
    }
}

void TrainingEvaluationPanel::RenderRegressionEvaluation() {
    if (ImGui::CollapsingHeader(ICON_FA_GAUGE_HIGH " Regression Metrics", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent(8);

        float col_width = ImGui::GetContentRegionAvail().x / 4.0f;

        auto MetricCard = [&](const char* label, double value, const char* fmt, ImVec4 color) {
            ImGui::BeginGroup();
            ImGui::PushStyleColor(ImGuiCol_Text, color);
            ImGui::Text(fmt, value);
            ImGui::PopStyleColor();
            ImGui::TextDisabled("%s", label);
            ImGui::EndGroup();
            ImGui::SameLine(0, col_width - ImGui::GetItemRectSize().x);
        };

        MetricCard("MSE", eval_regression_.mse, "%.6f",
                   ImVec4(0.3f, 0.9f, 0.3f, 1.0f));
        MetricCard("RMSE", eval_regression_.rmse, "%.6f",
                   ImVec4(0.3f, 0.7f, 1.0f, 1.0f));
        MetricCard("MAE", eval_regression_.mae, "%.6f",
                   ImVec4(1.0f, 0.7f, 0.3f, 1.0f));
        MetricCard("R\xc2\xb2", eval_regression_.r_squared, "%.4f",
                   ImVec4(0.9f, 0.3f, 0.9f, 1.0f));
        ImGui::NewLine();

        ImGui::Unindent(8);
    }

    ImGui::Spacing();

    // Predicted vs Actual scatter plot
    if (!eval_regression_.y_true.empty() &&
        ImGui::CollapsingHeader(ICON_FA_CHART_LINE " Predicted vs Actual", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent(8);
        if (ImPlot::BeginPlot("##PredVsActual", ImVec2(-1, 300))) {
            ImPlot::SetupAxes("Actual", "Predicted");
            ImPlot::PlotScatter("Predictions",
                                eval_regression_.y_true.data(),
                                eval_regression_.y_pred.data(),
                                (int)eval_regression_.y_true.size());

            // Perfect prediction line
            if (!eval_regression_.y_true.empty()) {
                double mn = *std::min_element(eval_regression_.y_true.begin(), eval_regression_.y_true.end());
                double mx = *std::max_element(eval_regression_.y_true.begin(), eval_regression_.y_true.end());
                double xs[] = {mn, mx};
                double ys[] = {mn, mx};
                ImPlot::PlotLine("Perfect", xs, ys, 2);
            }
            ImPlot::EndPlot();
        }
        ImGui::Unindent(8);
    }

    ImGui::Spacing();

    // Residual plot
    if (!eval_regression_.residuals.empty() &&
        ImGui::CollapsingHeader(ICON_FA_CHART_BAR " Residual Plot")) {
        ImGui::Indent(8);
        if (ImPlot::BeginPlot("##Residuals", ImVec2(-1, 250))) {
            ImPlot::SetupAxes("Predicted", "Residual");
            ImPlot::PlotScatter("Residuals",
                                eval_regression_.y_pred.data(),
                                eval_regression_.residuals.data(),
                                (int)eval_regression_.residuals.size());
            // Zero line
            double xs[] = {-1e9, 1e9};
            double ys[] = {0, 0};
            ImPlot::PlotLine("##zero", xs, ys, 2);
            ImPlot::EndPlot();
        }
        ImGui::Unindent(8);
    }

    // Learning curves (shared with classification)
    if (eval_has_learning_curves_ && ImGui::CollapsingHeader(ICON_FA_GRADUATION_CAP " Learning Curves")) {
        ImGui::Indent(8);
        RenderInlineLearningCurves();
        ImGui::Unindent(8);
    }
}

void TrainingEvaluationPanel::RenderInlineConfusionMatrix() {
    const auto& cm = eval_classification_;
    if (cm.n_classes <= 0) return;

    ImGui::Checkbox("Normalize", &eval_cm_normalize_);
    ImGui::Spacing();

    // Render as a colored table
    if (ImGui::BeginTable("##ConfMatrix", cm.n_classes + 1,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_SizingFixedFit)) {
        // Header row
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 80);
        for (int j = 0; j < cm.n_classes; j++) {
            std::string name = j < (int)cm.class_names.size() ? cm.class_names[j] : std::to_string(j);
            ImGui::TableSetupColumn(name.c_str(), ImGuiTableColumnFlags_WidthFixed, 60);
        }
        ImGui::TableHeadersRow();

        int max_val = 1;
        for (auto& row : cm.matrix)
            for (auto v : row)
                if (v > max_val) max_val = v;

        for (int i = 0; i < cm.n_classes; i++) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            std::string name = i < (int)cm.class_names.size() ? cm.class_names[i] : std::to_string(i);
            ImGui::Text("%s", name.c_str());

            int row_total = 0;
            for (int j = 0; j < cm.n_classes; j++)
                row_total += cm.matrix[i][j];

            for (int j = 0; j < cm.n_classes; j++) {
                ImGui::TableNextColumn();
                float intensity;
                if (eval_cm_normalize_ && row_total > 0)
                    intensity = (float)cm.matrix[i][j] / (float)row_total;
                else
                    intensity = (float)cm.matrix[i][j] / (float)max_val;

                // Color: diagonal = green, off-diagonal = red
                ImVec4 bg;
                if (i == j)
                    bg = ImVec4(0.0f, intensity * 0.8f, 0.0f, 0.5f + intensity * 0.3f);
                else
                    bg = ImVec4(intensity * 0.8f, 0.0f, 0.0f, 0.3f + intensity * 0.3f);

                ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, ImGui::GetColorU32(bg));

                if (eval_cm_normalize_ && row_total > 0)
                    ImGui::Text("%.1f%%", 100.0f * cm.matrix[i][j] / (float)row_total);
                else
                    ImGui::Text("%d", cm.matrix[i][j]);
            }
        }
        ImGui::EndTable();
    }
}

void TrainingEvaluationPanel::RenderPerClassMetrics() {
    const auto& cm = eval_classification_;
    if (cm.n_classes <= 0) return;

    if (ImGui::BeginTable("##PerClass", 5,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
        ImGui::TableSetupColumn("Class");
        ImGui::TableSetupColumn("Precision");
        ImGui::TableSetupColumn("Recall");
        ImGui::TableSetupColumn("F1 Score");
        ImGui::TableSetupColumn("Support");
        ImGui::TableHeadersRow();

        for (int i = 0; i < cm.n_classes; i++) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            std::string name = i < (int)cm.class_names.size() ? cm.class_names[i] : std::to_string(i);
            ImGui::Text("%s", name.c_str());

            ImGui::TableNextColumn();
            if (i < (int)cm.precision.size())
                ImGui::Text("%.3f", cm.precision[i]);

            ImGui::TableNextColumn();
            if (i < (int)cm.recall.size())
                ImGui::Text("%.3f", cm.recall[i]);

            ImGui::TableNextColumn();
            if (i < (int)cm.f1_scores.size())
                ImGui::Text("%.3f", cm.f1_scores[i]);

            ImGui::TableNextColumn();
            if (i < (int)cm.support.size())
                ImGui::Text("%d", cm.support[i]);
        }
        ImGui::EndTable();
    }
}

void TrainingEvaluationPanel::RenderInlineROCCurve() {
    if (!eval_has_roc_data_) return;

    if (ImPlot::BeginPlot("##ROCCurve", ImVec2(-1, 300))) {
        ImPlot::SetupAxes("False Positive Rate", "True Positive Rate");
        ImPlot::SetupAxesLimits(0, 1, 0, 1);

        // Plot per-class ROC if multi-class
        if (!eval_roc_.class_fpr.empty()) {
            for (size_t c = 0; c < eval_roc_.class_fpr.size(); c++) {
                std::string label = "Class " + std::to_string(c);
                if (!eval_roc_.class_auc.empty())
                    label += " (AUC=" + std::to_string(eval_roc_.class_auc[c]).substr(0, 5) + ")";
                ImPlot::PlotLine(label.c_str(),
                                 eval_roc_.class_fpr[c].data(),
                                 eval_roc_.class_tpr[c].data(),
                                 (int)eval_roc_.class_fpr[c].size());
            }
        } else if (!eval_roc_.fpr.empty()) {
            std::string label = "ROC (AUC=" + std::to_string(eval_roc_.auc).substr(0, 5) + ")";
            ImPlot::PlotLine(label.c_str(),
                             eval_roc_.fpr.data(),
                             eval_roc_.tpr.data(),
                             (int)eval_roc_.fpr.size());
        }

        // Diagonal reference
        double xs[] = {0, 1};
        double ys[] = {0, 1};
        ImPlot::SetNextLineStyle(ImVec4(0.5f, 0.5f, 0.5f, 0.5f));
        ImPlot::PlotLine("Random", xs, ys, 2);

        ImPlot::EndPlot();
    }

    ImGui::Text("AUC: %.4f", eval_roc_.auc);
}

void TrainingEvaluationPanel::RenderInlinePRCurve() {
    if (!eval_has_pr_data_) return;

    if (ImPlot::BeginPlot("##PRCurve", ImVec2(-1, 300))) {
        ImPlot::SetupAxes("Recall", "Precision");
        ImPlot::SetupAxesLimits(0, 1, 0, 1);

        if (!eval_pr_.recall.empty()) {
            std::string label = "PR (AP=" + std::to_string(eval_pr_.average_precision).substr(0, 5) + ")";
            ImPlot::PlotLine(label.c_str(),
                             eval_pr_.recall.data(),
                             eval_pr_.precision.data(),
                             (int)eval_pr_.recall.size());
        }

        ImPlot::EndPlot();
    }

    ImGui::Text("Average Precision: %.4f", eval_pr_.average_precision);
}

void TrainingEvaluationPanel::RenderInlineLearningCurves() {
    if (!eval_has_learning_curves_) return;

    if (ImPlot::BeginPlot("##LearningCurves", ImVec2(-1, 300))) {
        ImPlot::SetupAxes("Training Size", eval_learning_curves_.scoring_metric.c_str());

        if (!eval_learning_curves_.train_sizes.empty()) {
            // Convert int sizes to double for plotting
            std::vector<double> sizes_d(eval_learning_curves_.train_sizes.begin(),
                                         eval_learning_curves_.train_sizes.end());

            if (!eval_learning_curves_.train_scores_mean.empty()) {
                ImPlot::PlotLine("Train",
                                 sizes_d.data(),
                                 eval_learning_curves_.train_scores_mean.data(),
                                 (int)sizes_d.size());
            }
            if (!eval_learning_curves_.val_scores_mean.empty()) {
                ImPlot::PlotLine("Validation",
                                 sizes_d.data(),
                                 eval_learning_curves_.val_scores_mean.data(),
                                 (int)sizes_d.size());
            }
        }

        ImPlot::EndPlot();
    }
}

} // namespace gui
