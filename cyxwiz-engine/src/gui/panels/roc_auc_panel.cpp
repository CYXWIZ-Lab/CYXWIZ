#include "roc_auc_panel.h"
#include "../icons.h"
#include "../../core/data_registry.h"
#include "../../data/data_table.h"
#include "../../core/data_analyzer.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace cyxwiz {

ROCAUCPanel::ROCAUCPanel() : Panel("ROC Curve / AUC", true) {}

void ROCAUCPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin((std::string(ICON_FA_CHART_AREA) + " ROC Curve / AUC###ROCAUCPanel").c_str(), &visible_)) {
        if (data_registry_) available_tables_ = data_registry_->GetTableNames();

        ImGui::BeginChild("ConfigPanel", ImVec2(250, 0), true);
        RenderDataSelector();
        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);
        if (roc_result_.success) {
            if (ImGui::BeginTabBar("ROCTabs")) {
                if (ImGui::BeginTabItem(ICON_FA_CHART_AREA " ROC Curve")) {
                    RenderROCCurve();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_SLIDERS " Threshold Analysis")) {
                    RenderThresholdAnalysis();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_CHART_BAR " Metrics")) {
                    RenderMetrics();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        } else {
            ImGui::Text("Select true labels and prediction scores, then click Compute.");
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

void ROCAUCPanel::RenderDataSelector() {
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
        ImGui::Text(ICON_FA_TAGS " Column Selection");
        ImGui::Spacing();

        std::vector<const char*> col_names_c;
        for (const auto& name : column_names_) col_names_c.push_back(name.c_str());

        ImGui::Combo("True Labels (0/1)", &true_label_column_, col_names_c.data(), static_cast<int>(col_names_c.size()));
        ImGui::Combo("Prediction Scores", &score_column_, col_names_c.data(), static_cast<int>(col_names_c.size()));

        ImGui::Spacing();
        ImGui::Separator();

        if (ImGui::Button(ICON_FA_CALCULATOR " Compute ROC", ImVec2(-1, 30))) {
            ComputeROC();
        }
    }

    if (roc_result_.success) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Text(ICON_FA_CIRCLE_INFO " Results");
        ImGui::BulletText("AUC: %.4f", roc_result_.auc);
        ImGui::BulletText("Points: %d", static_cast<int>(roc_result_.fpr.size()));
    }

    if (!status_message_.empty()) {
        ImGui::Spacing();
        ImGui::TextWrapped("%s", status_message_.c_str());
    }
}

void ROCAUCPanel::RenderROCCurve() {
    if (!roc_result_.success) return;

    ImGui::Text("ROC Curve (AUC = %.4f)", roc_result_.auc);
    ImGui::Spacing();

    if (ImPlot::BeginPlot("##ROCCurve", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("False Positive Rate", "True Positive Rate");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, 1);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1);

        // Random classifier line (diagonal)
        double diag_x[] = {0.0, 1.0};
        double diag_y[] = {0.0, 1.0};
        ImPlot::SetNextLineStyle(ImVec4(0.5f, 0.5f, 0.5f, 0.7f), 1.0f);
        ImPlot::PlotLine("Random", diag_x, diag_y, 2);

        // ROC curve
        ImPlot::SetNextLineStyle(ImVec4(0.0f, 0.5f, 1.0f, 1.0f), 2.0f);
        ImPlot::PlotLine("ROC", roc_result_.fpr.data(), roc_result_.tpr.data(), static_cast<int>(roc_result_.fpr.size()));

        // Shade area under curve
        ImPlot::SetNextFillStyle(ImVec4(0.0f, 0.5f, 1.0f, 0.2f));
        ImPlot::PlotShaded("AUC Area", roc_result_.fpr.data(), roc_result_.tpr.data(), static_cast<int>(roc_result_.fpr.size()), 0);

        // Current threshold point
        if (!roc_result_.thresholds.empty()) {
            // Find closest threshold
            int best_idx = 0;
            double best_diff = 1e9;
            for (size_t i = 0; i < roc_result_.thresholds.size(); ++i) {
                double diff = std::abs(roc_result_.thresholds[i] - selected_threshold_);
                if (diff < best_diff) {
                    best_diff = diff;
                    best_idx = static_cast<int>(i);
                }
            }

            double point_x = roc_result_.fpr[best_idx];
            double point_y = roc_result_.tpr[best_idx];
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 8, ImVec4(1.0f, 0.0f, 0.0f, 1.0f), 2);
            ImPlot::PlotScatter("Current", &point_x, &point_y, 1);
        }

        ImPlot::EndPlot();
    }
}

void ROCAUCPanel::RenderThresholdAnalysis() {
    if (!roc_result_.success) return;

    ImGui::Text(ICON_FA_SLIDERS " Threshold Selection");
    ImGui::Spacing();

    if (ImGui::SliderFloat("Threshold", (float*)&selected_threshold_, 0.0f, 1.0f, "%.3f")) {
        // Recompute metrics at new threshold
        std::vector<int> y_true;
        std::vector<double> y_scores;

        if (current_table_) {
            int n_rows = static_cast<int>(current_table_->GetRowCount());
            for (int row = 0; row < n_rows; ++row) {
                auto true_val = DataAnalyzer::ToDouble(current_table_->GetCell(row, true_label_column_));
                auto score_val = DataAnalyzer::ToDouble(current_table_->GetCell(row, score_column_));
                if (true_val.has_value() && score_val.has_value()) {
                    y_true.push_back(static_cast<int>(true_val.value()));
                    y_scores.push_back(score_val.value());
                }
            }
            current_metrics_ = ModelEvaluation::ComputeBinaryMetrics(y_true, y_scores, selected_threshold_);
        }
    }

    ImGui::Spacing();
    ImGui::Separator();

    // Display metrics at current threshold
    ImGui::Text("Metrics at Threshold %.3f:", selected_threshold_);
    ImGui::Spacing();

    if (ImGui::BeginTable("ThresholdMetrics", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Metric", ImGuiTableColumnFlags_WidthFixed, 150.0f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableHeadersRow();

        auto add_row = [](const char* name, double value) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", name);
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", value);
        };

        add_row("Precision", current_metrics_.precision);
        add_row("Recall (Sensitivity)", current_metrics_.recall);
        add_row("Specificity", current_metrics_.specificity);
        add_row("F1 Score", current_metrics_.f1);
        add_row("Balanced Accuracy", current_metrics_.balanced_accuracy);
        add_row("MCC", current_metrics_.mcc);

        ImGui::EndTable();
    }

    ImGui::Spacing();
    ImGui::Text("Confusion at Threshold:");
    ImGui::BulletText("TP: %d, FP: %d", current_metrics_.tp, current_metrics_.fp);
    ImGui::BulletText("TN: %d, FN: %d", current_metrics_.tn, current_metrics_.fn);

    // Find optimal threshold button
    ImGui::Spacing();
    if (ImGui::Button(ICON_FA_BULLSEYE " Find Optimal (F1)", ImVec2(-1, 25))) {
        std::vector<int> y_true;
        std::vector<double> y_scores;

        if (current_table_) {
            int n_rows = static_cast<int>(current_table_->GetRowCount());
            for (int row = 0; row < n_rows; ++row) {
                auto true_val = DataAnalyzer::ToDouble(current_table_->GetCell(row, true_label_column_));
                auto score_val = DataAnalyzer::ToDouble(current_table_->GetCell(row, score_column_));
                if (true_val.has_value() && score_val.has_value()) {
                    y_true.push_back(static_cast<int>(true_val.value()));
                    y_scores.push_back(score_val.value());
                }
            }
            selected_threshold_ = ModelEvaluation::FindOptimalThreshold(y_true, y_scores, "f1");
            current_metrics_ = ModelEvaluation::ComputeBinaryMetrics(y_true, y_scores, selected_threshold_);
        }
    }
}

void ROCAUCPanel::RenderMetrics() {
    if (!roc_result_.success) return;

    ImGui::Text(ICON_FA_CHART_BAR " ROC Analysis Summary");
    ImGui::Separator();
    ImGui::Spacing();

    // AUC interpretation
    ImGui::Text("AUC Score: %.4f", roc_result_.auc);
    ImGui::Spacing();

    const char* interpretation;
    ImVec4 color;
    if (roc_result_.auc >= 0.9) {
        interpretation = "Excellent";
        color = ImVec4(0.0f, 0.8f, 0.0f, 1.0f);
    } else if (roc_result_.auc >= 0.8) {
        interpretation = "Good";
        color = ImVec4(0.6f, 0.8f, 0.0f, 1.0f);
    } else if (roc_result_.auc >= 0.7) {
        interpretation = "Fair";
        color = ImVec4(1.0f, 0.8f, 0.0f, 1.0f);
    } else if (roc_result_.auc >= 0.6) {
        interpretation = "Poor";
        color = ImVec4(1.0f, 0.5f, 0.0f, 1.0f);
    } else {
        interpretation = "Fail";
        color = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
    }

    ImGui::TextColored(color, "Interpretation: %s", interpretation);
    ImGui::Spacing();

    ImGui::Text("AUC Guidelines:");
    ImGui::BulletText("0.90 - 1.00: Excellent");
    ImGui::BulletText("0.80 - 0.89: Good");
    ImGui::BulletText("0.70 - 0.79: Fair");
    ImGui::BulletText("0.60 - 0.69: Poor");
    ImGui::BulletText("0.50 - 0.59: Fail (no better than random)");

    ImGui::Spacing();
    ImGui::Separator();

    // TPR vs FPR plot at different thresholds
    ImGui::Text("TPR and FPR vs Threshold");
    if (!roc_result_.thresholds.empty() && ImPlot::BeginPlot("##TPR_FPR", ImVec2(-1, 200))) {
        ImPlot::SetupAxes("Threshold", "Rate");
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1);

        // Need to reverse since thresholds are descending
        std::vector<double> thresh_rev(roc_result_.thresholds.rbegin(), roc_result_.thresholds.rend());
        std::vector<double> tpr_rev(roc_result_.tpr.rbegin(), roc_result_.tpr.rend());
        std::vector<double> fpr_rev(roc_result_.fpr.rbegin(), roc_result_.fpr.rend());

        ImPlot::SetNextLineStyle(ImVec4(0.0f, 0.8f, 0.0f, 1.0f), 2.0f);
        ImPlot::PlotLine("TPR", thresh_rev.data(), tpr_rev.data(), static_cast<int>(thresh_rev.size()));

        ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), 2.0f);
        ImPlot::PlotLine("FPR", thresh_rev.data(), fpr_rev.data(), static_cast<int>(thresh_rev.size()));

        ImPlot::EndPlot();
    }
}

void ROCAUCPanel::LoadSelectedData() {
    if (selected_table_idx_ < 0 || !data_registry_) return;

    current_table_ = data_registry_->GetTable(available_tables_[selected_table_idx_]);
    if (!current_table_) return;

    column_names_ = current_table_->GetHeaders();
    true_label_column_ = 0;
    int max_col = static_cast<int>(column_names_.size()) - 1;
    score_column_ = (1 < max_col) ? 1 : max_col;

    roc_result_ = ROCCurveData();
    current_metrics_ = BinaryMetrics();
    selected_threshold_ = 0.5;
    status_message_ = "Data loaded. Select label and score columns.";
}

void ROCAUCPanel::ComputeROC() {
    if (!current_table_) {
        status_message_ = "No data loaded.";
        return;
    }

    std::vector<int> y_true;
    std::vector<double> y_scores;
    int n_rows = static_cast<int>(current_table_->GetRowCount());

    for (int row = 0; row < n_rows; ++row) {
        auto true_val = DataAnalyzer::ToDouble(current_table_->GetCell(row, true_label_column_));
        auto score_val = DataAnalyzer::ToDouble(current_table_->GetCell(row, score_column_));

        if (true_val.has_value() && score_val.has_value()) {
            y_true.push_back(static_cast<int>(true_val.value()));
            y_scores.push_back(score_val.value());
        }
    }

    if (y_true.empty()) {
        status_message_ = "No valid data found.";
        return;
    }

    roc_result_ = ModelEvaluation::ComputeROC(y_true, y_scores);
    current_metrics_ = ModelEvaluation::ComputeBinaryMetrics(y_true, y_scores, selected_threshold_);

    if (roc_result_.success) {
        status_message_ = "ROC computed successfully.";
    } else {
        status_message_ = "Error: " + roc_result_.error_message;
    }
}

} // namespace cyxwiz
