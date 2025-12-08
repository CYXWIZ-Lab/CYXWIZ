#include "pr_curve_panel.h"
#include "../icons.h"
#include "../../core/data_registry.h"
#include "../../data/data_table.h"
#include "../../core/data_analyzer.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace cyxwiz {

PRCurvePanel::PRCurvePanel() : Panel("Precision-Recall Curve", true) {}

void PRCurvePanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin((std::string(ICON_FA_CHART_LINE) + " Precision-Recall Curve###PRCurvePanel").c_str(), &visible_)) {
        if (data_registry_) available_tables_ = data_registry_->GetTableNames();

        ImGui::BeginChild("ConfigPanel", ImVec2(250, 0), true);
        RenderDataSelector();
        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);
        if (pr_result_.success) {
            if (ImGui::BeginTabBar("PRTabs")) {
                if (ImGui::BeginTabItem(ICON_FA_CHART_LINE " PR Curve")) {
                    RenderPRCurve();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_CHART_AREA " F1 vs Threshold")) {
                    RenderF1Curve();
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

void PRCurvePanel::RenderDataSelector() {
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

        if (ImGui::Button(ICON_FA_CALCULATOR " Compute PR Curve", ImVec2(-1, 30))) {
            ComputePRCurve();
        }
    }

    if (pr_result_.success) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Text(ICON_FA_CIRCLE_INFO " Results");
        ImGui::BulletText("Avg Precision: %.4f", pr_result_.average_precision);
        ImGui::BulletText("Points: %d", static_cast<int>(pr_result_.precision.size()));
    }

    if (!status_message_.empty()) {
        ImGui::Spacing();
        ImGui::TextWrapped("%s", status_message_.c_str());
    }
}

void PRCurvePanel::RenderPRCurve() {
    if (!pr_result_.success) return;

    ImGui::Text("Precision-Recall Curve (AP = %.4f)", pr_result_.average_precision);
    ImGui::Spacing();

    if (ImPlot::BeginPlot("##PRCurve", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("Recall", "Precision");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, 1);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1);

        // PR curve
        ImPlot::SetNextLineStyle(ImVec4(0.0f, 0.5f, 1.0f, 1.0f), 2.0f);
        ImPlot::PlotLine("PR", pr_result_.recall.data(), pr_result_.precision.data(), static_cast<int>(pr_result_.recall.size()));

        // Shade area under curve
        ImPlot::SetNextFillStyle(ImVec4(0.0f, 0.5f, 1.0f, 0.2f));
        ImPlot::PlotShaded("AP Area", pr_result_.recall.data(), pr_result_.precision.data(), static_cast<int>(pr_result_.recall.size()), 0);

        // Iso-F1 curves
        for (double f1_val : {0.2, 0.4, 0.6, 0.8}) {
            std::vector<double> iso_r, iso_p;
            for (double r = 0.01; r <= 1.0; r += 0.01) {
                double p = f1_val * r / (2 * r - f1_val);
                if (p > 0 && p <= 1) {
                    iso_r.push_back(r);
                    iso_p.push_back(p);
                }
            }
            if (!iso_r.empty()) {
                ImPlot::SetNextLineStyle(ImVec4(0.7f, 0.7f, 0.7f, 0.5f), 1.0f);
                char label[32];
                snprintf(label, sizeof(label), "F1=%.1f", f1_val);
                ImPlot::PlotLine(label, iso_r.data(), iso_p.data(), static_cast<int>(iso_r.size()));
            }
        }

        ImPlot::EndPlot();
    }
}

void PRCurvePanel::RenderF1Curve() {
    if (!pr_result_.success || f1_scores_.empty()) return;

    ImGui::Text("F1 Score vs Threshold");
    ImGui::Spacing();

    // Find best F1 and its threshold
    double best_f1 = 0.0;
    int best_idx = 0;
    for (size_t i = 0; i < f1_scores_.size(); ++i) {
        if (f1_scores_[i] > best_f1) {
            best_f1 = f1_scores_[i];
            best_idx = static_cast<int>(i);
        }
    }

    double best_threshold = (best_idx < static_cast<int>(pr_result_.thresholds.size())) ?
                            pr_result_.thresholds[best_idx] : 0.5;

    ImGui::Text("Best F1: %.4f at threshold %.4f", best_f1, best_threshold);
    ImGui::Spacing();

    if (ImPlot::BeginPlot("##F1Curve", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("Threshold", "F1 Score");
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1);

        // Need thresholds that correspond to F1 scores
        std::vector<double> plot_thresholds;
        std::vector<double> plot_f1;

        // Reverse since thresholds are typically descending
        for (size_t i = 0; i < f1_scores_.size() && i < pr_result_.thresholds.size(); ++i) {
            plot_thresholds.push_back(pr_result_.thresholds[i]);
            plot_f1.push_back(f1_scores_[i]);
        }

        // Sort by threshold
        std::vector<size_t> order(plot_thresholds.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return plot_thresholds[a] < plot_thresholds[b];
        });

        std::vector<double> sorted_t(plot_thresholds.size()), sorted_f1(plot_f1.size());
        for (size_t i = 0; i < order.size(); ++i) {
            sorted_t[i] = plot_thresholds[order[i]];
            sorted_f1[i] = plot_f1[order[i]];
        }

        ImPlot::SetNextLineStyle(ImVec4(0.0f, 0.8f, 0.2f, 1.0f), 2.0f);
        ImPlot::PlotLine("F1", sorted_t.data(), sorted_f1.data(), static_cast<int>(sorted_t.size()));

        // Mark best point
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 8, ImVec4(1.0f, 0.0f, 0.0f, 1.0f), 2);
        ImPlot::PlotScatter("Best", &best_threshold, &best_f1, 1);

        ImPlot::EndPlot();
    }
}

void PRCurvePanel::RenderMetrics() {
    if (!pr_result_.success) return;

    ImGui::Text(ICON_FA_CHART_BAR " PR Curve Summary");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Average Precision (AP): %.4f", pr_result_.average_precision);
    ImGui::Spacing();

    // AP interpretation
    const char* interpretation;
    ImVec4 color;
    if (pr_result_.average_precision >= 0.9) {
        interpretation = "Excellent";
        color = ImVec4(0.0f, 0.8f, 0.0f, 1.0f);
    } else if (pr_result_.average_precision >= 0.7) {
        interpretation = "Good";
        color = ImVec4(0.6f, 0.8f, 0.0f, 1.0f);
    } else if (pr_result_.average_precision >= 0.5) {
        interpretation = "Fair";
        color = ImVec4(1.0f, 0.8f, 0.0f, 1.0f);
    } else {
        interpretation = "Poor";
        color = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
    }

    ImGui::TextColored(color, "Interpretation: %s", interpretation);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("PR Curve vs ROC:");
    ImGui::BulletText("PR is more informative for imbalanced datasets");
    ImGui::BulletText("ROC can be overly optimistic with class imbalance");
    ImGui::BulletText("AP summarizes the PR curve as a single number");

    ImGui::Spacing();
    ImGui::Separator();

    // Show precision at different recall levels
    ImGui::Text("Precision at Recall Levels:");
    if (ImGui::BeginTable("PrecisionAtRecall", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Recall", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableSetupColumn("Precision", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableHeadersRow();

        // Find precision at specific recall levels
        std::vector<double> recall_levels = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

        for (double target_recall : recall_levels) {
            // Find closest recall
            double best_prec = 0.0;
            for (size_t i = 0; i < pr_result_.recall.size(); ++i) {
                if (pr_result_.recall[i] >= target_recall) {
                    if (pr_result_.precision[i] > best_prec) {
                        best_prec = pr_result_.precision[i];
                    }
                }
            }

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%.1f", target_recall);
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", best_prec);
        }

        ImGui::EndTable();
    }
}

void PRCurvePanel::LoadSelectedData() {
    if (selected_table_idx_ < 0 || !data_registry_) return;

    current_table_ = data_registry_->GetTable(available_tables_[selected_table_idx_]);
    if (!current_table_) return;

    column_names_ = current_table_->GetHeaders();
    true_label_column_ = 0;
    int max_col = static_cast<int>(column_names_.size()) - 1;
    score_column_ = (1 < max_col) ? 1 : max_col;

    pr_result_ = PRCurveData();
    f1_scores_.clear();
    status_message_ = "Data loaded. Select label and score columns.";
}

void PRCurvePanel::ComputePRCurve() {
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

    pr_result_ = ModelEvaluation::ComputePRCurve(y_true, y_scores);

    if (pr_result_.success) {
        // Compute F1 at each threshold
        f1_scores_.clear();
        f1_scores_.reserve(pr_result_.precision.size());
        for (size_t i = 0; i < pr_result_.precision.size(); ++i) {
            double p = pr_result_.precision[i];
            double r = pr_result_.recall[i];
            double f1 = (p + r > 0) ? 2 * p * r / (p + r) : 0.0;
            f1_scores_.push_back(f1);
        }

        status_message_ = "PR curve computed successfully.";
    } else {
        status_message_ = "Error: " + pr_result_.error_message;
    }
}

} // namespace cyxwiz
