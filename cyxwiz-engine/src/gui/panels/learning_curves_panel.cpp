#include "learning_curves_panel.h"
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

LearningCurvesPanel::LearningCurvesPanel() : Panel("Learning Curves", true) {}

void LearningCurvesPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin((std::string(ICON_FA_GRADUATION_CAP) + " Learning Curves###LearningCurvesPanel").c_str(), &visible_)) {
        if (data_registry_) available_tables_ = data_registry_->GetTableNames();

        ImGui::BeginChild("ConfigPanel", ImVec2(250, 0), true);
        RenderDataSelector();
        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);
        bool has_data = !result_.train_sizes.empty() || !manual_train_sizes_.empty();
        if (has_data) {
            if (ImGui::BeginTabBar("LearningCurveTabs")) {
                if (ImGui::BeginTabItem(ICON_FA_CHART_LINE " Learning Curve")) {
                    RenderLearningCurve();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_SCALE_BALANCED " Bias-Variance")) {
                    RenderBiasVarianceAnalysis();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_LIGHTBULB " Recommendations")) {
                    RenderRecommendations();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        } else {
            ImGui::Text("No learning curve data available.");
            ImGui::Text("Learning curves are populated during training.");
            ImGui::Spacing();
            ImGui::Text("Or load data from a CSV with columns:");
            ImGui::BulletText("train_size");
            ImGui::BulletText("train_score");
            ImGui::BulletText("val_score");
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

void LearningCurvesPanel::RenderDataSelector() {
    ImGui::Text(ICON_FA_TABLE " Data Source");
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

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Manual data entry
    ImGui::Text(ICON_FA_KEYBOARD " Manual Entry");
    ImGui::Spacing();

    static int new_train_size = 100;
    static float new_train_score = 0.9f;
    static float new_val_score = 0.8f;

    ImGui::InputInt("Train Size", &new_train_size);
    ImGui::SliderFloat("Train Score", &new_train_score, 0.0f, 1.0f);
    ImGui::SliderFloat("Val Score", &new_val_score, 0.0f, 1.0f);

    if (ImGui::Button(ICON_FA_PLUS " Add Point", ImVec2(-1, 25))) {
        AddDataPoint(new_train_size, new_train_score, new_val_score);
    }

    if (ImGui::Button(ICON_FA_TRASH " Clear Data", ImVec2(-1, 25))) {
        ClearData();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Show data summary
    size_t total_points = result_.train_sizes.empty() ? manual_train_sizes_.size() : result_.train_sizes.size();
    ImGui::Text("Data Points: %d", static_cast<int>(total_points));

    if (!status_message_.empty()) {
        ImGui::Spacing();
        ImGui::TextWrapped("%s", status_message_.c_str());
    }
}

void LearningCurvesPanel::RenderLearningCurve() {
    // Use result_ data if available, otherwise manual data
    const std::vector<int>& train_sizes = result_.train_sizes.empty() ? manual_train_sizes_ : result_.train_sizes;
    const std::vector<double>& train_scores = result_.train_sizes.empty() ? manual_train_scores_ : result_.train_scores_mean;
    const std::vector<double>& val_scores = result_.train_sizes.empty() ? manual_val_scores_ : result_.val_scores_mean;

    if (train_sizes.empty()) return;

    ImGui::Text("Learning Curve");
    ImGui::Text("Shows model performance vs training set size");
    ImGui::Spacing();

    // Convert train_sizes to double for plotting
    std::vector<double> sizes_d(train_sizes.begin(), train_sizes.end());

    if (ImPlot::BeginPlot("##LearningCurve", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("Training Set Size", "Score");
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1.1);

        // Training score curve
        ImPlot::SetNextLineStyle(ImVec4(0.2f, 0.6f, 1.0f, 1.0f), 2.0f);
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 5, ImVec4(0.2f, 0.6f, 1.0f, 1.0f), 1);
        ImPlot::PlotLine("Training Score", sizes_d.data(), train_scores.data(), static_cast<int>(train_scores.size()));

        // Validation score curve
        ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.4f, 0.2f, 1.0f), 2.0f);
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Square, 5, ImVec4(1.0f, 0.4f, 0.2f, 1.0f), 1);
        ImPlot::PlotLine("Validation Score", sizes_d.data(), val_scores.data(), static_cast<int>(val_scores.size()));

        // Plot error bands if std available
        if (!result_.train_scores_std.empty()) {
            std::vector<double> train_upper(train_scores.size()), train_lower(train_scores.size());
            std::vector<double> val_upper(val_scores.size()), val_lower(val_scores.size());

            for (size_t i = 0; i < train_scores.size(); ++i) {
                train_upper[i] = train_scores[i] + result_.train_scores_std[i];
                train_lower[i] = train_scores[i] - result_.train_scores_std[i];
            }
            for (size_t i = 0; i < val_scores.size(); ++i) {
                val_upper[i] = val_scores[i] + result_.val_scores_std[i];
                val_lower[i] = val_scores[i] - result_.val_scores_std[i];
            }

            ImPlot::SetNextFillStyle(ImVec4(0.2f, 0.6f, 1.0f, 0.2f));
            ImPlot::PlotShaded("Train +/- std", sizes_d.data(), train_lower.data(), train_upper.data(), static_cast<int>(sizes_d.size()));

            ImPlot::SetNextFillStyle(ImVec4(1.0f, 0.4f, 0.2f, 0.2f));
            ImPlot::PlotShaded("Val +/- std", sizes_d.data(), val_lower.data(), val_upper.data(), static_cast<int>(sizes_d.size()));
        }

        ImPlot::EndPlot();
    }
}

void LearningCurvesPanel::RenderBiasVarianceAnalysis() {
    const std::vector<double>& train_scores = result_.train_sizes.empty() ? manual_train_scores_ : result_.train_scores_mean;
    const std::vector<double>& val_scores = result_.train_sizes.empty() ? manual_val_scores_ : result_.val_scores_mean;

    if (train_scores.empty() || val_scores.empty()) return;

    ImGui::Text(ICON_FA_SCALE_BALANCED " Bias-Variance Analysis");
    ImGui::Separator();
    ImGui::Spacing();

    // Calculate bias and variance indicators
    double final_train = train_scores.back();
    double final_val = val_scores.back();
    double gap = final_train - final_val;

    // Estimate bias (1 - train_score) and variance (train - val)
    double bias_estimate = 1.0 - final_train;
    double variance_estimate = gap;

    ImGui::Text("Final Training Score: %.4f", final_train);
    ImGui::Text("Final Validation Score: %.4f", final_val);
    ImGui::Text("Gap (Train - Val): %.4f", gap);
    ImGui::Spacing();

    // Bias-Variance decomposition visualization
    if (ImPlot::BeginPlot("##BiasVariance", ImVec2(-1, 200))) {
        ImPlot::SetupAxes("Component", "Magnitude");
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1);

        double positions[] = {0.0, 1.0};
        double values[] = {bias_estimate, variance_estimate};
        const char* labels[] = {"Bias", "Variance"};

        ImPlot::SetNextFillStyle(ImVec4(0.8f, 0.3f, 0.2f, 0.8f));
        ImPlot::PlotBars("##bars", positions, values, 2, 0.6);

        // Add custom tick labels
        ImPlot::SetupAxisTicks(ImAxis_X1, positions, 2, labels);

        ImPlot::EndPlot();
    }

    ImGui::Spacing();
    ImGui::Separator();

    // Diagnosis
    ImGui::Text(ICON_FA_STETHOSCOPE " Diagnosis:");
    ImGui::Spacing();

    if (bias_estimate > 0.2 && variance_estimate < 0.1) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "High Bias (Underfitting)");
        ImGui::BulletText("Model is too simple");
        ImGui::BulletText("Consider: more features, complex model");
    } else if (bias_estimate < 0.1 && variance_estimate > 0.15) {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "High Variance (Overfitting)");
        ImGui::BulletText("Model is memorizing training data");
        ImGui::BulletText("Consider: more data, regularization, simpler model");
    } else if (bias_estimate < 0.1 && variance_estimate < 0.1) {
        ImGui::TextColored(ImVec4(0.0f, 0.8f, 0.0f, 1.0f), "Good Fit");
        ImGui::BulletText("Model has good balance");
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Mixed");
        ImGui::BulletText("Model may benefit from tuning");
    }
}

void LearningCurvesPanel::RenderRecommendations() {
    const std::vector<int>& train_sizes = result_.train_sizes.empty() ? manual_train_sizes_ : result_.train_sizes;
    const std::vector<double>& train_scores = result_.train_sizes.empty() ? manual_train_scores_ : result_.train_scores_mean;
    const std::vector<double>& val_scores = result_.train_sizes.empty() ? manual_val_scores_ : result_.val_scores_mean;

    if (train_sizes.size() < 2) {
        ImGui::Text("Need at least 2 data points for recommendations.");
        return;
    }

    ImGui::Text(ICON_FA_LIGHTBULB " Recommendations");
    ImGui::Separator();
    ImGui::Spacing();

    double final_train = train_scores.back();
    double final_val = val_scores.back();
    double gap = final_train - final_val;

    // Check if curves are converging
    double prev_gap = train_scores[train_scores.size() - 2] - val_scores[val_scores.size() - 2];
    bool converging = gap < prev_gap;

    // Check trend
    double val_improvement = val_scores.back() - val_scores.front();

    ImGui::Text(ICON_FA_ARROW_TREND_UP " Trend Analysis:");
    ImGui::Spacing();

    if (converging) {
        ImGui::TextColored(ImVec4(0.0f, 0.8f, 0.0f, 1.0f), "Curves are converging");
        ImGui::BulletText("More data may continue to help");
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Curves are not converging");
        ImGui::BulletText("Model changes may be more effective than more data");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text(ICON_FA_LIST_CHECK " Action Items:");
    ImGui::Spacing();

    // High bias recommendations
    if (final_train < 0.8) {
        ImGui::Text("To reduce underfitting:");
        ImGui::BulletText("Add more features or polynomial features");
        ImGui::BulletText("Use a more complex model");
        ImGui::BulletText("Reduce regularization");
        ImGui::BulletText("Train longer (more epochs)");
    }

    // High variance recommendations
    if (gap > 0.1) {
        ImGui::Text("To reduce overfitting:");
        ImGui::BulletText("Get more training data");
        ImGui::BulletText("Add regularization (L1, L2, dropout)");
        ImGui::BulletText("Use simpler model architecture");
        ImGui::BulletText("Use data augmentation");
        ImGui::BulletText("Early stopping");
    }

    // Good fit
    if (final_val > 0.85 && gap < 0.1) {
        ImGui::TextColored(ImVec4(0.0f, 0.8f, 0.0f, 1.0f), "Model appears well-tuned!");
        ImGui::BulletText("Consider ensembling for marginal gains");
        ImGui::BulletText("Focus on feature engineering");
    }

    ImGui::Spacing();
    ImGui::Separator();

    // Data efficiency
    ImGui::Text(ICON_FA_DATABASE " Data Efficiency:");
    if (val_improvement > 0 && train_sizes.size() >= 3) {
        // Extrapolate how much more data might help
        double recent_improvement = val_scores.back() - val_scores[val_scores.size() - 2];
        if (recent_improvement > 0.01) {
            ImGui::TextColored(ImVec4(0.0f, 0.8f, 0.0f, 1.0f), "More data is still helping");
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Diminishing returns from more data");
        }
    }
}

void LearningCurvesPanel::LoadSelectedData() {
    if (selected_table_idx_ < 0 || !data_registry_) return;

    current_table_ = data_registry_->GetTable(available_tables_[selected_table_idx_]);
    if (!current_table_) return;

    // Try to load learning curve data from table
    auto headers = current_table_->GetHeaders();

    int size_col = -1, train_col = -1, val_col = -1;
    for (size_t i = 0; i < headers.size(); ++i) {
        std::string h = headers[i];
        std::transform(h.begin(), h.end(), h.begin(), ::tolower);
        if (h.find("size") != std::string::npos || h.find("samples") != std::string::npos) size_col = static_cast<int>(i);
        if (h.find("train") != std::string::npos) train_col = static_cast<int>(i);
        if (h.find("val") != std::string::npos || h.find("test") != std::string::npos) val_col = static_cast<int>(i);
    }

    if (size_col >= 0 && train_col >= 0 && val_col >= 0) {
        manual_train_sizes_.clear();
        manual_train_scores_.clear();
        manual_val_scores_.clear();

        int n_rows = static_cast<int>(current_table_->GetRowCount());
        for (int row = 0; row < n_rows; ++row) {
            auto size_val = DataAnalyzer::ToDouble(current_table_->GetCell(row, size_col));
            auto train_val = DataAnalyzer::ToDouble(current_table_->GetCell(row, train_col));
            auto val_val = DataAnalyzer::ToDouble(current_table_->GetCell(row, val_col));

            if (size_val.has_value() && train_val.has_value() && val_val.has_value()) {
                manual_train_sizes_.push_back(static_cast<int>(size_val.value()));
                manual_train_scores_.push_back(train_val.value());
                manual_val_scores_.push_back(val_val.value());
            }
        }

        status_message_ = "Loaded " + std::to_string(manual_train_sizes_.size()) + " data points.";
    } else {
        status_message_ = "Could not find required columns (size, train, val).";
    }
}

void LearningCurvesPanel::AddDataPoint(int train_size, double train_score, double val_score) {
    manual_train_sizes_.push_back(train_size);
    manual_train_scores_.push_back(train_score);
    manual_val_scores_.push_back(val_score);

    // Sort by train size
    std::vector<size_t> order(manual_train_sizes_.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [this](size_t a, size_t b) {
        return manual_train_sizes_[a] < manual_train_sizes_[b];
    });

    std::vector<int> sorted_sizes(manual_train_sizes_.size());
    std::vector<double> sorted_train(manual_train_scores_.size());
    std::vector<double> sorted_val(manual_val_scores_.size());

    for (size_t i = 0; i < order.size(); ++i) {
        sorted_sizes[i] = manual_train_sizes_[order[i]];
        sorted_train[i] = manual_train_scores_[order[i]];
        sorted_val[i] = manual_val_scores_[order[i]];
    }

    manual_train_sizes_ = sorted_sizes;
    manual_train_scores_ = sorted_train;
    manual_val_scores_ = sorted_val;

    status_message_ = "Added point. Total: " + std::to_string(manual_train_sizes_.size());
}

void LearningCurvesPanel::ClearData() {
    manual_train_sizes_.clear();
    manual_train_scores_.clear();
    manual_val_scores_.clear();
    result_ = LearningCurveData();
    status_message_ = "Data cleared.";
}

} // namespace cyxwiz
