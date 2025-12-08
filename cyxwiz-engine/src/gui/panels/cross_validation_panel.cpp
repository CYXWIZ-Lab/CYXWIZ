#include "cross_validation_panel.h"
#include "../icons.h"
#include "../../core/data_registry.h"
#include "../../data/data_table.h"
#include "../../core/data_analyzer.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>

namespace cyxwiz {

CrossValidationPanel::CrossValidationPanel() : Panel("Cross-Validation", true) {}

CrossValidationPanel::~CrossValidationPanel() {
    if (compute_thread_.joinable()) compute_thread_.join();
}

void CrossValidationPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin((std::string(ICON_FA_REPEAT) + " Cross-Validation###CrossValidationPanel").c_str(), &visible_)) {
        if (data_registry_) available_tables_ = data_registry_->GetTableNames();

        ImGui::BeginChild("ConfigPanel", ImVec2(280, 0), true);
        RenderDataSelector();
        ImGui::Separator();
        RenderCVSettings();
        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);
        if (!fold_splits_.empty()) {
            if (ImGui::BeginTabBar("CVTabs")) {
                if (ImGui::BeginTabItem(ICON_FA_CHART_BAR " Results")) {
                    RenderResults();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_LIST " Fold Details")) {
                    RenderFoldDetails();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        } else {
            ImGui::Text("Configure settings and click 'Generate Folds' or 'Run CV'.");
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

void CrossValidationPanel::RenderDataSelector() {
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
        std::vector<const char*> col_names_c;
        for (const auto& name : column_names_) col_names_c.push_back(name.c_str());

        ImGui::Combo("Target Column", &target_column_, col_names_c.data(), static_cast<int>(col_names_c.size()));

        ImGui::Spacing();
        ImGui::Text("Feature Columns:");
        ImGui::BeginChild("FeatureSelect", ImVec2(0, 120), true);
        for (size_t i = 0; i < column_names_.size(); ++i) {
            if (static_cast<int>(i) == target_column_) continue;  // Skip target
            bool selected = selected_features_[i];
            if (ImGui::Checkbox(column_names_[i].c_str(), &selected)) {
                selected_features_[i] = selected;
            }
        }
        ImGui::EndChild();
    }
}

void CrossValidationPanel::RenderCVSettings() {
    ImGui::Text(ICON_FA_COG " CV Settings");
    ImGui::Spacing();

    ImGui::SliderInt("K Folds", &n_folds_, 2, 20);

    ImGui::Checkbox("Stratified", &stratified_);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Maintain class proportions in each fold");
    }

    ImGui::Checkbox("Shuffle", &shuffle_);

    if (shuffle_) {
        ImGui::InputInt("Random Seed", &random_seed_);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    bool can_run = !is_computing_ && current_table_ != nullptr;
    if (!can_run) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_LAYER_GROUP " Generate Folds", ImVec2(-1, 30))) {
        RunCrossValidation();
    }

    if (!can_run) ImGui::EndDisabled();

    if (is_computing_) {
        ImGui::Text("Computing...");
    }

    if (!status_message_.empty()) {
        ImGui::Spacing();
        ImGui::TextWrapped("%s", status_message_.c_str());
    }
}

void CrossValidationPanel::RenderResults() {
    if (fold_splits_.empty()) return;

    ImGui::Text(ICON_FA_CHART_BAR " Cross-Validation Summary");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Folds: %d", n_folds_);
    ImGui::Text("Total Samples: %d", current_table_ ? static_cast<int>(current_table_->GetRowCount()) : 0);

    ImGui::Spacing();

    // Show fold sizes
    std::vector<double> train_sizes(n_folds_), val_sizes(n_folds_);
    std::vector<double> fold_indices(n_folds_);

    for (int i = 0; i < n_folds_; ++i) {
        fold_indices[i] = i + 1;
        train_sizes[i] = static_cast<double>(fold_splits_[i].first.size());
        val_sizes[i] = static_cast<double>(fold_splits_[i].second.size());
    }

    ImGui::Text("Fold Size Distribution:");
    if (ImPlot::BeginPlot("##FoldSizes", ImVec2(-1, 200))) {
        ImPlot::SetupAxes("Fold", "Samples");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0.5, n_folds_ + 0.5);

        ImPlot::SetNextFillStyle(ImVec4(0.2f, 0.5f, 0.8f, 0.8f));
        ImPlot::PlotBars("Train", fold_indices.data(), train_sizes.data(), n_folds_, 0.4, -0.2);

        ImPlot::SetNextFillStyle(ImVec4(0.8f, 0.3f, 0.2f, 0.8f));
        ImPlot::PlotBars("Validation", fold_indices.data(), val_sizes.data(), n_folds_, 0.4, 0.2);

        ImPlot::EndPlot();
    }

    // Summary statistics
    double mean_train = std::accumulate(train_sizes.begin(), train_sizes.end(), 0.0) / n_folds_;
    double mean_val = std::accumulate(val_sizes.begin(), val_sizes.end(), 0.0) / n_folds_;

    ImGui::Spacing();
    ImGui::Text("Average Train Size: %.1f samples", mean_train);
    ImGui::Text("Average Validation Size: %.1f samples", mean_val);

    // If stratified, show class distribution
    if (stratified_ && current_table_) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Text("Class Distribution (Stratified):");

        // Count classes in original data
        std::map<int, int> class_counts;
        int n_rows = static_cast<int>(current_table_->GetRowCount());
        for (int row = 0; row < n_rows; ++row) {
            auto val = DataAnalyzer::ToDouble(current_table_->GetCell(row, target_column_));
            if (val.has_value()) {
                int cls = static_cast<int>(val.value());
                class_counts[cls]++;
            }
        }

        for (const auto& [cls, count] : class_counts) {
            double ratio = static_cast<double>(count) / n_rows * 100;
            ImGui::BulletText("Class %d: %d (%.1f%%)", cls, count, ratio);
        }
    }
}

void CrossValidationPanel::RenderFoldDetails() {
    if (fold_splits_.empty()) return;

    ImGui::Text(ICON_FA_LIST " Fold Details");
    ImGui::Separator();
    ImGui::Spacing();

    for (int f = 0; f < n_folds_; ++f) {
        char header[64];
        snprintf(header, sizeof(header), "Fold %d (Train: %d, Val: %d)",
                 f + 1,
                 static_cast<int>(fold_splits_[f].first.size()),
                 static_cast<int>(fold_splits_[f].second.size()));

        if (ImGui::CollapsingHeader(header)) {
            ImGui::Indent();

            // Show first few validation indices
            ImGui::Text("Validation Indices (first 20):");
            std::string indices_str;
            int show_count = (fold_splits_[f].second.size() < 20) ?
                             static_cast<int>(fold_splits_[f].second.size()) : 20;
            for (int i = 0; i < show_count; ++i) {
                if (i > 0) indices_str += ", ";
                indices_str += std::to_string(fold_splits_[f].second[i]);
            }
            if (fold_splits_[f].second.size() > 20) {
                indices_str += "...";
            }
            ImGui::TextWrapped("%s", indices_str.c_str());

            // Class distribution in this fold's validation set
            if (current_table_) {
                std::map<int, int> val_class_counts;
                for (int idx : fold_splits_[f].second) {
                    if (idx < static_cast<int>(current_table_->GetRowCount())) {
                        auto val = DataAnalyzer::ToDouble(current_table_->GetCell(idx, target_column_));
                        if (val.has_value()) {
                            val_class_counts[static_cast<int>(val.value())]++;
                        }
                    }
                }

                ImGui::Text("Validation Class Distribution:");
                for (const auto& [cls, count] : val_class_counts) {
                    double ratio = static_cast<double>(count) / fold_splits_[f].second.size() * 100;
                    ImGui::BulletText("Class %d: %d (%.1f%%)", cls, count, ratio);
                }
            }

            ImGui::Unindent();
        }
    }
}

void CrossValidationPanel::LoadSelectedData() {
    if (selected_table_idx_ < 0 || !data_registry_) return;

    current_table_ = data_registry_->GetTable(available_tables_[selected_table_idx_]);
    if (!current_table_) return;

    column_names_ = current_table_->GetHeaders();
    selected_features_.assign(column_names_.size(), true);

    // Default: last column is target
    target_column_ = static_cast<int>(column_names_.size()) - 1;
    if (target_column_ >= 0) {
        selected_features_[target_column_] = false;
    }

    fold_splits_.clear();
    result_ = CrossValidationResult();
    status_message_ = "Data loaded. Configure CV settings.";
}

void CrossValidationPanel::RunCrossValidation() {
    if (!current_table_ || is_computing_) return;

    int n_samples = static_cast<int>(current_table_->GetRowCount());

    if (n_samples < n_folds_) {
        status_message_ = "Not enough samples for " + std::to_string(n_folds_) + " folds.";
        return;
    }

    is_computing_ = true;
    status_message_ = "Generating folds...";

    if (compute_thread_.joinable()) compute_thread_.join();

    compute_thread_ = std::thread([this, n_samples]() {
        if (stratified_) {
            // Get labels for stratified split
            std::vector<int> labels;
            labels.reserve(n_samples);
            for (int row = 0; row < n_samples; ++row) {
                auto val = DataAnalyzer::ToDouble(current_table_->GetCell(row, target_column_));
                labels.push_back(val.has_value() ? static_cast<int>(val.value()) : 0);
            }
            fold_splits_ = ModelEvaluation::StratifiedKFoldSplit(labels, n_folds_, shuffle_, static_cast<unsigned int>(random_seed_));
        } else {
            fold_splits_ = ModelEvaluation::KFoldSplit(n_samples, n_folds_, shuffle_, static_cast<unsigned int>(random_seed_));
        }

        result_.n_folds = n_folds_;
        is_computing_ = false;
        status_message_ = std::to_string(n_folds_) + " folds generated successfully.";
    });
}

} // namespace cyxwiz
