#include "cross_validation_panel.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"
#include "../../core/data_registry.h"
#include "../../data/data_table.h"
#include "../../core/table_class_labels.h"
#include "../../core/async_task_manager.h"
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
        ui::BeginSettingsColumn("cross_validation", 300.0f);
        RenderDataSelector();
        RenderCVSettings();
        ui::BeginResultsArea();
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
            ui::EmptyState(ICON_FA_REPEAT, "No folds yet",
                           "Pick a dataset, set the CV settings, then Generate Folds.");
        }
        ui::EndResultsArea();
    }
    ImGui::End();
}

void CrossValidationPanel::RenderDataSelector() {
    ui::SettingsSection("Data");
    source_.Render("cross_validation_dataset", [this](std::shared_ptr<DataTable> table) { UseTable(std::move(table)); });

    if (!column_names_.empty()) {
        ui::SettingsSection("Columns");
        std::vector<const char*> col_names_c;
        for (const auto& name : column_names_) col_names_c.push_back(name.c_str());

        ui::FieldLabel("Target column");
        ImGui::Combo("##target_column", &target_column_, col_names_c.data(), static_cast<int>(col_names_c.size()));

        ui::FieldLabel("Feature columns");
        ui::BeginInsetList("##columns", 120.0f);
        for (size_t i = 0; i < column_names_.size(); ++i) {
            if (static_cast<int>(i) == target_column_) continue;  // Skip target
            bool selected = selected_features_[i];
            if (ImGui::Checkbox(column_names_[i].c_str(), &selected)) {
                selected_features_[i] = selected;
            }
        }
        ui::EndInsetList();
    }
}

void CrossValidationPanel::RenderCVSettings() {
    ui::SettingsSection("CV settings");

    ui::FieldLabel("K folds");
    ImGui::SliderInt("##k_folds", &n_folds_, 2, 20);

    ImGui::Checkbox("Stratified", &stratified_);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Maintain class proportions in each fold");
    }

    ImGui::Checkbox("Shuffle", &shuffle_);

    if (shuffle_) {
        ui::FieldLabel("Random seed");
        ImGui::InputInt("##random_seed", &random_seed_);
    }

    ImGui::Spacing();

    bool can_run = !is_computing_ && current_table_ != nullptr;
    const char* run_reason = is_computing_ ? "Generating folds" : "Pick a dataset first";
    if (ui::PrimaryButton("Generate Folds", can_run, run_reason, ui::ButtonSize::Regular, -FLT_MIN)) {
        RunCrossValidation();
    }

    ImGui::PushStyleColor(ImGuiCol_Text, ui::CurrentTokens().text_dim);
    if (is_computing_) {
        ImGui::TextWrapped("Computing...");
    }

    if (!status_message_.empty()) {
        ImGui::TextWrapped("%s", status_message_.c_str());
    }
    ImGui::PopStyleColor();
}

void CrossValidationPanel::RenderResults() {
    if (fold_splits_.empty()) return;

    const ui::Tokens& t = ui::CurrentTokens();

    ImGui::Text(ICON_FA_CHART_BAR " Cross-Validation Summary");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Folds: %d", n_folds_);
    ImGui::Text("Total Samples: %d", current_table_ ? static_cast<int>(current_table_->GetRowCount()) : 0);

    ImGui::Spacing();

    // Show fold sizes
    std::vector<double> train_sizes(n_folds_), val_sizes(n_folds_);
    std::vector<double> train_positions(n_folds_), val_positions(n_folds_);

    for (int i = 0; i < n_folds_; ++i) {
        const double fold_position = static_cast<double>(i + 1);
        train_positions[i] = fold_position - 0.2;
        val_positions[i] = fold_position + 0.2;
        train_sizes[i] = static_cast<double>(fold_splits_[i].first.size());
        val_sizes[i] = static_cast<double>(fold_splits_[i].second.size());
    }

    ImGui::Text("Fold Size Distribution:");
    if (ImPlot::BeginPlot("##FoldSizes", ImVec2(-1, 200))) {
        ImPlot::SetupAxes("Fold", "Samples");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0.5, n_folds_ + 0.5);
        ImPlot::SetupAxisTicks(ImAxis_X1, 1, n_folds_, n_folds_);  // one tick per fold

        ImPlot::SetNextFillStyle(ui::WithAlpha(t.series[1], 0.8f));
        ImPlot::PlotBars("Train", train_positions.data(), train_sizes.data(),
                         n_folds_, 0.4);

        ImPlot::SetNextFillStyle(ui::WithAlpha(t.series[4], 0.8f));
        ImPlot::PlotBars("Validation", val_positions.data(), val_sizes.data(),
                         n_folds_, 0.4);

        ImPlot::EndPlot();
    }

    // Summary statistics
    double mean_train = std::accumulate(train_sizes.begin(), train_sizes.end(), 0.0) / n_folds_;
    double mean_val = std::accumulate(val_sizes.begin(), val_sizes.end(), 0.0) / n_folds_;

    ImGui::Spacing();
    ImGui::Text("Average Train Size: %.1f samples", mean_train);
    ImGui::Text("Average Validation Size: %.1f samples", mean_val);

    // If the folds were stratified, show the class distribution they kept
    if (!class_labels_.columns.empty()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Text("Class Distribution (Stratified):");

        const auto& labels = class_labels_.columns[0];
        std::map<int, int> class_counts;
        for (const auto& label : labels) {
            if (label) class_counts[*label]++;
        }

        for (const auto& [cls, count] : class_counts) {
            double ratio = static_cast<double>(count) / labels.size() * 100;
            ImGui::BulletText("%s: %d (%.1f%%)", ClassName(cls).c_str(), count, ratio);
        }
    }
}

std::string CrossValidationPanel::ClassName(int cls) const {
    return class_labels_.IsText() ? class_labels_.Name(cls) : "Class " + class_labels_.Name(cls);
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
            if (!class_labels_.columns.empty()) {
                const auto& labels = class_labels_.columns[0];
                std::map<int, int> val_class_counts;
                for (int idx : fold_splits_[f].second) {
                    if (idx < static_cast<int>(labels.size()) && labels[idx]) {
                        val_class_counts[*labels[idx]]++;
                    }
                }

                ImGui::Text("Validation Class Distribution:");
                for (const auto& [cls, count] : val_class_counts) {
                    double ratio = static_cast<double>(count) / fold_splits_[f].second.size() * 100;
                    ImGui::BulletText("%s: %d (%.1f%%)", ClassName(cls).c_str(), count, ratio);
                }
            }

            ImGui::Unindent();
        }
    }
}

void CrossValidationPanel::UseTable(std::shared_ptr<DataTable> table) {
    current_table_ = std::move(table);
    column_names_ = current_table_->GetHeaders();
    selected_features_.assign(column_names_.size(), true);

    // Default: last column is target
    target_column_ = static_cast<int>(column_names_.size()) - 1;
    if (target_column_ >= 0) {
        selected_features_[target_column_] = false;
    }

    fold_splits_.clear();
    class_labels_ = TableClassLabels{};
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

    // The worker gets copies of its inputs and hands the folds to the UI thread.
    compute_thread_ = std::thread([this, n_samples, table = current_table_, target = target_column_, stratified = stratified_,
                                   folds = n_folds_, shuffle = shuffle_, seed = static_cast<unsigned int>(random_seed_)]() {
        std::vector<std::pair<std::vector<int>, std::vector<int>>> splits;
        TableClassLabels class_labels;
        if (stratified) {
            // Stratify on the target's classes, text ("north") or numbers;
            // rows without a label form their own stratum.
            class_labels = ReadTableClassLabels(*table, {target});
            std::vector<int> labels;
            labels.reserve(n_samples);
            for (const auto& label : class_labels.columns[0]) labels.push_back(label.value_or(-1));
            splits = ModelEvaluation::StratifiedKFoldSplit(labels, folds, shuffle, seed);
        } else {
            splits = ModelEvaluation::KFoldSplit(n_samples, folds, shuffle, seed);
        }
        AsyncTaskManager::Instance().PostToMainThread(alive_, [this, splits = std::move(splits),
                                                               class_labels = std::move(class_labels), folds]() mutable {
            fold_splits_ = std::move(splits);
            class_labels_ = std::move(class_labels);
            result_.n_folds = folds;
            status_message_ = std::to_string(folds) + " folds generated successfully.";
            is_computing_ = false;
        });
    });
}

} // namespace cyxwiz
