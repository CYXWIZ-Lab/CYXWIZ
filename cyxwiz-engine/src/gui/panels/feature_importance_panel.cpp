#include "feature_importance_panel.h"
#include "../../core/label_vocabulary.h"
#include "../../data/data_table.h"
#include "../../core/data_analyzer.h"
#include <cyxwiz/sequential.h>
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"
#include "../../core/training_manager.h"
#include "../../core/executable_model.h"
#include <cyxwiz/device.h>
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <filesystem>

namespace cyxwiz {

FeatureImportancePanel::FeatureImportancePanel() {
    std::memset(export_path_, 0, sizeof(export_path_));
}

FeatureImportancePanel::~FeatureImportancePanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        is_computing_ = false;
        compute_thread_->join();
    }
}

void FeatureImportancePanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(900, 700), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_STAR " Feature Importance###FeatureImportance", &visible_)) {
        RenderToolbar();

        // The active model: the last trained model or a loaded checkpoint, the
        // same one Test and Inference use. A result for another model is stale.
        auto& training = TrainingManager::Instance();
        const std::shared_ptr<IExecutableModel> model = training.GetActiveModel();
        if (model.get() != result_model_ && !is_computing_.load()) {
            has_result_ = false;
            result_model_ = nullptr;
        }

        ui::BeginSettingsColumn("feature_importance", 300.0f);
        RenderModel(model != nullptr, training.GetActiveModelInfo(), model ? model->LayerCount() : 0);
        SuggestTarget(training.GetActiveModelInfo().effective_label_column);
        RenderDataSelector();
        RenderConfiguration(model != nullptr, !model || model->AsSequentialModel() != nullptr);
        ui::BeginResultsArea();
        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else if (has_result_) {
            RenderResults();
        } else if (!model) {
            ui::EmptyState(ICON_FA_STAR, "Nothing to measure yet",
                           "Run training (Train menu) or load a checkpoint. The model then shows here, "
                           "and in Test and Inference.");
        } else {
            ui::EmptyState(ICON_FA_STAR, "No importance yet",
                           "Pick a dataset, the target and the features, then Compute importance.");
        }
        ui::EndResultsArea();
    }
    ImGui::End();
}

void FeatureImportancePanel::RenderToolbar() {
    if (ui::SecondaryButton(ICON_FA_FILE_EXPORT " Export", has_result_, "Compute importance first")) {
        ImGui::OpenPopup("ExportImportance");
    }
    RenderExportOptions();
}

void FeatureImportancePanel::RenderModel(bool has_model, const ActiveModelInfo& info, size_t layers) {
    const ui::Tokens& t = ui::CurrentTokens();
    ui::SettingsSection("Model");
    ui::BeginCard("active_model");
    if (has_model) {
        ImGui::TextColored(t.success, ICON_FA_CIRCLE);
        ImGui::SameLine();
        ImGui::TextUnformatted("Active model");
        std::string origin = "Trained this session";
        if (info.origin == ActiveModelOrigin::LoadedCheckpoint) {
            origin = "Checkpoint " + std::filesystem::path(info.checkpoint_path).filename().string();
        }
        ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
        ImGui::PushTextWrapPos(0.0f);
        ImGui::Text("%s - %zu layers", origin.c_str(), layers);
        ImGui::TextUnformatted("The same model Test and Inference use.");
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();
    } else {
        ImGui::TextColored(t.pending, ICON_FA_CIRCLE);
        ImGui::SameLine();
        ImGui::TextUnformatted("No active model yet");
        ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
        ImGui::TextWrapped("Train a graph, or load a checkpoint. Feature Importance measures the active model.");
        ImGui::PopStyleColor();
    }
    ui::EndCard();

    if (ui::SecondaryButton("Load checkpoint...", load_checkpoint_ != nullptr && !is_computing_.load(),
                            "Wait for the computation to finish", ui::ButtonSize::Regular, -FLT_MIN)) {
        load_checkpoint_();
    }
}

void FeatureImportancePanel::RenderDataSelector() {
    ui::SettingsSection("Data");
    source_.Render("importance_dataset", [this](std::shared_ptr<DataTable> table) { SetDataset(std::move(table)); });
    if (!current_table_) return;

    // Feature selection
    ui::FieldLabel("Features");
    std::string preview = std::to_string(selected_features_.size()) + " selected";
    if (ImGui::BeginCombo("##features", preview.c_str())) {
        for (size_t i = 0; i < feature_columns_.size(); i++) {
            bool is_selected = std::find(selected_features_.begin(), selected_features_.end(),
                                        static_cast<int>(i)) != selected_features_.end();
            if (ImGui::Checkbox(feature_columns_[i].c_str(), &is_selected)) {
                if (is_selected) {
                    selected_features_.push_back(static_cast<int>(i));
                } else {
                    selected_features_.erase(
                        std::remove(selected_features_.begin(), selected_features_.end(), static_cast<int>(i)),
                        selected_features_.end());
                }
                has_result_ = false;
            }
        }
        ImGui::EndCombo();
    }

    // Target column
    const auto& headers = current_table_->GetHeaders();
    std::string target_preview = target_column_ >= 0 && target_column_ < static_cast<int>(headers.size())
        ? headers[target_column_] : "Select...";
    ui::FieldLabel("Target column");
    if (ImGui::BeginCombo("##target", target_preview.c_str())) {
        for (size_t i = 0; i < headers.size(); i++) {
            if (ImGui::Selectable(headers[i].c_str(), target_column_ == static_cast<int>(i))) {
                target_column_ = static_cast<int>(i);
                has_result_ = false;
            }
        }
        ImGui::EndCombo();
    }
}

void FeatureImportancePanel::RenderConfiguration(bool has_model, bool layer_stack) {
    ui::SettingsSection("Method");
    const char* methods[] = {"Permutation Importance", "Drop-Column Importance", "Weight-Based"};
    ui::FieldLabel("Method");
    if (ImGui::Combo("##method", &method_, methods, IM_ARRAYSIZE(methods))) {
        has_result_ = false;
    }

    // Method-specific options
    if (method_ == 0) {
        ui::FieldLabel("Repeats (more = less variance)");
        if (ImGui::SliderInt("##repeats", &n_repeats_, 1, 50)) {
            has_result_ = false;
        }
    }

    // Scoring method (for permutation and drop-column)
    if (method_ < 2) {
        const char* scoring_methods[] = {"Accuracy", "MSE"};
        ui::FieldLabel("Scoring");
        if (ImGui::Combo("##scoring", &scoring_method_, scoring_methods, IM_ARRAYSIZE(scoring_methods))) {
            has_result_ = false;
        }
    }

    ImGui::Spacing();
    const char* reason = !has_model                 ? "Needs an active model"
                       : method_ == 2 && !layer_stack
                           ? "Weight-Based reads the first layer; a graph with branches feeds its inputs to several. "
                             "Use Permutation or Drop-Column."
                       : !current_table_            ? "Pick a dataset first"
                       : target_column_ < 0         ? "Pick the target column"
                       : selected_features_.empty() ? "Select at least one feature"
                                                    : nullptr;
    if (ui::PrimaryButton("Compute importance", reason == nullptr && !is_computing_.load(), reason,
                          ui::ButtonSize::Regular, -FLT_MIN)) {
        RunImportanceAnalysis();
    }
}

void FeatureImportancePanel::RenderLoadingIndicator() {
    ImGui::Spacing();
    ImGui::Text("%s Computing feature importance...", ICON_FA_SPINNER);

    if (method_ == 0) {
        // Permutation importance progress
        int current = progress_feature_.load();
        int total = total_features_.load();
        if (total > 0) {
            ImGui::Text("Feature: %d / %d", current, total);
            float progress = static_cast<float>(current) / total;
            ImGui::ProgressBar(progress, ImVec2(-1, 0));
        }
    }
}

void FeatureImportancePanel::RenderResults() {
    if (ImGui::BeginTabBar("ImportanceTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_CHART_BAR " Chart")) {
            RenderImportanceChart();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(ICON_FA_TABLE " Ranking")) {
            RenderRankingTable();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }
}

void FeatureImportancePanel::RenderImportanceChart() {
    std::lock_guard<std::mutex> lock(result_mutex_);

    if (!result_.success) {
        ImGui::TextColored(ui::CurrentTokens().error, "Error: %s", result_.error_message.c_str());
        return;
    }

    ImGui::Text("Method: %s", result_.method.c_str());
    ImGui::Text("Baseline Score: %.4f", result_.baseline_score);
    ImGui::Spacing();

    // Prepare sorted data for horizontal bar chart
    std::vector<int> sorted_indices = result_.ranking;
    std::vector<double> sorted_importances;
    std::vector<const char*> sorted_names;

    for (int idx : sorted_indices) {
        sorted_importances.push_back(result_.importances[idx]);
        sorted_names.push_back(result_.feature_names[idx].c_str());
    }

    // Horizontal bar chart, most important at the top; the feature names
    // are the axis ticks (labels drawn beside the bars sat on the wrong bars
    // and under the legend).
    if (ImPlot::BeginPlot("Feature Importance", ImVec2(-1, std::max(200.0f, 28.0f * sorted_indices.size())),
                          ImPlotFlags_NoLegend)) {
        ImPlot::SetupAxes("Importance", nullptr, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

        std::vector<double> positions(sorted_indices.size());
        for (size_t i = 0; i < positions.size(); i++) {
            positions[i] = static_cast<double>(sorted_indices.size() - 1 - i);
        }
        if (!positions.empty()) {
            ImPlot::SetupAxisTicks(ImAxis_Y1, positions.data(), static_cast<int>(positions.size()), sorted_names.data());
        }
        ImPlot::PlotBars("Importance", sorted_importances.data(), positions.data(),
                         static_cast<int>(sorted_importances.size()), 0.6, ImPlotBarsFlags_Horizontal);

        // Error bars if available
        if (!result_.importances_std.empty() && method_ == 0) {
            std::vector<double> sorted_std;
            for (int idx : sorted_indices) {
                sorted_std.push_back(result_.importances_std[idx]);
            }

            // ImPlot::SetNextErrorBarStyle(ImVec4(1, 0, 0, 1));
            // ImPlot::PlotErrorBarsH() - simplified since ImPlot may not have horizontal error bars
        }

        ImPlot::EndPlot();
    }
}

void FeatureImportancePanel::RenderRankingTable() {
    std::lock_guard<std::mutex> lock(result_mutex_);

    const ui::Tokens& t = ui::CurrentTokens();

    if (!result_.success) {
        ImGui::TextColored(t.error, "Error: %s", result_.error_message.c_str());
        return;
    }

    int n_cols = result_.importances_std.empty() ? 3 : 4;

    if (ImGui::BeginTable("RankingTable", n_cols,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY,
                          ImVec2(0, 400))) {

        ImGui::TableSetupColumn("Rank", ImGuiTableColumnFlags_WidthFixed, 50);
        ImGui::TableSetupColumn("Feature", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Importance", ImGuiTableColumnFlags_WidthFixed, 100);
        if (!result_.importances_std.empty()) {
            ImGui::TableSetupColumn("Std Dev", ImGuiTableColumnFlags_WidthFixed, 80);
        }
        ImGui::TableHeadersRow();

        for (size_t rank = 0; rank < result_.ranking.size(); rank++) {
            int idx = result_.ranking[rank];

            ImGui::TableNextRow();

            // Color based on importance
            float importance = static_cast<float>(result_.importances[idx]);
            ImVec4 color;
            if (importance >= 0.7f) {
                color = t.success;  // high
            } else if (importance >= 0.3f) {
                color = t.warning;  // medium
            } else {
                color = t.caution;  // low
            }

            ImGui::TableNextColumn();
            ImGui::Text("%zu", rank + 1);

            ImGui::TableNextColumn();
            ImGui::Text("%s", result_.feature_names[idx].c_str());

            ImGui::TableNextColumn();
            ImGui::TextColored(color, "%.4f", result_.importances[idx]);

            if (!result_.importances_std.empty()) {
                ImGui::TableNextColumn();
                ImGui::Text("+/- %.4f", result_.importances_std[idx]);
            }
        }

        ImGui::EndTable();
    }

    // Summary
    ImGui::Spacing();
    ImGui::Text("Top 3 Features:");
    for (size_t i = 0; i < std::min(size_t(3), result_.ranking.size()); i++) {
        int idx = result_.ranking[i];
        ImGui::BulletText("%s (%.2f%%)", result_.feature_names[idx].c_str(),
                         result_.importances[idx] * 100);
    }
}

void FeatureImportancePanel::RenderExportOptions() {
    if (ImGui::BeginPopup("ExportImportance")) {
        ImGui::Text("Export Feature Importance");
        ImGui::Separator();

        ImGui::InputText("File Path", export_path_, sizeof(export_path_));

        if (ImGui::Button("Save CSV")) {
            std::lock_guard<std::mutex> lock(result_mutex_);

            std::ofstream file(export_path_);
            if (file) {
                file << "Rank,Feature,Importance";
                if (!result_.importances_std.empty()) {
                    file << ",Std_Dev";
                }
                file << "\n";

                for (size_t rank = 0; rank < result_.ranking.size(); rank++) {
                    int idx = result_.ranking[rank];
                    file << (rank + 1) << "," << result_.feature_names[idx] << ","
                         << result_.importances[idx];
                    if (!result_.importances_std.empty()) {
                        file << "," << result_.importances_std[idx];
                    }
                    file << "\n";
                }

                spdlog::info("Exported feature importance to: {}", export_path_);
            }

            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

void FeatureImportancePanel::SuggestTarget(const std::string& label_column) {
    if (target_suggested_ || !current_table_ || target_column_ >= 0 || label_column.empty()) return;
    target_suggested_ = true;
    const auto& headers = current_table_->GetHeaders();
    const auto header = std::find(headers.begin(), headers.end(), label_column);
    if (header == headers.end()) return;
    target_column_ = static_cast<int>(header - headers.begin());
    // A numeric label is also a feature column; it is the target, not an input.
    const auto feature = std::find(feature_columns_.begin(), feature_columns_.end(), label_column);
    if (feature != feature_columns_.end()) {
        const int index = static_cast<int>(feature - feature_columns_.begin());
        selected_features_.erase(std::remove(selected_features_.begin(), selected_features_.end(), index),
                                 selected_features_.end());
    }
}

void FeatureImportancePanel::SetDataset(std::shared_ptr<DataTable> table) {
    current_table_ = table;
    selected_features_.clear();
    target_column_ = -1;
    target_suggested_ = false;
    has_result_ = false;

    feature_columns_.clear();
    if (current_table_) {
        const auto& headers = current_table_->GetHeaders();
        for (size_t i = 0; i < current_table_->GetColumnCount(); i++) {
            auto dtype = DataAnalyzer::DetectColumnType(*current_table_, i);
            if (dtype == ColumnDataType::Numeric || dtype == ColumnDataType::Integer) {
                feature_columns_.push_back(i < headers.size() ? headers[i] : "Column " + std::to_string(i));
                selected_features_.push_back(static_cast<int>(feature_columns_.size()) - 1);
            }
        }
    }
}

void FeatureImportancePanel::PrepareData() {
    if (!current_table_ || target_column_ < 0) return;

    X_data_.clear();
    y_data_.clear();
    feature_names_.clear();

    // Get numeric column indices for selected features
    std::vector<size_t> feature_indices;
    int count = 0;
    for (size_t i = 0; i < current_table_->GetColumnCount(); i++) {
        auto dtype = DataAnalyzer::DetectColumnType(*current_table_, i);
        if (dtype == ColumnDataType::Numeric || dtype == ColumnDataType::Integer) {
            if (std::find(selected_features_.begin(), selected_features_.end(), count) != selected_features_.end()) {
                feature_indices.push_back(i);
                feature_names_.push_back(feature_columns_[count]);
            }
            count++;
        }
    }

    // Extract X data
    size_t n_rows = current_table_->GetRowCount();
    X_data_.resize(n_rows);

    for (size_t row = 0; row < n_rows; row++) {
        X_data_[row].resize(feature_indices.size());
        for (size_t j = 0; j < feature_indices.size(); j++) {
            auto val = DataAnalyzer::ToDouble(current_table_->GetCell(row, feature_indices[j]));
            X_data_[row][j] = val.value_or(0.0);
        }
    }

    // Extract y data. A text target (class names) maps through the sorted
    // label vocabulary training uses, so the classes match the model's
    // outputs; reading it as numbers made every target 0 (TOFIX145).
    y_data_.resize(n_rows);
    bool text_target = false;
    for (size_t row = 0; row < n_rows && !text_target; row++) {
        const auto cell = current_table_->GetCell(row, target_column_);
        text_target = !DataAnalyzer::ToDouble(cell).has_value() &&
                      !current_table_->GetCellAsString(row, static_cast<size_t>(target_column_)).empty();
    }
    if (text_target) {
        std::vector<std::string> names(n_rows);
        for (size_t row = 0; row < n_rows; row++)
            names[row] = current_table_->GetCellAsString(row, static_cast<size_t>(target_column_));
        const LabelVocabulary vocabulary = BuildLabelVocabulary(names);
        for (size_t row = 0; row < n_rows; row++)
            y_data_[row] = std::max(0, vocabulary.IndexOf(names[row]));
    } else {
        for (size_t row = 0; row < n_rows; row++) {
            auto val = DataAnalyzer::ToDouble(current_table_->GetCell(row, target_column_));
            y_data_[row] = val.value_or(0.0);
        }
    }
}

void FeatureImportancePanel::RunImportanceAnalysis() {
    auto& training = TrainingManager::Instance();
    std::shared_ptr<IExecutableModel> model = training.GetActiveModel();
    if (is_computing_.load() || !model || !current_table_ || target_column_ < 0) return;
    const std::optional<ProcessDeviceSelection> model_device = training.GetActiveModelInfo().model_device;

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    // Prepare data
    PrepareData();

    if (X_data_.empty() || y_data_.empty()) {
        spdlog::error("No data for importance analysis");
        return;
    }

    is_computing_ = true;
    progress_feature_ = 0;
    total_features_ = static_cast<int>(X_data_[0].size());

    result_model_ = model.get();
    auto X = X_data_;
    auto y = y_data_;
    auto names = feature_names_;
    int method = method_;
    int repeats = n_repeats_;
    std::string scoring = scoring_method_ == 0 ? "accuracy" : "mse";

    compute_thread_ = std::make_unique<std::thread>([this, model, model_device, X, y, names, method, repeats,
                                                     scoring]() {
        try {
            // A new thread starts on ArrayFire's default backend, which can differ
            // from the one holding the weights (as in the export dialog).
            if (model_device) {
                const auto activation = Device(model_device->type, model_device->device_id).ActivateExact(false);
                if (!activation.success) {
                    std::lock_guard<std::mutex> lock(result_mutex_);
                    result_ = FeatureImportanceResult{};
                    result_.error_message = "Could not switch to the device that holds the model: " +
                                            activation.message;
                    has_result_ = true;
                    is_computing_ = false;
                    return;
                }
            }
            FeatureImportanceResult result;
            // Any model's forward pass in evaluation mode (a layer stack or a
            // graph with branches).
            model->SetTraining(false);
            const FeatureImportanceAnalyzer::ModelForward forward = [&model](const Tensor& input) {
                return model->Forward(input);
            };

            switch (method) {
                case 0: {
                    // Permutation importance
                    spdlog::info("Computing permutation importance with {} repeats", repeats);

                    auto callback = [this](int current, int total) {
                        progress_feature_ = current;
                        total_features_ = total;
                    };

                    result = FeatureImportanceAnalyzer::ComputePermutationImportance(
                        forward, X, y, names, repeats, scoring, callback);
                    break;
                }

                case 1: {
                    // Drop-column importance
                    spdlog::info("Computing drop-column importance");
                    result = FeatureImportanceAnalyzer::ComputeDropColumnImportance(
                        forward, X, y, names, scoring);
                    break;
                }

                case 2: {
                    // Weight-based importance
                    spdlog::info("Computing weight-based importance");
                    if (SequentialModel* stack = model->AsSequentialModel()) {
                        result = FeatureImportanceAnalyzer::ComputeWeightImportance(*stack, names);
                    } else {
                        result.success = false;
                        result.error_message = "Weight-Based reads the first layer; a graph with branches feeds its "
                                               "inputs to several. Use Permutation or Drop-Column.";
                    }
                    break;
                }
            }

            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                result_ = std::move(result);
                has_result_ = result_.success;
            }

            if (has_result_) {
                spdlog::info("Feature importance complete. Top feature: {}",
                            result_.feature_names[result_.ranking[0]]);
            } else {
                spdlog::error("Feature importance failed: {}", result_.error_message);
            }

        } catch (const std::exception& e) {
            spdlog::error("Feature importance error: {}", e.what());
        }

        is_computing_ = false;
    });
}

} // namespace cyxwiz
