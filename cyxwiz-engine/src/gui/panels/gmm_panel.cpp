#include "gmm_panel.h"
#include <cyxwiz/device.h>
#include "../../core/async_task_manager.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"
#include "../../core/data_registry.h"
#include "../../core/data_analyzer.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace cyxwiz {

GMMPanel::GMMPanel() : Panel("GMM Clustering", true) {}

GMMPanel::~GMMPanel() {
    if (compute_thread_.joinable()) compute_thread_.join();
}

void GMMPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(950, 750), ImGuiCond_FirstUseEver);

    if (ImGui::Begin((std::string(ICON_FA_CHART_PIE) + " Gaussian Mixture Models###GMMPanel").c_str(), &visible_)) {
        ui::BeginSettingsColumn("gmm", 320.0f);
        RenderDataSelector();
        RenderParameters();
        RenderModelSelection();
        ui::BeginResultsArea();
        RenderResults();
        ui::EndResultsArea();
    }
    ImGui::End();
}

void GMMPanel::RenderDataSelector() {
    ui::SettingsSection("Data");
    source_.Render("gmm_dataset", [this](std::shared_ptr<DataTable> table) { UseTable(std::move(table)); });

    if (!column_names_.empty()) {
        ui::SettingsSection("Columns");
        ImGui::TextColored(ui::CurrentTokens().text_dim, "Feature columns");
        ui::BeginInsetList("##columns", 100);
        for (size_t i = 0; i < column_names_.size(); ++i) {
            bool selected = selected_columns_[i];
            if (ImGui::Checkbox(column_names_[i].c_str(), &selected)) {
                selected_columns_[i] = selected;
            }
        }
        ui::EndInsetList();

        std::vector<const char*> col_names_c;
        for (const auto& name : column_names_) col_names_c.push_back(name.c_str());
        if (ImGui::BeginTable("##xy", 2, ImGuiTableFlags_SizingStretchSame)) {
            ImGui::TableNextColumn();
            ui::FieldLabel("X axis");
            ImGui::Combo("##x_axis", &x_axis_column_, col_names_c.data(), static_cast<int>(col_names_c.size()));
            ImGui::TableNextColumn();
            ui::FieldLabel("Y axis");
            ImGui::Combo("##y_axis", &y_axis_column_, col_names_c.data(), static_cast<int>(col_names_c.size()));
            ImGui::EndTable();
        }
    }
}

void GMMPanel::RenderParameters() {
    ui::SettingsSection("Parameters");

    ui::FieldLabel("Components");
    ImGui::SliderInt("##n_components", &n_components_, 2, 20);
    ui::FieldLabel("Max iterations");
    ImGui::SliderInt("##max_iter", &max_iter_, 10, 500);
    ui::FieldLabel("Initializations");
    ImGui::SliderInt("##n_init", &n_init_, 1, 10);

    const char* cov_types[] = {"Full", "Tied", "Diagonal", "Spherical"};
    ui::FieldLabel("Covariance type");
    ImGui::Combo("##covariance_type", &covariance_type_, cov_types, 4);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Full: Each component has its own covariance matrix\n"
                         "Tied: All components share the same covariance\n"
                         "Diagonal: Off-diagonal elements are zero\n"
                         "Spherical: Single variance value per component");
    }

    ImGui::Spacing();

    bool can_run = !is_computing_ && current_table_ != nullptr;
    const char* run_reason = is_computing_ ? "GMM is running" : "Pick a dataset first";
    if (ui::PrimaryButton("Run GMM", can_run, run_reason, ui::ButtonSize::Regular, -FLT_MIN)) {
        RunClustering();
    }

    const ui::Tokens& t = ui::CurrentTokens();
    if (is_computing_) {
        ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
        ImGui::TextWrapped("Computing... Iteration %d", progress_iteration_.load());
        ImGui::PopStyleColor();
    }

    if (!status_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
        ImGui::TextWrapped("%s", status_message_.c_str());
        ImGui::PopStyleColor();
    }
}

void GMMPanel::RenderModelSelection() {
    ui::SettingsSection("Model selection");

    if (ImGui::BeginTable("##component_range", 2, ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextColumn();
        ui::FieldLabel("Min components");
        ImGui::SliderInt("##model_select_min", &model_select_min_, 1, 10);
        ImGui::TableNextColumn();
        ui::FieldLabel("Max components");
        ImGui::SliderInt("##model_select_max", &model_select_max_, model_select_min_ + 1, 20);
        ImGui::EndTable();
    }

    bool can_run = !is_computing_ && current_table_ != nullptr;
    const char* run_reason = is_computing_ ? "GMM is running" : "Pick a dataset first";
    if (ui::SecondaryButton("Find Optimal Components", can_run, run_reason, ui::ButtonSize::Regular, -FLT_MIN)) {
        RunModelSelection();
    }

    if (!model_k_values_.empty()) {
        ImGui::Text("Suggested: %d components", suggested_components_);

        if (ui::SecondaryButton("Use Suggested")) {
            n_components_ = suggested_components_;
        }

        if (ImPlot::BeginPlot("##BIC_AIC", ImVec2(-1, 120))) {
            ImPlot::SetupAxes("Components", "Score");
            ImPlot::SetupLegend(ImPlotLocation_NorthEast);

            std::vector<double> k_vals(model_k_values_.begin(), model_k_values_.end());
            ImPlot::PlotLine("BIC", k_vals.data(), model_bic_values_.data(), static_cast<int>(k_vals.size()));
            ImPlot::PlotLine("AIC", k_vals.data(), model_aic_values_.data(), static_cast<int>(k_vals.size()));

            ImPlot::EndPlot();
        }
    }
}

void GMMPanel::RenderResults() {
    if (!result_.success) {
        ui::EmptyState(ICON_FA_CHART_PIE, "No mixture model yet",
                       "Pick a dataset and feature columns, then Run GMM.");
        return;
    }

    if (ImGui::BeginTabBar("ResultTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_CHART_SCATTER " Scatter Plot")) {
            RenderScatterPlot();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(ICON_FA_CIRCLE_INFO " Components")) {
            RenderComponentInfo();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(ICON_FA_CHART_BAR " Statistics")) {
            ImGui::Text("GMM Results");
            ImGui::Separator();

            ImGui::Text("Components: %d", result_.n_components);
            ImGui::Text("Iterations: %d", result_.n_iterations);
            ImGui::Text("Converged: %s", result_.converged ? "Yes" : "No");
            ImGui::Text("Log-Likelihood: %.4f", result_.log_likelihood);
            ImGui::Text("BIC: %.4f", result_.bic);
            ImGui::Text("AIC: %.4f", result_.aic);

            // Component weights
            ImGui::Spacing();
            ImGui::Text("Component Weights:");

            if (!result_.weights.empty() && ImPlot::BeginPlot("##Weights", ImVec2(-1, 150))) {
                ImPlot::SetupAxes("Component", "Weight");
                std::vector<double> x_pos(result_.n_components);
                for (int i = 0; i < result_.n_components; ++i) x_pos[i] = i;
                ImPlot::PlotBars("Weight", x_pos.data(), result_.weights.data(), result_.n_components, 0.6);
                ImPlot::EndPlot();
            }

            // Cluster sizes (hard assignment)
            ImGui::Spacing();
            ImGui::Text("Cluster Sizes (Hard Assignment):");

            std::vector<int> cluster_sizes(result_.n_components, 0);
            for (int label : result_.labels) {
                if (label >= 0 && label < result_.n_components) cluster_sizes[label]++;
            }

            if (ImPlot::BeginPlot("##ClusterSizes", ImVec2(-1, 150))) {
                ImPlot::SetupAxes("Component", "Count");
                std::vector<double> x_pos(result_.n_components), sizes(result_.n_components);
                for (int i = 0; i < result_.n_components; ++i) {
                    x_pos[i] = i;
                    sizes[i] = cluster_sizes[i];
                }
                ImPlot::PlotBars("Count", x_pos.data(), sizes.data(), result_.n_components, 0.6);
                ImPlot::EndPlot();
            }

            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void GMMPanel::RenderScatterPlot() {
    if (input_data_.empty() || result_.labels.empty()) return;

    std::vector<int> selected_col_indices;
    for (size_t i = 0; i < selected_columns_.size(); ++i) {
        if (selected_columns_[i]) selected_col_indices.push_back(static_cast<int>(i));
    }

    if (selected_col_indices.size() < 2) {
        ImGui::Text("Select at least 2 columns.");
        return;
    }

    int max_y = static_cast<int>(input_data_[0].size()) - 1;
    int x_idx = 0, y_idx = (1 < max_y) ? 1 : max_y;
    for (size_t i = 0; i < selected_col_indices.size(); ++i) {
        if (selected_col_indices[i] == x_axis_column_) x_idx = static_cast<int>(i);
        if (selected_col_indices[i] == y_axis_column_) y_idx = static_cast<int>(i);
    }

    const ui::Tokens& t = ui::CurrentTokens();

    if (ImPlot::BeginPlot("##ScatterPlot", ImVec2(-1, -1))) {
        ImPlot::SetupAxes(column_names_[x_axis_column_].c_str(), column_names_[y_axis_column_].c_str());

        // Plot points colored by hard assignment
        for (int k = 0; k < result_.n_components; ++k) {
            std::vector<double> x_data, y_data;
            for (size_t i = 0; i < input_data_.size(); ++i) {
                if (result_.labels[i] == k) {
                    x_data.push_back(input_data_[i][x_idx]);
                    y_data.push_back(input_data_[i][y_idx]);
                }
            }
            if (!x_data.empty()) {
                ImVec4 color = ui::SeriesColour(t, static_cast<size_t>(k));
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4, color, IMPLOT_AUTO, color);
                ImPlot::PlotScatter(("Component " + std::to_string(k)).c_str(), x_data.data(), y_data.data(), static_cast<int>(x_data.size()));
            }
        }

        // Plot component means
        if (!result_.means.empty()) {
            std::vector<double> mx, my;
            for (const auto& mean : result_.means) {
                if (x_idx < static_cast<int>(mean.size()) && y_idx < static_cast<int>(mean.size())) {
                    mx.push_back(mean[x_idx]);
                    my.push_back(mean[y_idx]);
                }
            }
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 10, t.text_bright, 2, t.text_bright);
            ImPlot::PlotScatter("Means", mx.data(), my.data(), static_cast<int>(mx.size()));
        }

        ImPlot::EndPlot();
    }
}

void GMMPanel::RenderComponentInfo() {
    if (result_.means.empty()) return;

    std::vector<std::string> feature_names;
    for (size_t i = 0; i < selected_columns_.size(); ++i) {
        if (selected_columns_[i]) feature_names.push_back(column_names_[i]);
    }

    const ui::Tokens& t = ui::CurrentTokens();

    ImGui::Text("Component Means:");
    ImGui::Spacing();

    if (ImGui::BeginTable("MeansTable", static_cast<int>(feature_names.size()) + 2,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
        ImGui::TableSetupColumn("Component");
        ImGui::TableSetupColumn("Weight");
        for (const auto& name : feature_names) ImGui::TableSetupColumn(name.c_str());
        ImGui::TableHeadersRow();

        for (int k = 0; k < result_.n_components; ++k) {
            ImGui::TableNextRow();

            ImGui::TableNextColumn();
            ImVec4 color = ui::SeriesColour(t, static_cast<size_t>(k));
            ImGui::TextColored(color, "%d", k);

            ImGui::TableNextColumn();
            ImGui::Text("%.3f", result_.weights[k]);

            for (size_t f = 0; f < result_.means[k].size() && f < feature_names.size(); ++f) {
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", result_.means[k][f]);
            }
        }

        ImGui::EndTable();
    }
}

void GMMPanel::UseTable(std::shared_ptr<DataTable> table) {
    current_table_ = std::move(table);
    column_names_ = current_table_->GetHeaders();
    selected_columns_.assign(column_names_.size(), false);

    int selected_count = 0;
    for (size_t i = 0; i < column_names_.size() && selected_count < 4; ++i) {
        if (current_table_->GetRowCount() > 0) {
            auto val = DataAnalyzer::ToDouble(current_table_->GetCell(0, static_cast<int>(i)));
            if (val.has_value()) {
                selected_columns_[i] = true;
                selected_count++;
            }
        }
    }

    x_axis_column_ = 0;
    int max_col = static_cast<int>(column_names_.size()) - 1;
    y_axis_column_ = (1 < max_col) ? 1 : max_col;

    result_ = GMMResult();
    input_data_.clear();
    model_k_values_.clear();
    model_bic_values_.clear();
    model_aic_values_.clear();
    status_message_ = "Data loaded.";
}

// Rows of the selected columns that parse as numbers; false (with a
// status message) when there is nothing to cluster.
bool GMMPanel::CollectInputData() {
    std::vector<int> col_indices;
    for (size_t i = 0; i < selected_columns_.size(); ++i) {
        if (selected_columns_[i]) col_indices.push_back(static_cast<int>(i));
    }

    if (col_indices.size() < 2) {
        status_message_ = "Select at least 2 columns.";
        return false;
    }

    input_data_.clear();
    int n_rows = static_cast<int>(current_table_->GetRowCount());
    for (int row = 0; row < n_rows; ++row) {
        std::vector<double> row_data;
        bool valid = true;
        for (int col : col_indices) {
            auto val = DataAnalyzer::ToDouble(current_table_->GetCell(row, col));
            if (val.has_value()) row_data.push_back(val.value());
            else { valid = false; break; }
        }
        if (valid) input_data_.push_back(row_data);
    }

    if (input_data_.empty()) {
        status_message_ = "No valid data.";
        return false;
    }
    return true;
}

void GMMPanel::RunClustering() {
    if (!current_table_ || is_computing_ || !CollectInputData()) return;

    is_computing_ = true;
    status_message_ = "Computing...";

    if (compute_thread_.joinable()) compute_thread_.join();

    const char* cov_types[] = {"full", "tied", "diag", "spherical"};
    compute_thread_ = std::thread([this, data = input_data_, k = n_components_, cov = std::string(cov_types[covariance_type_]),
                                   max_iter = max_iter_, n_init = n_init_]() {
        Device::ActivateProcessDeviceOnThisThread();
        GMMResult result = Clustering::GMM(
            data,
            k,
            cov,
            max_iter,
            1e-3,
            n_init,
            0,
            [this](int iter, double) { progress_iteration_ = iter; }
        );
        AsyncTaskManager::Instance().PostToMainThread(alive_, [this, result = std::move(result)]() mutable {
            result_ = std::move(result);
            if (result_.success) {
                status_message_ = std::to_string(result_.n_components) + " components, BIC=" +
                                 std::to_string(static_cast<int>(result_.bic));
            } else {
                status_message_ = "Failed: " + result_.error_message;
            }
            is_computing_ = false;
        });
    });
}

void GMMPanel::RunModelSelection() {
    if (!current_table_ || is_computing_ || !CollectInputData()) return;

    is_computing_ = true;
    status_message_ = "Running model selection...";

    if (compute_thread_.joinable()) compute_thread_.join();

    const char* cov_types[] = {"full", "tied", "diag", "spherical"};
    compute_thread_ = std::thread([this, data = input_data_, cov = std::string(cov_types[covariance_type_]),
                                   k_min = model_select_min_, k_max = model_select_max_]() {
        Device::ActivateProcessDeviceOnThisThread();
        std::vector<int> k_values;
        std::vector<double> bic_values, aic_values;
        double best_bic = std::numeric_limits<double>::max();
        int suggested = k_min;

        for (int k = k_min; k <= k_max; ++k) {
            auto gmm_result = Clustering::GMM(data, k, cov, 50, 1e-3, 1, 0);

            if (gmm_result.success) {
                k_values.push_back(k);
                bic_values.push_back(gmm_result.bic);
                aic_values.push_back(gmm_result.aic);

                if (gmm_result.bic < best_bic) {
                    best_bic = gmm_result.bic;
                    suggested = k;
                }
            }
        }
        AsyncTaskManager::Instance().PostToMainThread(alive_, [this, k_values = std::move(k_values), bic_values = std::move(bic_values),
                                                               aic_values = std::move(aic_values), suggested]() mutable {
            model_k_values_ = std::move(k_values);
            model_bic_values_ = std::move(bic_values);
            model_aic_values_ = std::move(aic_values);
            suggested_components_ = suggested;
            status_message_ = "Model selection complete. Suggested: " + std::to_string(suggested_components_);
            is_computing_ = false;
        });
    });
}

} // namespace cyxwiz
