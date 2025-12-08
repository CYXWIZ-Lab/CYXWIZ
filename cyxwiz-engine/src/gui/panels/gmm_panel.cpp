#include "gmm_panel.h"
#include "../icons.h"
#include "../../core/data_registry.h"
#include "../../core/data_analyzer.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace cyxwiz {

const ImU32 GMMPanel::CLUSTER_COLORS[MAX_CLUSTERS] = {
    IM_COL32(31, 119, 180, 255),  IM_COL32(255, 127, 14, 255),
    IM_COL32(44, 160, 44, 255),   IM_COL32(214, 39, 40, 255),
    IM_COL32(148, 103, 189, 255), IM_COL32(140, 86, 75, 255),
    IM_COL32(227, 119, 194, 255), IM_COL32(127, 127, 127, 255),
    IM_COL32(188, 189, 34, 255),  IM_COL32(23, 190, 207, 255),
    IM_COL32(174, 199, 232, 255), IM_COL32(255, 187, 120, 255),
    IM_COL32(152, 223, 138, 255), IM_COL32(255, 152, 150, 255),
    IM_COL32(197, 176, 213, 255), IM_COL32(196, 156, 148, 255),
    IM_COL32(247, 182, 210, 255), IM_COL32(199, 199, 199, 255),
    IM_COL32(219, 219, 141, 255), IM_COL32(158, 218, 229, 255),
};

GMMPanel::GMMPanel() : Panel("GMM Clustering", true) {}

GMMPanel::~GMMPanel() {
    if (compute_thread_.joinable()) compute_thread_.join();
}

void GMMPanel::Render() {
    if (!is_open_) return;

    ImGui::SetNextWindowSize(ImVec2(950, 750), ImGuiCond_FirstUseEver);

    if (ImGui::Begin((std::string(ICON_FA_CHART_PIE) + " Gaussian Mixture Models###GMMPanel").c_str(), &is_open_)) {
        if (data_registry_) available_tables_ = data_registry_->GetTableNames();

        ImGui::BeginChild("ConfigPanel", ImVec2(320, 0), true);
        RenderDataSelector();
        ImGui::Separator();
        RenderParameters();
        ImGui::Separator();
        RenderModelSelection();
        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);
        RenderResults();
        ImGui::EndChild();
    }
    ImGui::End();
}

void GMMPanel::RenderDataSelector() {
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
        ImGui::Text("Feature Columns:");
        ImGui::BeginChild("ColumnSelect", ImVec2(0, 100), true);
        for (size_t i = 0; i < column_names_.size(); ++i) {
            bool selected = selected_columns_[i];
            if (ImGui::Checkbox(column_names_[i].c_str(), &selected)) {
                selected_columns_[i] = selected;
            }
        }
        ImGui::EndChild();

        std::vector<const char*> col_names_c;
        for (const auto& name : column_names_) col_names_c.push_back(name.c_str());
        ImGui::Combo("X Axis", &x_axis_column_, col_names_c.data(), static_cast<int>(col_names_c.size()));
        ImGui::Combo("Y Axis", &y_axis_column_, col_names_c.data(), static_cast<int>(col_names_c.size()));
    }
}

void GMMPanel::RenderParameters() {
    ImGui::Text(ICON_FA_SLIDERS " Parameters");
    ImGui::Spacing();

    ImGui::SliderInt("Components", &n_components_, 2, 20);
    ImGui::SliderInt("Max Iterations", &max_iter_, 10, 500);
    ImGui::SliderInt("Initializations", &n_init_, 1, 10);

    const char* cov_types[] = {"Full", "Tied", "Diagonal", "Spherical"};
    ImGui::Combo("Covariance Type", &covariance_type_, cov_types, 4);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Full: Each component has its own covariance matrix\n"
                         "Tied: All components share the same covariance\n"
                         "Diagonal: Off-diagonal elements are zero\n"
                         "Spherical: Single variance value per component");
    }

    ImGui::Spacing();

    bool can_run = !is_computing_ && !input_data_.empty();
    if (!can_run) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_PLAY " Run GMM", ImVec2(-1, 30))) {
        RunClustering();
    }

    if (!can_run) ImGui::EndDisabled();

    if (is_computing_) {
        ImGui::Text("Computing... Iteration %d", progress_iteration_.load());
    }

    if (!status_message_.empty()) {
        ImGui::TextWrapped("%s", status_message_.c_str());
    }
}

void GMMPanel::RenderModelSelection() {
    ImGui::Text(ICON_FA_CHART_LINE " Model Selection");
    ImGui::Spacing();

    ImGui::SliderInt("Min Components", &model_select_min_, 1, 10);
    ImGui::SliderInt("Max Components", &model_select_max_, model_select_min_ + 1, 20);

    bool can_run = !is_computing_ && !input_data_.empty();
    if (!can_run) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_MAGNIFYING_GLASS_CHART " Find Optimal Components", ImVec2(-1, 25))) {
        RunModelSelection();
    }

    if (!can_run) ImGui::EndDisabled();

    if (!model_k_values_.empty()) {
        ImGui::Text("Suggested: %d components", suggested_components_);

        if (ImGui::Button("Use Suggested")) {
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
        ImGui::Text("Run GMM clustering to see results.");
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
                ImVec4 color = ImGui::ColorConvertU32ToFloat4(CLUSTER_COLORS[k % MAX_CLUSTERS]);
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4, color);
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
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 10, ImVec4(0, 0, 0, 1), 2);
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
            ImVec4 color = ImGui::ColorConvertU32ToFloat4(CLUSTER_COLORS[k % MAX_CLUSTERS]);
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

void GMMPanel::LoadSelectedData() {
    if (selected_table_idx_ < 0 || !data_registry_) return;

    current_table_ = data_registry_->GetTable(available_tables_[selected_table_idx_]);
    if (!current_table_) return;

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

void GMMPanel::RunClustering() {
    if (!current_table_ || is_computing_) return;

    std::vector<int> col_indices;
    for (size_t i = 0; i < selected_columns_.size(); ++i) {
        if (selected_columns_[i]) col_indices.push_back(static_cast<int>(i));
    }

    if (col_indices.size() < 2) {
        status_message_ = "Select at least 2 columns.";
        return;
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
        return;
    }

    is_computing_ = true;
    status_message_ = "Computing...";

    if (compute_thread_.joinable()) compute_thread_.join();

    compute_thread_ = std::thread([this]() {
        const char* cov_types[] = {"full", "tied", "diag", "spherical"};

        result_ = Clustering::GMM(
            input_data_,
            n_components_,
            cov_types[covariance_type_],
            max_iter_,
            1e-3,
            n_init_,
            0,
            [this](int iter, double) { progress_iteration_ = iter; }
        );

        is_computing_ = false;

        if (result_.success) {
            status_message_ = std::to_string(result_.n_components) + " components, BIC=" +
                             std::to_string(static_cast<int>(result_.bic));
        } else {
            status_message_ = "Failed: " + result_.error_message;
        }
    });
}

void GMMPanel::RunModelSelection() {
    if (input_data_.empty() || is_computing_) return;

    is_computing_ = true;
    status_message_ = "Running model selection...";

    if (compute_thread_.joinable()) compute_thread_.join();

    compute_thread_ = std::thread([this]() {
        model_k_values_.clear();
        model_bic_values_.clear();
        model_aic_values_.clear();

        const char* cov_types[] = {"full", "tied", "diag", "spherical"};
        double best_bic = std::numeric_limits<double>::max();
        suggested_components_ = model_select_min_;

        for (int k = model_select_min_; k <= model_select_max_; ++k) {
            auto gmm_result = Clustering::GMM(input_data_, k, cov_types[covariance_type_], 50, 1e-3, 1, 0);

            if (gmm_result.success) {
                model_k_values_.push_back(k);
                model_bic_values_.push_back(gmm_result.bic);
                model_aic_values_.push_back(gmm_result.aic);

                if (gmm_result.bic < best_bic) {
                    best_bic = gmm_result.bic;
                    suggested_components_ = k;
                }
            }
        }

        is_computing_ = false;
        status_message_ = "Model selection complete. Suggested: " + std::to_string(suggested_components_);
    });
}

} // namespace cyxwiz
