#include "dbscan_panel.h"
#include "../icons.h"
#include "../../core/data_registry.h"
#include "../../core/data_analyzer.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace cyxwiz {

const ImU32 DBSCANPanel::CLUSTER_COLORS[MAX_CLUSTERS] = {
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

DBSCANPanel::DBSCANPanel() : Panel("DBSCAN Clustering", true) {}

DBSCANPanel::~DBSCANPanel() {
    if (compute_thread_.joinable()) compute_thread_.join();
}

void DBSCANPanel::Render() {
    if (!is_open_) return;

    ImGui::SetNextWindowSize(ImVec2(900, 700), ImGuiCond_FirstUseEver);

    if (ImGui::Begin((std::string(ICON_FA_CIRCLE_NODES) + " DBSCAN Clustering###DBSCANPanel").c_str(), &is_open_)) {
        if (data_registry_) available_tables_ = data_registry_->GetTableNames();

        ImGui::BeginChild("ConfigPanel", ImVec2(300, 0), true);
        RenderDataSelector();
        ImGui::Separator();
        RenderParameters();
        ImGui::Separator();
        RenderKDistanceAnalysis();
        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);
        RenderResults();
        ImGui::EndChild();
    }
    ImGui::End();
}

void DBSCANPanel::RenderDataSelector() {
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
        ImGui::BeginChild("ColumnSelect", ImVec2(0, 120), true);
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

void DBSCANPanel::RenderParameters() {
    ImGui::Text(ICON_FA_SLIDERS " Parameters");
    ImGui::Spacing();

    ImGui::SliderFloat("Epsilon (eps)", (float*)&eps_, 0.01f, 10.0f, "%.3f");
    ImGui::SliderInt("Min Samples", &min_samples_, 1, 50);

    const char* metrics[] = {"Euclidean", "Manhattan", "Cosine"};
    ImGui::Combo("Distance Metric", &metric_idx_, metrics, 3);

    ImGui::Spacing();

    bool can_run = !is_computing_ && !input_data_.empty();
    if (!can_run) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_PLAY " Run DBSCAN", ImVec2(-1, 30))) {
        RunClustering();
    }

    if (!can_run) ImGui::EndDisabled();

    if (is_computing_) {
        ImGui::Text("Computing...");
    }

    if (!status_message_.empty()) {
        ImGui::TextWrapped("%s", status_message_.c_str());
    }
}

void DBSCANPanel::RenderKDistanceAnalysis() {
    ImGui::Text(ICON_FA_CHART_LINE " K-Distance Graph");
    ImGui::Spacing();

    ImGui::SliderInt("K", &k_distance_k_, 1, 20);

    bool can_run = !is_computing_ && !input_data_.empty();
    if (!can_run) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_MAGNIFYING_GLASS_CHART " Compute K-Distances", ImVec2(-1, 25))) {
        ComputeKDistances();
    }

    if (!can_run) ImGui::EndDisabled();

    if (!k_distances_.empty()) {
        ImGui::Text("Suggested eps: %.3f", suggested_eps_);

        if (ImGui::Button("Use Suggested eps")) {
            eps_ = suggested_eps_;
        }

        if (ImPlot::BeginPlot("##KDistPlot", ImVec2(-1, 150))) {
            ImPlot::SetupAxes("Point Index (sorted)", "K-Distance");
            ImPlot::PlotLine("K-Distance", k_distances_.data(), static_cast<int>(k_distances_.size()));

            double suggested_line = suggested_eps_;
            ImPlot::PlotInfLines("Suggested", &suggested_line, 1, ImPlotInfLinesFlags_Horizontal);
            ImPlot::EndPlot();
        }
    }
}

void DBSCANPanel::RenderResults() {
    if (!result_.success) {
        ImGui::Text("Run DBSCAN clustering to see results.");
        return;
    }

    if (ImGui::BeginTabBar("ResultTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_CHART_SCATTER " Scatter Plot")) {
            RenderScatterPlot();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(ICON_FA_CHART_BAR " Statistics")) {
            ImGui::Text("DBSCAN Results");
            ImGui::Separator();

            ImGui::Text("Clusters Found: %d", result_.n_clusters);
            ImGui::Text("Noise Points: %d", result_.n_noise_points);
            ImGui::Text("Core Samples: %d", static_cast<int>(std::count(result_.core_samples.begin(), result_.core_samples.end(), true)));

            // Cluster sizes
            std::vector<int> cluster_sizes(result_.n_clusters, 0);
            for (int label : result_.labels) {
                if (label >= 0 && label < result_.n_clusters) {
                    cluster_sizes[label]++;
                }
            }

            if (result_.n_clusters > 0 && ImPlot::BeginPlot("##ClusterSizes", ImVec2(-1, 200))) {
                ImPlot::SetupAxes("Cluster", "Count");
                std::vector<double> x_pos(result_.n_clusters), sizes(result_.n_clusters);
                for (int i = 0; i < result_.n_clusters; ++i) {
                    x_pos[i] = i;
                    sizes[i] = cluster_sizes[i];
                }
                ImPlot::PlotBars("Size", x_pos.data(), sizes.data(), result_.n_clusters, 0.6);
                ImPlot::EndPlot();
            }

            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void DBSCANPanel::RenderScatterPlot() {
    if (input_data_.empty() || result_.labels.empty()) return;

    std::vector<int> selected_col_indices;
    for (size_t i = 0; i < selected_columns_.size(); ++i) {
        if (selected_columns_[i]) selected_col_indices.push_back(static_cast<int>(i));
    }

    if (selected_col_indices.size() < 2) {
        ImGui::Text("Select at least 2 columns.");
        return;
    }

    int x_idx = 0, y_idx = (std::min)(1, static_cast<int>(input_data_[0].size()) - 1);
    for (size_t i = 0; i < selected_col_indices.size(); ++i) {
        if (selected_col_indices[i] == x_axis_column_) x_idx = static_cast<int>(i);
        if (selected_col_indices[i] == y_axis_column_) y_idx = static_cast<int>(i);
    }

    if (ImPlot::BeginPlot("##ScatterPlot", ImVec2(-1, -1))) {
        ImPlot::SetupAxes(column_names_[x_axis_column_].c_str(), column_names_[y_axis_column_].c_str());

        // Plot noise points (gray)
        std::vector<double> noise_x, noise_y;
        for (size_t i = 0; i < input_data_.size(); ++i) {
            if (result_.labels[i] == -1) {
                noise_x.push_back(input_data_[i][x_idx]);
                noise_y.push_back(input_data_[i][y_idx]);
            }
        }
        if (!noise_x.empty()) {
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 3, ImVec4(0.5f, 0.5f, 0.5f, 0.5f));
            ImPlot::PlotScatter("Noise", noise_x.data(), noise_y.data(), static_cast<int>(noise_x.size()));
        }

        // Plot clusters
        for (int k = 0; k < result_.n_clusters; ++k) {
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
                ImPlot::PlotScatter(("Cluster " + std::to_string(k)).c_str(), x_data.data(), y_data.data(), static_cast<int>(x_data.size()));
            }
        }

        ImPlot::EndPlot();
    }
}

void DBSCANPanel::LoadSelectedData() {
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
    y_axis_column_ = (std::min)(1, static_cast<int>(column_names_.size()) - 1);

    result_ = DBSCANResult();
    input_data_.clear();
    k_distances_.clear();
    status_message_ = "Data loaded.";
}

void DBSCANPanel::RunClustering() {
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
        const char* metrics[] = {"euclidean", "manhattan", "cosine"};
        result_ = Clustering::DBSCAN(input_data_, eps_, min_samples_, metrics[metric_idx_]);
        is_computing_ = false;

        if (result_.success) {
            status_message_ = std::to_string(result_.n_clusters) + " clusters, " + std::to_string(result_.n_noise_points) + " noise points.";
        } else {
            status_message_ = "Failed: " + result_.error_message;
        }
    });
}

void DBSCANPanel::ComputeKDistances() {
    if (input_data_.empty() || is_computing_) return;

    is_computing_ = true;
    if (compute_thread_.joinable()) compute_thread_.join();

    compute_thread_ = std::thread([this]() {
        k_distances_ = Clustering::ComputeKDistances(input_data_, k_distance_k_);

        // Find elbow point
        if (k_distances_.size() > 10) {
            size_t elbow_idx = k_distances_.size() * 9 / 10;  // 90th percentile
            suggested_eps_ = k_distances_[elbow_idx];
        } else if (!k_distances_.empty()) {
            suggested_eps_ = k_distances_.back() * 0.9;
        }

        is_computing_ = false;
    });
}

} // namespace cyxwiz
