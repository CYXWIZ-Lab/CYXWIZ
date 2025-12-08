#include "hierarchical_panel.h"
#include "../icons.h"
#include "../../core/data_registry.h"
#include "../../core/data_analyzer.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <set>

namespace cyxwiz {

const ImU32 HierarchicalPanel::CLUSTER_COLORS[MAX_CLUSTERS] = {
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

HierarchicalPanel::HierarchicalPanel() : Panel("Hierarchical Clustering", true) {}

HierarchicalPanel::~HierarchicalPanel() {
    if (compute_thread_.joinable()) compute_thread_.join();
}

void HierarchicalPanel::Render() {
    if (!is_open_) return;

    ImGui::SetNextWindowSize(ImVec2(900, 700), ImGuiCond_FirstUseEver);

    if (ImGui::Begin((std::string(ICON_FA_SITEMAP) + " Hierarchical Clustering###HierarchicalPanel").c_str(), &is_open_)) {
        if (data_registry_) available_tables_ = data_registry_->GetTableNames();

        ImGui::BeginChild("ConfigPanel", ImVec2(300, 0), true);
        RenderDataSelector();
        ImGui::Separator();
        RenderParameters();
        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);
        RenderResults();
        ImGui::EndChild();
    }
    ImGui::End();
}

void HierarchicalPanel::RenderDataSelector() {
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

void HierarchicalPanel::RenderParameters() {
    ImGui::Text(ICON_FA_SLIDERS " Parameters");
    ImGui::Spacing();

    ImGui::SliderInt("Number of Clusters", &n_clusters_, 2, 20);

    const char* linkage_methods[] = {"Ward", "Complete", "Average", "Single"};
    ImGui::Combo("Linkage", &linkage_method_, linkage_methods, 4);

    const char* metrics[] = {"Euclidean", "Manhattan", "Cosine"};
    ImGui::Combo("Distance Metric", &metric_idx_, metrics, 3);

    ImGui::Spacing();

    bool can_run = !is_computing_ && !input_data_.empty();
    if (!can_run) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_PLAY " Run Clustering", ImVec2(-1, 30))) {
        RunClustering();
    }

    if (!can_run) ImGui::EndDisabled();

    // Cut height slider (after clustering)
    if (result_.success && !result_.linkage_matrix.empty()) {
        ImGui::Separator();
        ImGui::Text("Cut Dendrogram:");

        if (ImGui::SliderFloat("Cut Height", (float*)&cut_height_, 0.0f, (float)max_height_, "%.2f")) {
            // Recompute labels based on cut height
            result_.labels = Clustering::CutDendrogram(result_.linkage_matrix, cut_height_, static_cast<int>(input_data_.size()));
            std::set<int> unique_labels(result_.labels.begin(), result_.labels.end());
            result_.n_clusters = static_cast<int>(unique_labels.size());
        }
    }

    if (is_computing_) {
        ImGui::Text("Computing...");
    }

    if (!status_message_.empty()) {
        ImGui::TextWrapped("%s", status_message_.c_str());
    }
}

void HierarchicalPanel::RenderResults() {
    if (!result_.success) {
        ImGui::Text("Run hierarchical clustering to see results.");
        return;
    }

    if (ImGui::BeginTabBar("ResultTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_SITEMAP " Dendrogram")) {
            RenderDendrogram();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(ICON_FA_CHART_SCATTER " Scatter Plot")) {
            RenderScatterPlot();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(ICON_FA_CHART_BAR " Statistics")) {
            ImGui::Text("Hierarchical Clustering Results");
            ImGui::Separator();
            ImGui::Text("Clusters: %d", result_.n_clusters);

            std::vector<int> cluster_sizes(result_.n_clusters, 0);
            for (int label : result_.labels) {
                if (label >= 0 && label < result_.n_clusters) cluster_sizes[label]++;
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

void HierarchicalPanel::RenderDendrogram() {
    if (result_.linkage_matrix.empty()) {
        ImGui::Text("No dendrogram data available.");
        return;
    }

    // Simplified dendrogram visualization using ImPlot
    // Shows merge heights as a step plot
    if (ImPlot::BeginPlot("##Dendrogram", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("Merge Order", "Distance");

        std::vector<double> merge_order, heights;
        for (size_t i = 0; i < result_.linkage_matrix.size(); ++i) {
            merge_order.push_back(static_cast<double>(i));
            heights.push_back(result_.linkage_matrix[i][2]);
        }

        ImPlot::PlotStairs("Merge Height", merge_order.data(), heights.data(), static_cast<int>(merge_order.size()));

        // Show cut line
        double cut_line = cut_height_;
        ImPlot::PlotInfLines("Cut", &cut_line, 1, ImPlotInfLinesFlags_Horizontal);

        ImPlot::EndPlot();
    }
}

void HierarchicalPanel::RenderScatterPlot() {
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

void HierarchicalPanel::LoadSelectedData() {
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

    result_ = HierarchicalResult();
    input_data_.clear();
    cut_height_ = 0.0;
    max_height_ = 1.0;
    status_message_ = "Data loaded.";
}

void HierarchicalPanel::RunClustering() {
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
        const char* linkages[] = {"ward", "complete", "average", "single"};
        const char* metrics[] = {"euclidean", "manhattan", "cosine"};

        result_ = Clustering::Hierarchical(input_data_, n_clusters_, linkages[linkage_method_], metrics[metric_idx_]);
        is_computing_ = false;

        if (result_.success) {
            // Find max height for slider
            max_height_ = 0.0;
            for (const auto& link : result_.linkage_matrix) {
                if (link[2] > max_height_) max_height_ = link[2];
            }
            cut_height_ = max_height_ * 0.5;
            status_message_ = std::to_string(result_.n_clusters) + " clusters found.";
        } else {
            status_message_ = "Failed: " + result_.error_message;
        }
    });
}

} // namespace cyxwiz
