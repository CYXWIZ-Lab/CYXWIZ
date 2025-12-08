#include "kmeans_panel.h"
#include "../icons.h"
#include "../../core/data_registry.h"
#include "../../core/data_analyzer.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace cyxwiz {

// Cluster colors (distinct colors for up to 20 clusters)
const ImU32 KMeansPanel::CLUSTER_COLORS[MAX_CLUSTERS] = {
    IM_COL32(31, 119, 180, 255),   // Blue
    IM_COL32(255, 127, 14, 255),   // Orange
    IM_COL32(44, 160, 44, 255),    // Green
    IM_COL32(214, 39, 40, 255),    // Red
    IM_COL32(148, 103, 189, 255),  // Purple
    IM_COL32(140, 86, 75, 255),    // Brown
    IM_COL32(227, 119, 194, 255),  // Pink
    IM_COL32(127, 127, 127, 255),  // Gray
    IM_COL32(188, 189, 34, 255),   // Olive
    IM_COL32(23, 190, 207, 255),   // Cyan
    IM_COL32(174, 199, 232, 255),  // Light Blue
    IM_COL32(255, 187, 120, 255),  // Light Orange
    IM_COL32(152, 223, 138, 255),  // Light Green
    IM_COL32(255, 152, 150, 255),  // Light Red
    IM_COL32(197, 176, 213, 255),  // Light Purple
    IM_COL32(196, 156, 148, 255),  // Light Brown
    IM_COL32(247, 182, 210, 255),  // Light Pink
    IM_COL32(199, 199, 199, 255),  // Light Gray
    IM_COL32(219, 219, 141, 255),  // Light Olive
    IM_COL32(158, 218, 229, 255),  // Light Cyan
};

KMeansPanel::KMeansPanel() : Panel("K-Means Clustering", true) {
}

KMeansPanel::~KMeansPanel() {
    cancel_requested_ = true;
    if (compute_thread_.joinable()) {
        compute_thread_.join();
    }
}

void KMeansPanel::Render() {
    if (!is_open_) return;

    ImGui::SetNextWindowSize(ImVec2(900, 700), ImGuiCond_FirstUseEver);

    if (ImGui::Begin((std::string(ICON_FA_BULLSEYE) + " K-Means Clustering###KMeansPanel").c_str(), &is_open_)) {
        // Update available tables
        if (data_registry_) {
            available_tables_ = data_registry_->GetTableNames();
        }

        // Left panel - Configuration
        ImGui::BeginChild("ConfigPanel", ImVec2(300, 0), true);

        RenderDataSelector();
        ImGui::Separator();
        RenderParameters();
        ImGui::Separator();
        RenderElbowAnalysis();

        ImGui::EndChild();

        ImGui::SameLine();

        // Right panel - Results
        ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);

        RenderResults();

        ImGui::EndChild();
    }
    ImGui::End();
}

void KMeansPanel::RenderDataSelector() {
    ImGui::Text(ICON_FA_TABLE " Data Selection");
    ImGui::Spacing();

    // Table selector
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

    // Column selection
    if (!column_names_.empty()) {
        ImGui::Text("Feature Columns:");
        ImGui::BeginChild("ColumnSelect", ImVec2(0, 150), true);

        for (size_t i = 0; i < column_names_.size(); ++i) {
            bool selected = selected_columns_[i];
            if (ImGui::Checkbox(column_names_[i].c_str(), &selected)) {
                selected_columns_[i] = selected;
            }
        }

        ImGui::EndChild();

        // X/Y axis for scatter plot
        ImGui::Text("Scatter Plot Axes:");

        std::vector<const char*> col_names_c;
        for (const auto& name : column_names_) {
            col_names_c.push_back(name.c_str());
        }

        ImGui::Combo("X Axis", &x_axis_column_, col_names_c.data(), static_cast<int>(col_names_c.size()));
        ImGui::Combo("Y Axis", &y_axis_column_, col_names_c.data(), static_cast<int>(col_names_c.size()));

        // Optional label column
        std::vector<const char*> label_options = {"None"};
        for (const auto& name : column_names_) {
            label_options.push_back(name.c_str());
        }
        int label_idx = label_column_idx_ + 1;  // Offset for "None"
        if (ImGui::Combo("Label Column", &label_idx, label_options.data(), static_cast<int>(label_options.size()))) {
            label_column_idx_ = label_idx - 1;
        }
    }
}

void KMeansPanel::RenderParameters() {
    ImGui::Text(ICON_FA_SLIDERS " Parameters");
    ImGui::Spacing();

    ImGui::SliderInt("Number of Clusters (k)", &n_clusters_, 2, 20);
    ImGui::SliderInt("Max Iterations", &max_iter_, 10, 1000);
    ImGui::SliderInt("Initializations", &n_init_, 1, 20);

    const char* init_methods[] = {"Random", "K-Means++"};
    ImGui::Combo("Initialization", &init_method_, init_methods, 2);

    ImGui::Spacing();

    // Run button
    bool can_run = !is_computing_ && !input_data_.empty();
    if (!can_run) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_PLAY " Run K-Means", ImVec2(-1, 30))) {
        RunClustering();
    }

    if (!can_run) ImGui::EndDisabled();

    // Cancel button
    if (is_computing_) {
        if (ImGui::Button(ICON_FA_STOP " Cancel", ImVec2(-1, 30))) {
            cancel_requested_ = true;
        }

        ImGui::Text("Computing... Iteration %d", progress_iteration_.load());
        ImGui::Text("Inertia: %.4f", progress_inertia_.load());
    }

    if (!status_message_.empty()) {
        ImGui::TextWrapped("%s", status_message_.c_str());
    }
}

void KMeansPanel::RenderElbowAnalysis() {
    ImGui::Text(ICON_FA_CHART_LINE " Elbow Analysis");
    ImGui::Spacing();

    ImGui::SliderInt("K Min", &elbow_k_min_, 1, 10);
    ImGui::SliderInt("K Max", &elbow_k_max_, elbow_k_min_ + 1, 20);

    bool can_run = !is_computing_ && !input_data_.empty();
    if (!can_run) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_MAGNIFYING_GLASS_CHART " Find Optimal K", ImVec2(-1, 25))) {
        RunElbowAnalysis();
    }

    if (!can_run) ImGui::EndDisabled();

    // Show elbow result
    if (!elbow_result_.k_values.empty()) {
        ImGui::Text("Suggested K: %d", elbow_result_.suggested_k);

        // Elbow plot
        if (ImPlot::BeginPlot("##ElbowPlot", ImVec2(-1, 150))) {
            ImPlot::SetupAxes("K", "Inertia");

            std::vector<double> k_vals(elbow_result_.k_values.begin(), elbow_result_.k_values.end());
            ImPlot::PlotLine("Inertia", k_vals.data(), elbow_result_.inertias.data(),
                            static_cast<int>(k_vals.size()));
            ImPlot::PlotScatter("Inertia", k_vals.data(), elbow_result_.inertias.data(),
                               static_cast<int>(k_vals.size()));

            // Mark suggested k
            double suggested_x = elbow_result_.suggested_k;
            ImPlot::PlotInfLines("Suggested K", &suggested_x, 1);

            ImPlot::EndPlot();
        }
    }
}

void KMeansPanel::RenderResults() {
    if (!result_.success) {
        ImGui::Text("Run K-Means clustering to see results.");
        return;
    }

    // Tabs for different views
    if (ImGui::BeginTabBar("ResultTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_CHART_SCATTER " Scatter Plot")) {
            RenderScatterPlot();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(ICON_FA_TABLE " Centroids")) {
            RenderCentroidsTable();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(ICON_FA_CHART_BAR " Statistics")) {
            ImGui::Text("Clustering Statistics");
            ImGui::Separator();

            ImGui::Text("Number of Clusters: %d", result_.n_clusters);
            ImGui::Text("Iterations: %d", result_.n_iterations);
            ImGui::Text("Converged: %s", result_.converged ? "Yes" : "No");
            ImGui::Text("Inertia: %.4f", result_.inertia);

            // Cluster sizes
            ImGui::Spacing();
            ImGui::Text("Cluster Sizes:");

            std::vector<int> cluster_sizes(result_.n_clusters, 0);
            for (int label : result_.labels) {
                if (label >= 0 && label < result_.n_clusters) {
                    cluster_sizes[label]++;
                }
            }

            if (ImPlot::BeginPlot("##ClusterSizes", ImVec2(-1, 200))) {
                ImPlot::SetupAxes("Cluster", "Count");

                std::vector<double> x_pos(result_.n_clusters);
                std::vector<double> sizes(result_.n_clusters);
                for (int i = 0; i < result_.n_clusters; ++i) {
                    x_pos[i] = i;
                    sizes[i] = cluster_sizes[i];
                }

                ImPlot::PlotBars("Size", x_pos.data(), sizes.data(), result_.n_clusters, 0.6);

                ImPlot::EndPlot();
            }

            // Inertia history plot
            if (!result_.inertia_history.empty()) {
                ImGui::Spacing();
                ImGui::Text("Convergence:");

                if (ImPlot::BeginPlot("##InertiaHistory", ImVec2(-1, 150))) {
                    ImPlot::SetupAxes("Iteration", "Inertia");
                    ImPlot::PlotLine("Inertia", result_.inertia_history.data(),
                                    static_cast<int>(result_.inertia_history.size()));
                    ImPlot::EndPlot();
                }
            }

            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }
}

void KMeansPanel::RenderScatterPlot() {
    if (input_data_.empty() || result_.labels.empty()) return;

    if (x_axis_column_ >= static_cast<int>(column_names_.size()) ||
        y_axis_column_ >= static_cast<int>(column_names_.size())) {
        ImGui::Text("Invalid axis columns selected.");
        return;
    }

    // Find indices of selected columns in the input data
    std::vector<int> selected_col_indices;
    for (size_t i = 0; i < selected_columns_.size(); ++i) {
        if (selected_columns_[i]) {
            selected_col_indices.push_back(static_cast<int>(i));
        }
    }

    if (selected_col_indices.size() < 2) {
        ImGui::Text("Select at least 2 columns.");
        return;
    }

    // Map x_axis_column_ and y_axis_column_ to input data indices
    int x_idx = -1, y_idx = -1;
    for (size_t i = 0; i < selected_col_indices.size(); ++i) {
        if (selected_col_indices[i] == x_axis_column_) x_idx = static_cast<int>(i);
        if (selected_col_indices[i] == y_axis_column_) y_idx = static_cast<int>(i);
    }

    if (x_idx < 0 || y_idx < 0) {
        // Use first two columns as fallback
        x_idx = 0;
        y_idx = (std::min)(1, static_cast<int>(input_data_[0].size()) - 1);
    }

    if (ImPlot::BeginPlot("##ScatterPlot", ImVec2(-1, -1))) {
        ImPlot::SetupAxes(column_names_[x_axis_column_].c_str(),
                         column_names_[y_axis_column_].c_str());

        // Plot each cluster with different color
        for (int k = 0; k < result_.n_clusters; ++k) {
            std::vector<double> x_data, y_data;

            for (size_t i = 0; i < input_data_.size(); ++i) {
                if (result_.labels[i] == k) {
                    if (x_idx < static_cast<int>(input_data_[i].size()) &&
                        y_idx < static_cast<int>(input_data_[i].size())) {
                        x_data.push_back(input_data_[i][x_idx]);
                        y_data.push_back(input_data_[i][y_idx]);
                    }
                }
            }

            if (!x_data.empty()) {
                ImVec4 color = ImGui::ColorConvertU32ToFloat4(CLUSTER_COLORS[k % MAX_CLUSTERS]);
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4, color);

                std::string label = "Cluster " + std::to_string(k);
                ImPlot::PlotScatter(label.c_str(), x_data.data(), y_data.data(),
                                   static_cast<int>(x_data.size()));
            }
        }

        // Plot centroids
        if (!result_.centroids.empty() && result_.centroids[0].size() >= 2) {
            std::vector<double> cx, cy;
            for (const auto& centroid : result_.centroids) {
                if (x_idx < static_cast<int>(centroid.size()) &&
                    y_idx < static_cast<int>(centroid.size())) {
                    cx.push_back(centroid[x_idx]);
                    cy.push_back(centroid[y_idx]);
                }
            }

            ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, 10, ImVec4(0, 0, 0, 1), 2);
            ImPlot::PlotScatter("Centroids", cx.data(), cy.data(), static_cast<int>(cx.size()));
        }

        ImPlot::EndPlot();
    }
}

void KMeansPanel::RenderCentroidsTable() {
    if (result_.centroids.empty()) return;

    // Get column names for selected features
    std::vector<std::string> feature_names;
    for (size_t i = 0; i < selected_columns_.size(); ++i) {
        if (selected_columns_[i]) {
            feature_names.push_back(column_names_[i]);
        }
    }

    ImGui::Text("Cluster Centroids:");
    ImGui::Spacing();

    if (ImGui::BeginTable("CentroidsTable", static_cast<int>(feature_names.size()) + 1,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
        // Header
        ImGui::TableSetupColumn("Cluster");
        for (const auto& name : feature_names) {
            ImGui::TableSetupColumn(name.c_str());
        }
        ImGui::TableHeadersRow();

        // Data rows
        for (int k = 0; k < result_.n_clusters; ++k) {
            ImGui::TableNextRow();

            ImGui::TableNextColumn();
            ImVec4 color = ImGui::ColorConvertU32ToFloat4(CLUSTER_COLORS[k % MAX_CLUSTERS]);
            ImGui::TextColored(color, "%d", k);

            for (size_t f = 0; f < result_.centroids[k].size() && f < feature_names.size(); ++f) {
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", result_.centroids[k][f]);
            }
        }

        ImGui::EndTable();
    }
}

void KMeansPanel::LoadSelectedData() {
    if (selected_table_idx_ < 0 || selected_table_idx_ >= static_cast<int>(available_tables_.size())) {
        return;
    }

    if (!data_registry_) return;

    current_table_ = data_registry_->GetTable(available_tables_[selected_table_idx_]);
    if (!current_table_) return;

    // Get column names
    column_names_ = current_table_->GetHeaders();
    selected_columns_.assign(column_names_.size(), false);

    // Select first few numeric columns by default
    int selected_count = 0;
    for (size_t i = 0; i < column_names_.size() && selected_count < 4; ++i) {
        // Check if column is numeric by trying first value
        if (current_table_->GetRowCount() > 0) {
            auto val = DataAnalyzer::ToDouble(current_table_->GetCell(0, static_cast<int>(i)));
            if (val.has_value()) {
                selected_columns_[i] = true;
                selected_count++;
            }
        }
    }

    // Reset axis selections
    x_axis_column_ = 0;
    y_axis_column_ = (std::min)(1, static_cast<int>(column_names_.size()) - 1);
    label_column_idx_ = -1;

    // Clear previous results
    result_ = KMeansResult();
    input_data_.clear();
    data_labels_.clear();

    status_message_ = "Data loaded. Select columns and run clustering.";
}

void KMeansPanel::RunClustering() {
    if (!current_table_ || is_computing_) return;

    // Collect selected column indices
    std::vector<int> col_indices;
    for (size_t i = 0; i < selected_columns_.size(); ++i) {
        if (selected_columns_[i]) {
            col_indices.push_back(static_cast<int>(i));
        }
    }

    if (col_indices.size() < 2) {
        status_message_ = "Please select at least 2 columns.";
        return;
    }

    // Extract data
    input_data_.clear();
    data_labels_.clear();

    int n_rows = static_cast<int>(current_table_->GetRowCount());
    for (int row = 0; row < n_rows; ++row) {
        std::vector<double> row_data;
        bool valid = true;

        for (int col : col_indices) {
            auto val = DataAnalyzer::ToDouble(current_table_->GetCell(row, col));
            if (val.has_value()) {
                row_data.push_back(val.value());
            } else {
                valid = false;
                break;
            }
        }

        if (valid) {
            input_data_.push_back(row_data);

            // Get label if specified
            if (label_column_idx_ >= 0 && label_column_idx_ < static_cast<int>(column_names_.size())) {
                data_labels_.push_back(current_table_->GetCellAsString(row, label_column_idx_));
            }
        }
    }

    if (input_data_.empty()) {
        status_message_ = "No valid numeric data found.";
        return;
    }

    // Start clustering in background thread
    is_computing_ = true;
    cancel_requested_ = false;
    status_message_ = "Computing...";

    if (compute_thread_.joinable()) {
        compute_thread_.join();
    }

    compute_thread_ = std::thread([this]() {
        std::string init_method = (init_method_ == 1) ? "kmeans++" : "random";

        result_ = Clustering::KMeans(
            input_data_,
            n_clusters_,
            max_iter_,
            init_method,
            n_init_,
            tolerance_,
            0,  // Random seed
            [this](int iter, double inertia) {
                progress_iteration_ = iter;
                progress_inertia_ = inertia;
            }
        );

        is_computing_ = false;

        if (result_.success) {
            status_message_ = "Clustering complete. " + std::to_string(result_.n_clusters) +
                             " clusters found in " + std::to_string(result_.n_iterations) + " iterations.";
        } else {
            status_message_ = "Clustering failed: " + result_.error_message;
        }
    });
}

void KMeansPanel::RunElbowAnalysis() {
    if (!current_table_ || is_computing_ || input_data_.empty()) return;

    is_computing_ = true;
    cancel_requested_ = false;
    status_message_ = "Running elbow analysis...";

    if (compute_thread_.joinable()) {
        compute_thread_.join();
    }

    compute_thread_ = std::thread([this]() {
        elbow_result_ = Clustering::ComputeElbowAnalysis(
            input_data_,
            elbow_k_min_,
            elbow_k_max_,
            [this](int current, int total) {
                progress_iteration_ = current;
            }
        );

        is_computing_ = false;

        if (elbow_result_.success) {
            n_clusters_ = elbow_result_.suggested_k;
            status_message_ = "Elbow analysis complete. Suggested K: " + std::to_string(elbow_result_.suggested_k);
        } else {
            status_message_ = "Elbow analysis failed: " + elbow_result_.error_message;
        }
    });
}

} // namespace cyxwiz
