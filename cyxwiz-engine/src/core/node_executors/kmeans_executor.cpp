#include "kmeans_executor.h"
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <algorithm>

namespace cyxwiz {

// Cluster colors for visualization (up to 20 clusters)
const ImU32 KMeansExecutor::CLUSTER_COLORS[MAX_CLUSTERS] = {
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
    IM_COL32(31, 119, 180, 180),   // Light Blue
    IM_COL32(255, 127, 14, 180),   // Light Orange
    IM_COL32(44, 160, 44, 180),    // Light Green
    IM_COL32(214, 39, 40, 180),    // Light Red
    IM_COL32(148, 103, 189, 180),  // Light Purple
    IM_COL32(140, 86, 75, 180),    // Light Brown
    IM_COL32(227, 119, 194, 180),  // Light Pink
    IM_COL32(127, 127, 127, 180),  // Light Gray
    IM_COL32(188, 189, 34, 180),   // Light Olive
    IM_COL32(23, 190, 207, 180)    // Light Cyan
};

KMeansExecutor::~KMeansExecutor() {
    cancel_requested_ = true;
    if (execution_thread_.joinable()) {
        execution_thread_.join();
    }
}

void KMeansExecutor::Execute() {
    if (input_data_.empty()) {
        state_ = ExecutorState::Error;
        error_message_ = "No input data provided";
        return;
    }

    if (input_data_.size() < static_cast<size_t>(config_.n_clusters)) {
        state_ = ExecutorState::Error;
        error_message_ = fmt::format("Need at least {} samples for {} clusters",
                                     config_.n_clusters, config_.n_clusters);
        return;
    }

    state_ = ExecutorState::Executing;
    cancel_requested_ = false;
    ReportProgress(0.0f, "Starting K-Means clustering...");

    spdlog::info("KMeansExecutor: Running with k={}, {} samples, {} features",
                 config_.n_clusters, input_data_.size(),
                 input_data_.empty() ? 0 : input_data_[0].size());

    // Call backend clustering
    std::string init_method = config_.init_method == 0 ? "random" : "kmeans++";

    result_ = Clustering::KMeans(
        input_data_,
        config_.n_clusters,
        config_.max_iter,
        init_method,
        config_.n_init,
        config_.tolerance,
        0,  // Random seed
        [this](int iter, double inertia) {
            if (cancel_requested_) return;
            float progress = static_cast<float>(iter) / config_.max_iter;
            ReportProgress(progress, fmt::format("Iteration {}, inertia: {:.2f}", iter, inertia));
        }
    );

    if (cancel_requested_) {
        state_ = ExecutorState::Cancelled;
        status_message_ = "Cancelled";
        return;
    }

    if (result_.success) {
        state_ = ExecutorState::Completed;
        ReportProgress(1.0f, fmt::format("Completed: {} clusters, inertia: {:.2f}",
                                         result_.n_clusters, result_.inertia));
        spdlog::info("KMeansExecutor: Completed in {} iterations, inertia: {:.2f}",
                     result_.n_iterations, result_.inertia);
    } else {
        state_ = ExecutorState::Error;
        error_message_ = result_.error_message;
        spdlog::error("KMeansExecutor: Failed - {}", result_.error_message);
    }
}

void KMeansExecutor::ExecuteAsync() {
    if (is_async_running_) return;

    if (execution_thread_.joinable()) {
        execution_thread_.join();
    }

    is_async_running_ = true;
    execution_thread_ = std::thread([this]() {
        Execute();
        is_async_running_ = false;
    });
}

void KMeansExecutor::RunElbowAnalysis(int k_min, int k_max) {
    if (input_data_.empty()) {
        elbow_result_.success = false;
        elbow_result_.error_message = "No input data";
        return;
    }

    spdlog::info("KMeansExecutor: Running elbow analysis k=[{}, {}]", k_min, k_max);

    elbow_result_ = Clustering::ComputeElbowAnalysis(
        input_data_,
        k_min,
        k_max,
        [this](int current_k, int total_k) {
            ReportProgress(static_cast<float>(current_k) / total_k,
                          fmt::format("Elbow analysis: k={}/{}", current_k, total_k));
        }
    );

    if (elbow_result_.success) {
        spdlog::info("KMeansExecutor: Elbow analysis complete, suggested k={}",
                     elbow_result_.suggested_k);
    }
}

// ==================== UI Rendering ====================

void KMeansExecutor::RenderConfigUI() {
    RenderParametersSection();
    ImGui::Separator();
    RenderElbowSection();
}

void KMeansExecutor::RenderResultsUI() {
    if (state_ == ExecutorState::Executing) {
        ImGui::ProgressBar(progress_, ImVec2(-1, 0), status_message_.c_str());
        if (ImGui::Button("Cancel")) {
            Cancel();
        }
        return;
    }

    if (state_ == ExecutorState::Error) {
        ImGui::TextColored(ImVec4(1, 0.3f, 0.3f, 1), "Error: %s", error_message_.c_str());
        return;
    }

    if (state_ != ExecutorState::Completed || !result_.success) {
        ImGui::TextDisabled("No results yet. Configure parameters and click 'Run'.");
        return;
    }

    // Results available
    RenderMetricsSection();
    ImGui::Separator();
    RenderScatterPlot();
    ImGui::Separator();
    RenderCentroidsTable();
}

void KMeansExecutor::RenderParametersSection() {
    ImGui::Text("Parameters");
    ImGui::Spacing();

    // Number of clusters
    ImGui::SliderInt("Clusters (k)", &config_.n_clusters, 2, 20);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Number of clusters to form");
    }

    // Initialization method
    const char* init_methods[] = { "Random", "K-Means++" };
    ImGui::Combo("Initialization", &config_.init_method, init_methods, 2);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("K-Means++ is recommended (smarter initial centroid selection)");
    }

    // Advanced options (collapsible)
    if (ImGui::TreeNode("Advanced Options")) {
        ImGui::SliderInt("Max Iterations", &config_.max_iter, 10, 1000);
        ImGui::SliderInt("N Init", &config_.n_init, 1, 20);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Number of times to run with different centroid seeds");
        }

        float tol = static_cast<float>(config_.tolerance);
        if (ImGui::SliderFloat("Tolerance", &tol, 1e-6f, 1e-2f, "%.2e", ImGuiSliderFlags_Logarithmic)) {
            config_.tolerance = tol;
        }

        ImGui::TreePop();
    }

    // Column selection for visualization
    if (!column_names_.empty()) {
        ImGui::Separator();
        ImGui::Text("Visualization Axes");

        std::vector<const char*> col_names;
        for (const auto& name : column_names_) {
            col_names.push_back(name.c_str());
        }

        ImGui::Combo("X Axis", &config_.x_column, col_names.data(), static_cast<int>(col_names.size()));
        ImGui::Combo("Y Axis", &config_.y_column, col_names.data(), static_cast<int>(col_names.size()));
    }

    ImGui::Spacing();

    // Run button
    bool can_run = !input_data_.empty() && state_ != ExecutorState::Executing;
    if (!can_run) ImGui::BeginDisabled();

    if (ImGui::Button("Run Clustering", ImVec2(-1, 30))) {
        ExecuteAsync();
    }

    if (!can_run) ImGui::EndDisabled();

    if (input_data_.empty()) {
        ImGui::TextColored(ImVec4(1, 0.6f, 0, 1), "Connect input data to run");
    }
}

void KMeansExecutor::RenderElbowSection() {
    ImGui::Checkbox("Show Elbow Analysis", &show_elbow_plot_);

    if (show_elbow_plot_) {
        static int elbow_k_min = 2;
        static int elbow_k_max = 10;

        ImGui::SliderInt("K Min", &elbow_k_min, 2, 5);
        ImGui::SliderInt("K Max", &elbow_k_max, 5, 20);

        if (ImGui::Button("Run Elbow Analysis")) {
            RunElbowAnalysis(elbow_k_min, elbow_k_max);
        }

        // Plot elbow curve
        if (elbow_result_.success && !elbow_result_.k_values.empty()) {
            if (ImPlot::BeginPlot("Elbow Method", ImVec2(-1, 200))) {
                ImPlot::SetupAxes("Number of Clusters (k)", "Inertia");

                std::vector<double> k_double(elbow_result_.k_values.begin(),
                                             elbow_result_.k_values.end());

                ImPlot::PlotLine("Inertia", k_double.data(), elbow_result_.inertias.data(),
                                static_cast<int>(k_double.size()));

                // Mark suggested k
                if (elbow_result_.suggested_k > 0) {
                    double suggested_x = elbow_result_.suggested_k;
                    double suggested_y = elbow_result_.inertias[elbow_result_.suggested_k - elbow_k_min];
                    ImPlot::PlotScatter("Suggested k", &suggested_x, &suggested_y, 1);
                }

                ImPlot::EndPlot();
            }

            ImGui::Text("Suggested k: %d", elbow_result_.suggested_k);

            if (ImGui::Button("Use Suggested k")) {
                config_.n_clusters = elbow_result_.suggested_k;
            }
        }
    }
}

void KMeansExecutor::RenderScatterPlot() {
    if (input_data_.empty() || result_.labels.empty()) return;

    int x_col = std::clamp(config_.x_column, 0, static_cast<int>(input_data_[0].size()) - 1);
    int y_col = std::clamp(config_.y_column, 0, static_cast<int>(input_data_[0].size()) - 1);

    if (ImPlot::BeginPlot("Cluster Scatter Plot", ImVec2(-1, 300))) {
        std::string x_label = x_col < static_cast<int>(column_names_.size()) ? column_names_[x_col] : "X";
        std::string y_label = y_col < static_cast<int>(column_names_.size()) ? column_names_[y_col] : "Y";
        ImPlot::SetupAxes(x_label.c_str(), y_label.c_str());

        // Plot each cluster separately for different colors
        for (int cluster = 0; cluster < config_.n_clusters; cluster++) {
            std::vector<double> x_vals, y_vals;

            for (size_t i = 0; i < input_data_.size(); i++) {
                if (result_.labels[i] == cluster) {
                    x_vals.push_back(input_data_[i][x_col]);
                    y_vals.push_back(input_data_[i][y_col]);
                }
            }

            if (!x_vals.empty()) {
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4,
                    ImGui::ColorConvertU32ToFloat4(CLUSTER_COLORS[cluster % MAX_CLUSTERS]));

                std::string label = fmt::format("Cluster {}", cluster);
                ImPlot::PlotScatter(label.c_str(), x_vals.data(), y_vals.data(),
                                   static_cast<int>(x_vals.size()));
            }
        }

        // Plot centroids
        if (!result_.centroids.empty()) {
            std::vector<double> cx, cy;
            for (const auto& centroid : result_.centroids) {
                if (x_col < static_cast<int>(centroid.size()) &&
                    y_col < static_cast<int>(centroid.size())) {
                    cx.push_back(centroid[x_col]);
                    cy.push_back(centroid[y_col]);
                }
            }

            if (!cx.empty()) {
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, 10, ImVec4(0, 0, 0, 1), 3);
                ImPlot::PlotScatter("Centroids", cx.data(), cy.data(), static_cast<int>(cx.size()));
            }
        }

        ImPlot::EndPlot();
    }
}

void KMeansExecutor::RenderCentroidsTable() {
    ImGui::Checkbox("Show Centroids Table", &show_centroids_table_);

    if (!show_centroids_table_ || result_.centroids.empty()) return;

    if (ImGui::BeginTable("Centroids", static_cast<int>(result_.centroids[0].size()) + 1,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX)) {

        // Header
        ImGui::TableSetupColumn("Cluster");
        for (size_t i = 0; i < result_.centroids[0].size(); i++) {
            std::string col_name = i < column_names_.size() ? column_names_[i] : fmt::format("Dim {}", i);
            ImGui::TableSetupColumn(col_name.c_str());
        }
        ImGui::TableHeadersRow();

        // Data rows
        for (size_t c = 0; c < result_.centroids.size(); c++) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(CLUSTER_COLORS[c % MAX_CLUSTERS]),
                              "Cluster %zu", c);

            for (size_t d = 0; d < result_.centroids[c].size(); d++) {
                ImGui::TableSetColumnIndex(static_cast<int>(d) + 1);
                ImGui::Text("%.4f", result_.centroids[c][d]);
            }
        }

        ImGui::EndTable();
    }
}

void KMeansExecutor::RenderMetricsSection() {
    ImGui::Text("Results");
    ImGui::Spacing();

    ImGui::Text("Clusters: %d", result_.n_clusters);
    ImGui::Text("Iterations: %d", result_.n_iterations);
    ImGui::Text("Inertia: %.4f", result_.inertia);
    ImGui::Text("Converged: %s", result_.converged ? "Yes" : "No");

    // Cluster sizes
    if (!result_.labels.empty()) {
        std::vector<int> cluster_counts(config_.n_clusters, 0);
        for (int label : result_.labels) {
            if (label >= 0 && label < config_.n_clusters) {
                cluster_counts[label]++;
            }
        }

        ImGui::Separator();
        ImGui::Text("Cluster Sizes:");
        for (int i = 0; i < config_.n_clusters; i++) {
            ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(CLUSTER_COLORS[i % MAX_CLUSTERS]),
                              "  Cluster %d: %d samples", i, cluster_counts[i]);
        }
    }
}

// ==================== Code Generation ====================

std::string KMeansExecutor::GenerateCode(CodeFramework framework) const {
    switch (framework) {
        case CodeFramework::Sklearn:
            return GenerateSklearnCode();
        case CodeFramework::PyCyxWiz:
            return GeneratePyCyxWizCode();
        default:
            return GenerateSklearnCode();  // Default to sklearn
    }
}

std::string KMeansExecutor::GenerateSklearnCode() const {
    return fmt::format(R"(
from sklearn.cluster import KMeans
import numpy as np

# K-Means Clustering
kmeans = KMeans(
    n_clusters={},
    max_iter={},
    init='{}',
    n_init={},
    tol={:.2e},
    random_state=42
)

# Fit and predict
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_

print(f"Clusters: {{len(centroids)}}")
print(f"Inertia: {{inertia:.4f}}")
)",
        config_.n_clusters,
        config_.max_iter,
        config_.init_method == 0 ? "random" : "k-means++",
        config_.n_init,
        config_.tolerance
    );
}

std::string KMeansExecutor::GeneratePyCyxWizCode() const {
    return fmt::format(R"(
import pycyxwiz as cyx

# K-Means Clustering (GPU-accelerated)
result = cyx.clustering.kmeans(
    X,
    n_clusters={},
    max_iter={},
    init='{}',
    n_init={},
    tol={:.2e}
)

labels = result.labels
centroids = result.centroids
inertia = result.inertia

print(f"Clusters: {{len(centroids)}}")
print(f"Inertia: {{inertia:.4f}}")
print(f"Converged: {{result.converged}}")
)",
        config_.n_clusters,
        config_.max_iter,
        config_.init_method == 0 ? "random" : "kmeans++",
        config_.n_init,
        config_.tolerance
    );
}

} // namespace cyxwiz
