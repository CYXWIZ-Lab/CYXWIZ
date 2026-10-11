#pragma once

#include "../panel.h"
#include "../dataset_source.h"
#include <cyxwiz/clustering.h>
#include "../../data/data_table.h"
#include <imgui.h>
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <atomic>

namespace cyxwiz {

/**
 * KMeansPanel - K-Means clustering UI panel
 *
 * Features:
 * - Table/column selector for input data
 * - Number of clusters (k) slider with elbow method plot
 * - Init method: random, k-means++
 * - 2D scatter plot with cluster colors
 * - Centroid markers
 * - Inertia display
 */
class KMeansPanel : public Panel {
public:
    KMeansPanel();
    ~KMeansPanel() override;

    void Render() override;

    bool IsOpen() const { return visible_; }
    void SetOpen(bool open) { visible_ = open; }

private:
    void RenderDataSelector();
    void RenderParameters();
    void RenderElbowAnalysis();
    void RenderResults();
    void RenderScatterPlot();
    void RenderCentroidsTable();

    void UseTable(std::shared_ptr<DataTable> table);  // a picked dataset was read
    void RunClustering();
    void RunElbowAnalysis();

    // State
    std::shared_ptr<DataTable> current_table_;

    // Data selection
    ui::DatasetSource source_;  // the picked dataset (Data Input or a node result)
    std::vector<std::string> column_names_;
    std::vector<bool> selected_columns_;
    int label_column_idx_ = -1;  // Optional label column for coloring
    int x_axis_column_ = 0;      // Column for X axis in scatter plot
    int y_axis_column_ = 1;      // Column for Y axis in scatter plot

    // K-Means parameters
    int n_clusters_ = 3;
    int max_iter_ = 300;
    int n_init_ = 10;
    int init_method_ = 1;  // 0 = random, 1 = kmeans++
    double tolerance_ = 1e-4;

    // Elbow analysis
    int elbow_k_min_ = 2;
    int elbow_k_max_ = 10;
    bool show_elbow_plot_ = false;
    ElbowAnalysis elbow_result_;

    // Results
    KMeansResult result_;
    std::vector<std::vector<double>> input_data_;
    std::vector<std::string> data_labels_;  // Original labels if available

    // Async execution
    std::thread compute_thread_;
    std::atomic<bool> is_computing_{false};
    std::shared_ptr<int> alive_ = std::make_shared<int>(0);  // results posted to the UI thread stop once the panel is gone
    std::atomic<bool> cancel_requested_{false};
    std::atomic<int> progress_iteration_{0};
    std::atomic<int> progress_total_{0};
    std::atomic<double> progress_inertia_{0.0};
    std::string status_message_;

    // Visualization colors
    static constexpr int MAX_CLUSTERS = 20;
    static const ImU32 CLUSTER_COLORS[MAX_CLUSTERS];
};

} // namespace cyxwiz
