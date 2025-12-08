#pragma once

#include "../panel.h"
#include <cyxwiz/clustering.h>
#include "../../data/data_table.h"
#include <imgui.h>
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <atomic>

namespace cyxwiz {

class DataTableRegistry;

/**
 * DBSCANPanel - DBSCAN clustering UI panel
 *
 * Features:
 * - Epsilon (eps) slider with distance histogram
 * - Min samples control
 * - Noise point visualization (gray)
 * - Auto-eps suggestion based on k-distance graph
 */
class DBSCANPanel : public Panel {
public:
    DBSCANPanel();
    ~DBSCANPanel() override;

    void Render() override;
    void SetDataTableRegistry(DataTableRegistry* registry) { data_registry_ = registry; }

    bool IsOpen() const { return is_open_; }
    void SetOpen(bool open) { is_open_ = open; }

private:
    void RenderDataSelector();
    void RenderParameters();
    void RenderKDistanceAnalysis();
    void RenderResults();
    void RenderScatterPlot();

    void LoadSelectedData();
    void RunClustering();
    void ComputeKDistances();

    bool is_open_ = false;
    DataTableRegistry* data_registry_ = nullptr;
    std::shared_ptr<DataTable> current_table_;

    // Data selection
    std::vector<std::string> available_tables_;
    int selected_table_idx_ = -1;
    std::vector<std::string> column_names_;
    std::vector<bool> selected_columns_;
    int x_axis_column_ = 0;
    int y_axis_column_ = 1;

    // DBSCAN parameters
    double eps_ = 0.5;
    int min_samples_ = 5;
    int metric_idx_ = 0;  // 0 = euclidean, 1 = manhattan, 2 = cosine

    // K-distance analysis
    int k_distance_k_ = 5;
    std::vector<double> k_distances_;
    double suggested_eps_ = 0.5;

    // Results
    DBSCANResult result_;
    std::vector<std::vector<double>> input_data_;

    // Async
    std::thread compute_thread_;
    std::atomic<bool> is_computing_{false};
    std::string status_message_;

    static constexpr int MAX_CLUSTERS = 20;
    static const ImU32 CLUSTER_COLORS[MAX_CLUSTERS];
};

} // namespace cyxwiz
