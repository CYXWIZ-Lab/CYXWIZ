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
 * HierarchicalPanel - Hierarchical clustering UI panel
 *
 * Features:
 * - Dendrogram visualization
 * - Linkage method: ward, complete, average, single
 * - Cut height slider for cluster selection
 */
class HierarchicalPanel : public Panel {
public:
    HierarchicalPanel();
    ~HierarchicalPanel() override;

    void Render() override;
    void SetDataTableRegistry(DataTableRegistry* registry) { data_registry_ = registry; }

    bool IsOpen() const { return is_open_; }
    void SetOpen(bool open) { is_open_ = open; }

private:
    void RenderDataSelector();
    void RenderParameters();
    void RenderResults();
    void RenderDendrogram();
    void RenderScatterPlot();

    void LoadSelectedData();
    void RunClustering();

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

    // Hierarchical parameters
    int n_clusters_ = 3;
    int linkage_method_ = 0;  // 0=ward, 1=complete, 2=average, 3=single
    int metric_idx_ = 0;
    double cut_height_ = 0.0;
    double max_height_ = 1.0;

    // Results
    HierarchicalResult result_;
    std::vector<std::vector<double>> input_data_;

    // Async
    std::thread compute_thread_;
    std::atomic<bool> is_computing_{false};
    std::string status_message_;

    static constexpr int MAX_CLUSTERS = 20;
    static const ImU32 CLUSTER_COLORS[MAX_CLUSTERS];
};

} // namespace cyxwiz
