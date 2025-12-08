#pragma once

#include "../panel.h"
#include <cyxwiz/clustering.h>
#include "../../data/data_table.h"
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <atomic>

namespace cyxwiz {

class DataTableRegistry;

/**
 * ClusterEvalPanel - Cluster quality evaluation UI panel
 *
 * Features:
 * - Silhouette score and plot
 * - Davies-Bouldin index
 * - Calinski-Harabasz score
 * - Compare multiple clustering results
 */
class ClusterEvalPanel : public Panel {
public:
    ClusterEvalPanel();
    ~ClusterEvalPanel() override;

    void Render() override;
    void SetDataTableRegistry(DataTableRegistry* registry) { data_registry_ = registry; }

    bool IsOpen() const { return is_open_; }
    void SetOpen(bool open) { is_open_ = open; }

    // Allow setting data and labels from other panels
    void SetClusteringData(const std::vector<std::vector<double>>& data,
                          const std::vector<int>& labels,
                          const std::string& name);

private:
    void RenderDataSelector();
    void RenderMetrics();
    void RenderSilhouettePlot();
    void RenderComparison();

    void LoadSelectedData();
    void ComputeMetrics();

    bool is_open_ = false;
    DataTableRegistry* data_registry_ = nullptr;
    std::shared_ptr<DataTable> current_table_;

    // Data selection
    std::vector<std::string> available_tables_;
    int selected_table_idx_ = -1;
    std::vector<std::string> column_names_;
    std::vector<bool> selected_columns_;
    int label_column_idx_ = -1;

    // Current evaluation
    std::vector<std::vector<double>> input_data_;
    std::vector<int> cluster_labels_;
    std::string current_name_;
    ClusterMetrics metrics_;

    // Comparison history
    struct EvalRecord {
        std::string name;
        int n_clusters;
        double silhouette;
        double davies_bouldin;
        double calinski_harabasz;
    };
    std::vector<EvalRecord> eval_history_;

    // Async
    std::thread compute_thread_;
    std::atomic<bool> is_computing_{false};
    std::string status_message_;
};

} // namespace cyxwiz
