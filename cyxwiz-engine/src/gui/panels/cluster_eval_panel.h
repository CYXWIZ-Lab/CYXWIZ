#pragma once

#include "../panel.h"
#include "../dataset_source.h"
#include <cyxwiz/clustering.h>
#include "../../data/data_table.h"
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <atomic>

namespace cyxwiz {

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

    bool IsOpen() const { return visible_; }
    void SetOpen(bool open) { visible_ = open; }

    // Allow setting data and labels from other panels
    void SetClusteringData(const std::vector<std::vector<double>>& data,
                          const std::vector<int>& labels,
                          const std::string& name);

private:
    void RenderDataSelector();
    void RenderMetrics();
    void RenderSilhouettePlot();
    void RenderComparison();

    void UseTable(std::shared_ptr<DataTable> table);  // a picked dataset was read
    void ComputeMetrics();

    std::shared_ptr<DataTable> current_table_;

    // Data selection
    ui::DatasetSource source_;  // the picked dataset (Data Input or a node result)
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
    static constexpr size_t kMaxEvalHistory = 100;

    // Async
    std::thread compute_thread_;
    std::atomic<bool> is_computing_{false};
    std::string status_message_;
};

} // namespace cyxwiz
