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
 * GMMPanel - Gaussian Mixture Model clustering UI panel
 *
 * Features:
 * - Number of components
 * - Covariance type selection
 * - BIC/AIC for model selection
 * - Soft cluster assignments visualization
 */
class GMMPanel : public Panel {
public:
    GMMPanel();
    ~GMMPanel() override;

    void Render() override;

    bool IsOpen() const { return visible_; }
    void SetOpen(bool open) { visible_ = open; }

private:
    void RenderDataSelector();
    void RenderParameters();
    void RenderResults();
    void RenderScatterPlot();
    void RenderComponentInfo();
    void RenderModelSelection();

    void UseTable(std::shared_ptr<DataTable> table);  // a picked dataset was read
    void RunClustering();
    void RunModelSelection();

    std::shared_ptr<DataTable> current_table_;

    // Data selection
    ui::DatasetSource source_;  // the picked dataset (Data Input or a node result)
    std::vector<std::string> column_names_;
    std::vector<bool> selected_columns_;
    int x_axis_column_ = 0;
    int y_axis_column_ = 1;

    // GMM parameters
    int n_components_ = 3;
    int covariance_type_ = 0;  // 0=full, 1=tied, 2=diag, 3=spherical
    int max_iter_ = 100;
    int n_init_ = 1;

    // Model selection
    int model_select_min_ = 2;
    int model_select_max_ = 10;
    std::vector<int> model_k_values_;
    std::vector<double> model_bic_values_;
    std::vector<double> model_aic_values_;
    int suggested_components_ = 3;

    // Results
    GMMResult result_;
    std::vector<std::vector<double>> input_data_;

    // Async
    std::thread compute_thread_;
    std::atomic<bool> is_computing_{false};
    std::shared_ptr<int> alive_ = std::make_shared<int>(0);  // results posted to the UI thread stop once the panel is gone
    std::atomic<int> progress_iteration_{0};
    std::string status_message_;

    static constexpr int MAX_CLUSTERS = 20;
    static const ImU32 CLUSTER_COLORS[MAX_CLUSTERS];
};

} // namespace cyxwiz
