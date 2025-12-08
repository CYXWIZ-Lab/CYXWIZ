#pragma once

#include <cyxwiz/feature_importance.h>
#include <imgui.h>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>

namespace cyxwiz {

class DataTable;
class SequentialModel;

/**
 * FeatureImportancePanel - Feature Importance Analysis Tool
 *
 * Features:
 * - Permutation importance (model-agnostic)
 * - Drop-column importance
 * - Weight-based importance (first layer)
 * - Horizontal bar chart visualization
 * - Error bars for standard deviation
 * - Feature ranking table
 * - Progress indicator for computation
 * - Export results to CSV
 */
class FeatureImportancePanel {
public:
    FeatureImportancePanel();
    ~FeatureImportancePanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

    // Set the trained model
    void SetModel(std::shared_ptr<SequentialModel> model);

    // Set the dataset for evaluation
    void SetDataset(const std::string& table_name);
    void SetDataset(std::shared_ptr<DataTable> table);

private:
    void RenderToolbar();
    void RenderDataSelector();
    void RenderConfiguration();
    void RenderLoadingIndicator();
    void RenderResults();
    void RenderImportanceChart();
    void RenderRankingTable();
    void RenderExportOptions();

    void RunImportanceAnalysis();
    void PrepareData();

    bool visible_ = false;

    // Model
    std::shared_ptr<SequentialModel> model_;
    bool has_model_ = false;

    // Dataset
    std::string selected_table_;
    std::shared_ptr<DataTable> current_table_;
    std::vector<std::string> feature_columns_;
    std::vector<int> selected_features_;
    int target_column_ = -1;

    // Configuration
    int method_ = 0;  // 0=Permutation, 1=Drop-Column, 2=Weight
    int n_repeats_ = 10;
    int scoring_method_ = 0;  // 0=Accuracy, 1=MSE

    // Prepared data
    std::vector<std::vector<double>> X_data_;
    std::vector<double> y_data_;
    std::vector<std::string> feature_names_;

    // Results
    FeatureImportanceResult result_;
    bool has_result_ = false;

    // Async computation
    std::atomic<bool> is_computing_{false};
    std::atomic<int> progress_feature_{0};
    std::atomic<int> total_features_{0};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;

    // Export
    char export_path_[256] = "";
};

} // namespace cyxwiz
