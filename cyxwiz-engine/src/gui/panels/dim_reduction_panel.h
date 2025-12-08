#pragma once

#include <cyxwiz/dimensionality_reduction.h>
#include <imgui.h>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>

namespace cyxwiz {

class DataTable;

/**
 * DimReductionPanel - Dimensionality Reduction Visualization Tool
 *
 * Features:
 * - PCA, t-SNE, and UMAP algorithms
 * - Table/column selector for input data
 * - Algorithm-specific parameter configuration
 * - 2D scatter plot visualization with ImPlot
 * - Color by label column
 * - Explained variance chart (PCA)
 * - Progress indicator for iterative methods
 * - Export embeddings to CSV
 */
class DimReductionPanel {
public:
    DimReductionPanel();
    ~DimReductionPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

    void AnalyzeTable(const std::string& table_name);
    void AnalyzeTable(std::shared_ptr<DataTable> table);

private:
    void RenderToolbar();
    void RenderDataSelector();
    void RenderAlgorithmConfig();
    void RenderPCAConfig();
    void RenderTSNEConfig();
    void RenderUMAPConfig();
    void RenderLoadingIndicator();
    void RenderResults();
    void RenderScatterPlot();
    void RenderExplainedVarianceChart();
    void RenderEmbeddingsTable();
    void RenderExportOptions();

    void RunReduction();
    void PrepareData();

    bool visible_ = false;
    std::string selected_table_;
    std::shared_ptr<DataTable> current_table_;
    std::vector<std::string> numeric_columns_;
    std::vector<int> selected_columns_;  // Columns to use for reduction
    int label_column_ = -1;              // Column for coloring points

    // Algorithm selection
    int algorithm_ = 0;  // 0=PCA, 1=t-SNE, 2=UMAP

    // PCA parameters
    int pca_n_components_ = 2;

    // t-SNE parameters
    int tsne_n_dims_ = 2;
    int tsne_perplexity_ = 30;
    int tsne_iterations_ = 1000;
    double tsne_learning_rate_ = 200.0;

    // UMAP parameters
    int umap_n_dims_ = 2;
    int umap_n_neighbors_ = 15;
    double umap_min_dist_ = 0.1;

    // Input data
    std::vector<std::vector<double>> input_data_;
    std::vector<std::string> labels_;

    // Results
    PCAResult pca_result_;
    tSNEResult tsne_result_;
    UMAPResult umap_result_;

    bool has_result_ = false;
    int last_algorithm_ = -1;  // Track which algorithm produced current result

    // Async computation
    std::atomic<bool> is_computing_{false};
    std::atomic<int> progress_iteration_{0};
    std::atomic<double> progress_kl_{0.0};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;

    // Export
    char export_path_[256] = "";
};

} // namespace cyxwiz
