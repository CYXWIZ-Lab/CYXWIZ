#pragma once

#include "node_executor.h"
#include <cyxwiz/clustering.h>
#include <imgui.h>
#include <implot.h>
#include <atomic>
#include <thread>

namespace cyxwiz {

/**
 * KMeansExecutor - Executor for K-Means clustering node
 *
 * Uses backend Clustering::KMeans for GPU-accelerated computation.
 * Provides ImGui UI for configuration and scatter plot visualization.
 */
class KMeansExecutor : public INodeExecutor {
public:
    KMeansExecutor() = default;
    ~KMeansExecutor() override;

    // === Identity ===
    std::string GetName() const override { return "K-Means Clustering"; }
    std::string GetCategory() const override { return "Analytics"; }

    // === Configuration ===
    struct Config {
        int n_clusters = 3;
        int max_iter = 300;
        int n_init = 10;
        int init_method = 1;  // 0 = random, 1 = kmeans++
        double tolerance = 1e-4;
        int x_column = 0;     // Column for X axis in scatter plot
        int y_column = 1;     // Column for Y axis in scatter plot
    };

    Config& GetConfig() { return config_; }
    const Config& GetConfig() const { return config_; }

    // === Execution ===
    void Execute() override;
    void ExecuteAsync();  // Non-blocking version

    // Get results
    const KMeansResult& GetResult() const { return result_; }
    const ElbowAnalysis& GetElbowResult() const { return elbow_result_; }

    // === Elbow Analysis ===
    void RunElbowAnalysis(int k_min = 2, int k_max = 10);
    bool HasElbowResult() const { return elbow_result_.success; }

    // === UI Rendering ===
    void RenderConfigUI() override;
    void RenderResultsUI() override;
    bool HasVisualization() const override { return true; }

    // === Code Generation ===
    std::string GenerateCode(CodeFramework framework) const override;

private:
    // Configuration
    Config config_;

    // Results
    KMeansResult result_;
    ElbowAnalysis elbow_result_;

    // Async execution
    std::thread execution_thread_;
    std::atomic<bool> is_async_running_{false};

    // UI state
    bool show_elbow_plot_ = false;
    bool show_centroids_table_ = true;

    // Cluster colors for visualization
    static constexpr int MAX_CLUSTERS = 20;
    static const ImU32 CLUSTER_COLORS[MAX_CLUSTERS];

    // Helper methods
    void RenderParametersSection();
    void RenderElbowSection();
    void RenderScatterPlot();
    void RenderCentroidsTable();
    void RenderMetricsSection();

    std::string GenerateSklearnCode() const;
    std::string GeneratePyCyxWizCode() const;
};

} // namespace cyxwiz
