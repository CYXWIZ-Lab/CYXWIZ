#pragma once

#include "../panel.h"
#include "../../plotting/plot_manager.h"
#include "../../plotting/test_data_generator.h"
#include <string>
#include <vector>
#include <memory>

namespace cyxwiz {

/**
 * PlotTestPanel - GUI for testing and demonstrating the plotting system
 * Supports both real-time (ImPlot) and offline (matplotlib) plotting
 */
class PlotTestPanel : public Panel {
public:
    PlotTestPanel();
    ~PlotTestPanel() override;

    void Render() override;

private:
    // UI State
    bool show_realtime_demo_ = true;
    bool show_offline_demo_ = true;
    bool show_test_controls_ = true;

    // Plot IDs
    std::string training_loss_plot_;
    std::string training_acc_plot_;
    std::string histogram_plot_;
    std::string scatter_plot_;
    std::string heatmap_plot_;

    // Test Data State
    int num_epochs_ = 100;
    int num_samples_ = 1000;
    int num_bins_ = 50;
    float noise_level_ = 0.05f;
    bool is_training_ = false;
    int current_epoch_ = 0;

    // Plot Type Selection
    int selected_plot_type_ = 0;  // 0=Line, 1=Scatter, 2=Bar, 3=Histogram, 4=Heatmap
    int selected_backend_ = 0;    // 0=ImPlot, 1=Matplotlib
    int selected_data_gen_ = 0;   // 0=Normal, 1=Sine, 2=Training, 3=Cluster, etc.

    // Initialization
    void InitializePlots();
    void ShutdownPlots();

    // Rendering Sections
    void RenderRealtimeDemo();
    void RenderOfflineDemo();
    void RenderTestControls();

    // Real-time Plotting Examples
    void RenderTrainingCurves();
    void RenderHistogram();
    void RenderScatter();
    void RenderHeatmap();

    // Offline Plotting Examples
    void RenderStatisticalPlots();
    void RenderKDEPlot();
    void RenderQQPlot();
    void RenderBoxPlot();

    // Test Actions
    void GenerateTestPlot();
    void SimulateTraining();
    void ExportPlotToFile();
    void ClearAllPlots();

    // Data Generation Helpers
    void UpdateTrainingData();
    void GenerateHistogramData();
    void GenerateScatterData();
    void GenerateHeatmapData();
};

} // namespace cyxwiz
