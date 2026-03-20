#pragma once

#include <imgui.h>
#include <implot.h>
#include <string>
#include <vector>
#include <map>

namespace cyxwiz {

/**
 * Visualizer - Interactive data visualization for Data Studio
 *
 * Phase 1, Week 1: Core Infrastructure
 *
 * This component provides interactive data visualization using ImPlot.
 * Users can create various chart types from dataset columns.
 *
 * Features:
 * - Line plots, scatter plots, bar charts, histograms
 * - Multi-series support
 * - Interactive zooming, panning
 * - Export to image (PNG)
 * - Customizable colors, labels, legends
 *
 * Architecture:
 *   Visualizer -> DuckDB (data query) -> ImPlot (rendering)
 */
class Visualizer {
public:
    Visualizer();
    ~Visualizer();

    /**
     * Render the visualizer UI
     * Called from DataStudioPanel's Visualize tab
     */
    void Render();

    /**
     * Set the active dataset for visualization
     */
    void SetActiveDataset(const std::string& dataset_name);

    /**
     * Create a new plot from selected columns
     */
    void CreatePlot(const std::string& plot_type,
                   const std::vector<std::string>& columns);

private:
    enum class PlotType {
        Line,
        Scatter,
        Bar,
        Histogram,
        Heatmap,
        Box
    };

    struct PlotConfig {
        int id;
        std::string name;
        PlotType type;
        std::vector<std::string> columns;  // Column names to plot
        std::string x_axis_label;
        std::string y_axis_label;
        bool show_legend;
        std::map<std::string, std::vector<double>> data;  // Column -> values
    };

    // State
    std::string current_dataset_;
    std::vector<PlotConfig> plots_;
    int next_plot_id_;
    int selected_plot_id_;

    // ImPlot context (separate from other plots in Engine)
    ImPlotContext* context_;

    // UI state
    bool show_create_plot_dialog_;
    std::vector<std::string> available_columns_;
    std::vector<bool> selected_columns_;
    int selected_plot_type_;

    // Rendering helpers
    void RenderPlotGallery();
    void RenderPlotCanvas();
    void RenderCreatePlotDialog();
    void RenderPlotToolbar();

    // Plot rendering
    void RenderLinePlot(const PlotConfig& plot);
    void RenderScatterPlot(const PlotConfig& plot);
    void RenderBarChart(const PlotConfig& plot);
    void RenderHistogram(const PlotConfig& plot);
    void RenderHeatmap(const PlotConfig& plot);
    void RenderBoxPlot(const PlotConfig& plot);

    // Data loading
    bool LoadPlotData(PlotConfig& plot);
    void DeletePlot(int plot_id);
    void ClearAllPlots();
};

} // namespace cyxwiz
