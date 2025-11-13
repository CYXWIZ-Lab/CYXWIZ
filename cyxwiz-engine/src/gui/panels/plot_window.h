#pragma once

#include "../panel.h"
#include "../../plotting/plot_manager.h"
#include "../../plotting/test_data_generator.h"
#include <string>
#include <functional>

namespace cyxwiz {

/**
 * PlotWindow - Reusable dockable window for displaying plots
 * Can be instantiated for any plot type with custom data and configuration
 */
class PlotWindow : public Panel {
public:
    enum class PlotWindowType {
        Line2D,
        Scatter2D,
        Bar,
        Stem,
        Stair,
        Histogram,
        PieChart,
        BoxPlot,
        Polar,
        Heatmap,
        Surface3D,
        Scatter3D,
        Line3D,
        Parametric
    };

    /**
     * Constructor
     * @param title Window title
     * @param type Type of plot to display
     * @param auto_generate If true, automatically generates sample data
     */
    PlotWindow(const std::string& title, PlotWindowType type, bool auto_generate = true);
    ~PlotWindow() override;

    void Render() override;

    // Data management
    void SetDataGenerator(std::function<void()> generator);
    void RegenerateData();

    // Plot access
    std::string GetPlotId() const { return plot_id_; }

    // Export
    void SaveToFile(const std::string& filepath);

private:
    PlotWindowType type_;
    std::string plot_id_;
    std::function<void()> data_generator_;
    bool auto_generated_;
    bool show_controls_;

    // Generation parameters (for UI controls)
    int num_points_;
    float noise_level_;
    int num_bins_;

    // Helper methods
    void InitializePlot();
    void GenerateDefaultData();
    void RenderControls();
    void RenderPlot();

    // Type-specific data generators
    void GenerateLineData();
    void GenerateScatterData();
    void GenerateBarData();
    void GenerateStemData();
    void GenerateStairData();
    void GenerateHistogramData();
    void GeneratePieData();
    void GenerateBoxPlotData();
    void GeneratePolarData();
    void GenerateHeatmapData();
    void Generate3DSurfaceData();
    void Generate3DScatterData();
    void Generate3DLineData();
    void GenerateParametricData();
};

} // namespace cyxwiz
