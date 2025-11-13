#pragma once

#include "plot_backend.h"
#include <string>

namespace cyxwiz::plotting {

/**
 * ImPlotBackend - Real-time plotting using Dear ImPlot
 * Integrated into ImGui render loop
 */
class ImPlotBackend : public PlotBackend {
public:
    ImPlotBackend();
    ~ImPlotBackend() override;

    // Lifecycle
    bool Initialize(int width, int height) override;
    void Shutdown() override;

    // Plot lifecycle
    void BeginPlot(const char* title) override;
    void EndPlot() override;

    // Basic plotting primitives
    void PlotLine(const char* label, const double* x_data,
                 const double* y_data, int count) override;
    void PlotScatter(const char* label, const double* x_data,
                    const double* y_data, int count) override;
    void PlotBars(const char* label, const double* x_data,
                 const double* y_data, int count) override;
    void PlotHistogram(const char* label, const double* values,
                      int count, int bins) override;

    // Advanced plot types
    void PlotHeatmap(const char* label, const double* values,
                    int rows, int cols) override;
    void PlotBoxPlot(const char* label, const double* values,
                    int count) override;

    // New plot types
    void PlotStems(const char* label, const double* x_data,
                  const double* y_data, int count) override;
    void PlotStairs(const char* label, const double* x_data,
                   const double* y_data, int count) override;
    void PlotPieChart(const char* label, const double* values,
                     const char* const* labels, int count) override;
    void PlotPolarLine(const char* label, const double* theta,
                      const double* r, int count) override;

    // Axis configuration
    void SetAxisLabel(int axis, const char* label) override;
    void SetAxisLimits(int axis, double min, double max) override;
    void SetAxisAutoFit(int axis, bool enabled) override;

    // Plot appearance
    void SetTitle(const char* title) override;
    void SetLegendVisible(bool visible) override;
    void SetGridVisible(bool visible) override;

    // Export
    bool SaveToFile(const char* filepath) override;

    // Backend info
    const char* GetBackendName() const override { return "ImPlot"; }
    bool IsRealtime() const override { return true; }

private:
    struct ImPlotState;
    ImPlotState* state_ = nullptr;

    std::string current_title_;
    int width_ = 800;
    int height_ = 600;
    bool show_legend_ = true;
    bool show_grid_ = true;
    bool initialized_ = false;
};

} // namespace cyxwiz::plotting
