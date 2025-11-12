#pragma once

#include <string>

namespace cyxwiz::plotting {

/**
 * PlotBackend - Abstract interface for plotting backends
 * Implementations: ImPlotBackend (real-time), MatplotlibBackend (offline)
 */
class PlotBackend {
public:
    virtual ~PlotBackend() = default;

    // Lifecycle
    virtual bool Initialize(int width, int height) = 0;
    virtual void Shutdown() = 0;

    // Plot lifecycle
    virtual void BeginPlot(const char* title) = 0;
    virtual void EndPlot() = 0;

    // Basic plotting primitives
    virtual void PlotLine(const char* label, const double* x_data,
                         const double* y_data, int count) = 0;
    virtual void PlotScatter(const char* label, const double* x_data,
                            const double* y_data, int count) = 0;
    virtual void PlotBars(const char* label, const double* x_data,
                         const double* y_data, int count) = 0;
    virtual void PlotHistogram(const char* label, const double* values,
                               int count, int bins) = 0;

    // Advanced plot types
    virtual void PlotHeatmap(const char* label, const double* values,
                            int rows, int cols) = 0;
    virtual void PlotBoxPlot(const char* label, const double* values,
                            int count) = 0;

    // Axis configuration
    virtual void SetAxisLabel(int axis, const char* label) = 0;  // axis: 0=X, 1=Y
    virtual void SetAxisLimits(int axis, double min, double max) = 0;
    virtual void SetAxisAutoFit(int axis, bool enabled) = 0;

    // Plot appearance
    virtual void SetTitle(const char* title) = 0;
    virtual void SetLegendVisible(bool visible) = 0;
    virtual void SetGridVisible(bool visible) = 0;

    // Export
    virtual bool SaveToFile(const char* filepath) = 0;

    // Backend info
    virtual const char* GetBackendName() const = 0;
    virtual bool IsRealtime() const = 0;
};

} // namespace cyxwiz::plotting
