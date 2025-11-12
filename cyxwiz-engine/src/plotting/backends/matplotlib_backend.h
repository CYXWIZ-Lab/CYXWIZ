#pragma once

#include "plot_backend.h"
#include <string>
#include <memory>

namespace cyxwiz::plotting {

/**
 * MatplotlibBackend - Offline plotting using Python's matplotlib
 * Requires Python interpreter and matplotlib installation
 */
class MatplotlibBackend : public PlotBackend {
public:
    MatplotlibBackend();
    ~MatplotlibBackend() override;

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

    // Statistical plots (matplotlib-specific)
    void PlotKDE(const char* label, const double* values, int count);
    void PlotQQPlot(const char* label, const double* values, int count);
    void PlotViolin(const char* label, const double* values, int count);
    void PlotMosaic(const char* label, const double* categories,
                   int rows, int cols);

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
    const char* GetBackendName() const override { return "Matplotlib"; }
    bool IsRealtime() const override { return false; }

    // Display (shows plot in window)
    bool Show();

private:
    struct PythonState;
    std::unique_ptr<PythonState> py_state_;

    std::string current_title_;
    std::string x_label_;
    std::string y_label_;
    int width_ = 800;
    int height_ = 600;
    bool show_legend_ = true;
    bool show_grid_ = true;
    bool initialized_ = false;
    bool in_plot_ = false;

    // Python command queue
    void ExecutePythonCommand(const std::string& cmd);
    std::string python_commands_;
};

} // namespace cyxwiz::plotting
