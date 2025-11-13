#include "implot_backend.h"
#include <implot.h>
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace cyxwiz::plotting {

// ============================================================================
// Internal State
// ============================================================================

struct ImPlotBackend::ImPlotState {
    ImPlotContext* context = nullptr;
    std::string x_label;
    std::string y_label;
    double x_min = 0.0, x_max = 1.0;
    double y_min = 0.0, y_max = 1.0;
    bool x_auto_fit = true;
    bool y_auto_fit = true;
    bool in_plot = false;
};

// ============================================================================
// Lifecycle
// ============================================================================

ImPlotBackend::ImPlotBackend() {
    state_ = new ImPlotState();
}

ImPlotBackend::~ImPlotBackend() {
    Shutdown();
    delete state_;
}

bool ImPlotBackend::Initialize(int width, int height) {
    if (initialized_) {
        spdlog::warn("ImPlotBackend already initialized");
        return true;
    }

    width_ = width;
    height_ = height;

    // Create ImPlot context (if not already created globally)
    if (ImPlot::GetCurrentContext() == nullptr) {
        state_->context = ImPlot::CreateContext();
        ImPlot::SetCurrentContext(state_->context);
    }

    initialized_ = true;
    spdlog::info("ImPlotBackend initialized ({}x{})", width, height);
    return true;
}

void ImPlotBackend::Shutdown() {
    if (!initialized_) {
        return;
    }

    if (state_->context) {
        ImPlot::DestroyContext(state_->context);
        state_->context = nullptr;
    }

    initialized_ = false;
    spdlog::info("ImPlotBackend shutdown");
}

// ============================================================================
// Plot Lifecycle
// ============================================================================

void ImPlotBackend::BeginPlot(const char* title) {
    if (!initialized_) {
        spdlog::error("ImPlotBackend not initialized");
        return;
    }

    current_title_ = title ? title : "Plot";

    // Set up ImPlot flags
    ImPlotFlags plot_flags = 0;
    if (!show_legend_) {
        plot_flags |= ImPlotFlags_NoLegend;
    }

    // Set up axis flags
    ImPlotAxisFlags axis_flags = 0;
    if (!show_grid_) {
        axis_flags |= ImPlotAxisFlags_NoGridLines;
    }
    if (state_->x_auto_fit) {
        axis_flags |= ImPlotAxisFlags_AutoFit;
    }

    // Begin plot
    if (ImPlot::BeginPlot(current_title_.c_str(),
                         ImVec2(static_cast<float>(width_),
                               static_cast<float>(height_)),
                         plot_flags)) {
        state_->in_plot = true;

        // Set axis labels
        if (!state_->x_label.empty()) {
            ImPlot::SetupAxis(ImAxis_X1, state_->x_label.c_str());
        }
        if (!state_->y_label.empty()) {
            ImPlot::SetupAxis(ImAxis_Y1, state_->y_label.c_str());
        }

        // Set axis limits if not auto-fit
        if (!state_->x_auto_fit) {
            ImPlot::SetupAxisLimits(ImAxis_X1, state_->x_min, state_->x_max,
                                   ImPlotCond_Always);
        }
        if (!state_->y_auto_fit) {
            ImPlot::SetupAxisLimits(ImAxis_Y1, state_->y_min, state_->y_max,
                                   ImPlotCond_Always);
        }
    } else {
        state_->in_plot = false;
    }
}

void ImPlotBackend::EndPlot() {
    if (state_->in_plot) {
        ImPlot::EndPlot();
        state_->in_plot = false;
    }
}

// ============================================================================
// Basic Plotting Primitives
// ============================================================================

void ImPlotBackend::PlotLine(const char* label, const double* x_data,
                             const double* y_data, int count) {
    if (!state_->in_plot || count <= 0) {
        return;
    }

    ImPlot::PlotLine(label, x_data, y_data, count);
}

void ImPlotBackend::PlotScatter(const char* label, const double* x_data,
                                const double* y_data, int count) {
    if (!state_->in_plot || count <= 0) {
        return;
    }

    ImPlot::PlotScatter(label, x_data, y_data, count);
}

void ImPlotBackend::PlotBars(const char* label, const double* x_data,
                            const double* y_data, int count) {
    if (!state_->in_plot || count <= 0) {
        return;
    }

    // ImPlot::PlotBars expects different signature - use bar width
    double bar_width = 0.67;  // Default bar width
    ImPlot::PlotBars(label, x_data, y_data, count, bar_width);
}

void ImPlotBackend::PlotHistogram(const char* label, const double* values,
                                  int count, int bins) {
    if (!state_->in_plot || count <= 0) {
        return;
    }

    // Calculate histogram range
    double min_val = *std::min_element(values, values + count);
    double max_val = *std::max_element(values, values + count);

    ImPlot::PlotHistogram(label, values, count, bins, 1.0,
                         ImPlotRange(min_val, max_val));
}

// ============================================================================
// Advanced Plot Types
// ============================================================================

void ImPlotBackend::PlotHeatmap(const char* label, const double* values,
                                int rows, int cols) {
    if (!state_->in_plot || rows <= 0 || cols <= 0) {
        return;
    }

    ImPlot::PlotHeatmap(label, values, rows, cols, 0, 0,
                       nullptr, ImPlotPoint(0, 0), ImPlotPoint(cols, rows));
}

void ImPlotBackend::PlotBoxPlot(const char* label, const double* values,
                                int count) {
    if (!state_->in_plot || count <= 0) {
        return;
    }

    // Calculate box plot statistics
    std::vector<double> sorted(values, values + count);
    std::sort(sorted.begin(), sorted.end());

    double min = sorted.front();
    double q1 = sorted[count / 4];
    double median = sorted[count / 2];
    double q3 = sorted[3 * count / 4];
    double max = sorted.back();

    // ImPlot doesn't have built-in box plots, so we draw using error bars
    // TODO: Implement custom box plot rendering
    spdlog::warn("Box plot rendering not fully implemented yet");
}

// ============================================================================
// New Plot Types
// ============================================================================

void ImPlotBackend::PlotStems(const char* label, const double* x_data,
                              const double* y_data, int count) {
    if (!state_->in_plot || count <= 0) {
        return;
    }

    ImPlot::PlotStems(label, x_data, y_data, count);
}

void ImPlotBackend::PlotStairs(const char* label, const double* x_data,
                               const double* y_data, int count) {
    if (!state_->in_plot || count <= 0) {
        return;
    }

    ImPlot::PlotStairs(label, x_data, y_data, count);
}

void ImPlotBackend::PlotPieChart(const char* label, const double* values,
                                 const char* const* labels, int count) {
    if (!state_->in_plot || count <= 0) {
        return;
    }

    // ImPlot::PlotPieChart signature: (labels_array, values, count, x, y, radius, format, angle0)
    ImPlot::PlotPieChart(labels, values, count, 0.0, 0.0, 1.0, "%.1f", 90.0);
}

void ImPlotBackend::PlotPolarLine(const char* label, const double* theta,
                                  const double* r, int count) {
    if (!state_->in_plot || count <= 0) {
        return;
    }

    // Convert polar to cartesian for plotting
    // ImPlot doesn't have direct polar support in this version
    // We'll need to convert theta/r to x/y
    std::vector<double> x_cart(count);
    std::vector<double> y_cart(count);

    for (int i = 0; i < count; ++i) {
        x_cart[i] = r[i] * std::cos(theta[i]);
        y_cart[i] = r[i] * std::sin(theta[i]);
    }

    ImPlot::PlotLine(label, x_cart.data(), y_cart.data(), count);
}

// ============================================================================
// Axis Configuration
// ============================================================================

void ImPlotBackend::SetAxisLabel(int axis, const char* label) {
    if (axis == 0) {
        state_->x_label = label ? label : "";
    } else if (axis == 1) {
        state_->y_label = label ? label : "";
    }
}

void ImPlotBackend::SetAxisLimits(int axis, double min, double max) {
    if (axis == 0) {
        state_->x_min = min;
        state_->x_max = max;
        state_->x_auto_fit = false;
    } else if (axis == 1) {
        state_->y_min = min;
        state_->y_max = max;
        state_->y_auto_fit = false;
    }
}

void ImPlotBackend::SetAxisAutoFit(int axis, bool enabled) {
    if (axis == 0) {
        state_->x_auto_fit = enabled;
    } else if (axis == 1) {
        state_->y_auto_fit = enabled;
    }
}

// ============================================================================
// Plot Appearance
// ============================================================================

void ImPlotBackend::SetTitle(const char* title) {
    current_title_ = title ? title : "";
}

void ImPlotBackend::SetLegendVisible(bool visible) {
    show_legend_ = visible;
}

void ImPlotBackend::SetGridVisible(bool visible) {
    show_grid_ = visible;
}

// ============================================================================
// Export
// ============================================================================

bool ImPlotBackend::SaveToFile(const char* filepath) {
    // ImPlot doesn't have built-in file export
    // TODO: Implement screenshot capture using framebuffer
    spdlog::warn("ImPlot SaveToFile not implemented - use matplotlib backend for export");
    return false;
}

} // namespace cyxwiz::plotting
