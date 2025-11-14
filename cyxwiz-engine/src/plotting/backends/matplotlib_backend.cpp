#include "matplotlib_backend.h"
#include <spdlog/spdlog.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <sstream>
#include <vector>

namespace py = pybind11;

namespace cyxwiz::plotting {

// ============================================================================
// Internal Python State
// ============================================================================

struct MatplotlibBackend::PythonState {
    py::object plt_module;
    py::object np_module;
    py::object fig;
    py::object ax;
    bool python_available = false;
};

// ============================================================================
// Lifecycle
// ============================================================================

MatplotlibBackend::MatplotlibBackend() {
    py_state_ = std::make_unique<PythonState>();
}

MatplotlibBackend::~MatplotlibBackend() {
    Shutdown();
}

bool MatplotlibBackend::Initialize(int width, int height) {
    if (initialized_) {
        spdlog::warn("MatplotlibBackend already initialized");
        return true;
    }

    width_ = width;
    height_ = height;

    try {
        // Import matplotlib and set non-interactive backend BEFORE importing pyplot
        // This prevents matplotlib from trying to create Tk GUI windows
        py::module_ mpl = py::module_::import("matplotlib");
        mpl.attr("use")("Agg");  // Use Agg backend (non-interactive, image-only)

        // Now import matplotlib.pyplot and numpy
        py::module_ plt = py::module_::import("matplotlib.pyplot");
        py::module_ np = py::module_::import("numpy");

        py_state_->plt_module = plt;
        py_state_->np_module = np;
        py_state_->python_available = true;

        spdlog::info("MatplotlibBackend initialized with Agg backend (non-interactive)");
        initialized_ = true;
        return true;
    } catch (const py::error_already_set& e) {
        spdlog::error("Failed to initialize matplotlib: {}", e.what());
        spdlog::error("Make sure matplotlib and numpy are installed: pip install matplotlib numpy");
        py_state_->python_available = false;
        initialized_ = true;  // Mark as initialized but non-functional
        return false;
    } catch (const std::exception& e) {
        spdlog::error("Failed to initialize matplotlib: {}", e.what());
        py_state_->python_available = false;
        initialized_ = true;
        return false;
    }
}

void MatplotlibBackend::Shutdown() {
    if (!initialized_) {
        return;
    }

    python_commands_.clear();
    py_state_->python_available = false;
    initialized_ = false;

    spdlog::info("MatplotlibBackend shutdown");
}

// ============================================================================
// Plot Lifecycle
// ============================================================================

void MatplotlibBackend::BeginPlot(const char* title) {
    if (!initialized_) {
        spdlog::error("MatplotlibBackend not initialized");
        return;
    }

    if (!py_state_->python_available) {
        spdlog::warn("Python not available, matplotlib plotting disabled");
        return;
    }

    current_title_ = title ? title : "Plot";
    python_commands_.clear();
    in_plot_ = true;

    // Build matplotlib figure
    std::ostringstream cmd;
    cmd << "import matplotlib.pyplot as plt\n";
    cmd << "import numpy as np\n";
    // Note: Don't close figures here - we need to save them later
    cmd << "fig, ax = plt.subplots(figsize=("
        << (width_ / 100.0) << ", " << (height_ / 100.0) << "))\n";
    cmd << "ax.set_title('" << current_title_ << "')\n";

    if (!x_label_.empty()) {
        cmd << "ax.set_xlabel('" << x_label_ << "')\n";
    }
    if (!y_label_.empty()) {
        cmd << "ax.set_ylabel('" << y_label_ << "')\n";
    }

    if (show_grid_) {
        cmd << "ax.grid(True)\n";
    }

    python_commands_ += cmd.str();
}

void MatplotlibBackend::EndPlot() {
    if (!in_plot_) {
        return;
    }

    if (show_legend_) {
        python_commands_ += "ax.legend()\n";
    }

    // Execute accumulated Python commands
    ExecutePythonCommand(python_commands_);

    in_plot_ = false;
}

// ============================================================================
// Basic Plotting Primitives
// ============================================================================

void MatplotlibBackend::PlotLine(const char* label, const double* x_data,
                                 const double* y_data, int count) {
    if (!in_plot_ || count <= 0) {
        return;
    }

    std::ostringstream cmd;
    cmd << "x = np.array([";
    for (int i = 0; i < count; ++i) {
        if (i > 0) cmd << ", ";
        cmd << x_data[i];
    }
    cmd << "])\n";

    cmd << "y = np.array([";
    for (int i = 0; i < count; ++i) {
        if (i > 0) cmd << ", ";
        cmd << y_data[i];
    }
    cmd << "])\n";

    cmd << "ax.plot(x, y, label='" << label << "')\n";
    python_commands_ += cmd.str();
}

void MatplotlibBackend::PlotScatter(const char* label, const double* x_data,
                                    const double* y_data, int count) {
    if (!in_plot_ || count <= 0) {
        return;
    }

    std::ostringstream cmd;
    cmd << "x = np.array([";
    for (int i = 0; i < count; ++i) {
        if (i > 0) cmd << ", ";
        cmd << x_data[i];
    }
    cmd << "])\n";

    cmd << "y = np.array([";
    for (int i = 0; i < count; ++i) {
        if (i > 0) cmd << ", ";
        cmd << y_data[i];
    }
    cmd << "])\n";

    cmd << "ax.scatter(x, y, label='" << label << "')\n";
    python_commands_ += cmd.str();
}

void MatplotlibBackend::PlotBars(const char* label, const double* x_data,
                                 const double* y_data, int count) {
    if (!in_plot_ || count <= 0) {
        return;
    }

    std::ostringstream cmd;
    cmd << "x = np.array([";
    for (int i = 0; i < count; ++i) {
        if (i > 0) cmd << ", ";
        cmd << x_data[i];
    }
    cmd << "])\n";

    cmd << "y = np.array([";
    for (int i = 0; i < count; ++i) {
        if (i > 0) cmd << ", ";
        cmd << y_data[i];
    }
    cmd << "])\n";

    cmd << "ax.bar(x, y, label='" << label << "')\n";
    python_commands_ += cmd.str();
}

void MatplotlibBackend::PlotHistogram(const char* label, const double* values,
                                      int count, int bins) {
    if (!in_plot_ || count <= 0) {
        return;
    }

    std::ostringstream cmd;
    cmd << "data = np.array([";
    for (int i = 0; i < count; ++i) {
        if (i > 0) cmd << ", ";
        cmd << values[i];
    }
    cmd << "])\n";

    cmd << "ax.hist(data, bins=" << bins << ", label='" << label
        << "', alpha=0.7, edgecolor='black')\n";
    python_commands_ += cmd.str();
}

// ============================================================================
// Advanced Plot Types
// ============================================================================

void MatplotlibBackend::PlotHeatmap(const char* label, const double* values,
                                    int rows, int cols) {
    if (!in_plot_ || rows <= 0 || cols <= 0) {
        return;
    }

    std::ostringstream cmd;
    cmd << "import matplotlib.pyplot as plt\n";
    cmd << "data = np.array([";
    for (int i = 0; i < rows * cols; ++i) {
        if (i > 0) cmd << ", ";
        cmd << values[i];
    }
    cmd << "]).reshape(" << rows << ", " << cols << ")\n";
    cmd << "im = ax.imshow(data, cmap='viridis', aspect='auto')\n";
    cmd << "plt.colorbar(im, ax=ax)\n";

    python_commands_ += cmd.str();
}

void MatplotlibBackend::PlotBoxPlot(const char* label, const double* values,
                                    int count) {
    if (!in_plot_ || count <= 0) {
        return;
    }

    std::ostringstream cmd;
    cmd << "data = np.array([";
    for (int i = 0; i < count; ++i) {
        if (i > 0) cmd << ", ";
        cmd << values[i];
    }
    cmd << "])\n";

    cmd << "ax.boxplot([data], labels=['" << label << "'])\n";
    python_commands_ += cmd.str();
}

void MatplotlibBackend::PlotKDE(const char* label, const double* values, int count) {
    if (!in_plot_ || count <= 0) {
        return;
    }

    std::ostringstream cmd;
    cmd << "from scipy import stats\n";
    cmd << "data = np.array([";
    for (int i = 0; i < count; ++i) {
        if (i > 0) cmd << ", ";
        cmd << values[i];
    }
    cmd << "])\n";

    cmd << "kde = stats.gaussian_kde(data)\n";
    cmd << "x_range = np.linspace(data.min(), data.max(), 100)\n";
    cmd << "ax.plot(x_range, kde(x_range), label='" << label << "')\n";

    python_commands_ += cmd.str();
}

void MatplotlibBackend::PlotQQPlot(const char* label, const double* values, int count) {
    if (!in_plot_ || count <= 0) {
        return;
    }

    std::ostringstream cmd;
    cmd << "from scipy import stats\n";
    cmd << "data = np.array([";
    for (int i = 0; i < count; ++i) {
        if (i > 0) cmd << ", ";
        cmd << values[i];
    }
    cmd << "])\n";

    cmd << "stats.probplot(data, dist='norm', plot=ax)\n";
    python_commands_ += cmd.str();
}

void MatplotlibBackend::PlotViolin(const char* label, const double* values, int count) {
    if (!in_plot_ || count <= 0) {
        return;
    }

    std::ostringstream cmd;
    cmd << "data = np.array([";
    for (int i = 0; i < count; ++i) {
        if (i > 0) cmd << ", ";
        cmd << values[i];
    }
    cmd << "])\n";

    cmd << "ax.violinplot([data], showmeans=True, showmedians=True)\n";
    python_commands_ += cmd.str();
}

void MatplotlibBackend::PlotMosaic(const char* label, const double* categories,
                                   int rows, int cols) {
    // Mosaic plots are complex and typically require categorical data
    spdlog::warn("Mosaic plot not yet implemented in matplotlib backend");
}

// ============================================================================
// New Plot Types (Stubs for now)
// ============================================================================

void MatplotlibBackend::PlotStems(const char* label, const double* x_data,
                                  const double* y_data, int count) {
    if (!in_plot_ || count <= 0) {
        return;
    }

    std::ostringstream cmd;
    cmd << "x = np.array([";
    for (int i = 0; i < count; ++i) {
        if (i > 0) cmd << ", ";
        cmd << x_data[i];
    }
    cmd << "])\n";

    cmd << "y = np.array([";
    for (int i = 0; i < count; ++i) {
        if (i > 0) cmd << ", ";
        cmd << y_data[i];
    }
    cmd << "])\n";

    cmd << "ax.stem(x, y, label='" << label << "')\n";
    python_commands_ += cmd.str();
}

void MatplotlibBackend::PlotStairs(const char* label, const double* x_data,
                                   const double* y_data, int count) {
    if (!in_plot_ || count <= 0) {
        return;
    }

    std::ostringstream cmd;
    cmd << "x = np.array([";
    for (int i = 0; i < count; ++i) {
        if (i > 0) cmd << ", ";
        cmd << x_data[i];
    }
    cmd << "])\n";

    cmd << "y = np.array([";
    for (int i = 0; i < count; ++i) {
        if (i > 0) cmd << ", ";
        cmd << y_data[i];
    }
    cmd << "])\n";

    cmd << "ax.stairs(y, x, label='" << label << "')\n";
    python_commands_ += cmd.str();
}

void MatplotlibBackend::PlotPieChart(const char* label, const double* values,
                                     const char* const* labels, int count) {
    if (!in_plot_ || count <= 0) {
        return;
    }

    std::ostringstream cmd;
    cmd << "values = np.array([";
    for (int i = 0; i < count; ++i) {
        if (i > 0) cmd << ", ";
        cmd << values[i];
    }
    cmd << "])\n";

    cmd << "labels = [";
    for (int i = 0; i < count; ++i) {
        if (i > 0) cmd << ", ";
        cmd << "'" << labels[i] << "'";
    }
    cmd << "]\n";

    cmd << "ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)\n";
    python_commands_ += cmd.str();
}

void MatplotlibBackend::PlotPolarLine(const char* label, const double* theta,
                                      const double* r, int count) {
    if (!in_plot_ || count <= 0) {
        return;
    }

    std::ostringstream cmd;
    cmd << "theta = np.array([";
    for (int i = 0; i < count; ++i) {
        if (i > 0) cmd << ", ";
        cmd << theta[i];
    }
    cmd << "])\n";

    cmd << "r = np.array([";
    for (int i = 0; i < count; ++i) {
        if (i > 0) cmd << ", ";
        cmd << r[i];
    }
    cmd << "])\n";

    cmd << "ax = plt.subplot(projection='polar')\n";
    cmd << "ax.plot(theta, r, label='" << label << "')\n";
    python_commands_ += cmd.str();
}

// ============================================================================
// Axis Configuration
// ============================================================================

void MatplotlibBackend::SetAxisLabel(int axis, const char* label) {
    if (axis == 0) {
        x_label_ = label ? label : "";
    } else if (axis == 1) {
        y_label_ = label ? label : "";
    }
}

void MatplotlibBackend::SetAxisLimits(int axis, double min, double max) {
    std::ostringstream cmd;
    if (axis == 0) {
        cmd << "ax.set_xlim(" << min << ", " << max << ")\n";
    } else if (axis == 1) {
        cmd << "ax.set_ylim(" << min << ", " << max << ")\n";
    }
    python_commands_ += cmd.str();
}

void MatplotlibBackend::SetAxisAutoFit(int axis, bool enabled) {
    if (enabled) {
        std::ostringstream cmd;
        cmd << "ax.autoscale(enable=True, axis='";
        cmd << (axis == 0 ? "x" : "y") << "')\n";
        python_commands_ += cmd.str();
    }
}

// ============================================================================
// Plot Appearance
// ============================================================================

void MatplotlibBackend::SetTitle(const char* title) {
    current_title_ = title ? title : "";
}

void MatplotlibBackend::SetLegendVisible(bool visible) {
    show_legend_ = visible;
}

void MatplotlibBackend::SetGridVisible(bool visible) {
    show_grid_ = visible;
}

// ============================================================================
// Export and Display
// ============================================================================

bool MatplotlibBackend::SaveToFile(const char* filepath) {
    if (!initialized_ || !py_state_->python_available) {
        spdlog::error("Cannot save plot: matplotlib not available");
        return false;
    }

    // Create save command with necessary imports (plt might be out of scope)
    std::string cmd = "import matplotlib.pyplot as plt\n";
    cmd += "plt.savefig('" + std::string(filepath) +
           "', dpi=300, bbox_inches='tight')\n";

    // Execute save command separately (don't use python_commands_ which has plt.close)
    ExecutePythonCommand(cmd);

    spdlog::info("Matplotlib plot saved to: {}", filepath);
    return true;
}

bool MatplotlibBackend::Show() {
    if (!initialized_ || !py_state_->python_available) {
        spdlog::error("Cannot show plot: matplotlib not available");
        return false;
    }

    python_commands_ += "plt.show()\n";

    // Execute Python commands to display the plot
    ExecutePythonCommand(python_commands_);

    return true;
}

// ============================================================================
// Internal Helpers
// ============================================================================

void MatplotlibBackend::ExecutePythonCommand(const std::string& cmd) {
    if (!py_state_->python_available) {
        spdlog::warn("Python not available, skipping command execution");
        return;
    }

    try {
        py::exec(cmd);
        spdlog::debug("Executed Python command:\n{}", cmd);
    } catch (const py::error_already_set& e) {
        spdlog::error("Python execution error: {}", e.what());
    } catch (const std::exception& e) {
        spdlog::error("Python execution error: {}", e.what());
    }
}

} // namespace cyxwiz::plotting
