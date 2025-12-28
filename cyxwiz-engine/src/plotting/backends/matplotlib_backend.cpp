#include "matplotlib_backend.h"
#include <spdlog/spdlog.h>
#include <sstream>
#include <vector>

#ifdef CYXWIZ_HAS_PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
namespace py = pybind11;
#endif

namespace cyxwiz::plotting {

#ifdef CYXWIZ_HAS_PYTHON
struct MatplotlibBackend::PythonState {
    py::object plt_module;
    py::object np_module;
    py::object fig;
    py::object ax;
    bool python_available = false;
};
#else
struct MatplotlibBackend::PythonState {
    bool python_available = false;
};
#endif

MatplotlibBackend::MatplotlibBackend() { py_state_ = std::make_unique<PythonState>(); }
MatplotlibBackend::~MatplotlibBackend() { Shutdown(); }

bool MatplotlibBackend::Initialize(int width, int height) {
    if (initialized_) { spdlog::warn("MatplotlibBackend already initialized"); return true; }
    width_ = width; height_ = height;
#ifdef CYXWIZ_HAS_PYTHON
    try {
        py::module_ mpl = py::module_::import("matplotlib");
        mpl.attr("use")("Agg");
        py_state_->plt_module = py::module_::import("matplotlib.pyplot");
        py_state_->np_module = py::module_::import("numpy");
        py_state_->python_available = true;
        spdlog::info("MatplotlibBackend initialized");
        initialized_ = true; return true;
    } catch (const std::exception& e) {
        spdlog::error("Failed to initialize matplotlib: {}", e.what());
        py_state_->python_available = false; initialized_ = true; return false;
    }
#else
    spdlog::warn("MatplotlibBackend: Python support not compiled in");
    py_state_->python_available = false; initialized_ = true; return false;
#endif
}

void MatplotlibBackend::Shutdown() {
    if (!initialized_) return;
    python_commands_.clear(); py_state_->python_available = false; initialized_ = false;
}

void MatplotlibBackend::BeginPlot(const char* title) {
    if (!initialized_ || !py_state_->python_available) return;
    current_title_ = title ? title : "Plot"; python_commands_.clear(); in_plot_ = true;
    std::ostringstream cmd;
    cmd << "import matplotlib.pyplot as plt\nimport numpy as np\n";
    cmd << "fig, ax = plt.subplots(figsize=(" << (width_/100.0) << "," << (height_/100.0) << "))\n";
    cmd << "ax.set_title('" << current_title_ << "')\n";
    if (!x_label_.empty()) cmd << "ax.set_xlabel('" << x_label_ << "')\n";
    if (!y_label_.empty()) cmd << "ax.set_ylabel('" << y_label_ << "')\n";
    if (show_grid_) cmd << "ax.grid(True)\n";
    python_commands_ += cmd.str();
}

void MatplotlibBackend::EndPlot() {
    if (!in_plot_) return;
    if (show_legend_) python_commands_ += "ax.legend()\n";
    ExecutePythonCommand(python_commands_); in_plot_ = false;
}

void MatplotlibBackend::PlotLine(const char* label, const double* x, const double* y, int n) {
    if (!in_plot_ || n <= 0) return;
    std::ostringstream cmd;
    cmd << "x = np.array(["; for(int i=0;i<n;++i){if(i>0)cmd<<",";cmd<<x[i];} cmd << "])\n";
    cmd << "y = np.array(["; for(int i=0;i<n;++i){if(i>0)cmd<<",";cmd<<y[i];} cmd << "])\n";
    cmd << "ax.plot(x, y, label='" << label << "')\n";
    python_commands_ += cmd.str();
}

void MatplotlibBackend::PlotScatter(const char* label, const double* x, const double* y, int n) {
    if (!in_plot_ || n <= 0) return;
    std::ostringstream cmd;
    cmd << "x = np.array(["; for(int i=0;i<n;++i){if(i>0)cmd<<",";cmd<<x[i];} cmd << "])\n";
    cmd << "y = np.array(["; for(int i=0;i<n;++i){if(i>0)cmd<<",";cmd<<y[i];} cmd << "])\n";
    cmd << "ax.scatter(x, y, label='" << label << "')\n";
    python_commands_ += cmd.str();
}

void MatplotlibBackend::PlotBars(const char* label, const double* x, const double* y, int n) {
    if (!in_plot_ || n <= 0) return;
    std::ostringstream cmd;
    cmd << "x = np.array(["; for(int i=0;i<n;++i){if(i>0)cmd<<",";cmd<<x[i];} cmd << "])\n";
    cmd << "y = np.array(["; for(int i=0;i<n;++i){if(i>0)cmd<<",";cmd<<y[i];} cmd << "])\n";
    cmd << "ax.bar(x, y, label='" << label << "')\n";
    python_commands_ += cmd.str();
}

void MatplotlibBackend::PlotHistogram(const char* label, const double* v, int n, int bins) {
    if (!in_plot_ || n <= 0) return;
    std::ostringstream cmd;
    cmd << "d = np.array(["; for(int i=0;i<n;++i){if(i>0)cmd<<",";cmd<<v[i];} cmd << "])\n";
    cmd << "ax.hist(d, bins=" << bins << ", label='" << label << "', alpha=0.7)\n";
    python_commands_ += cmd.str();
}

void MatplotlibBackend::PlotHeatmap(const char* label, const double* v, int r, int c) {
    if (!in_plot_ || r <= 0 || c <= 0) return;
    std::ostringstream cmd;
    cmd << "d = np.array(["; for(int i=0;i<r*c;++i){if(i>0)cmd<<",";cmd<<v[i];} cmd << "]).reshape(" << r << "," << c << ")\n";
    cmd << "ax.imshow(d, cmap='viridis', aspect='auto')\n";
    python_commands_ += cmd.str();
}

void MatplotlibBackend::PlotBoxPlot(const char* label, const double* v, int n) {
    if (!in_plot_ || n <= 0) return;
    std::ostringstream cmd;
    cmd << "d = np.array(["; for(int i=0;i<n;++i){if(i>0)cmd<<",";cmd<<v[i];} cmd << "])\n";
    cmd << "ax.boxplot([d], labels=['" << label << "'])\n";
    python_commands_ += cmd.str();
}

void MatplotlibBackend::PlotKDE(const char* label, const double* v, int n) { spdlog::warn("PlotKDE requires scipy"); }
void MatplotlibBackend::PlotQQPlot(const char* label, const double* v, int n) { spdlog::warn("PlotQQPlot requires scipy"); }
void MatplotlibBackend::PlotViolin(const char* label, const double* v, int n) { spdlog::warn("PlotViolin stub"); }
void MatplotlibBackend::PlotMosaic(const char* label, const double* c, int r, int cols) { spdlog::warn("PlotMosaic stub"); }
void MatplotlibBackend::PlotStems(const char* label, const double* x, const double* y, int n) { spdlog::warn("PlotStems stub"); }
void MatplotlibBackend::PlotStairs(const char* label, const double* x, const double* y, int n) { spdlog::warn("PlotStairs stub"); }
void MatplotlibBackend::PlotPieChart(const char* label, const double* v, const char* const* l, int n) { spdlog::warn("PlotPieChart stub"); }
void MatplotlibBackend::PlotPolarLine(const char* label, const double* t, const double* r, int n) { spdlog::warn("PlotPolarLine stub"); }

void MatplotlibBackend::SetAxisLabel(int axis, const char* label) {
    if (axis == 0) x_label_ = label ? label : "";
    else if (axis == 1) y_label_ = label ? label : "";
}

void MatplotlibBackend::SetAxisLimits(int axis, double min, double max) {
    std::ostringstream cmd;
    if (axis == 0) cmd << "ax.set_xlim(" << min << "," << max << ")\n";
    else cmd << "ax.set_ylim(" << min << "," << max << ")\n";
    python_commands_ += cmd.str();
}

void MatplotlibBackend::SetAxisAutoFit(int axis, bool enabled) {
    if (enabled) python_commands_ += std::string("ax.autoscale(enable=True, axis='") + (axis==0?"x":"y") + "')\n";
}

void MatplotlibBackend::SetTitle(const char* title) { current_title_ = title ? title : ""; }
void MatplotlibBackend::SetLegendVisible(bool v) { show_legend_ = v; }
void MatplotlibBackend::SetGridVisible(bool v) { show_grid_ = v; }

bool MatplotlibBackend::SaveToFile(const char* filepath) {
    if (!initialized_ || !py_state_->python_available) return false;
    ExecutePythonCommand("import matplotlib.pyplot as plt\nplt.savefig('" + std::string(filepath) + "', dpi=300)\n");
    return true;
}

bool MatplotlibBackend::Show() {
    if (!initialized_ || !py_state_->python_available) return false;
    python_commands_ += "plt.show()\n";
    ExecutePythonCommand(python_commands_);
    return true;
}

void MatplotlibBackend::ExecutePythonCommand(const std::string& cmd) {
    if (!py_state_->python_available) return;
#ifdef CYXWIZ_HAS_PYTHON
    try { py::exec(cmd); } catch (const std::exception& e) { spdlog::error("Python error: {}", e.what()); }
#endif
}

} // namespace cyxwiz::plotting
