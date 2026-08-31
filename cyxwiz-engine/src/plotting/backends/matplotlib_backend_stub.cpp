#include "matplotlib_backend.h"

#include <spdlog/spdlog.h>

namespace cyxwiz::plotting {

struct MatplotlibBackend::PythonState {};

MatplotlibBackend::MatplotlibBackend()
    : py_state_(std::make_unique<PythonState>()) {}

MatplotlibBackend::~MatplotlibBackend() = default;

bool MatplotlibBackend::Initialize(int width, int height) {
    width_ = width;
    height_ = height;
    initialized_ = true;
    spdlog::warn(
        "Matplotlib backend is unavailable because this Engine build "
        "excludes Python scripting");
    return false;
}

void MatplotlibBackend::Shutdown() {
    initialized_ = false;
    in_plot_ = false;
    python_commands_.clear();
}

void MatplotlibBackend::BeginPlot(const char*) {}
void MatplotlibBackend::EndPlot() {}
void MatplotlibBackend::PlotLine(
    const char*, const double*, const double*, int) {}
void MatplotlibBackend::PlotScatter(
    const char*, const double*, const double*, int) {}
void MatplotlibBackend::PlotBars(
    const char*, const double*, const double*, int) {}
void MatplotlibBackend::PlotHistogram(const char*, const double*, int, int) {}
void MatplotlibBackend::PlotHeatmap(const char*, const double*, int, int) {}
void MatplotlibBackend::PlotBoxPlot(const char*, const double*, int) {}
void MatplotlibBackend::PlotStems(
    const char*, const double*, const double*, int) {}
void MatplotlibBackend::PlotStairs(
    const char*, const double*, const double*, int) {}
void MatplotlibBackend::PlotPieChart(
    const char*, const double*, const char* const*, int) {}
void MatplotlibBackend::PlotPolarLine(
    const char*, const double*, const double*, int) {}
void MatplotlibBackend::PlotKDE(const char*, const double*, int) {}
void MatplotlibBackend::PlotQQPlot(const char*, const double*, int) {}
void MatplotlibBackend::PlotViolin(const char*, const double*, int) {}
void MatplotlibBackend::PlotMosaic(const char*, const double*, int, int) {}
void MatplotlibBackend::SetAxisLabel(int, const char*) {}
void MatplotlibBackend::SetAxisLimits(int, double, double) {}
void MatplotlibBackend::SetAxisAutoFit(int, bool) {}
void MatplotlibBackend::SetTitle(const char*) {}
void MatplotlibBackend::SetLegendVisible(bool) {}
void MatplotlibBackend::SetGridVisible(bool) {}
bool MatplotlibBackend::SaveToFile(const char*) { return false; }
bool MatplotlibBackend::Show() { return false; }
void MatplotlibBackend::ExecutePythonCommand(const std::string&) {}

}  // namespace cyxwiz::plotting
