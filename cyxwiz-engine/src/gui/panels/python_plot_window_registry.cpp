#include "python_plot_window_registry.h"
#include "plot_window.h"

namespace cyxwiz {

namespace {
    std::vector<std::shared_ptr<PlotWindow>> g_python_plot_windows;
}

const std::vector<std::shared_ptr<PlotWindow>>& GetPythonPlotWindows() {
    return g_python_plot_windows;
}

void AddPythonPlotWindow(const std::shared_ptr<PlotWindow>& window) {
    g_python_plot_windows.push_back(window);
}

void ClearPythonPlotWindows() {
    g_python_plot_windows.clear();
}

} // namespace cyxwiz
