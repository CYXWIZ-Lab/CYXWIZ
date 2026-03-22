#pragma once

#include <memory>
#include <vector>

namespace cyxwiz {

class PlotWindow;

const std::vector<std::shared_ptr<PlotWindow>>& GetPythonPlotWindows();
void AddPythonPlotWindow(const std::shared_ptr<PlotWindow>& window);
void ClearPythonPlotWindows();

} // namespace cyxwiz
