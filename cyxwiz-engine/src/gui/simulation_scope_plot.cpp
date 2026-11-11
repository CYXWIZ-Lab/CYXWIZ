#include "simulation_scope_plot.h"

#include <implot.h>

namespace gui {

void DrawScopeTrace(const char* id, const char* series_label, const std::vector<float>& times,
                    const std::vector<float>& values, const ImVec2& size, bool auto_scale, const ImVec4& colour) {
    if (times.empty() || times.size() != values.size()) return;
    if (!ImPlot::BeginPlot(id, size, ImPlotFlags_NoTitle)) return;
    const ImPlotAxisFlags y_flags = auto_scale ? (ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_NoLabel) : ImPlotAxisFlags_NoLabel;
    ImPlot::SetupAxes("Time (s)", "Value", ImPlotAxisFlags_NoLabel, y_flags);
    // X follows the newest sample
    const float t_max = times.back();
    float t_window = times.size() > 1 ? times.back() - times.front() : 2.0f;
    if (t_window < 2.0f) t_window = 2.0f;
    ImPlot::SetupAxisLimits(ImAxis_X1, t_max - t_window, t_max, ImGuiCond_Always);
    ImPlot::PushStyleColor(ImPlotCol_Line, colour);
    ImPlot::PlotLine(series_label, times.data(), values.data(), static_cast<int>(times.size()));
    ImPlot::PopStyleColor();
    ImPlot::EndPlot();
}

}  // namespace gui
