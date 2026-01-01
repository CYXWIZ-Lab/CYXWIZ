#pragma once

#include "../panel.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <vector>
#include <string>

namespace cyxwiz {

enum class VizChartType { Histogram, Scatter, Bar, Box };

/**
 * VisualizationPanel - Dockable panel for data visualization
 *
 * Receives data from DataExplorerPanel and displays interactive charts.
 * Can be docked anywhere in the workspace.
 */
class VisualizationPanel : public Panel {
public:
    VisualizationPanel();
    ~VisualizationPanel() override;

    void Render() override;
    const char* GetIcon() const override { return ICON_FA_CHART_SIMPLE; }

    // Data intake from DataExplorer
    void SetData(const std::vector<std::string>& column_names,
                 const std::vector<std::vector<std::string>>& rows);
    void ClearData();
    bool HasData() const { return !column_names_.empty() && !rows_.empty(); }

    // Get data info
    size_t GetRowCount() const { return rows_.size(); }
    size_t GetColumnCount() const { return column_names_.size(); }

private:
    void RenderToolbar();
    void RenderHistogramChart();
    void RenderScatterChart();
    void RenderBarChart();
    void RenderBoxChart();

    // Helper to convert string column to doubles
    std::vector<double> GetColumnAsDoubles(int col_index) const;

    // State
    VizChartType chart_type_ = VizChartType::Histogram;
    std::vector<std::string> column_names_;
    std::vector<std::vector<std::string>> rows_;
    int viz_x_column_ = 0;
    int viz_y_column_ = 1;
};

} // namespace cyxwiz
