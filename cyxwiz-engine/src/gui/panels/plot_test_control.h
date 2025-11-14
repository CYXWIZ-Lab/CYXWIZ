#pragma once

#include "../panel.h"
#include "plot_window.h"
#include <memory>
#include <vector>

namespace cyxwiz {

/**
 * Plot Test Control Panel
 * Dockable panel for testing the plotting system
 * Allows selection of plot type, backend, and test data
 */
class PlotTestControlPanel : public Panel {
public:
    PlotTestControlPanel();
    ~PlotTestControlPanel() override = default;

    void Render() override;

    // Access to created plot windows
    const std::vector<std::shared_ptr<PlotWindow>>& GetPlotWindows() const { return plot_windows_; }

private:
    void GeneratePlot();
    void ClearAllPlots();

    // Selection state
    int selected_plot_type_;
    int selected_backend_;
    int selected_test_data_;

    // Plot windows created by this panel
    std::vector<std::shared_ptr<PlotWindow>> plot_windows_;

    // Track current plot window to reuse
    std::shared_ptr<PlotWindow> current_plot_window_;
};

} // namespace cyxwiz
