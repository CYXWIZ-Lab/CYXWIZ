#pragma once

#include "../core/plot/python_plot_request.h"
#include "../core/thread_inbox.h"

#include <memory>
#include <string>
#include <vector>

namespace arrow {
class Table;
}

namespace scripting {

/**
 * Captured plot/image from matplotlib or other plotting libraries
 */
struct CapturedPlot {
    std::vector<unsigned char> png_data;  // PNG image data
    int width = 0;
    int height = 0;
    std::string label;  // Optional label (e.g., figure title)
};

// Figures a finished script published for the Plot Output window
// (TOFIX134 P0 item 6). The script worker publishes, the UI thread takes
// every frame, so no figure depends on the window being visible or on the
// UI seeing the run's running -> finished edge.
using PlotInbox = cyxwiz::ThreadInbox<CapturedPlot>;

// A plot a script asked for with the cyxwiz module (TOFIX134 P5): checked on
// the script thread, its columns as an Arrow table, for Plot Output. With
// `close` set it closes the window of that title instead.
struct PythonPlot {
    cyxwiz::plot::PythonPlotRequest request;
    std::shared_ptr<arrow::Table> table;
    std::string close;
};
using PythonPlotInbox = cyxwiz::ThreadInbox<PythonPlot>;

// One value a script reported with pycyxwiz.rl_update_metric (TOFIX134 P0
// item 7), for the RL Training Dashboard.
struct RLMetricSample {
    std::string name;
    float value = 0.0f;
};

}  // namespace scripting
