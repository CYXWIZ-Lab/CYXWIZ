#pragma once

#include "../core/thread_inbox.h"

#include <string>
#include <vector>

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

// One value a script reported with pycyxwiz.rl_update_metric (TOFIX134 P0
// item 7), for the RL Training Dashboard.
struct RLMetricSample {
    std::string name;
    float value = 0.0f;
};

}  // namespace scripting
