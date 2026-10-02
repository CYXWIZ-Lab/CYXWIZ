#pragma once

#include <mutex>
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
class PlotInbox {
public:
    void Publish(std::vector<CapturedPlot> plots) {
        if (plots.empty()) return;
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& plot : plots) plots_.push_back(std::move(plot));
    }

    std::vector<CapturedPlot> Take() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<CapturedPlot> taken;
        taken.swap(plots_);
        return taken;
    }

private:
    std::mutex mutex_;
    std::vector<CapturedPlot> plots_;
};

}  // namespace scripting
