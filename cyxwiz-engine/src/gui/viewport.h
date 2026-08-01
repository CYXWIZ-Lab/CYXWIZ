#pragma once

#include <memory>
#include <vector>
#include <string>

namespace cyxwiz {
    class TrainingPlotPanel;
    struct DeviceInfo;
}

namespace gui {

// Simple struct to cache device info without pulling in device.h
struct CachedDeviceInfo {
    int type = -1;
    int device_id = -1;
    std::string name;
    size_t memory_total = 0;
    size_t memory_available = 0;
    int compute_units = 0;
    bool supports_fp64 = false;
    bool supports_fp16 = false;
};

class Viewport {
public:
    Viewport();
    ~Viewport();

    void Render();

    // Set training panel for live metrics display
    void SetTrainingPanel(cyxwiz::TrainingPlotPanel* panel) { training_panel_ = panel; }

    // Visibility control for sidebar integration
    bool* GetVisiblePtr() { return &show_window_; }

    // Refresh device list (call when user requests refresh)
    void RefreshDevices();

private:
    bool show_window_;
    cyxwiz::TrainingPlotPanel* training_panel_ = nullptr;

    // Cached device list to avoid querying on every frame
    std::vector<CachedDeviceInfo> cached_devices_;
    bool devices_initialized_ = false;
};

} // namespace gui
