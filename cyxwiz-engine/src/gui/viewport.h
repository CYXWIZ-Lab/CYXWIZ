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
    int kind = 0;
    int identity_confidence = 0;
    std::string provider;
    std::string driver_version;
    std::string physical_fingerprint;
    bool provider_known = false;
    bool driver_version_known = false;
    bool physical_fingerprint_known = false;
    int metadata_status = 0;
    bool device_selectable = false;
    bool execution_validated = false;
    bool name_is_fallback = false;
    bool memory_total_known = false;
    bool name_from_qualification = false;
    std::string identity_source;
    bool qualification_evidence_available = false;
    bool matrix_qualified = false;
    bool training_authorized = false;
    int training_authorization_status = 0;
    std::string qualification_matrix_id;
    std::string qualification_message;
    std::string training_authorization_message;
    std::string failure_category;
    std::string failed_operation;
    std::string observed_failure;
    std::string failure_interpretation;
    std::string recommended_action;
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
