#pragma once

// Tools > Monitoring (TOFIX129 D13): one Monitor window with four tabs,
// shown through core/monitor_presentation.
//   System       the machine, the compute device and this process, with trends
//   Memory       the Engine's memory over time and a now / lowest / highest table
//   Loaded data  every dataset the Engine holds (monitor_panel_loaded_data.cpp)
//   Frames       frame times (gui/frame_metrics)
// The readings come from a sampler on its own thread, which runs for the life
// of the Engine, also while the window is closed. The window never lists the
// compute devices on a timer: it asks for the process device once and again
// only when that changes.

#include "../../core/loaded_data_report.h"
#include "../../core/monitor_presentation.h"
#include "../../core/monitor_sampler.h"
#include "../../core/storage_caches.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

// Label and value lines without a frame (the Monitor window's cards and facts).
void MonitorRows(const char* id, const std::vector<std::pair<std::string, std::string>>& rows);

class MonitorPanel {
public:
    // Asks Python to collect its garbage; `done` gets the number of objects
    // collected (on any thread). False when Python is busy or not started.
    using CollectPython = std::function<bool(std::function<void(long long)> done)>;

    MonitorPanel();
    ~MonitorPanel();

    // Every frame, also while the window is closed: reads what the Engine
    // holds on the device (on this, the interface thread) once per interval,
    // and draws the Clear caches dialog when it is open.
    void Render();
    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    // Opens the window on a tab ("system", "memory", "loaded_data", "frames").
    void Open(const std::string& tab_id);
    // Tools > Monitoring > Clear Cache.
    void OpenClearCaches();
    // Tools > Monitoring > Free unused memory.
    void FreeUnusedMemory();

    void SetOpenProfiler(std::function<void()> open) { open_profiler_ = std::move(open); }
    void SetCollectPython(CollectPython collect) { collect_python_ = std::move(collect); }

private:
    void RenderBar();
    void RenderSystem();
    void RenderMemory();
    void RenderLoadedData();
    void RenderFrames();
    void RenderClearCaches();
    void RefreshHistory();
    void ReadHeldMemory();
    void RefreshDevice();
    void SaveDefaults();
    void ExportCsv();
    void MeasureCaches();
    void SetInterval(double seconds);

    // What the sampler thread needs of the device (written on the interface
    // thread, read on the sampler's).
    struct DeviceForSampler {
        std::mutex mutex;
        bool present = false;
        std::string hardware_uuid;
        double held_stale_after = 3.0;  // seconds: an older held reading is not known
    };

    bool visible_ = false;
    monitor_view::Tab tab_ = monitor_view::Tab::System;
    bool select_tab_ = false;
    std::string message_;
    bool message_error_ = false;

    std::shared_ptr<DeviceForSampler> device_for_sampler_;
    std::unique_ptr<MonitorSampler> sampler_;
    std::vector<MonitorSample> history_;  // a copy, refreshed when the sampler has a new reading
    std::uint64_t history_generation_ = ~0ULL;
    double interval_ = 1.0;
    int keep_choice_ = 1;  // 5 min, 15 min, 1 hour
    std::chrono::steady_clock::time_point last_held_read_{};

    // The process device, asked once and when it changes.
    monitor_view::DeviceLabel device_;
    int device_type_seen_ = -2;  // -2: never asked
    int device_id_seen_ = -1;

    // Memory tab.
    bool series_on_[5] = {true, true, true, true, true};

    // Free unused memory.
    std::string released_;
    bool released_waiting_ = false;
    bool released_device_ran_ = false;
    std::uint64_t released_bytes_ = 0;
    std::shared_ptr<std::atomic<long long>> python_collected_;  // -2 until Python answers

    // Loaded data (rebuilt when the dataset catalog changes).
    std::shared_ptr<int> alive_ = std::make_shared<int>(0);
    int catalog_subscription_ = 0;
    bool loaded_stale_ = true;
    LoadedDataReport loaded_;
    std::string loaded_pick_;

    // Clear caches dialog.
    bool caches_open_request_ = false;
    bool caches_measuring_ = false;
    bool caches_measure_again_ = false;  // asked while a measure was in flight
    bool caches_clearing_ = false;
    std::vector<StorageCache> caches_;
    std::vector<std::string> caches_selected_;
    std::string caches_note_;

    std::function<void()> open_profiler_;
    CollectPython collect_python_;
};

}  // namespace cyxwiz
