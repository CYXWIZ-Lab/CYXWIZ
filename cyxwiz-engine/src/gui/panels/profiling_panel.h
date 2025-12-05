#pragma once

#include "../panel.h"
#include <imgui.h>
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <chrono>
#include <functional>

namespace cyxwiz {

/**
 * Layer profiling data for a single forward/backward pass
 */
struct LayerProfile {
    std::string name;
    int node_id = -1;
    double forward_time_ms = 0.0;
    double backward_time_ms = 0.0;
    size_t memory_bytes = 0;
    size_t param_count = 0;
};

/**
 * A snapshot of profiling data at a specific training step
 */
struct ProfilingSnapshot {
    int epoch = 0;
    int step = 0;
    double timestamp = 0.0;  // Seconds since training started
    double total_forward_ms = 0.0;
    double total_backward_ms = 0.0;
    double data_loading_ms = 0.0;
    double optimizer_step_ms = 0.0;
    std::vector<LayerProfile> layer_profiles;
};

/**
 * Performance Profiling Panel
 * Provides detailed timing visualization for neural network training.
 * Shows per-layer forward/backward pass timing, memory usage, and historical trends.
 */
class ProfilingPanel : public Panel {
public:
    ProfilingPanel();
    ~ProfilingPanel() override = default;

    void Render() override;

    // Data collection interface (called from TrainingExecutor)
    void BeginProfiling();
    void EndProfiling();
    void RecordLayerForward(const std::string& layer_name, int node_id, double time_ms, size_t memory_bytes = 0);
    void RecordLayerBackward(const std::string& layer_name, int node_id, double time_ms);
    void RecordDataLoading(double time_ms);
    void RecordOptimizerStep(double time_ms);
    void FinalizeStep(int epoch, int step);

    // Clear all profiling data
    void Clear();

    // Export to CSV for external analysis
    bool ExportToCSV(const std::string& path);

    // Set callback for highlighting a node in the editor when clicked
    using NodeHighlightCallback = std::function<void(int node_id)>;
    void SetNodeHighlightCallback(NodeHighlightCallback callback) { node_highlight_callback_ = callback; }

    // Check if profiling is active
    bool IsProfilingActive() const { return is_profiling_; }

private:
    // Rendering functions
    void RenderToolbar();
    void RenderTimelineView();
    void RenderLayerBreakdown();
    void RenderHistoryChart();
    void RenderSummaryStats();

    // Helper functions
    void SortLayerProfiles();
    double GetTotalTime(const ProfilingSnapshot& snapshot) const;
    std::string FormatTime(double ms) const;
    std::string FormatMemory(size_t bytes) const;

    // Profiling state
    bool is_profiling_ = false;
    std::chrono::steady_clock::time_point profiling_start_time_;

    // Current step data (being collected)
    ProfilingSnapshot current_snapshot_;
    std::map<std::string, LayerProfile> current_layer_data_;
    mutable std::mutex data_mutex_;

    // Historical data
    std::vector<ProfilingSnapshot> history_;
    static constexpr size_t kMaxHistorySize = 1000;

    // UI state
    int selected_snapshot_index_ = -1;  // -1 = latest
    int sort_column_ = 0;               // 0=name, 1=forward, 2=backward, 3=total, 4=memory
    bool sort_ascending_ = true;
    bool auto_scroll_timeline_ = true;
    bool show_forward_time_ = true;
    bool show_backward_time_ = true;
    bool show_memory_ = true;
    int view_mode_ = 0;  // 0=timeline, 1=table, 2=history

    // Filter
    char layer_filter_[128] = "";

    // Timeline visualization
    float timeline_zoom_ = 1.0f;
    float timeline_scroll_ = 0.0f;

    // Colors
    ImVec4 forward_color_ = ImVec4(0.2f, 0.6f, 1.0f, 1.0f);   // Blue
    ImVec4 backward_color_ = ImVec4(1.0f, 0.5f, 0.2f, 1.0f);  // Orange
    ImVec4 memory_color_ = ImVec4(0.3f, 0.8f, 0.3f, 1.0f);    // Green

    // Callback
    NodeHighlightCallback node_highlight_callback_;
};

} // namespace cyxwiz
