// analytics_panel.h - Historical metrics and trend analysis
#pragma once

#include "gui/server_panel.h"
#include "core/metrics_storage.h"
#include <vector>
#include <map>
#include <string>
#include <chrono>

namespace cyxwiz::servernode::gui {

// Time range presets for historical queries
enum class TimeRange {
    HOUR_1,     // Last 1 hour
    HOUR_6,     // Last 6 hours
    HOUR_24,    // Last 24 hours
    DAY_7,      // Last 7 days
    DAY_30,     // Last 30 days
    CUSTOM      // Custom range
};

class AnalyticsPanel : public ServerPanel {
public:
    AnalyticsPanel();
    void Render() override;

private:
    // Main layout sections
    void RenderTimeRangeSelector();
    void RenderMetricSelector();
    void RenderHistoricalChart();
    void RenderTrendAnalysis();
    void RenderStatsSummary();
    void RenderJobCorrelationSection();

    // Chart rendering helpers
    void RenderAreaChart(const char* title,
                         const std::vector<core::MetricPoint>& data,
                         ImVec4 color,
                         float min_val = 0.0f,
                         float max_val = 100.0f);
    void RenderMultiLineChart(const char* title,
                              const std::map<core::MetricType, std::vector<core::MetricPoint>>& data_sets);

    // Data fetching
    void RefreshData();
    void LoadHistoricalData(core::MetricType type);
    core::AggregationLevel GetAggregationLevel() const;
    int64_t GetStartTimestamp() const;
    std::string FormatTimestamp(int64_t timestamp) const;
    std::string GetTimeRangeLabel() const;

    // Time range controls
    TimeRange selected_range_ = TimeRange::HOUR_24;
    int64_t custom_start_time_ = 0;
    int64_t custom_end_time_ = 0;

    // Metric selection
    std::vector<core::MetricType> available_metrics_ = {
        core::MetricType::CPU_USAGE,
        core::MetricType::GPU_USAGE,
        core::MetricType::RAM_USAGE,
        core::MetricType::VRAM_USAGE,
        core::MetricType::NETWORK_RX,
        core::MetricType::NETWORK_TX,
        core::MetricType::TEMPERATURE,
        core::MetricType::POWER_WATTS
    };
    std::vector<bool> metric_enabled_;
    core::MetricType primary_metric_ = core::MetricType::CPU_USAGE;

    // Cached chart data
    std::map<core::MetricType, std::vector<core::MetricPoint>> chart_data_;
    std::map<core::MetricType, core::MetricsStorage::MetricsStats> stats_cache_;

    // Job correlation data
    std::vector<core::JobMetricsRecord> job_metrics_;
    std::string selected_job_id_;
    std::vector<std::string> available_jobs_;

    // Database info
    int64_t db_size_bytes_ = 0;
    int64_t total_records_ = 0;

    // Update timing
    std::chrono::steady_clock::time_point last_refresh_;
    bool needs_refresh_ = true;
    bool loading_data_ = false;

    // Display options
    bool show_min_max_ = true;
    bool show_trend_line_ = false;
    bool auto_refresh_ = true;
    int auto_refresh_seconds_ = 60;
};

} // namespace cyxwiz::servernode::gui
