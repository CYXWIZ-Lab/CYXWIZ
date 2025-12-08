// analytics_panel.cpp
#include "gui/panels/analytics_panel.h"
#include "gui/icons.h"
#include <imgui.h>
#include <implot.h>
#include <ctime>
#include <algorithm>
#include <spdlog/spdlog.h>

namespace cyxwiz::servernode::gui {

// Bring core namespace types into scope for helper functions
namespace core = cyxwiz::servernode::core;

namespace {

const char* GetMetricName(core::MetricType type) {
    switch (type) {
        case core::MetricType::CPU_USAGE:    return "CPU Usage";
        case core::MetricType::GPU_USAGE:    return "GPU Usage";
        case core::MetricType::RAM_USAGE:    return "RAM Usage";
        case core::MetricType::VRAM_USAGE:   return "VRAM Usage";
        case core::MetricType::NETWORK_RX:   return "Network RX";
        case core::MetricType::NETWORK_TX:   return "Network TX";
        case core::MetricType::TEMPERATURE:  return "Temperature";
        case core::MetricType::POWER_WATTS:  return "Power (Watts)";
        default:                             return "Unknown";
    }
}

const char* GetMetricUnit(core::MetricType type) {
    switch (type) {
        case core::MetricType::CPU_USAGE:
        case core::MetricType::GPU_USAGE:
        case core::MetricType::RAM_USAGE:
        case core::MetricType::VRAM_USAGE:   return "%";
        case core::MetricType::NETWORK_RX:
        case core::MetricType::NETWORK_TX:   return "Mbps";
        case core::MetricType::TEMPERATURE:  return "C";
        case core::MetricType::POWER_WATTS:  return "W";
        default:                             return "";
    }
}

ImVec4 GetMetricColor(core::MetricType type) {
    switch (type) {
        case core::MetricType::CPU_USAGE:    return ImVec4(0.0f, 0.47f, 0.84f, 1.0f);
        case core::MetricType::GPU_USAGE:    return ImVec4(0.46f, 0.72f, 0.0f, 1.0f);
        case core::MetricType::RAM_USAGE:    return ImVec4(0.58f, 0.44f, 0.86f, 1.0f);
        case core::MetricType::VRAM_USAGE:   return ImVec4(0.0f, 0.75f, 0.75f, 1.0f);
        case core::MetricType::NETWORK_RX:   return ImVec4(0.96f, 0.62f, 0.12f, 1.0f);
        case core::MetricType::NETWORK_TX:   return ImVec4(0.91f, 0.30f, 0.24f, 1.0f);
        case core::MetricType::TEMPERATURE:  return ImVec4(0.95f, 0.77f, 0.06f, 1.0f);
        case core::MetricType::POWER_WATTS:  return ImVec4(0.56f, 0.56f, 0.56f, 1.0f);
        default:                             return ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    }
}

} // anonymous namespace

AnalyticsPanel::AnalyticsPanel()
    : ServerPanel("Analytics") {
    // Initialize metric enabled states
    metric_enabled_.resize(available_metrics_.size(), false);
    if (!metric_enabled_.empty()) {
        metric_enabled_[0] = true;  // CPU enabled by default
    }
}

void AnalyticsPanel::Render() {
    // Header
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[2]);
    ImGui::Text("%s Analytics Dashboard", ICON_FA_CHART_LINE);
    ImGui::PopFont();
    ImGui::Separator();
    ImGui::Spacing();

    // Check if metrics storage is available
    if (!core::MetricsStorageSingleton::IsInitialized()) {
        ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.2f, 1.0f),
            ICON_FA_TRIANGLE_EXCLAMATION " Metrics storage not initialized");
        ImGui::TextWrapped("Historical metrics require the daemon to be running and connected.");
        return;
    }

    // Auto-refresh logic
    auto now = std::chrono::steady_clock::now();
    if (auto_refresh_ && !loading_data_) {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_refresh_).count();
        if (elapsed >= auto_refresh_seconds_ || needs_refresh_) {
            RefreshData();
        }
    }

    // Main layout with left sidebar for controls
    float sidebar_width = 280.0f;

    // Left sidebar - controls
    ImGui::BeginChild("AnalyticsSidebar", ImVec2(sidebar_width, 0), true);
    RenderTimeRangeSelector();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    RenderMetricSelector();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    RenderStatsSummary();
    ImGui::EndChild();

    ImGui::SameLine();

    // Main area - charts
    ImGui::BeginChild("AnalyticsMain", ImVec2(0, 0), true);
    RenderHistoricalChart();
    ImGui::Spacing();
    RenderTrendAnalysis();
    ImGui::EndChild();
}

void AnalyticsPanel::RenderTimeRangeSelector() {
    ImGui::Text("%s Time Range", ICON_FA_CLOCK);
    ImGui::Spacing();

    const char* range_labels[] = {
        "Last Hour",
        "Last 6 Hours",
        "Last 24 Hours",
        "Last 7 Days",
        "Last 30 Days",
        "Custom Range"
    };

    int range_idx = static_cast<int>(selected_range_);
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##TimeRange", &range_idx, range_labels, IM_ARRAYSIZE(range_labels))) {
        selected_range_ = static_cast<TimeRange>(range_idx);
        needs_refresh_ = true;
    }

    // Show aggregation level info
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
        "Aggregation: %s",
        selected_range_ <= TimeRange::HOUR_6 ? "Raw (1s)" :
        selected_range_ == TimeRange::HOUR_24 ? "1 minute" :
        selected_range_ == TimeRange::DAY_7 ? "1 hour" : "1 day");

    ImGui::Spacing();

    // Refresh controls
    if (ImGui::Checkbox("Auto Refresh", &auto_refresh_)) {
        if (auto_refresh_) needs_refresh_ = true;
    }

    if (auto_refresh_) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(60);
        if (ImGui::InputInt("##RefreshSec", &auto_refresh_seconds_)) {
            auto_refresh_seconds_ = std::clamp(auto_refresh_seconds_, 10, 300);
        }
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "sec");
    }

    if (ImGui::Button(ICON_FA_ROTATE " Refresh Now", ImVec2(-1, 0))) {
        needs_refresh_ = true;
    }

    if (loading_data_) {
        ImGui::SameLine();
        ImGui::Text(ICON_FA_SPINNER " Loading...");
    }
}

void AnalyticsPanel::RenderMetricSelector() {
    ImGui::Text("%s Metrics", ICON_FA_CHART_SIMPLE);
    ImGui::Spacing();

    for (size_t i = 0; i < available_metrics_.size(); ++i) {
        auto type = available_metrics_[i];
        auto color = GetMetricColor(type);

        ImGui::PushStyleColor(ImGuiCol_CheckMark, color);

        // Use a regular bool variable since std::vector<bool> doesn't work with ImGui
        bool enabled = metric_enabled_[i];
        if (ImGui::Checkbox(GetMetricName(type), &enabled)) {
            metric_enabled_[i] = enabled;
            if (enabled) {
                LoadHistoricalData(type);
            }
        }
        ImGui::PopStyleColor();

        // Color indicator
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 20);
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 pos = ImGui::GetCursorScreenPos();
        draw_list->AddRectFilled(
            pos,
            ImVec2(pos.x + 15, pos.y + 15),
            ImGui::ColorConvertFloat4ToU32(color),
            3.0f
        );
        ImGui::Dummy(ImVec2(15, 15));
    }

    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
        "Select metrics to display in chart");
}

void AnalyticsPanel::RenderStatsSummary() {
    ImGui::Text("%s Statistics", ICON_FA_CALCULATOR);
    ImGui::Spacing();

    if (stats_cache_.empty()) {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No data available");
        return;
    }

    // Show stats for enabled metrics
    for (size_t i = 0; i < available_metrics_.size(); ++i) {
        if (!metric_enabled_[i]) continue;

        auto type = available_metrics_[i];
        auto it = stats_cache_.find(type);
        if (it == stats_cache_.end()) continue;

        const auto& stats = it->second;
        auto color = GetMetricColor(type);
        const char* unit = GetMetricUnit(type);

        if (ImGui::TreeNodeEx(GetMetricName(type), ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextColored(color, "Avg: %.1f%s", stats.average, unit);
            ImGui::Text("Min: %.1f%s  Max: %.1f%s", stats.min, unit, stats.max, unit);
            ImGui::Text("Std Dev: %.2f", stats.std_dev);
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                "Samples: %lld", static_cast<long long>(stats.sample_count));
            ImGui::TreePop();
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Database info
    ImGui::Text("%s Database Info", ICON_FA_DATABASE);
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
        "Size: %.2f MB", db_size_bytes_ / (1024.0 * 1024.0));
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
        "Records: %lld", static_cast<long long>(total_records_));
}

void AnalyticsPanel::RenderHistoricalChart() {
    ImGui::Text("%s Historical Data - %s", ICON_FA_CHART_AREA, GetTimeRangeLabel().c_str());
    ImGui::Spacing();

    // Count enabled metrics
    int enabled_count = 0;
    for (bool enabled : metric_enabled_) {
        if (enabled) enabled_count++;
    }

    if (enabled_count == 0) {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
            "Select metrics from the sidebar to display");
        return;
    }

    // Check if we have data
    bool has_data = false;
    for (size_t i = 0; i < available_metrics_.size(); ++i) {
        if (metric_enabled_[i]) {
            auto it = chart_data_.find(available_metrics_[i]);
            if (it != chart_data_.end() && !it->second.empty()) {
                has_data = true;
                break;
            }
        }
    }

    if (!has_data) {
        ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.2f, 1.0f),
            "No historical data available for selected time range");
        return;
    }

    // Create ImPlot chart
    ImVec2 chart_size = ImVec2(-1, 300);

    if (ImPlot::BeginPlot("##HistoricalChart", chart_size, ImPlotFlags_NoMouseText)) {
        // Setup axes
        ImPlot::SetupAxes("Time", "Value", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisFormat(ImAxis_X1, [](double value, char* buff, int size, void*) {
            time_t t = static_cast<time_t>(value);
            struct tm* tm_info = localtime(&t);
            strftime(buff, size, "%H:%M", tm_info);
            return size;
        });

        // Plot each enabled metric
        for (size_t i = 0; i < available_metrics_.size(); ++i) {
            if (!metric_enabled_[i]) continue;

            auto type = available_metrics_[i];
            auto it = chart_data_.find(type);
            if (it == chart_data_.end() || it->second.empty()) continue;

            const auto& data = it->second;
            auto color = GetMetricColor(type);

            // Prepare arrays for ImPlot
            std::vector<double> times, values;
            times.reserve(data.size());
            values.reserve(data.size());

            for (const auto& point : data) {
                times.push_back(static_cast<double>(point.timestamp));
                values.push_back(point.avg > 0 ? point.avg : point.value);
            }

            ImPlot::SetNextLineStyle(color, 2.0f);
            ImPlot::PlotLine(GetMetricName(type), times.data(), values.data(),
                static_cast<int>(times.size()));

            // Show min/max as shaded area if aggregated
            if (show_min_max_ && data[0].min != data[0].max) {
                std::vector<double> mins, maxs;
                mins.reserve(data.size());
                maxs.reserve(data.size());

                for (const auto& point : data) {
                    mins.push_back(point.min);
                    maxs.push_back(point.max);
                }

                ImVec4 shade_color = color;
                shade_color.w = 0.2f;
                ImPlot::SetNextFillStyle(shade_color);
                ImPlot::PlotShaded(GetMetricName(type), times.data(), mins.data(),
                    maxs.data(), static_cast<int>(times.size()));
            }
        }

        ImPlot::EndPlot();
    }

    // Display options
    ImGui::Checkbox("Show Min/Max Range", &show_min_max_);
    ImGui::SameLine();
    ImGui::Checkbox("Show Trend Line", &show_trend_line_);
}

void AnalyticsPanel::RenderTrendAnalysis() {
    if (ImGui::CollapsingHeader(ICON_FA_CHART_LINE " Trend Analysis", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Simple trend indicators for each metric
        for (size_t i = 0; i < available_metrics_.size(); ++i) {
            if (!metric_enabled_[i]) continue;

            auto type = available_metrics_[i];
            auto it = chart_data_.find(type);
            if (it == chart_data_.end() || it->second.size() < 2) continue;

            const auto& data = it->second;
            auto color = GetMetricColor(type);

            // Calculate simple trend (compare first and last quartile averages)
            size_t quarter = data.size() / 4;
            if (quarter < 1) quarter = 1;

            double first_avg = 0, last_avg = 0;
            for (size_t j = 0; j < quarter; ++j) {
                first_avg += (data[j].avg > 0 ? data[j].avg : data[j].value);
            }
            for (size_t j = data.size() - quarter; j < data.size(); ++j) {
                last_avg += (data[j].avg > 0 ? data[j].avg : data[j].value);
            }
            first_avg /= quarter;
            last_avg /= quarter;

            double change_pct = ((last_avg - first_avg) / (first_avg + 0.001)) * 100;

            ImGui::Text("%s:", GetMetricName(type));
            ImGui::SameLine(150);

            if (change_pct > 5) {
                ImGui::TextColored(ImVec4(0.91f, 0.30f, 0.24f, 1.0f),
                    ICON_FA_ARROW_UP " +%.1f%%", change_pct);
            } else if (change_pct < -5) {
                ImGui::TextColored(ImVec4(0.46f, 0.72f, 0.0f, 1.0f),
                    ICON_FA_ARROW_DOWN " %.1f%%", change_pct);
            } else {
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                    ICON_FA_MINUS " %.1f%%", change_pct);
            }

            ImGui::SameLine(250);
            ImGui::TextColored(color, "%.1f%s -> %.1f%s",
                first_avg, GetMetricUnit(type),
                last_avg, GetMetricUnit(type));
        }
    }
}

void AnalyticsPanel::RenderJobCorrelationSection() {
    if (ImGui::CollapsingHeader(ICON_FA_LINK " Job Correlation")) {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
            "Compare metrics with job execution");

        // Job selector would go here
        ImGui::Text("Coming soon: Select jobs to overlay on charts");
    }
}

void AnalyticsPanel::RefreshData() {
    if (loading_data_) return;

    loading_data_ = true;
    last_refresh_ = std::chrono::steady_clock::now();
    needs_refresh_ = false;

    auto& storage = core::MetricsStorageSingleton::Instance();

    // Get database info
    db_size_bytes_ = storage.GetDatabaseSize();
    total_records_ = storage.GetTotalRecords();

    // Load data for each enabled metric
    for (size_t i = 0; i < available_metrics_.size(); ++i) {
        if (metric_enabled_[i]) {
            LoadHistoricalData(available_metrics_[i]);
        }
    }

    loading_data_ = false;
}

void AnalyticsPanel::LoadHistoricalData(core::MetricType type) {
    auto& storage = core::MetricsStorageSingleton::Instance();

    int64_t now = std::time(nullptr);
    int64_t start = GetStartTimestamp();
    auto level = GetAggregationLevel();

    // Load chart data
    chart_data_[type] = storage.GetMetricsHistory(type, start, now, level);

    // Load stats
    stats_cache_[type] = storage.GetMetricsStats(type, start, now);
}

core::AggregationLevel AnalyticsPanel::GetAggregationLevel() const {
    switch (selected_range_) {
        case TimeRange::HOUR_1:
        case TimeRange::HOUR_6:
            return core::AggregationLevel::RAW;
        case TimeRange::HOUR_24:
            return core::AggregationLevel::MINUTE;
        case TimeRange::DAY_7:
            return core::AggregationLevel::HOUR;
        case TimeRange::DAY_30:
        case TimeRange::CUSTOM:
        default:
            return core::AggregationLevel::DAY;
    }
}

int64_t AnalyticsPanel::GetStartTimestamp() const {
    int64_t now = std::time(nullptr);

    switch (selected_range_) {
        case TimeRange::HOUR_1:  return now - 3600;
        case TimeRange::HOUR_6:  return now - 6 * 3600;
        case TimeRange::HOUR_24: return now - 24 * 3600;
        case TimeRange::DAY_7:   return now - 7 * 24 * 3600;
        case TimeRange::DAY_30:  return now - 30 * 24 * 3600;
        case TimeRange::CUSTOM:  return custom_start_time_;
        default:                 return now - 24 * 3600;
    }
}

std::string AnalyticsPanel::GetTimeRangeLabel() const {
    switch (selected_range_) {
        case TimeRange::HOUR_1:  return "Last Hour";
        case TimeRange::HOUR_6:  return "Last 6 Hours";
        case TimeRange::HOUR_24: return "Last 24 Hours";
        case TimeRange::DAY_7:   return "Last 7 Days";
        case TimeRange::DAY_30:  return "Last 30 Days";
        case TimeRange::CUSTOM:  return "Custom Range";
        default:                 return "Unknown";
    }
}

std::string AnalyticsPanel::FormatTimestamp(int64_t timestamp) const {
    time_t t = static_cast<time_t>(timestamp);
    struct tm* tm_info = localtime(&t);
    char buffer[64];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm_info);
    return buffer;
}

} // namespace cyxwiz::servernode::gui
