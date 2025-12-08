#pragma once

#include "../../core/data_analyzer.h"
#include <imgui.h>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>

namespace cyxwiz {

class DataTable;

/**
 * DescriptiveStatsPanel - Summary statistics calculator
 *
 * Features:
 * - Column selector with data table dropdown
 * - Comprehensive statistics display (mean, median, std, etc.)
 * - Box plot visualization
 * - Histogram with normal curve overlay
 * - Percentile table
 * - Export to CSV
 */
class DescriptiveStatsPanel {
public:
    DescriptiveStatsPanel();
    ~DescriptiveStatsPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

    void AnalyzeTable(const std::string& table_name);
    void AnalyzeTable(std::shared_ptr<DataTable> table);

private:
    void RenderToolbar();
    void RenderDataSelector();
    void RenderColumnSelector();
    void RenderLoadingIndicator();
    void RenderStatisticsTable();
    void RenderPercentileTable();
    void RenderBoxPlot();
    void RenderHistogram();
    void RenderExportOptions();

    void ComputeAsync();

    bool visible_ = false;
    std::string selected_table_;
    std::shared_ptr<DataTable> current_table_;
    int selected_column_ = -1;
    std::vector<std::string> numeric_columns_;

    DescriptiveStats stats_;
    std::vector<double> column_data_;  // Current column's data for visualization
    bool has_stats_ = false;

    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex stats_mutex_;

    // Export
    char export_path_[256] = "";
};

} // namespace cyxwiz
