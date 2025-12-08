#pragma once

#include "../../core/data_analyzer.h"
#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>

namespace cyxwiz {

class DataTable;

/**
 * DataProfilerPanel - Comprehensive dataset analysis panel
 *
 * Features:
 * - Dataset/table selector dropdown
 * - Summary statistics overview
 * - Per-column detailed profiles
 * - Histograms for numeric columns
 * - Top values for categorical columns
 * - Export to CSV/JSON
 * - Async profiling with progress indicator
 */
class DataProfilerPanel {
public:
    DataProfilerPanel();
    ~DataProfilerPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

    // Analyze a specific table
    void AnalyzeTable(const std::string& table_name);
    void AnalyzeTable(std::shared_ptr<DataTable> table);

    // Check if analysis is in progress
    bool IsAnalyzing() const { return is_analyzing_.load(); }

private:
    void RenderToolbar();
    void RenderDataSelector();
    void RenderSummaryTab();
    void RenderColumnsTab();
    void RenderColumnDetail(const ColumnProfile& profile);
    void RenderHistogram(const ColumnProfile& profile);
    void RenderTopValues(const ColumnProfile& profile);
    void RenderLoadingIndicator();
    void RenderExportOptions();

    // Async profiling
    void StartProfileAsync(std::shared_ptr<DataTable> table);
    void OnProfileComplete();

    // Export functions
    void ExportToCSV(const std::string& filepath);
    void ExportToJSON(const std::string& filepath);

    bool visible_ = false;

    // Current data source
    std::string selected_table_;
    std::shared_ptr<DataTable> current_table_;

    // Analysis results
    DataProfile profile_;
    bool has_profile_ = false;

    // Async state
    std::atomic<bool> is_analyzing_{false};
    std::atomic<float> analysis_progress_{0.0f};
    std::string analysis_status_;
    std::unique_ptr<std::thread> analysis_thread_;
    std::mutex profile_mutex_;

    // UI state
    int current_tab_ = 0;  // 0 = Summary, 1 = Columns
    int selected_column_ = -1;
    bool auto_refresh_ = false;
    int histogram_bins_ = 20;
    int top_n_values_ = 10;

    // Export dialog
    bool show_export_dialog_ = false;
    int export_format_ = 0;  // 0 = CSV, 1 = JSON
    char export_path_[512] = {0};

    // Column filter
    char column_filter_[128] = {0};
};

} // namespace cyxwiz
