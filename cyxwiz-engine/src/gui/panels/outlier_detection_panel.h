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
 * OutlierDetectionPanel - Detect and visualize outliers
 *
 * Features:
 * - Column selection
 * - Multiple detection methods (IQR, Z-Score, Modified Z-Score)
 * - Adjustable parameters
 * - Scatter plot with outliers highlighted
 * - Outlier table with row indices
 */
class OutlierDetectionPanel {
public:
    OutlierDetectionPanel();
    ~OutlierDetectionPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

    void AnalyzeTable(const std::string& table_name);
    void AnalyzeTable(std::shared_ptr<DataTable> table);

    bool IsAnalyzing() const { return is_analyzing_.load(); }

private:
    void RenderToolbar();
    void RenderDataSelector();
    void RenderColumnSelector();
    void RenderMethodSelector();
    void RenderResults();
    void RenderScatterPlot();
    void RenderOutlierTable();
    void RenderLoadingIndicator();

    void DetectAsync();

    bool visible_ = false;

    // Current data
    std::string selected_table_;
    std::shared_ptr<DataTable> current_table_;

    // Column selection
    std::vector<std::string> numeric_columns_;
    int selected_column_idx_ = -1;

    // Detection settings
    OutlierMethod method_ = OutlierMethod::IQR;
    float iqr_factor_ = 1.5f;
    float zscore_threshold_ = 3.0f;
    float modified_zscore_threshold_ = 3.5f;

    // Results
    OutlierResult result_;
    bool has_result_ = false;

    // Async state
    std::atomic<bool> is_analyzing_{false};
    std::unique_ptr<std::thread> analysis_thread_;
    std::mutex result_mutex_;

    // UI state
    int sort_column_ = 0;
    bool sort_ascending_ = false;
    int page_ = 0;
    int rows_per_page_ = 50;
};

} // namespace cyxwiz
