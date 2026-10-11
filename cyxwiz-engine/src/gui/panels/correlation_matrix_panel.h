#pragma once

#include "../../core/data_analyzer.h"
#include "../dataset_source.h"
#include <imgui.h>
#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>

namespace cyxwiz {

class DataTable;

/**
 * CorrelationMatrixPanel - Correlation heatmap visualization
 *
 * Features:
 * - Interactive heatmap on the theme's diverging scale
 * - Hover tooltips with exact values
 * - Column selection/filtering
 * - Export to CSV/image
 * - Async computation
 */
class CorrelationMatrixPanel {
public:
    CorrelationMatrixPanel();
    ~CorrelationMatrixPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

    // Analyze a specific table
    void AnalyzeTable(std::shared_ptr<DataTable> table);

    bool IsComputing() const { return is_computing_.load(); }

private:
    void RenderToolbar();
    void RenderDataSelector();
    void RenderHeatmap();
    void RenderLegend();
    void RenderLoadingIndicator();
    void RenderSelectedInfo();

    // Async computation
    void ComputeAsync(std::shared_ptr<DataTable> table);

    // Export
    void ExportToCSV(const std::string& filepath);

    bool visible_ = false;

    // Current data source
    ui::DatasetSource source_;  // the picked dataset (Data Input or a node result)
    std::shared_ptr<DataTable> current_table_;

    // Correlation results
    CorrelationMatrix correlation_;
    bool has_correlation_ = false;

    // Async state
    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex corr_mutex_;

    // UI state
    int hovered_i_ = -1;
    int hovered_j_ = -1;
    float cell_size_ = 30.0f;
    bool show_values_ = true;
    bool show_legend_ = true;

    // Export
    char export_path_[512] = {0};
};

} // namespace cyxwiz
