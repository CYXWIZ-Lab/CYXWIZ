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
 * MissingValuePanel - Analyze and visualize missing values
 *
 * Features:
 * - Summary statistics (total missing, percentage)
 * - Per-column missing breakdown
 * - Row completeness visualization
 * - Pattern detection (which rows have missing)
 * - Imputation suggestions
 */
class MissingValuePanel {
public:
    MissingValuePanel();
    ~MissingValuePanel();

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
    void RenderSummary();
    void RenderColumnTable();
    void RenderMissingPattern();
    void RenderImputationSuggestions();
    void RenderLoadingIndicator();

    void AnalyzeAsync(std::shared_ptr<DataTable> table);

    bool visible_ = false;

    // Current data
    std::string selected_table_;
    std::shared_ptr<DataTable> current_table_;

    // Analysis results
    MissingValueAnalysis analysis_;
    bool has_analysis_ = false;

    // Async state
    std::atomic<bool> is_analyzing_{false};
    std::unique_ptr<std::thread> analysis_thread_;
    std::mutex analysis_mutex_;

    // UI state
    int sort_column_ = -1;
    bool sort_ascending_ = true;
    int selected_column_ = -1;
};

} // namespace cyxwiz
