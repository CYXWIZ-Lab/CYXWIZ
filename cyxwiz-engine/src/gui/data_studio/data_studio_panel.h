#pragma once

#include "query_editor.h"
#include "analyzer.h"
#include "visualizer.h"
#include <imgui.h>
#include <memory>
#include <string>

namespace cyxwiz {

/**
 * DataStudioPanel - Data analysis and query interface
 *
 * Unified Canvas Phase 5: Simplified Data Studio
 *
 * This panel provides SQL querying, statistical analysis, and visualization
 * tools for datasets. The visual pipeline building capability has been moved
 * to the Node Editor (Unified Canvas) for a single, unified experience.
 *
 * Architecture:
 *   DataStudioPanel (Container)
 *     ├─ QueryEditor (SQL query interface)
 *     ├─ Analyzer (Statistical analysis)
 *     └─ Visualizer (Interactive plots)
 *
 * Tab Structure:
 *   - Query: SQL query editor with DuckDB backend
 *   - Analyze: Statistical analysis and profiling
 *   - Visualize: Interactive data visualization
 *
 * Integration Points:
 *   - DataRegistry: Load datasets for analysis
 *   - DuckDBConnector: Execute SQL queries
 *   - ArrowDataset: Zero-copy data access
 *
 * Note: Visual pipeline building is now in Node Editor's "Data Pipeline" mode
 */
class DataStudioPanel {
public:
    DataStudioPanel();
    ~DataStudioPanel() = default;

    /**
     * Render the Data Studio panel
     * Called every frame from MainWindow
     */
    void Render();

    /**
     * Set the active dataset for all components
     * @param dataset_name Name of dataset in DataRegistry
     */
    void SetActiveDataset(const std::string& dataset_name);

    /**
     * Get the currently active dataset
     */
    const std::string& GetActiveDataset() const { return active_dataset_; }

    /**
     * Check if the panel is visible
     */
    bool IsVisible() const { return visible_; }

    /**
     * Show/hide the panel
     */
    void SetVisible(bool visible) { visible_ = visible; }

    /**
     * Get pointer to visibility flag (for sidebar integration)
     */
    bool* GetVisiblePtr() { return &visible_; }

private:
    // Component instances (Unified Canvas Phase 5: Removed pipeline_canvas_)
    std::unique_ptr<QueryEditor> query_editor_;
    std::unique_ptr<Analyzer> analyzer_;
    std::unique_ptr<Visualizer> visualizer_;

    // State
    std::string active_dataset_;
    int selected_tab_;
    bool visible_;

    // Tab indices (Unified Canvas Phase 5: Removed Pipeline tab)
    enum class Tab {
        Query = 0,
        Analyze = 1,
        Visualize = 2
    };

    // Rendering helpers
    void RenderToolbar();
    void RenderDatasetSelector();
    void RenderTabBar();
    void RenderStatusBar();
};

} // namespace cyxwiz
