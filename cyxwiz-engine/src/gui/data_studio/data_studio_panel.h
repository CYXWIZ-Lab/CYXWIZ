#pragma once

#include "pipeline_canvas.h"
#include "query_editor.h"
#include "analyzer.h"
#include "visualizer.h"
#include <imgui.h>
#include <memory>
#include <string>

// Forward declaration
namespace gui {
class MainWindow;
}

namespace cyxwiz {

/**
 * DataStudioPanel - Main container for Data Studio feature
 *
 * Phase 1, Week 1: Core Infrastructure
 *
 * This is the main panel that integrates all Data Studio components
 * into a unified tabbed interface. It provides a visual, no-code
 * data exploration and transformation experience similar to KNIME,
 * RapidMiner, and Orange.
 *
 * Architecture:
 *   DataStudioPanel (Container)
 *     ├─ PipelineCanvas (Visual pipeline builder)
 *     ├─ QueryEditor (SQL query interface)
 *     ├─ Analyzer (Statistical analysis)
 *     └─ Visualizer (Interactive plots)
 *
 * Tab Structure:
 *   - Pipeline: Visual node-based pipeline editor
 *   - Query: SQL query editor with DuckDB backend
 *   - Analyze: Statistical analysis and profiling
 *   - Visualize: Interactive data visualization
 *
 * Integration Points:
 *   - DataRegistry: Load datasets for analysis
 *   - DuckDBConnector: Execute SQL queries
 *   - ArrowDataset: Zero-copy data access
 *   - MainWindow: Dockable panel in main UI
 */
class DataStudioPanel {
public:
    DataStudioPanel();
    ~DataStudioPanel() = default;

    /**
     * Set MainWindow reference for Node Editor access (Phase 5 Week 7)
     * Must be called after construction
     */
    void SetMainWindow(gui::MainWindow* main_window) { main_window_ = main_window; }

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
    // Component instances
    std::unique_ptr<PipelineCanvas> pipeline_canvas_;
    std::unique_ptr<QueryEditor> query_editor_;
    std::unique_ptr<Analyzer> analyzer_;
    std::unique_ptr<Visualizer> visualizer_;

    // State
    std::string active_dataset_;
    int selected_tab_;
    bool visible_;

    // MainWindow reference for Node Editor handoff (Phase 5 Week 7)
    gui::MainWindow* main_window_ = nullptr;

    // Tab indices
    enum class Tab {
        Pipeline = 0,
        Query = 1,
        Analyze = 2,
        Visualize = 3
    };

    // Rendering helpers
    void RenderToolbar();
    void RenderDatasetSelector();
    void RenderTabBar();
    void RenderStatusBar();

    // Phase 5 Week 7 - Node Editor Handoff
    void OnDeployToNodeEditor();
};

} // namespace cyxwiz
