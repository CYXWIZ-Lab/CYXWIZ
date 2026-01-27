#pragma once

#include "../panel.h"
#include "../../data/data_table.h"
#include "../../core/async_task_manager.h"
#include <imgui.h>
#include <string>
#include <memory>
#include <vector>
#include <atomic>
#include <mutex>
#include <map>

namespace cyxwiz {

// ============================================================================
// Colormap Types for conditional formatting
// ============================================================================
enum class ColorMapType {
    None,       // No colormap
    Viridis,    // Purple → Blue → Green → Yellow
    Inferno,    // Black → Magenta → Orange → Yellow
    RdYlGn,     // Red → Yellow → Green (diverging)
    Blues,      // Light blue → Dark blue
    Plasma,     // Purple → Pink → Orange → Yellow
    Magma,      // Black → Purple → Pink → White
    Custom      // User-defined range
};

// ============================================================================
// Quick Plot Types
// ============================================================================
enum class QuickPlotType {
    // Basic charts
    Histogram,      // Distribution of single column
    Bar,            // Categorical/value bars
    Scatter,        // Two columns X vs Y
    Line,           // Trend/sequence
    Box,            // Quartiles, outliers

    // Extended charts
    Pie,            // Category proportions
    Stairs,         // Step function
    Stem,           // Discrete values
    Area,           // Filled line chart

    // Advanced charts
    Heatmap,        // Matrix/grid data
    Histogram2D     // 2D density
};

// ============================================================================
// Column colormap configuration
// ============================================================================
struct ColumnColorMap {
    ColorMapType type = ColorMapType::None;
    bool apply_to_background = true;  // vs text color
    float min_val = 0, max_val = 1;   // Custom range (auto-computed if not set)
    bool auto_range = true;           // Use column min/max
};

// ============================================================================
// Number formatting for display
// ============================================================================
struct ColumnFormat {
    int decimal_places = -1;          // -1 = auto
    bool thousands_separator = false;
    bool as_percentage = false;
    std::string prefix;               // e.g., "$"
    std::string suffix;               // e.g., "%"
};

// ============================================================================
// Selection range for multi-selection
// ============================================================================
struct SelectionRange {
    int row_start = -1, row_end = -1;  // Inclusive
    int col_start = -1, col_end = -1;  // Inclusive

    bool Contains(int row, int col) const {
        return row >= row_start && row <= row_end &&
               col >= col_start && col <= col_end;
    }
};

// ============================================================================
// Quick plot popup state
// ============================================================================
struct PlotPopup {
    bool open = false;
    QuickPlotType type = QuickPlotType::Histogram;
    int x_column = -1;
    int y_column = -1;
    std::string title;
    std::vector<double> x_data;
    std::vector<double> y_data;
    std::vector<std::string> labels;  // For categorical data
};

/**
 * Column statistics for Table Viewer data analysis
 */
struct TableColumnStats {
    std::string type = "Unknown";  // "Numeric", "Text", "Date", "Mixed"
    size_t count = 0;
    size_t null_count = 0;
    double min_val = 0, max_val = 0;
    double sum = 0, avg = 0;
    double std_dev = 0;                 // Standard deviation
    double median = 0;                  // Median value
    double q1 = 0, q3 = 0;              // Quartiles
    std::vector<std::pair<std::string, int>> top_values;  // For text columns
    std::vector<int> histogram_bins;    // For mini histogram (16 bins)
    bool computed = false;
};

/**
 * TableViewer Panel - Display tabular data from CSV, Excel, HDF5
 * Uses ImGui tables for efficient rendering of large datasets
 * Supports multiple tabs and async loading
 */
class TableViewerPanel : public Panel {
public:
    TableViewerPanel();
    ~TableViewerPanel() override = default;

    void Render() override;

    // Load data from files (async by default for large files)
    bool LoadCSV(const std::string& filepath);
    bool LoadTXT(const std::string& filepath, char delimiter = '\t');
    bool LoadHDF5(const std::string& filepath, const std::string& dataset_name = "data");
    bool LoadExcel(const std::string& filepath, const std::string& sheet_name = "");

    // Set table to display (opens in new tab)
    void SetTable(std::shared_ptr<DataTable> table);
    void SetTableByName(const std::string& name);

    // Close current tab
    void CloseCurrentTab();
    void CloseTab(int index);
    void CloseAllTabs();

    // Check if file is already open
    bool IsFileOpen(const std::string& filepath) const;
    void FocusTab(const std::string& filepath);

    // Clear current table
    void Clear();

private:
    // Tab structure
    struct TableTab {
        std::string filename;           // Display name
        std::string filepath;           // Full path
        std::shared_ptr<DataTable> table;
        int current_page = 0;
        std::string filter_text;
        char filter_buffer[256] = {0};
        bool is_loading = false;        // Async loading in progress
        float load_progress = 0.0f;
        std::string load_status;

        // Column statistics and sorting
        std::vector<TableColumnStats> column_stats;
        int selected_column = -1;
        int sort_column = -1;
        bool sort_ascending = true;
        std::vector<size_t> sorted_indices;  // For virtual sorting

        // Cell selection (per-tab)
        int selected_row = -1;
        int selected_col = -1;

        // ═══════════════════════════════════════════════════════════
        // NEW: Multi-selection support
        // ═══════════════════════════════════════════════════════════
        std::vector<SelectionRange> selections;

        // ═══════════════════════════════════════════════════════════
        // NEW: True filtering (hide rows vs highlight)
        // ═══════════════════════════════════════════════════════════
        std::vector<size_t> filtered_indices;  // Subset of sorted_indices
        bool filter_mode_hide = false;         // true = hide non-matching rows

        // ═══════════════════════════════════════════════════════════
        // NEW: Per-column colormaps
        // ═══════════════════════════════════════════════════════════
        std::vector<ColumnColorMap> column_colormaps;

        // ═══════════════════════════════════════════════════════════
        // NEW: Per-column formatting
        // ═══════════════════════════════════════════════════════════
        std::vector<ColumnFormat> column_formats;

        // ═══════════════════════════════════════════════════════════
        // NEW: Column freeze for horizontal scrolling
        // ═══════════════════════════════════════════════════════════
        int frozen_columns = 0;

        TableTab() {
            std::memset(filter_buffer, 0, sizeof(filter_buffer));
        }
    };

    void RenderTabBar();
    void RenderToolbar();
    void RenderTable();
    void RenderStatusBar();
    void RenderLoadingIndicator();
    void RenderStatsSidebar(TableTab* tab);

    // Column statistics and sorting
    void ComputeColumnStats(TableTab* tab);
    void SortByColumn(TableTab* tab, int column);

    // ═══════════════════════════════════════════════════════════════
    // NEW: Context menus
    // ═══════════════════════════════════════════════════════════════
    void RenderColumnContextMenu(TableTab* tab, int column);
    void RenderCellContextMenu(TableTab* tab, int row, int col);

    // ═══════════════════════════════════════════════════════════════
    // NEW: Colormaps
    // ═══════════════════════════════════════════════════════════════
    void ApplyColorMap(TableTab* tab, int column, ColorMapType type);
    void ClearColorMap(TableTab* tab, int column);
    ImVec4 GetColorMapColor(double normalized, ColorMapType type) const;

    // ═══════════════════════════════════════════════════════════════
    // NEW: Filtering
    // ═══════════════════════════════════════════════════════════════
    void ApplyFilter(TableTab* tab);
    void ClearFilter(TableTab* tab);

    // ═══════════════════════════════════════════════════════════════
    // NEW: Quick Plot
    // ═══════════════════════════════════════════════════════════════
    void ShowQuickPlotPopup(TableTab* tab);
    void RenderQuickPlot();
    void RenderMiniHistogram(TableTab* tab, int column);
    std::vector<double> GetColumnAsDoubles(TableTab* tab, int column) const;

    // ═══════════════════════════════════════════════════════════════
    // NEW: Clipboard operations
    // ═══════════════════════════════════════════════════════════════
    void CopyToClipboard(const std::string& text);
    void CopyCellToClipboard(TableTab* tab, int row, int col);
    void CopyRowToClipboard(TableTab* tab, int row);
    void CopyColumnToClipboard(TableTab* tab, int col);
    void CopySelectionToClipboard(TableTab* tab);

    // Async loading helpers
    void LoadFileAsync(const std::string& filepath, const std::string& type, char delimiter = ',');
    void OnLoadComplete(int tab_index, std::shared_ptr<DataTable> table, bool success, const std::string& error);

    // Tab management
    int FindTabByPath(const std::string& filepath) const;
    TableTab* GetActiveTab();
    const TableTab* GetActiveTab() const;

    // Tabs
    std::vector<std::unique_ptr<TableTab>> tabs_;
    int active_tab_index_ = -1;
    int close_tab_index_ = -1;  // Tab to close after render (-1 = none)

    // UI state
    bool show_line_numbers_ = true;
    int rows_per_page_ = 100;

    // Stats sidebar
    bool show_stats_sidebar_ = true;
    float stats_sidebar_width_ = 180.0f;

    // Data bars
    bool show_data_bars_ = true;
    float data_bar_alpha_ = 0.3f;

    // ═══════════════════════════════════════════════════════════════
    // NEW: Context menu state
    // ═══════════════════════════════════════════════════════════════
    int context_menu_column_ = -1;
    int context_menu_row_ = -1;
    int context_menu_col_ = -1;

    // ═══════════════════════════════════════════════════════════════
    // NEW: Quick plot popup
    // ═══════════════════════════════════════════════════════════════
    PlotPopup plot_popup_;
    bool show_plot_popup_ = false;

    // ═══════════════════════════════════════════════════════════════
    // NEW: Dialog states
    // ═══════════════════════════════════════════════════════════════
    bool show_export_dialog_ = false;
    bool show_find_dialog_ = false;
    char find_buffer_[256] = {0};
    char replace_buffer_[256] = {0};

    // Async loading
    std::mutex tabs_mutex_;
    std::atomic<bool> has_pending_load_{false};
};

} // namespace cyxwiz
