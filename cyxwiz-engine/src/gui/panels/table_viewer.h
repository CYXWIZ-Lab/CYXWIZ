#pragma once

#include "../panel.h"
#include "../../data/data_table.h"
#include "../variables_view.h"
#include "../plot/plot_window.h"
#include <imgui.h>
#include <cstring>
#include <string>
#include <memory>
#include <vector>
#include <map>

namespace cyxwiz {

// Forward declaration
class VisualizationPanel;

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

struct DatasetProfile;

/**
 * A column's kind and range, for drawing the grid: sorting, the header icon,
 * number formats, data bars and colour maps. The column statistics in the
 * Stats sidebar are the column profile (core/dataset_profiler, TOFIX148).
 */
struct TableColumnKind {
    std::string type = "Unknown";  // "Numeric" (any number in it) or "Text"
    double min_val = 0, max_val = 0;
};

/**
 * TableViewer Panel - Display already-materialized tabular data.
 * Source-file and registered-dataset previews flow through Data Preview.
 * Supports multiple tabs for in-memory DataTable instances.
 */
class TableViewerPanel : public Panel {
public:
    TableViewerPanel();
    ~TableViewerPanel() override;

    void Render() override;

    // Set table to display (opens in new tab)
    void SetTable(std::shared_ptr<DataTable> table);

    // Close current tab
    void CloseCurrentTab();
    void CloseTab(int index);
    void CloseAllTabs();

    // Check if file is already open
    bool IsFileOpen(const std::string& filepath) const;
    void FocusTab(const std::string& filepath);

    // Clear current table
    void Clear();

    // A loaded dataset's first `max_rows` rows in a new tab (TOFIX140), read
    // through the data service: the dataset stays where its Data Input put it
    // (in memory or on disk); false + `error` when it cannot be read.
    bool OpenDataset(const std::string& dataset_name, int64_t max_rows, std::string& error);

    // Data Viewer (TOFIX133 P5, board 10): a tab fed from a Python value. A
    // second open of the same value replaces its tab; "Read again", a slice
    // change or "Read all rows" read it again through the variables worker.
    void SetScriptingEngine(scripting::ScriptingEngine* engine) { engine_ = engine; }
    void OpenVariable(const VariablesView::OpenRequest& request, const scripting::VariablesService::Result& result);

    // Visualization integration
    void SetVisualizationPanel(VisualizationPanel* viz) { visualization_panel_ = viz; }
    void SendToVisualizer(int column = -1);  // -1 = all columns, otherwise specific column

private:
    // Tab structure
    struct TableTab {
        // Stable tab identity: two tabs may show the same name (the same
        // table opened twice), and ImGui keys tabs by this, not the label.
        static int NewUid() { static int next = 0; return ++next; }
        const int uid = NewUid();
        std::string filename;           // Display name
        std::string filepath;           // Full path
        std::shared_ptr<DataTable> table;
        int current_page = 0;
        std::string filter_text;
        char filter_buffer[256] = {0};

        // Column kinds (grid drawing) and sorting
        std::vector<TableColumnKind> column_kinds;
        // The column profile for the Stats sidebar, computed in the
        // background; stale after an edit.
        std::shared_ptr<DatasetProfile> profile;
        bool profile_stale = true;
        uint64_t profile_task = 0;
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

        // ═══════════════════════════════════════════════════════════
        // NEW: Cell editing support
        // ═══════════════════════════════════════════════════════════
        bool is_dirty = false;              // Has unsaved changes
        int editing_row = -1;               // Row being edited (-1 = none)
        int editing_col = -1;               // Column being edited (-1 = none)
        char edit_buffer[1024] = {0};       // Text buffer for editing
        bool edit_just_started = false;     // Flag to focus input on first frame

        // A tab fed from a Python value (Data Viewer, board 10).
        struct Live {
            bool on = false;
            VariablesView::OpenRequest request;
            std::string kind;                  // frame, array, list
            std::vector<long long> shape;
            long long rows = 0;                // in the value
            long long shown = 0;               // read into this tab
            std::vector<std::string> dtypes;   // per table column ("index" for a frame's index)
            std::vector<int> slice;
            std::string read_at;
            std::string problem;               // why the last read again failed (the data shown stays)
            bool reading = false;
        } live;

        TableTab() {
            std::memset(filter_buffer, 0, sizeof(filter_buffer));
            std::memset(edit_buffer, 0, sizeof(edit_buffer));
        }

        // Unified accessors for the retained in-memory table display path.
        size_t GetRowCount() const {
            return table ? table->GetRowCount() : 0;
        }

        size_t GetColumnCount() const {
            return table ? table->GetColumnCount() : 0;
        }

        const std::vector<std::string>& GetHeaders() const {
            static std::vector<std::string> empty;
            return table ? table->GetHeaders() : empty;
        }

        std::string GetCellAsString(size_t row, size_t col) {
            return table ? table->GetCellAsString(row, col) : "";
        }

        DataTable::CellValue GetCell(size_t row, size_t col) {
            return table ? table->GetCell(row, col) : std::monostate{};
        }

        bool HasData() const {
            return table && table->GetRowCount() > 0;
        }
    };

    void RenderTabBar();
    void RenderLiveHeader(TableTab* tab);
    void ReadLive(TableTab* tab, std::vector<int> index, long long max_rows);
    void RenderToolbar();
    void RenderTable();
    void RenderStatusBar();
    void RenderStatsSidebar(TableTab* tab);
    // No tab: how tables get here, and the loaded-dataset picker.
    void RenderEmptyState();
    void RenderOpenDatasetPopup();
    bool open_dataset_popup_requested_ = false;
    std::string open_dataset_pick_;
    int open_dataset_rows_ = 10000;
    std::string open_dataset_error_;

    // Column kinds, the profile and sorting
    void ComputeColumnKinds(TableTab* tab);
    void StartProfile(TableTab* tab);
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
    // Plot window (TOFIX134 P1: replaces Quick Plot)
    // ═══════════════════════════════════════════════════════════════
    // Opens the Plot window on `tab` with `kind`; x/y are column indices
    // (-1: none).
    void OpenPlot(TableTab* tab, plot::Kind kind, int x_column, int y_column = -1);
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

    // ═══════════════════════════════════════════════════════════════
    // NEW: Number formatting
    // ═══════════════════════════════════════════════════════════════
    std::string FormatNumber(double value, const ColumnFormat& fmt);

    // ═══════════════════════════════════════════════════════════════
    // NEW: Export and Find dialogs
    // ═══════════════════════════════════════════════════════════════
    void RenderExportDialog();
    void RenderFindDialog();

    // ═══════════════════════════════════════════════════════════════
    // NEW: Cell editing
    // ═══════════════════════════════════════════════════════════════
    void BeginCellEdit(TableTab* tab, int row, int col);
    void EndCellEdit(TableTab* tab, bool save);
    void SaveTable(TableTab* tab);
    bool HasUnsavedChanges() const;

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
    std::shared_ptr<int> alive_ = std::make_shared<int>(0);  // scopes the profile tasks
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
    // Plot window
    // ═══════════════════════════════════════════════════════════════
    std::unique_ptr<plot::PlotWindow> plot_window_;

    // ═══════════════════════════════════════════════════════════════
    // NEW: Dialog states
    // ═══════════════════════════════════════════════════════════════
    bool show_export_dialog_ = false;
    bool show_find_dialog_ = false;
    char find_buffer_[256] = {0};
    char replace_buffer_[256] = {0};

    // Visualization integration
    VisualizationPanel* visualization_panel_ = nullptr;

    scripting::ScriptingEngine* engine_ = nullptr;
    int select_tab_ = -1;  // a replaced tab to bring to the front next frame
};

} // namespace cyxwiz
