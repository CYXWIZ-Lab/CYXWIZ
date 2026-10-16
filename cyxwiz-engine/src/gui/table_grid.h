#pragma once

// The Table Viewer's grid as a view (TOFIX148): one in-memory DataTable with
// sorting, filtering, cell and range selection, number formats, data bars,
// colour maps, frozen columns, copy and (when editable) cell editing. The
// Table Viewer panel adds tabs, files, the Stats sidebar and its toolbar; the
// SQL Workbench and Data Studio embed the grid for their results.

#include "../core/plot/plot_model.h"
#include "../data/data_table.h"

#include <imgui.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

// Colour maps for conditional formatting.
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

struct ColumnColorMap {
    ColorMapType type = ColorMapType::None;
    bool apply_to_background = true;  // vs text color
    float min_val = 0, max_val = 1;   // Custom range (auto-computed if not set)
    bool auto_range = true;           // Use column min/max
};

// Number formatting for display.
struct ColumnFormat {
    int decimal_places = -1;          // -1 = auto
    bool thousands_separator = false;
    bool as_percentage = false;
    std::string prefix;               // e.g., "$"
    std::string suffix;               // e.g., "%"
};

// A selected range of cells (inclusive).
struct SelectionRange {
    int row_start = -1, row_end = -1;
    int col_start = -1, col_end = -1;

    bool Contains(int row, int col) const {
        return row >= row_start && row <= row_end &&
               col >= col_start && col <= col_end;
    }
};

/**
 * A column's kind and range, for drawing the grid: sorting, the header icon,
 * number formats, data bars and colour maps. Column statistics are the column
 * profile (core/dataset_profiler, TOFIX148).
 */
struct TableColumnKind {
    std::string type = "Unknown";  // "Numeric" (any number in it) or "Text"
    double min_val = 0, max_val = 0;
};

class TableGrid {
public:
    struct Options {
        bool line_numbers = true;
        bool data_bars = true;
        float data_bar_alpha = 0.3f;
        int rows_per_page = 100;
        bool editable = false;      // a double-click edits a cell
        bool column_stats = false;  // the cell menu's "View Column Statistics" (a Stats sidebar beside the grid)
    };

    TableGrid() = default;
    TableGrid(const TableGrid&) = delete;
    TableGrid& operator=(const TableGrid&) = delete;

    // A fresh view of `t`: no sort, filter, selection, formats or edit.
    void SetTable(std::shared_ptr<DataTable> t);
    bool HasData() const { return table && table->GetRowCount() > 0; }

    // The grid and, past one page, its pager, in `size` (0: the space left).
    void Render(const char* id, const ImVec2& size = ImVec2(0.0f, 0.0f));

    // The rows in display order: sorted, and filtered when hiding non-matches.
    const std::vector<size_t>& DisplayIndices() const;

    void ComputeColumnKinds();
    void SortByColumn(int column);
    // The filter text: highlighted, or with filter_mode_hide the only rows shown.
    void SetFilter(const std::string& text);
    void ApplyFilter();
    void ClearFilter();
    bool Hiding() const { return filter_mode_hide && !filter_text.empty(); }
    // Whether a cell's text matches the filter (its case and whole-cell options).
    bool Matches(const std::string& text) const;
    void ApplyColorMap(int column, ColorMapType type);
    void ClearColorMap(int column);
    static ImVec4 GetColorMapColor(double normalized, ColorMapType type);
    static std::string FormatNumber(double value, const ColumnFormat& fmt);

    void BeginCellEdit(int row, int col);
    void EndCellEdit(bool save);
    bool IsEditing() const { return editing_row >= 0 && editing_col >= 0; }

    // A cell as text, "" when empty: a number in its shortest exact form
    // (106.4), or to 10 significant digits when that is long (2/3).
    std::string CellText(size_t row, size_t col) const;

    void CopyCellToClipboard(int row, int col) const;
    void CopyRowToClipboard(int row) const;
    void CopyColumnToClipboard(int col) const;
    void CopySelectionToClipboard() const;

    Options options;
    // A cell changed (editable grids); the column kinds already follow it.
    std::function<void()> on_edited;
    // The column menu's Plot items; hidden when not set.
    std::function<void(int column, plot::Kind kind)> on_plot;
    // A caption under each column name (a Python value's dtypes); "index"
    // draws none.
    std::vector<std::string> captions;

    std::shared_ptr<DataTable> table;
    int current_page = 0;

    // Filter: the toolbar's text, highlighted or (hide) the only rows shown.
    std::string filter_text;
    char filter_buffer[256] = {0};
    bool filter_mode_hide = false;
    bool filter_case_sensitive = false;
    bool filter_whole_cell = false;       // the whole cell, not a part of it
    std::vector<size_t> filtered_indices;  // Subset of sorted_indices

    std::vector<TableColumnKind> column_kinds;
    int sort_column = -1;
    bool sort_ascending = true;
    std::vector<size_t> sorted_indices;    // For virtual sorting

    // The column the Stats sidebar shows, the selected cell, Ctrl / Shift ranges.
    int selected_column = -1;
    int selected_row = -1;
    int selected_col = -1;
    std::vector<SelectionRange> selections;

    std::vector<ColumnColorMap> column_colormaps;
    std::vector<ColumnFormat> column_formats;
    int frozen_columns = 0;

    // Cell editing
    int editing_row = -1;
    int editing_col = -1;
    char edit_buffer[1024] = {0};
    bool edit_just_started = false;  // focus the input on its first frame

private:
    // Text columns open no wider than this; a long cell shows its start.
    static constexpr float kMaxInitialWidth = 420.0f;
    std::vector<float> initial_widths_;  // per column, 0 = fit the content
    void ComputeInitialWidths();
    void RenderColumnContextMenu(int column);
    void RenderCellContextMenu(int row, int col);
};

}  // namespace cyxwiz
