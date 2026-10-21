#pragma once

// The Table Viewer's grid as a view (TOFIX148): one in-memory DataTable with
// sorting, filtering, cell and range selection, number formats, data bars,
// colour maps, frozen columns, copy and (when editable) cell editing. The
// Table Viewer panel adds tabs, files, the Stats sidebar and its toolbar; the
// SQL Workbench and Data Studio embed the grid for their results.

#include "../core/plot/plot_model.h"
#include "../core/table_grid_rows.h"
#include "../data/data_table.h"

#include <imgui.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
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
    ~TableGrid();
    TableGrid(const TableGrid&) = delete;
    TableGrid& operator=(const TableGrid&) = delete;

    // A fresh view of `t`: no sort, filter, selection, formats or edit.
    // `kinds` (TableColumnKindsOf, worked out off the UI thread) saves the
    // grid working them out on its first frame.
    void SetTable(std::shared_ptr<DataTable> t, std::vector<TableColumnKind> kinds = {});
    bool HasData() const { return table && table->GetRowCount() > 0; }

    // The grid and, past one page, its pager, in `size` (0: the space left).
    void Render(const char* id, const ImVec2& size = ImVec2(0.0f, 0.0f));

    // The rows in display order: sorted, and filtered when hiding non-matches.
    const std::vector<size_t>& DisplayIndices() const;

    void ComputeColumnKinds();

    // Sorting, filtering and the Find counts read every row: a small table in
    // the frame, a large one in a task (the Task viewer lists it) while the
    // grid keeps its current rows and shows the progress with Cancel.
    // Sort by `column` (-1: the table's own order, reversed when descending).
    void SetOrder(int column, bool ascending);
    // The filter text: highlighted, or with Hide the only rows shown.
    void SetFilter(const std::string& text);
    // The case and whole-cell options changed.
    void RefreshFilter();
    void SetHide(bool hide);
    void ClearFilter();
    // Hide asked for (the rows shown follow when the work is done).
    bool Hiding() const { return filter_mode_hide && !filter_text.empty(); }
    // The rows shown are the filtered ones.
    bool ShowingFiltered() const { return showing_filtered_; }
    // Whether a cell's text matches the filter (its case and whole-cell options).
    bool Matches(const std::string& text) const { return filter_.Matches(text); }
    // Sorting or filtering is running; Cancel keeps the rows as they are.
    bool Busy() const { return order_task_ != 0; }
    void CancelWork();
    // While the returned token lives (a task reading the table: an export),
    // cells cannot be edited; an edit in progress is saved first.
    std::shared_ptr<void> HoldTable();
    bool TableHeld() const { return Busy() || holds_->load() > 0; }
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
    // A table wider than the grid holds (ImGui tables take 512 columns)
    // shows its frozen columns and one block of the others; the column
    // pager under the grid moves between blocks.
    static constexpr int kColumnBlock = 500;
    int column_page = 0;
    int drawn_column_page_ = -1;  // the block drawn last frame

    // Filter: the toolbar's text, highlighted or (hide) the only rows shown.
    std::string filter_text;
    char filter_buffer[256] = {0};
    bool filter_mode_hide = false;
    bool filter_case_sensitive = false;
    bool filter_whole_cell = false;       // the whole cell, not a part of it
    std::vector<size_t> filtered_indices;  // Subset of sorted_indices
    // Find: count the matching cells and rows (every cell read); the counts
    // of the last filter, -1 while not counted.
    bool count_matches = false;
    int64_t match_cells = -1;
    int64_t match_rows = 0;

    std::vector<TableColumnKind> column_kinds;
    // The order asked for; sorted_indices follow when the work is done.
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

    // The rows again: the order too when `reorder`, else only the filter
    // work on the current order.
    void Requery(bool reorder);
    void ApplyRows(TableRowOrder rows, bool filter_rows, bool counted, int sort_column, bool ascending);
    void RenderWorkLine();
    TableFilter filter_;
    bool showing_filtered_ = false;
    // The order the rows are in (sort_column / sort_ascending are asked for).
    int applied_sort_column_ = -1;
    bool applied_sort_ascending_ = true;
    // The header's sort arrow is set to sort_column on the next frame (a
    // menu sort, a cancelled one).
    bool sync_sort_header_ = false;
    uint64_t order_task_ = 0;
    bool order_task_reorders_ = false;
    std::string order_task_label_;
    uint64_t order_generation_ = 0;
    std::shared_ptr<int> alive_ = std::make_shared<int>(0);
    std::shared_ptr<std::atomic<int>> holds_ = std::make_shared<std::atomic<int>>(0);
    void RenderColumnContextMenu(int column);
    void RenderCellContextMenu(int row, int col);
};

}  // namespace cyxwiz
