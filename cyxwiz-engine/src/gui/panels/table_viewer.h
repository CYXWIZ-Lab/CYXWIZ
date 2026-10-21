#pragma once

#include "../panel.h"
#include "../table_grid.h"
#include "../../core/dataset_table_slice.h"
#include "../../data/data_table.h"
#include "../variables_view.h"
#include "../plot/plot_window.h"
#include <imgui.h>
#include <string>
#include <memory>
#include <vector>
#include <map>

namespace arrow {
class Table;
}

namespace cyxwiz {

// Forward declaration

struct DatasetProfile;

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

    // A loaded dataset's first `max_rows` rows in a new tab (TOFIX140). Read
    // in a task (TOFIX159: shown in the Task viewer, with progress and
    // Cancel; the viewer shows it too), so the Engine keeps drawing; the tab
    // opens when the rows are read. The dataset stays where its Data Input
    // put it (in memory or on disk).
    void OpenDataset(const std::string& dataset_name, int64_t max_rows);
    // The first `max_rows` rows (all when < 0) of an Arrow table (a step's
    // output, a query result) in a tab named `name`, converted in a task the
    // same way.
    void OpenArrow(std::shared_ptr<arrow::Table> table, const std::string& name, int64_t max_rows = -1);

    // Data Viewer (TOFIX133 P5, board 10): a tab fed from a Python value. A
    // second open of the same value replaces its tab; "Read again", a slice
    // change or "Read all rows" read it again through the variables worker.
    void SetScriptingEngine(scripting::ScriptingEngine* engine) { engine_ = engine; }
    void OpenVariable(const VariablesView::OpenRequest& request, const scripting::VariablesService::Result& result);

    // Visualization integration

private:
    // Tab structure
    struct TableTab {
        // Stable tab identity: two tabs may show the same name (the same
        // table opened twice), and ImGui keys tabs by this, not the label.
        static int NewUid() { static int next = 0; return ++next; }
        const int uid = NewUid();
        std::string filename;           // Display name
        std::string filepath;           // Full path
        // The table and its view: sort, filter, selection, formats, edit.
        TableGrid grid;
        bool is_dirty = false;  // Has unsaved changes
        // The rows as read from Arrow (OpenDataset, OpenArrow): the Stats
        // profile reads them while the table is unedited, so opening a large
        // table converts nothing on the UI thread.
        std::shared_ptr<arrow::Table> source;
        // The loaded dataset this tab shows whole (OpenDataset of all its
        // rows): while unedited it is profiled as Data Studio profiles it,
        // so either one's saved answers serve the other.
        std::string dataset;
        // The column profile for the Stats sidebar, computed in the
        // background; stale after an edit.
        std::shared_ptr<DatasetProfile> profile;
        bool profile_stale = true;
        uint64_t profile_task = 0;

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
            grid.options.editable = true;
            grid.options.column_stats = true;
            grid.on_edited = [this] {
                is_dirty = true;
                profile_stale = true;
            };
        }
        TableTab(const TableTab&) = delete;
        TableTab& operator=(const TableTab&) = delete;

        size_t GetRowCount() const { return grid.table ? grid.table->GetRowCount() : 0; }
        size_t GetColumnCount() const { return grid.table ? grid.table->GetColumnCount() : 0; }
        const std::vector<std::string>& GetHeaders() const {
            static std::vector<std::string> empty;
            return grid.table ? grid.table->GetHeaders() : empty;
        }
        bool HasData() const { return grid.HasData(); }
    };

    // A tab on `table`; the grid's Plot items open the Plot window on it.
    // `kinds` and `source` come with a table read in a task.
    std::unique_ptr<TableTab> MakeTab(std::shared_ptr<DataTable> table, std::string filename,
                                      std::vector<TableColumnKind> kinds = {},
                                      std::shared_ptr<arrow::Table> source = nullptr);
    TableTab* AddTab(std::shared_ptr<DataTable> table, std::vector<TableColumnKind> kinds,
                     std::shared_ptr<arrow::Table> source);
    // Reads (OpenDataset, OpenArrow) running in tasks, drawn under the
    // toolbar with their progress and Cancel.
    struct Opening {
        uint64_t task = 0;
        std::string name;
        std::shared_ptr<const void> token;
    };
    std::vector<Opening> opening_;
    // `dataset`: the loaded dataset read (OpenDataset), "" for a table.
    void StartOpen(const std::string& name, const std::string& dataset,
                   std::function<DatasetSlice(const SliceProgress&)> read);
    void RenderOpening();

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

    // The column profile for the Stats sidebar
    void StartProfile(TableTab* tab);

    // ═══════════════════════════════════════════════════════════════
    // Plot window (TOFIX134 P1: replaces Quick Plot)
    // ═══════════════════════════════════════════════════════════════
    // Opens the Plot window on `tab` with `kind`; x/y are column indices
    // (-1: none).
    void OpenPlot(TableTab* tab, plot::Kind kind, int x_column, int y_column = -1);
    void RenderMiniHistogram(TableTab* tab, int column);

    // Export: a menu on the toolbar's Export (CSV, TSV, JSON, Parquet to a
    // file, or CSV to the clipboard); Find: a window over the grid's filter.
    void RenderExportMenu(TableTab* tab);
    void ExportTable(TableTab* tab, int format);  // 0 CSV, 1 TSV, 2 JSON, 3 Parquet, 4 clipboard (CSV)
    void RenderFindDialog();

    // Saving an edited table to its file
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

    // Data bars (the grids' option)
    bool show_data_bars_ = true;

    // ═══════════════════════════════════════════════════════════════
    // Plot window
    // ═══════════════════════════════════════════════════════════════
    std::unique_ptr<plot::PlotWindow> plot_window_;

    // Export and Find
    bool export_shown_rows_only_ = false;  // the grid's rows (sorted, filtered) instead of the table
    bool export_headers_ = true;           // CSV / TSV: the column names first
    std::string status_note_;              // what the last export did, in the status bar
    bool status_note_error_ = false;
    bool show_find_dialog_ = false;
    char find_buffer_[256] = {0};
    int find_cells_ = -1;                  // cells matching (-1: nothing searched)
    size_t find_rows_ = 0;

    // Visualization integration

    scripting::ScriptingEngine* engine_ = nullptr;
    int select_tab_ = -1;  // a replaced tab to bring to the front next frame
};

} // namespace cyxwiz
