#pragma once

// The Plot window (TOFIX134 P1 step 1.5, approved board 1): a plot-type
// list, one PlotView, and the data and encodings of the plot. Replaces the
// Table Viewer's Quick Plot; later it hosts the Plot node (P2) and grows into
// the Dashboard (P3). Data is prepared off the UI thread.

#include "column_picker.h"
#include "plot_view.h"
#include "../../core/plot/plot_prepare.h"

#include <array>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace arrow {
class Table;
}

namespace cyxwiz {
class DataTable;
}

namespace cyxwiz::plot {

class PlotWindow {
public:
    explicit PlotWindow(std::string id);
    ~PlotWindow();

    // Shows `table` (named `source_name`) with `spec`. row_limit > 0: the
    // table holds only the first row_limit of total_rows rows.
    void Open(const std::string& source_name, std::shared_ptr<DataTable> table, PlotSpec spec, size_t row_limit = 0,
              size_t total_rows = 0);
    // New data for the same source (a read again): the spec is kept.
    void SetTable(std::shared_ptr<DataTable> table, size_t row_limit = 0, size_t total_rows = 0);
    const std::string& SourceName() const { return source_name_; }

    // Plot node (TOFIX134 P2): an Arrow table, read off the UI thread. With
    // no columns chosen yet a first plot is picked (a label column's counts,
    // else the first numeric column's histogram).
    void SetArrowTable(const std::string& source_name, std::shared_ptr<arrow::Table> table, size_t row_limit = 0,
                       size_t total_rows = 0);
    // No data (not connected, not available, not read yet): the message
    // shows where the plot would be.
    void ClearData(const std::string& source_name, const std::string& message);
    void SetSpec(PlotSpec spec);
    const PlotSpec& Spec() const { return spec_; }
    // Drawn instead of the source line (the node's status and actions).
    std::function<void()> draw_header;
    // The plot type, columns or labels changed (saved in the node).
    std::function<void(const PlotSpec&)> on_spec_changed;

    void Render();
    bool visible = false;

    // Optional: "Open in Visualizer" with the X column's index.
    std::function<void(int column)> on_open_visualizer;

private:
    void Rebuild();
    void Poll();
    void DrawKinds();
    void DrawSettings();
    // ROWS section (TOFIX134 P2 board 6); true when the selection changed.
    bool DrawRows(float width);
    // Column names and types of the table, then their summaries (range,
    // share not 0, one value) for the column picker, read off the UI thread
    // for an Arrow table.
    void ReadColumns();
    size_t TableRows() const;
    std::string PythonScript() const;
    int ColumnIndex(const std::string& name) const;

    std::string id_;
    std::string source_name_;
    std::shared_ptr<DataTable> table_;
    std::shared_ptr<arrow::Table> arrow_table_;
    std::string empty_message_ = "No table. Open a table in the Table Viewer and choose Plot on a column.";
    std::vector<std::string> headers_;
    std::vector<bool> numeric_;
    std::vector<ColumnSummary> columns_;
    std::future<std::vector<ColumnSummary>> columns_job_;
    ColumnPicker picker_;
    // Text of each filter condition's value (spec_.conditions[i].value).
    std::vector<std::array<char, 128>> condition_values_;
    size_t row_limit_ = 0, total_rows_ = 0;
    PlotSpec spec_;
    PlotView view_;
    std::future<Prepared> job_;
    bool busy_ = false;
    bool dirty_ = false;
    char title_buf_[256] = {};
};

}  // namespace cyxwiz::plot
