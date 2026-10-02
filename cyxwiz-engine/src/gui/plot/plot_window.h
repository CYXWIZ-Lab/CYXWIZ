#pragma once

// The Plot window (TOFIX134 P1 step 1.5, approved board 1): a plot-type
// list, one PlotView, and the data and encodings of the plot. Replaces the
// Table Viewer's Quick Plot; later it hosts the Plot node (P2) and grows into
// the Dashboard (P3). Data is prepared off the UI thread.

#include "plot_view.h"
#include "../../core/plot/plot_prepare.h"

#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>

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

    void Render();
    bool visible = false;

    // Optional: "Open in Visualizer" with the X column's index.
    std::function<void(int column)> on_open_visualizer;

private:
    void Rebuild();
    void Poll();
    void DrawKinds();
    void DrawSettings();
    std::string PythonScript() const;
    int ColumnIndex(const std::string& name) const;

    std::string id_;
    std::string source_name_;
    std::shared_ptr<DataTable> table_;
    std::vector<std::string> headers_;
    std::vector<bool> numeric_;
    size_t row_limit_ = 0, total_rows_ = 0;
    PlotSpec spec_;
    PlotView view_;
    std::future<Prepared> job_;
    bool busy_ = false;
    bool dirty_ = false;
    char title_buf_[256] = {};
};

}  // namespace cyxwiz::plot
