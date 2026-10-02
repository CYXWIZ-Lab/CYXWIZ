#include "table_viewer.h"

namespace cyxwiz {

// TOFIX134 P1 step 1.5: the Plot window replaces Quick Plot. Every Quick Plot
// entry point (column menu, Stats Plot, a variable opened to plot) opens it
// with the plot type and column; the window keeps Plot with Python and Open
// in Visualizer.
void TableViewerPanel::OpenPlot(TableTab* tab, plot::Kind kind, int x_column, int y_column) {
    if (!tab || !tab->table) return;
    if (!plot_window_) {
        plot_window_ = std::make_unique<plot::PlotWindow>("table_viewer_plot");
        plot_window_->on_open_visualizer = [this](int column) { SendToVisualizer(column); };
    }
    const auto& headers = tab->table->GetHeaders();
    const auto name = [&](int c) {
        return c >= 0 && c < static_cast<int>(headers.size()) ? headers[static_cast<size_t>(c)] : std::string();
    };
    plot::PlotSpec spec;
    spec.kind = kind;
    const plot::KindInfo& info = plot::Info(kind);
    if (kind == plot::Kind::Box || kind == plot::Kind::Violin ||
        ((info.required & plot::kEncY) && !(info.required & plot::kEncX) && y_column < 0)) {
        // Values-only kinds (line of a column, box): the column is the Y.
        if (!name(x_column).empty()) spec.y_columns = {name(x_column)};
    } else {
        spec.x_column = name(x_column);
        if (!name(y_column).empty()) spec.y_columns = {name(y_column)};
    }
    // A live variable read with a row limit says so on the plot.
    size_t limit = 0, total = 0;
    if (tab->live.on && tab->live.shown >= 0 && tab->live.shown < tab->live.rows) {
        limit = static_cast<size_t>(tab->live.shown);
        total = static_cast<size_t>(tab->live.rows);
    }
    plot_window_->Open(tab->filename, tab->table, std::move(spec), limit, total);
}

}  // namespace cyxwiz
