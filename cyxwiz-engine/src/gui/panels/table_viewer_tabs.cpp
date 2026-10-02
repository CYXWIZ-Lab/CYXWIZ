#include "table_viewer.h"
#include "../../core/variables_presentation.h"
#include <spdlog/spdlog.h>
#include <memory>

namespace cyxwiz {

void TableViewerPanel::SetTable(std::shared_ptr<DataTable> table) {
    if (!table) return;

    auto tab = std::make_unique<TableTab>();
    tab->filename = table->GetName();
    tab->filepath = "";  // In-memory table
    tab->table = table;

    tabs_.push_back(std::move(tab));
    active_tab_index_ = static_cast<int>(tabs_.size()) - 1;
}

TableViewerPanel::~TableViewerPanel() { VariablesView::CancelReads(this); }

void TableViewerPanel::OpenVariable(const VariablesView::OpenRequest& request,
                                    const scripting::VariablesService::Result& result) {
    if (!result.table) return;
    int index = -1;
    for (int i = 0; i < static_cast<int>(tabs_.size()); ++i) {
        const auto& live = tabs_[i]->live;
        if (live.on && live.request.scope.key == request.scope.key && live.request.path == request.path) index = i;
    }
    auto tab = std::make_unique<TableTab>();
    tab->filename = request.name + " \xC2\xB7 " + request.scope.label;
    tab->table = result.table;
    tab->live.on = true;
    tab->live.request = request;
    tab->live.request.plot = false;
    tab->live.kind = result.table_kind;
    tab->live.shape = result.shape;
    tab->live.rows = result.rows;
    tab->live.shown = result.shown;
    tab->live.dtypes = result.dtypes;
    tab->live.slice = result.slice;
    tab->live.read_at = ClockNow();
    // Stats start on the first data column, not a frame's index.
    if (result.table_kind == "frame" && result.table->GetColumnCount() > 1) tab->selected_column = 1;
    TableTab* opened = tab.get();
    if (index < 0) {
        tabs_.push_back(std::move(tab));
        index = static_cast<int>(tabs_.size()) - 1;
    } else {
        tabs_[index] = std::move(tab);
        select_tab_ = index;
    }
    active_tab_index_ = index;
    visible_ = true;
    if (request.plot) {
        // Plot: the first numeric column (not a frame's index), as a line for
        // a one-dimensional array, else a histogram.
        ComputeColumnStats(opened);
        const int first = result.table_kind == "frame" ? 1 : 0;
        for (int c = first; c < static_cast<int>(opened->column_stats.size()); ++c) {
            if (opened->column_stats[c].type != "Numeric") continue;
            const bool line = result.table_kind == "array" && result.shape.size() == 1;
            plot_popup_ = PlotPopup{};
            plot_popup_.type = line ? QuickPlotType::Line : QuickPlotType::Histogram;
            plot_popup_.x_column = c;
            plot_popup_.y_column = -1;
            const std::string column = opened->table->GetHeaders()[static_cast<size_t>(c)];
            plot_popup_.title = (line ? request.name : "Histogram of " + request.name + " " + column);
            plot_popup_.x_data = GetColumnAsDoubles(opened, c);
            show_plot_popup_ = true;
            break;
        }
    }
}

void TableViewerPanel::ReadLive(TableTab* tab, std::vector<int> index, long long max_rows) {
    if (!engine_ || !tab || !tab->live.on || tab->live.reading) return;
    VariablesView::OpenRequest request = tab->live.request;
    request.index = std::move(index);
    request.max_rows = max_rows;
    scripting::VariablesService::Request r;
    r.kind = scripting::VariablesService::Kind::Table;
    r.scope = request.scope.key;
    r.path_json = request.path;
    r.max_rows = request.max_rows;
    r.index = request.index;
    tab->live.reading = true;
    tab->live.problem.clear();
    VariablesView::Read(engine_, std::move(r), this, [this, request](const scripting::VariablesService::Result& result) {
        TableTab* t = nullptr;
        for (auto& candidate : tabs_)
            if (candidate->live.on && candidate->live.request.scope.key == request.scope.key &&
                candidate->live.request.path == request.path)
                t = candidate.get();
        if (t) t->live.reading = false;
        if (result.busy) {
            if (t) t->live.problem = "A run is active; read again when it finishes.";
            return;
        }
        if (!result.error.empty() || !result.table) {
            if (!t) return;
            const bool gone = result.error.rfind("This value is gone", 0) == 0;
            t->live.problem = gone ? request.name + " is no longer in " +
                                         (request.scope.label == "Python session" ? std::string("the Python session")
                                                                                  : request.scope.label) +
                                         " (Restart or del). This is the last read."
                                   : request.name + " could not be read: " + result.error;
            return;
        }
        OpenVariable(request, result);
    });
}

void TableViewerPanel::SetTableByName(const std::string& name) {
    auto table = DataTableRegistry::Instance().GetTable(name);
    if (table) {
        SetTable(table);
    } else {
        spdlog::warn("Table not found in registry: {}", name);
    }
}

void TableViewerPanel::CloseCurrentTab() {
    if (active_tab_index_ >= 0 && active_tab_index_ < static_cast<int>(tabs_.size())) {
        CloseTab(active_tab_index_);
    }
}

void TableViewerPanel::CloseTab(int index) {
    if (index >= 0 && index < static_cast<int>(tabs_.size())) {
        close_tab_index_ = index;
    }
}

void TableViewerPanel::CloseAllTabs() {
    tabs_.clear();
    active_tab_index_ = -1;
}

bool TableViewerPanel::IsFileOpen(const std::string& filepath) const {
    return FindTabByPath(filepath) >= 0;
}

void TableViewerPanel::FocusTab(const std::string& filepath) {
    int index = FindTabByPath(filepath);
    if (index >= 0) {
        active_tab_index_ = index;
    }
}

int TableViewerPanel::FindTabByPath(const std::string& filepath) const {
    for (int i = 0; i < static_cast<int>(tabs_.size()); i++) {
        if (tabs_[i]->filepath == filepath) {
            return i;
        }
    }
    return -1;
}

TableViewerPanel::TableTab* TableViewerPanel::GetActiveTab() {
    if (active_tab_index_ >= 0 && active_tab_index_ < static_cast<int>(tabs_.size())) {
        return tabs_[active_tab_index_].get();
    }
    return nullptr;
}

const TableViewerPanel::TableTab* TableViewerPanel::GetActiveTab() const {
    if (active_tab_index_ >= 0 && active_tab_index_ < static_cast<int>(tabs_.size())) {
        return tabs_[active_tab_index_].get();
    }
    return nullptr;
}

void TableViewerPanel::Clear() {
    CloseAllTabs();
}


}  // namespace cyxwiz

