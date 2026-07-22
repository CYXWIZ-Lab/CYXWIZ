#include "table_viewer.h"
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

