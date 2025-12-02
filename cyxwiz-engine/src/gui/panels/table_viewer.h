#pragma once

#include "../panel.h"
#include "../../data/data_table.h"
#include "../../core/async_task_manager.h"
#include <string>
#include <memory>
#include <vector>
#include <atomic>
#include <mutex>

namespace cyxwiz {

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

        TableTab() {
            std::memset(filter_buffer, 0, sizeof(filter_buffer));
        }
    };

    void RenderTabBar();
    void RenderToolbar();
    void RenderTable();
    void RenderStatusBar();
    void RenderLoadingIndicator();

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

    // Async loading
    std::mutex tabs_mutex_;
    std::atomic<bool> has_pending_load_{false};
};

} // namespace cyxwiz
