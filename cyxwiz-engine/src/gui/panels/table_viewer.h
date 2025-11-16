#pragma once

#include "../panel.h"
#include "../../data/data_table.h"
#include <string>
#include <memory>

namespace cyxwiz {

/**
 * TableViewer Panel - Display tabular data from CSV, Excel, HDF5
 * Uses ImGui tables for efficient rendering of large datasets
 */
class TableViewerPanel : public Panel {
public:
    TableViewerPanel();
    ~TableViewerPanel() override = default;

    void Render() override;

    // Load data from files
    bool LoadCSV(const std::string& filepath);
    bool LoadTXT(const std::string& filepath, char delimiter = '\t');
    bool LoadHDF5(const std::string& filepath, const std::string& dataset_name = "data");
    bool LoadExcel(const std::string& filepath, const std::string& sheet_name = "");

    // Set table to display
    void SetTable(std::shared_ptr<DataTable> table);
    void SetTableByName(const std::string& name);

    // Clear current table
    void Clear();

private:
    void RenderToolbar();
    void RenderTable();
    void RenderStatusBar();

    std::shared_ptr<DataTable> current_table_;
    std::string current_table_name_;

    // UI state
    bool show_line_numbers_;
    int rows_per_page_;
    int current_page_;
    std::string filter_text_;
    char filter_buffer_[256];
};

} // namespace cyxwiz
