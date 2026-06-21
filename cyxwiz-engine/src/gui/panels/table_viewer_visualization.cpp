#include "table_viewer.h"
#include "visualization_panel.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <string>
#include <vector>

namespace cyxwiz {
// ============================================================================
// Visualization Integration
// ============================================================================

void TableViewerPanel::SendToVisualizer(int column) {
    if (!visualization_panel_) {
        spdlog::warn("VisualizationPanel not connected to TableViewer");
        return;
    }

    TableTab* tab = GetActiveTab();
    if (!tab || !tab->table) {
        spdlog::warn("No active table to send to Visualizer");
        return;
    }

    const auto& headers = tab->table->GetHeaders();
    std::vector<std::string> col_names;
    std::vector<std::vector<std::string>> rows;

    // Determine which columns to send
    std::vector<int> columns_to_send;
    if (column >= 0 && column < static_cast<int>(headers.size())) {
        // Single column
        columns_to_send.push_back(column);
    } else {
        // All columns
        for (int i = 0; i < static_cast<int>(headers.size()); i++) {
            columns_to_send.push_back(i);
        }
    }

    // Build column names
    for (int col : columns_to_send) {
        col_names.push_back(headers[col]);
    }

    // Build rows - use filtered/sorted indices if active
    const std::vector<size_t>& display_indices = (tab->filter_mode_hide && !tab->filtered_indices.empty())
        ? tab->filtered_indices
        : tab->sorted_indices;

    // Limit to reasonable number of rows for visualization
    size_t max_rows = std::min(display_indices.size(), size_t(10000));

    for (size_t i = 0; i < max_rows; i++) {
        size_t actual_row = display_indices[i];
        std::vector<std::string> row_data;
        for (int col : columns_to_send) {
            row_data.push_back(tab->table->GetCellAsString(actual_row, col));
        }
        rows.push_back(row_data);
    }

    // Send to visualizer
    visualization_panel_->SetData(col_names, rows);
    visualization_panel_->Show();

    spdlog::info("Sent {} columns x {} rows to Visualizer", col_names.size(), rows.size());
}

}  // namespace cyxwiz

