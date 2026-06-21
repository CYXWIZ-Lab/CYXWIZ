#include "table_viewer.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace cyxwiz {
// ============================================================================
// Cell Editing
// ============================================================================

void TableViewerPanel::BeginCellEdit(TableTab* tab, int row, int col) {
    if (!tab || !tab->table) return;

    // End any current editing
    if (tab->editing_row >= 0 && tab->editing_col >= 0) {
        EndCellEdit(tab, true);  // Save current edit
    }

    // Get current cell value
    std::string current_value = tab->table->GetCellAsString(row, col);

    // Copy to edit buffer
    strncpy(tab->edit_buffer, current_value.c_str(), sizeof(tab->edit_buffer) - 1);
    tab->edit_buffer[sizeof(tab->edit_buffer) - 1] = '\0';

    // Set editing state
    tab->editing_row = row;
    tab->editing_col = col;
    tab->edit_just_started = true;

    spdlog::debug("Started editing cell [{}, {}]: '{}'", row + 1, col + 1, current_value);
}

void TableViewerPanel::EndCellEdit(TableTab* tab, bool save) {
    if (!tab || tab->editing_row < 0 || tab->editing_col < 0) return;

    if (save && tab->table) {
        std::string new_value = tab->edit_buffer;
        std::string old_value = tab->table->GetCellAsString(tab->editing_row, tab->editing_col);

        // Only update if value changed
        if (new_value != old_value) {
            if (tab->table->SetCellFromString(tab->editing_row, tab->editing_col, new_value)) {
                tab->is_dirty = true;
                spdlog::info("Cell [{}, {}] changed: '{}' -> '{}'",
                    tab->editing_row + 1, tab->editing_col + 1, old_value, new_value);

                // Recompute column stats for this column (value may affect min/max/avg)
                if (tab->editing_col < static_cast<int>(tab->column_stats.size())) {
                    tab->column_stats[tab->editing_col].computed = false;
                }
            } else {
                spdlog::warn("Failed to update cell [{}, {}]", tab->editing_row + 1, tab->editing_col + 1);
            }
        }
    }

    // Clear editing state
    tab->editing_row = -1;
    tab->editing_col = -1;
    std::memset(tab->edit_buffer, 0, sizeof(tab->edit_buffer));
    tab->edit_just_started = false;
}

void TableViewerPanel::SaveTable(TableTab* tab) {
    if (!tab || !tab->table || tab->filepath.empty()) {
        spdlog::warn("Cannot save: No table or filepath");
        return;
    }

    // Determine file type from extension
    std::string ext = fs::path(tab->filepath).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    bool success = false;
    if (ext == ".csv") {
        success = tab->table->SaveToCSV(tab->filepath);
    } else if (ext == ".txt" || ext == ".tsv") {
        char delim = (ext == ".tsv") ? '\t' : '\t';
        success = tab->table->SaveToTXT(tab->filepath, delim);
    } else if (ext == ".h5" || ext == ".hdf5") {
        success = tab->table->SaveToHDF5(tab->filepath);
    } else {
        // Default to CSV
        success = tab->table->SaveToCSV(tab->filepath);
    }

    if (success) {
        tab->is_dirty = false;
        spdlog::info("Saved table to: {}", tab->filepath);
    } else {
        spdlog::error("Failed to save table to: {}", tab->filepath);
    }
}

bool TableViewerPanel::HasUnsavedChanges() const {
    for (const auto& tab : tabs_) {
        if (tab && tab->is_dirty) {
            return true;
        }
    }
    return false;
}

}  // namespace cyxwiz

