#include "table_viewer.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace cyxwiz {

// Saving an edited table to its file (the grid edits the cells).
void TableViewerPanel::SaveTable(TableTab* tab) {
    if (!tab || !tab->grid.table || tab->filepath.empty()) {
        spdlog::warn("Cannot save: No table or filepath");
        return;
    }

    // Determine file type from extension
    std::string ext = fs::path(tab->filepath).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    bool success = false;
    if (ext == ".csv") {
        success = tab->grid.table->SaveToCSV(tab->filepath);
    } else if (ext == ".txt" || ext == ".tsv") {
        char delim = (ext == ".tsv") ? '\t' : '\t';
        success = tab->grid.table->SaveToTXT(tab->filepath, delim);
    } else if (ext == ".h5" || ext == ".hdf5") {
        success = tab->grid.table->SaveToHDF5(tab->filepath);
    } else {
        // Default to CSV
        success = tab->grid.table->SaveToCSV(tab->filepath);
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

