#pragma once

// The one dataset dropdown (TOFIX140 A): every dataset the catalog knows, as
// Data Studio shows it (name, origin, storage, size). Data enters through Data
// Input nodes only; tools pick from what they loaded and the tables graph nodes
// made from it.

#include <string>

namespace cyxwiz::ui {

struct DatasetPickerOptions {
    bool tabular_only = true;  // analysis tools read tables (in memory or on disk)
    float width = 0.0f;        // 0: fill the row up to 520 px
};

// Draws the dropdown; true when the user picked a different dataset (`selected`
// is then its catalog name).
bool DatasetPicker(const char* id, std::string& selected, const DatasetPickerOptions& options = {});

}  // namespace cyxwiz::ui
