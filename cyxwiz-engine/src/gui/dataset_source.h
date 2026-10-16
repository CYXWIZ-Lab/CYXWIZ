#pragma once

// A tool panel's dataset (TOFIX140 A, one data entry point): the shared
// dataset picker, then the picked catalog dataset read in the background
// through the session query service (in memory or on disk alike), handed to the
// panel as a DataTable on the UI thread. Data enters only through Data Input
// nodes; tools pick what they loaded and the tables graph nodes made from it.

#include <functional>
#include <memory>
#include <string>

namespace cyxwiz {
class DataTable;
}

namespace cyxwiz::ui {

class DatasetSource {
public:
    using Loaded = std::function<void(std::shared_ptr<DataTable>)>;

    // Draws a full row: "Dataset" and the picker, then under it "Reading..."
    // or the read error. After a new pick is read, `on_loaded` runs with the
    // table. Give it its own row (not inside a SameLine toolbar strip).
    void Render(const char* id, Loaded on_loaded);
    // Reads the picked dataset again (it was re-loaded); `on_loaded` as above.
    void Reload(Loaded on_loaded);

    const std::string& Name() const { return name_; }
    bool Loading() const { return loading_; }

private:
    void Read(Loaded on_loaded);

    std::string name_;      // catalog name of the pick
    std::string error_;     // the last read's failure
    bool loading_ = false;
    std::shared_ptr<int> alive_ = std::make_shared<int>(0);  // reads stop calling back once the panel is gone
};

}  // namespace cyxwiz::ui
