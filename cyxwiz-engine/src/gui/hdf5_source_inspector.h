#pragma once

#include "../core/hdf5_inspection_task.h"
#include "data_preview_table_renderer.h"

#include <map>
#include <string>

namespace gui {

// Dialog-owned, transient inspection only. It has no node/registry access.
class Hdf5SourceInspector {
public:
    ~Hdf5SourceInspector();
    Hdf5SourceInspector() = default;
    Hdf5SourceInspector(const Hdf5SourceInspector&) = delete;
    Hdf5SourceInspector& operator=(const Hdf5SourceInspector&) = delete;

    void SetSource(const std::string& path);
    void SetSelection(const cyxwiz::Hdf5TableSelection& selection);
    // Call after SetSource; restores paths without scheduling work or saving edits.
    void RestoreSettings(const std::map<std::string, std::string>& parameters);
    void Browse(const std::string& group = "/", uint64_t offset = 0, bool refresh = false);
    void Preview(uint64_t row_offset = 0, uint64_t column_offset = 0);
    void Poll();
    void Cancel();
    void Reset();
    void RenderSettings();
    void RenderPreview();

    bool Busy() const { return task_ != nullptr; }
    const std::string& Error() const { return error_; }
    const cyxwiz::Hdf5BrowsePage& Hierarchy() const { return hierarchy_; }
    const cyxwiz::DataPreviewPage& Page() const { return page_; }
    cyxwiz::Hdf5TableSelection Selection() const { return {data_path_, label_path_}; }

private:
    void Start(cyxwiz::Hdf5InspectionRequest request);
    void InvalidateSelection();
    void RenderError() const;

    std::string source_path_;
    char group_path_[4097] = "/";
    char data_path_[4097] = "/data";
    char label_path_[4097] = {};
    int selection_role_ = 0;
    int rows_per_page_ = 20;
    uint64_t row_offset_ = 0;
    uint64_t column_offset_ = 0;
    std::optional<cyxwiz::Hdf5SourceStamp> source_stamp_;
    cyxwiz::Hdf5BrowsePage hierarchy_;
    cyxwiz::DataPreviewPage page_;
    cyxwiz::Hdf5TableDatasetInfo data_;
    cyxwiz::Hdf5InspectionRequest request_;
    std::shared_ptr<cyxwiz::AsyncTask> task_;
    std::shared_ptr<cyxwiz::Hdf5InspectionTaskResult> state_;
    DataPreviewViewState view_;
    std::string error_;
};
} // namespace gui
