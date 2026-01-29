#pragma once

#include "../../utils/hdf5_browser.h"
#include <string>
#include <vector>

namespace gui {

/**
 * HDF5 file preview renderer
 * Shows tree structure, dataset info, and sample data preview
 */
class HDF5PreviewRenderer {
public:
    HDF5PreviewRenderer() = default;

    // Load HDF5 file structure
    void LoadFile(const std::string& filepath);

    // Render the preview
    void Render();

    // Clear state
    void Clear();

private:
    // Render tree node recursively
    void RenderTreeNode(cyxwiz::HDF5NodeInfo& node);

    // Render dataset details panel
    void RenderDatasetDetails();

    // Render sample preview for selected dataset
    void RenderSamplePreview();

    // File info
    std::string current_path_;
    cyxwiz::HDF5NodeInfo root_node_;
    bool file_loaded_ = false;

    // Selection state
    cyxwiz::HDF5NodeInfo* selected_node_ = nullptr;
    std::string selected_dataset_path_;

    // Preview data cache
    std::vector<float> preview_data_;
    std::vector<size_t> preview_shape_;
    bool preview_loaded_ = false;
    int preview_sample_idx_ = 0;
    static constexpr size_t MAX_PREVIEW_SAMPLES = 10;
};

} // namespace gui
