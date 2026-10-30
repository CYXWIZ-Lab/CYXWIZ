#pragma once

// Explore > Images (TOFIX178 1.3e): an image dataset's thumbnails from its
// index table, filtered by class and sorted, as a grid that fills the panel,
// a list, or one image in the viewer (image_viewer_view), with the picked
// image's details. Draws core/image_gallery_presentation.

#include "../../core/image_gallery_presentation.h"
#include "../../core/image_pipeline_preview.h"
#include "../image_thumbnails.h"
#include "image_viewer_view.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

class LiveGraphCompile;

class ImageGalleryView {
public:
    ImageGalleryView();
    ~ImageGalleryView();
    // The canvas graph's compile: "What the model sees" runs its image steps.
    void SetLiveCompile(const LiveGraphCompile* live_compile) { live_compile_ = live_compile; }

    void SetDataset(const std::string& dataset);
    void Render();
    // The first `count` images as a strip (the Data Input dialog's Preview).
    void RenderStrip(size_t count);

    // "Show it in the index table" (Data Studio opens the Table Viewer).
    std::function<void(const std::string& index_table)> on_open_index;

private:
    enum class Layout { Grid, List, Viewer };

    void Refresh();
    void RenderToolbar(const image_gallery::View& view);
    void RenderFilters(const image_gallery::View& view, bool sort_buttons);
    void RenderGrid(const image_gallery::View& view, float width, float height);
    void RenderList(const image_gallery::View& view, float width, float height);
    void RenderDetails(const image_gallery::View& view, float width);
    void RenderViewer(const image_gallery::View& view);
    void HandleKeys(const image_gallery::View& view);
    void Pick(const image_gallery::View& view, long long row);
    // Runs the graph's image steps on the picked image (once per image and
    // graph); RenderPipeline draws them.
    void UpdatePipeline(const image_gallery::Row& row);
    void RenderPipeline(const image_gallery::Row& row);
    ImageViewerView::ModelPicture ModelPicture() const;
    void ReleasePipeline();

    std::string dataset_;
    image_gallery::Input input_;
    const void* index_seen_ = nullptr;
    gui::ImageThumbnails tiles_;
    gui::ImageThumbnails preview_;
    ImageViewerView viewer_;

    Layout layout_ = Layout::Grid;
    Layout before_viewer_ = Layout::Grid;  // Esc in the viewer goes back to it
    int tile_ = 128;                       // px
    bool fill_ = true;                     // fill the tile (crop) or fit the whole image
    bool show_class_ = true, show_name_ = false, show_size_ = false, show_flags_ = true;
    bool model_sees_open_ = false;
    int columns_ = 1;                      // grid columns last frame (Up / Down move a row)

    const LiveGraphCompile* live_compile_ = nullptr;
    std::string pipeline_key_;            // the image + the graph's steps it was run for
    uint64_t pipeline_task_ = 0;
    ImagePipelinePreview pipeline_;
    std::vector<uint32_t> pipeline_textures_;  // decoded, fixed..., draws...
    std::shared_ptr<int> alive_ = std::make_shared<int>(0);
};

}  // namespace cyxwiz
