#pragma once

// Explore > Images (TOFIX178 1.3e): an image dataset's thumbnails from its
// index table, by page, filtered by class and sorted, with the picked
// image's details. Draws core/image_gallery_presentation.

#include "../../core/image_gallery_presentation.h"
#include "../image_thumbnails.h"

#include <functional>
#include <string>

namespace cyxwiz {

class ImageGalleryView {
public:
    ImageGalleryView();

    void SetDataset(const std::string& dataset);
    void Render();
    // The first `count` images as a strip (the Data Input dialog's Preview).
    void RenderStrip(size_t count);

    // "Show it in the index table" (Data Studio opens the Table Viewer).
    std::function<void(const std::string& index_table)> on_open_index;

private:
    void Refresh();

    std::string dataset_;
    image_gallery::Input input_;
    const void* index_seen_ = nullptr;
    gui::ImageThumbnails tiles_;
    gui::ImageThumbnails preview_;
};

}  // namespace cyxwiz
