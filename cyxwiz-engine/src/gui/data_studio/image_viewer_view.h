#pragma once

// Explore > Images > Viewer: one image at full size. Zoom (Fit, 100 %, a typed
// percent, - / +, the wheel), drag to pan, rotate and flip the view, one
// channel at a time, a pixel grid and a checkerboard; the pixel under the
// cursor, a histogram per channel and the pixels around the cursor. Nothing
// here changes the file. Draws core/image_viewer and core/image_pixels.

#include "../../core/image_pixels.h"
#include "../../core/image_viewer.h"

#include <imgui.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

class ImageViewerView {
public:
    ImageViewerView();
    ~ImageViewerView();
    ImageViewerView(const ImageViewerView&) = delete;
    ImageViewerView& operator=(const ImageViewerView&) = delete;

    // The graph's last image step on this image, for Compare.
    struct ModelPicture {
        ImTextureID texture = 0;
        int width = 0, height = 0;
        std::string title;  // "Gaussian Blur 3 x 3, 64 x 64"
    };

    // Zoom: Fit, 100 %, the percent and - / +, for the gallery's toolbar.
    void RenderZoomControls();
    // The viewer for the image at `path`, filling the space left; `detail`
    // is the gallery's facts about it, `strip` draws the filmstrip (its
    // height `strip_height`) under the picture.
    void Render(const std::string& path, const std::vector<std::pair<std::string, std::string>>& detail,
                const ModelPicture& model, float strip_height, const std::function<void()>& strip);

private:
    void Load(const std::string& path);
    void Release();
    void RenderControls(bool has_model);
    void RenderCanvas(const ImVec2& size, const ModelPicture& model);
    void RenderStatus();
    void RenderSide(const std::vector<std::pair<std::string, std::string>>& detail);

    std::string path_;
    uint64_t task_ = 0;
    bool loading_ = false;
    ImagePixels pixels_;
    ImageHistograms histograms_;
    uint32_t colour_ = 0, luminance_ = 0, alpha_ = 0;  // textures
    // The luminance and alpha views, uploaded the first time they are shown
    // (then freed); the display size when the image was shrunk to fit a texture.
    ImageDisplay display_;

    image_viewer::View view_;
    bool fitted_ = true;  // the zoom follows the area until the user zooms
    image_viewer::Orientation orientation_;
    int channel_ = 0;     // RGB, R, G, B, Luminance, Alpha
    bool pixel_grid_ = true;
    bool checkerboard_ = false;
    int compare_ = 0;     // Original, Model sees, Side by side
    double zoom_percent_ = 100.0;

    bool hovering_ = false;
    int hover_x_ = 0, hover_y_ = 0;
    std::shared_ptr<int> alive_ = std::make_shared<int>(0);
};

}  // namespace cyxwiz
