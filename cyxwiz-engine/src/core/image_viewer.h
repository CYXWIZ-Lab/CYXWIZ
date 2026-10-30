#pragma once

// Explore > Images > Viewer: zoom, pan, rotation and flip of one image, and
// which source pixel is under a point. Pure math, no ImGui; the renderer is
// gui/data_studio/image_viewer_view.

#include <array>
#include <string>

namespace cyxwiz::image_viewer {

// The steps - / + and the mouse wheel move through (1/16 to 16x).
inline constexpr std::array<float, 15> kZoomSteps = {0.0625f, 0.125f, 0.25f, 0.333f, 0.5f, 0.667f, 1.0f, 1.5f,
                                                      2.0f,    3.0f,   4.0f,  6.0f,   8.0f, 12.0f,  16.0f};
inline constexpr float kMinZoom = 0.0625f;
inline constexpr float kMaxZoom = 16.0f;
// From this zoom the pixel grid can show (each pixel 8 screen px or more).
inline constexpr float kPixelGridZoom = 8.0f;

// Rotation in quarter turns clockwise, then a mirror left to right.
struct Orientation {
    int quarter_turns = 0;  // 0..3
    bool flip = false;
};

// The image's size on screen at zoom 1 after rotating: width and height
// swap on a quarter or three-quarter turn.
void ShownSize(const Orientation& o, int width, int height, float* shown_w, float* shown_h);

// The zoom that shows the whole image in the area (it may enlarge a small
// image), within kMinZoom..kMaxZoom.
float FitZoom(const Orientation& o, int width, int height, float area_w, float area_h);

// The next step above (direction > 0) or below (< 0) `zoom`.
float StepZoom(float zoom, int direction);
float ClampZoom(float zoom);
std::string ZoomText(float zoom);  // "67 %", "1600 %"

// Zoom and pan of the shown image: `pan` is the image centre's offset from
// the area centre, in screen px.
struct View {
    float zoom = 1.0f;
    float pan_x = 0.0f, pan_y = 0.0f;
};
// `view` zoomed to `zoom` keeping the point `at` (from the area centre, in
// screen px) over the same image pixel.
View ZoomAt(const View& view, float zoom, float at_x, float at_y);

// The texture coordinates of the shown image's corners, clockwise from its
// top left on screen, for this orientation.
struct UV {
    float u = 0.0f, v = 0.0f;
};
std::array<UV, 4> CornerUVs(const Orientation& o);

// The source pixel under the point (x, y) of the shown image, measured in
// shown pixels (0 .. shown width). False outside the image.
bool SourcePixel(const Orientation& o, int width, int height, float x, float y, int* px, int* py);

// "#51473d".
std::string HexColour(int r, int g, int b);

}  // namespace cyxwiz::image_viewer
