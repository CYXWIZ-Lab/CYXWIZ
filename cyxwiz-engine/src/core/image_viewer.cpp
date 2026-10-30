#include "image_viewer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cyxwiz::image_viewer {

namespace {

int Turns(const Orientation& o) { return ((o.quarter_turns % 4) + 4) % 4; }

}  // namespace

void ShownSize(const Orientation& o, int width, int height, float* shown_w, float* shown_h) {
    const bool sideways = Turns(o) % 2 == 1;
    *shown_w = static_cast<float>(sideways ? height : width);
    *shown_h = static_cast<float>(sideways ? width : height);
}

float ClampZoom(float zoom) { return std::clamp(zoom, kMinZoom, kMaxZoom); }

float FitZoom(const Orientation& o, int width, int height, float area_w, float area_h) {
    float w = 0.0f, h = 0.0f;
    ShownSize(o, width, height, &w, &h);
    if (w <= 0.0f || h <= 0.0f || area_w <= 0.0f || area_h <= 0.0f) return 1.0f;
    return ClampZoom(std::min(area_w / w, area_h / h));
}

float StepZoom(float zoom, int direction) {
    // A step counts only when it is clearly past `zoom` (a fitted 0.6667 is
    // the 0.667 step, not one below it).
    if (direction > 0) {
        for (const float step : kZoomSteps)
            if (step > zoom * 1.01f) return step;
        return kMaxZoom;
    }
    for (auto it = kZoomSteps.rbegin(); it != kZoomSteps.rend(); ++it)
        if (*it < zoom * 0.99f) return *it;
    return kMinZoom;
}

std::string ZoomText(float zoom) {
    char text[16];
    std::snprintf(text, sizeof(text), "%.0f %%", std::round(zoom * 100.0f));
    return text;
}

View ZoomAt(const View& view, float zoom, float at_x, float at_y) {
    View out = view;
    out.zoom = ClampZoom(zoom);
    const float ratio = out.zoom / view.zoom;
    out.pan_x = at_x - (at_x - view.pan_x) * ratio;
    out.pan_y = at_y - (at_y - view.pan_y) * ratio;
    return out;
}

std::array<UV, 4> CornerUVs(const Orientation& o) {
    // The image's own corners, clockwise from its top left.
    std::array<UV, 4> corners = {UV{0.0f, 0.0f}, UV{1.0f, 0.0f}, UV{1.0f, 1.0f}, UV{0.0f, 1.0f}};
    // A mirror swaps left and right: TL<->TR, BL<->BR.
    if (o.flip) corners = {corners[1], corners[0], corners[3], corners[2]};
    // A clockwise quarter turn puts the image's bottom-left corner at the
    // shown top left.
    std::array<UV, 4> out = corners;
    for (int k = 0; k < Turns(o); ++k) out = {out[3], out[0], out[1], out[2]};
    return out;
}

bool SourcePixel(const Orientation& o, int width, int height, float x, float y, int* px, int* py) {
    float w = 0.0f, h = 0.0f;
    ShownSize(o, width, height, &w, &h);
    if (x < 0.0f || y < 0.0f || x >= w || y >= h) return false;
    // Bilinear blend of the corner UVs at (x / w, y / h) gives the source
    // coordinate; corners are axis aligned, so it is exact.
    const auto uv = CornerUVs(o);
    const float s = x / w, t = y / h;
    const float u = (1 - s) * (1 - t) * uv[0].u + s * (1 - t) * uv[1].u + s * t * uv[2].u + (1 - s) * t * uv[3].u;
    const float v = (1 - s) * (1 - t) * uv[0].v + s * (1 - t) * uv[1].v + s * t * uv[2].v + (1 - s) * t * uv[3].v;
    *px = std::clamp(static_cast<int>(u * width), 0, width - 1);
    *py = std::clamp(static_cast<int>(v * height), 0, height - 1);
    return true;
}

std::string HexColour(int r, int g, int b) {
    char text[8];
    std::snprintf(text, sizeof(text), "#%02x%02x%02x", r & 255, g & 255, b & 255);
    return text;
}

}  // namespace cyxwiz::image_viewer
