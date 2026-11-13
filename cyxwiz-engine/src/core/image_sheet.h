#pragma once

// An image as pictures (TOFIX209, approved mockup): grey and RGB images are
// one picture; any other image is a sheet, the colour picture first when its
// first three channels are still the photo, then every further channel as a
// grey tile on its own scale, named. Pure data: no ImGui, no device.

#include "image_channel_layout.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz {

struct ImageSheetTile {
    std::string name;  // "R, G, B", "channel 4 · edges"; empty for a lone picture
    std::vector<unsigned char> rgba;
    size_t width = 0;
    size_t height = 0;
};

struct ImageSheet {
    std::vector<ImageSheetTile> tiles;  // empty when the values do not fit the shape
};

// How the picture tile maps values to 0..255: as they are (pixels in 0..1),
// or stretched over the picture's own range (a normalized model input).
// Further channels are always stretched, each over its own range.
enum class SheetRange { Unit, Stretched };

namespace image_sheet_detail {

struct Range {
    float low = 0.0f;
    float span = 1.0f;
};

// The finite values' range over channels [first, first + count).
inline Range RangeOf(const float* values, size_t pixels, size_t channels, size_t first, size_t count) {
    float low = 0.0f, high = 0.0f;
    bool any = false;
    for (size_t p = 0; p < pixels; ++p) {
        for (size_t c = first; c < first + count; ++c) {
            const float v = values[p * channels + c];
            if (!std::isfinite(v)) continue;
            low = any ? std::min(low, v) : v;
            high = any ? std::max(high, v) : v;
            any = true;
        }
    }
    return {low, std::max(1e-6f, high - low)};
}

inline unsigned char Byte(float v, const Range& range) {
    return static_cast<unsigned char>(
        std::lround(std::clamp((std::isfinite(v) ? v - range.low : 0.0f) / range.span, 0.0f, 1.0f) * 255.0f));
}

}  // namespace image_sheet_detail

// `values` are height * width * channels, row-major with the channels
// interleaved. A layout that does not name every channel is replaced by the
// decoded one (RGB, grey or numbered).
inline ImageSheet BuildImageSheet(const float* values, size_t height, size_t width, size_t channels,
                                  ImageChannelLayout layout, SheetRange range) {
    using namespace image_sheet_detail;
    ImageSheet sheet;
    if (!values || height == 0 || width == 0 || channels == 0) return sheet;
    if (layout.names.size() != channels || layout.colour > channels) layout = DecodedChannelLayout(channels);
    const size_t pixels = height * width;
    const auto tile = [&](std::string name) -> ImageSheetTile& {
        sheet.tiles.push_back({std::move(name), std::vector<unsigned char>(pixels * 4, 255), width, height});
        return sheet.tiles.back();
    };
    size_t next = 0;
    if (layout.colour == 3) {
        ImageSheetTile& picture = tile(channels > 3 ? "R, G, B" : "");
        const Range r = range == SheetRange::Unit ? Range{} : RangeOf(values, pixels, channels, 0, 3);
        for (size_t p = 0; p < pixels; ++p)
            for (size_t c = 0; c < 3; ++c) picture.rgba[p * 4 + c] = Byte(values[p * channels + c], r);
        next = 3;
    }
    for (size_t c = next; c < channels; ++c) {
        const bool alone = channels == 1;
        const std::string& name = layout.names[c];
        const std::string numbered = "channel " + std::to_string(c + 1);
        ImageSheetTile& grey = tile(alone ? std::string() : name == numbered ? numbered : numbered + " \xC2\xB7 " + name);
        // A lone grey image keeps the picture's rule; a further channel is on its own scale.
        const Range r = alone && range == SheetRange::Unit ? Range{} : RangeOf(values, pixels, channels, c, 1);
        for (size_t p = 0; p < pixels; ++p) {
            const unsigned char v = Byte(values[p * channels + c], r);
            grey.rgba[p * 4] = grey.rgba[p * 4 + 1] = grey.rgba[p * 4 + 2] = v;
        }
    }
    return sheet;
}

}  // namespace cyxwiz
