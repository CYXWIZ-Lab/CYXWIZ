#pragma once

// Explore > Images > Viewer: one image at full size as 8-bit RGBA, its
// histograms and its single-channel views. Decoded with OpenCV, as the image
// index and training read images (any type in image_formats.h).

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz {

struct ImagePixels {
    int width = 0, height = 0;
    int channels = 0;        // 1 grey, 2 grey + alpha, 3 colour, 4 colour + alpha (a file with more
                             // channels, a multi-band TIFF, shows its first three as colour)
    int bits = 8;            // per channel in the file: 8, 16 or 32 (float)
    std::vector<uint8_t> rgba;  // width * height * 4, scaled to 8 bits
    std::string error;       // set when the file cannot be read
    bool HasAlpha() const { return channels == 2 || channels == 4; }
};

// The file at full size; `error` says why when it cannot be read.
ImagePixels DecodeImagePixels(const std::string& path);

// What the viewer uploads: the picture and its luminance and alpha views,
// shrunk (area average) so neither side is over `max_side` (a texture's
// limit). The readout and histograms keep the full-size pixels.
struct ImageDisplay {
    int width = 0, height = 0;
    std::vector<uint8_t> colour, luminance, alpha;  // RGBA each; alpha empty without one
};
ImageDisplay MakeImageDisplay(const ImagePixels& pixels, int max_side);

inline constexpr int kHistogramBins = 32;  // 8 values each

struct ChannelHistogram {
    std::array<uint64_t, kHistogramBins> bins{};
    double mean = 0.0, std_dev = 0.0;  // of the 0..255 values
};

// Red, green, blue, luminance (0.2989 R + 0.587 G + 0.114 B, as the Grayscale
// node), and alpha when the image has it.
struct ImageHistograms {
    ChannelHistogram red, green, blue, luminance, alpha;
};
ImageHistograms ComputeHistograms(const ImagePixels& pixels);

enum class ImagePlane { Luminance, Alpha };
// One channel as a grey RGBA picture, for the viewer's Luminance and Alpha
// views (red, green and blue are drawn by masking the colour picture).
std::vector<uint8_t> PlaneRgba(const ImagePixels& pixels, ImagePlane plane);

}  // namespace cyxwiz
