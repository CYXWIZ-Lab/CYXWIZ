#pragma once

// The image file types the Engine reads: the ones its OpenCV build decodes
// (vcpkg opencv4: JPEG, PNG, TIFF and WebP, plus the formats OpenCV reads
// without a library, GIF among them; it has no TGA reader). One list for
// every folder scan, loader and preview, so a type is either readable
// everywhere or listed nowhere.
// tests/test_image_formats decodes each one through the Engine's OpenCV.

#include <algorithm>
#include <array>
#include <cctype>
#include <string>
#include <string_view>

namespace cyxwiz::image_formats {

// Lower case, with the dot.
inline constexpr std::array<std::string_view, 19> kExtensions = {
    ".jpg", ".jpeg", ".jpe", ".png",  ".bmp", ".dib", ".tif", ".tiff", ".webp", ".gif",
    ".ppm", ".pgm",  ".pbm", ".pnm",  ".sr",  ".ras", ".hdr", ".pic",  ".pfm",
};

// `ext` with or without its dot, in any case: ".JPG", "png".
inline bool IsImageExtension(std::string_view ext) {
    std::string lower = ext.empty() || ext.front() == '.' ? std::string(ext) : "." + std::string(ext);
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return std::find(kExtensions.begin(), kExtensions.end(), lower) != kExtensions.end();
}

// A file path whose extension is an image type.
inline bool IsImagePath(std::string_view path) {
    const size_t dot = path.find_last_of('.');
    const size_t slash = path.find_last_of("/\\");
    if (dot == std::string_view::npos || (slash != std::string_view::npos && dot < slash)) return false;
    return IsImageExtension(path.substr(dot));
}

}  // namespace cyxwiz::image_formats
