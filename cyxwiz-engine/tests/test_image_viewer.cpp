// Explore > Images > Viewer: zoom, pan, rotation and flip (core/image_viewer)
// and the full-size pixels, histograms and channel planes (core/image_pixels).
#include "../src/core/image_pixels.h"
#include "../src/core/image_viewer.h"

#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

namespace fs = std::filesystem;
namespace iv = cyxwiz::image_viewer;

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

bool Near(float a, float b) { return std::fabs(a - b) < 1e-3f; }

void CheckZoom() {
    // 1023 x 768 in a 744 x 560 area: the width limits it (744 / 1023).
    Check(Near(iv::FitZoom({}, 1023, 768, 744.0f, 560.0f), 744.0f / 1023.0f), "fit by the width");
    // A quarter turn swaps the sides: 768 wide, 1023 tall.
    Check(Near(iv::FitZoom({1, false}, 1023, 768, 744.0f, 560.0f), 560.0f / 1023.0f), "fit after a quarter turn");
    Check(Near(iv::FitZoom({}, 20, 10, 800.0f, 800.0f), 16.0f), "a small image enlarges up to 16x");
    Check(Near(iv::StepZoom(0.6667f, 1), 1.0f) && Near(iv::StepZoom(0.6667f, -1), 0.5f), "a fitted 2/3 is the 67 % step");
    Check(Near(iv::StepZoom(16.0f, 1), 16.0f) && Near(iv::StepZoom(0.0625f, -1), 0.0625f), "steps stop at the ends");
    Check(iv::ZoomText(0.667f) == "67 %" && iv::ZoomText(16.0f) == "1600 %", "zoom text");

    // Zooming at a point keeps that point over the same pixel.
    const iv::View before{1.0f, 10.0f, -4.0f};
    const iv::View after = iv::ZoomAt(before, 4.0f, 50.0f, 30.0f);
    // The pixel under (50, 30) sits (50 - 10, 30 + 4) / zoom from the image centre.
    Check(Near((50.0f - after.pan_x) / after.zoom, (50.0f - before.pan_x) / before.zoom) &&
              Near((30.0f - after.pan_y) / after.zoom, (30.0f - before.pan_y) / before.zoom),
          "zoom at a point keeps its pixel");
}

void CheckOrientation() {
    int x = 0, y = 0;
    // 4 x 2 image. Shown as it is: (0.5, 0.5) is pixel (0, 0); (3.5, 1.5) is (3, 1).
    Check(iv::SourcePixel({}, 4, 2, 0.5f, 0.5f, &x, &y) && x == 0 && y == 0, "top left");
    Check(iv::SourcePixel({}, 4, 2, 3.5f, 1.5f, &x, &y) && x == 3 && y == 1, "bottom right");
    Check(!iv::SourcePixel({}, 4, 2, 4.5f, 0.5f, &x, &y), "outside the image");
    // Mirrored: the shown top left is the image's top right.
    Check(iv::SourcePixel({0, true}, 4, 2, 0.5f, 0.5f, &x, &y) && x == 3 && y == 0, "flip");
    // A clockwise quarter turn shows 2 wide, 4 tall; its top left is the
    // image's bottom left, its top right the image's top left.
    Check(iv::SourcePixel({1, false}, 4, 2, 0.5f, 0.5f, &x, &y) && x == 0 && y == 1, "quarter turn: top left");
    Check(iv::SourcePixel({1, false}, 4, 2, 1.5f, 0.5f, &x, &y) && x == 0 && y == 0, "quarter turn: top right");
    Check(iv::SourcePixel({1, false}, 4, 2, 1.5f, 3.5f, &x, &y) && x == 3 && y == 0, "quarter turn: bottom right");
    // A half turn: the shown top left is the image's bottom right.
    Check(iv::SourcePixel({2, false}, 4, 2, 0.5f, 0.5f, &x, &y) && x == 3 && y == 1, "half turn");
    const auto uv = iv::CornerUVs({1, false});
    Check(Near(uv[0].u, 0.0f) && Near(uv[0].v, 1.0f) && Near(uv[1].u, 0.0f) && Near(uv[1].v, 0.0f), "quarter turn corners");
    float w = 0, h = 0;
    iv::ShownSize({3, true}, 4, 2, &w, &h);
    Check(Near(w, 2.0f) && Near(h, 4.0f), "three quarter turn swaps the sides");
    Check(iv::HexColour(81, 71, 61) == "#51473d", "hex colour");
}

void CheckPixels(const fs::path& dir) {
    // 2 x 1 colour PNG: pure red, pure blue (OpenCV writes BGR).
    cv::Mat colour(1, 2, CV_8UC3);
    colour.at<cv::Vec3b>(0, 0) = {0, 0, 255};
    colour.at<cv::Vec3b>(0, 1) = {255, 0, 0};
    const std::string path = (dir / "two.png").string();
    cv::imwrite(path, colour);
    const auto pixels = cyxwiz::DecodeImagePixels(path);
    Check(pixels.error.empty() && pixels.width == 2 && pixels.height == 1 && pixels.channels == 3 && pixels.bits == 8,
          "colour PNG decodes");
    Check(pixels.rgba.size() == 8 && pixels.rgba[0] == 255 && pixels.rgba[2] == 0 && pixels.rgba[6] == 255 && pixels.rgba[3] == 255,
          "RGBA order, opaque");
    const auto h = cyxwiz::ComputeHistograms(pixels);
    Check(h.red.bins[31] == 1 && h.red.bins[0] == 1 && std::fabs(h.red.mean - 127.5) < 1e-9 &&
              std::fabs(h.red.std_dev - 127.5) < 1e-9,
          "red histogram, mean and std");
    // Luminance as the Grayscale node: red 0.2989 * 255 = 76, blue 0.114 * 255 = 29.
    const auto grey = cyxwiz::PlaneRgba(pixels, cyxwiz::ImagePlane::Luminance);
    Check(grey.size() == 8 && grey[0] == 76 && grey[4] == 29 && grey[1] == 76 && grey[3] == 255, "luminance plane");
    Check(h.luminance.bins[76 / 8] == 1 && h.luminance.bins[29 / 8] == 1, "luminance histogram");

    // 16-bit colour with alpha scales to 8 bits; alpha has its own histogram.
    cv::Mat grey_alpha(1, 1, CV_16UC4, cv::Scalar(65535, 65535, 65535, 32896));
    const std::string ga = (dir / "ga.png").string();
    cv::imwrite(ga, grey_alpha);
    const auto ga_pixels = cyxwiz::DecodeImagePixels(ga);
    Check(ga_pixels.channels == 4 && ga_pixels.bits == 16 && ga_pixels.HasAlpha() && ga_pixels.rgba[0] == 255 &&
              ga_pixels.rgba[3] == 128,
          "16-bit RGBA");
    Check(cyxwiz::ComputeHistograms(ga_pixels).alpha.bins[128 / 8] == 1, "alpha histogram");

    // The display copy: shrunk to the texture limit, aspect kept; the full
    // pixels stay for the readout.
    cv::Mat wide(2, 8, CV_8UC3, cv::Scalar(10, 20, 30));
    const std::string wide_path = (dir / "wide.png").string();
    cv::imwrite(wide_path, wide);
    const auto wide_pixels = cyxwiz::DecodeImagePixels(wide_path);
    const auto shrunk = cyxwiz::MakeImageDisplay(wide_pixels, 4);
    Check(shrunk.width == 4 && shrunk.height == 1 && shrunk.colour.size() == 16 && shrunk.luminance.size() == 16 &&
              shrunk.alpha.empty() && shrunk.colour[0] == 30 && wide_pixels.width == 8,
          "display shrunk to the limit");
    const auto whole = cyxwiz::MakeImageDisplay(wide_pixels, 4096);
    Check(whole.width == 8 && whole.height == 2 && whole.colour == wide_pixels.rgba, "no shrink under the limit");

    // Five channels (a multi-band TIFF): the first three show as colour. This
    // OpenCV cannot write one (imwrite takes 1, 3 or 4 channels), so the
    // check runs only where it can.
    cv::Mat bands(1, 1, CV_8UC(5));
    for (int c = 0; c < 5; ++c) bands.data[c] = static_cast<uchar>(c + 1);  // bands 1 .. 5
    const std::string bands_path = (dir / "bands.tif").string();
    bool written = false;
    try {
        written = cv::imwrite(bands_path, bands);
    } catch (const cv::Exception&) {
    }
    if (written) {
        const auto five = cyxwiz::DecodeImagePixels(bands_path);
        Check(five.error.empty() && five.channels == 3 && five.rgba[0] == 3 && five.rgba[2] == 1,
              "five bands show the first three: " + five.error);
    }

    std::ofstream(dir / "broken.png") << "not an image";
    Check(!cyxwiz::DecodeImagePixels((dir / "broken.png").string()).error.empty(), "a broken file says why");
}

}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "cyxwiz_test_image_viewer";
    fs::create_directories(dir);
    CheckZoom();
    CheckOrientation();
    CheckPixels(dir);
    std::error_code ec;
    fs::remove_all(dir, ec);
    if (failures) return 1;
    std::cout << "Image viewer passed\n";
    return 0;
}
