// The Engine's image types (core/image_formats.h): every listed type decodes
// through the Engine's OpenCV, and types it cannot decode are not listed.
#include "../src/core/image_formats.h"

#include <opencv2/imgcodecs.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

namespace fs = std::filesystem;
namespace formats = cyxwiz::image_formats;

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

// Writes a 5 x 3 picture as `ext` and reads it back.
bool Decodes(const std::string& ext, const fs::path& dir) {
    const bool is_float = ext == ".hdr" || ext == ".pic" || ext == ".pfm";
    const bool is_gray = ext == ".pbm" || ext == ".pgm";
    cv::Mat picture(3, 5, is_float ? CV_32FC3 : (is_gray ? CV_8UC1 : CV_8UC3), cv::Scalar::all(is_float ? 0.5 : 200));
    const fs::path file = dir / ("probe" + ext);
    try {
        if (!cv::imwrite(file.string(), picture)) return false;
    } catch (const cv::Exception&) {
        return false;
    }
    const cv::Mat back = cv::imread(file.string(), cv::IMREAD_UNCHANGED);
    return !back.empty() && back.cols == 5 && back.rows == 3;
}

}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "cyxwiz_test_image_formats";
    fs::create_directories(dir);

    for (const auto ext : formats::kExtensions)
        Check(Decodes(std::string(ext), dir), std::string(ext) + " is listed but OpenCV does not decode it");
    // Types an image tool might expect: listed only if they decode.
    for (const char* ext : {".tga", ".avif", ".jp2", ".exr", ".jxl", ".heic"}) {
        const bool decodes = Decodes(ext, dir);
        std::cout << ext << (decodes ? " decodes" : " does not decode") << '\n';
        Check(decodes == formats::IsImageExtension(ext),
              std::string(ext) + (decodes ? " decodes but is not listed" : " is listed but does not decode"));
    }

    Check(formats::IsImageExtension(".JPG") && formats::IsImageExtension("webp"), "any case, with or without the dot");
    Check(!formats::IsImageExtension("") && !formats::IsImageExtension(".csv"), "not an image type");
    Check(formats::IsImagePath("D:/pets/cat/Cat.0.TIFF"), "a path's extension");
    Check(!formats::IsImagePath("D:/pets.jpg/readme") && !formats::IsImagePath("jpg"), "a folder's dot or no dot");

    std::error_code ec;
    fs::remove_all(dir, ec);
    if (failures) return 1;
    std::cout << "Image formats passed\n";
    return 0;
}
