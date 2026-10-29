// "What the model sees" (TOFIX178 1.3e): an image through Resize, the fixed
// transforms and draws of the random ones, with the code training runs.

#include "../src/core/image_pipeline_preview.h"
#include "../src/core/prepared_image_set.h"

#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace cyxwiz;
namespace fs = std::filesystem;

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << "\n";
        std::exit(1);
    }
}

image::ImageOp Op(image::ImageOpKind kind) {
    image::ImageOp op;
    op.kind = kind;
    return op;
}

}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "cyxwiz_test_image_pipeline_preview";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    cv::Mat image(30, 40, CV_8UC3);
    for (int y = 0; y < image.rows; ++y)
        for (int x = 0; x < image.cols; ++x) image.at<cv::Vec3b>(y, x) = cv::Vec3b(static_cast<uchar>(x * 6), static_cast<uchar>(y * 8), static_cast<uchar>(100 + x + y));
    const std::string file = (dir / "picture.png").string();
    Check(cv::imwrite(file, image), "wrote the picture");

    const PreparedImageSpec spec{24, 16, ResizeMode::Exact};
    image::ImageOp crop = Op(image::ImageOpKind::CenterCrop);
    crop.width = 12;
    crop.height = 8;
    image::ImageOp flip = Op(image::ImageOpKind::HorizontalFlip);
    flip.probability = 1.0f;
    image::ImageOp rotate = Op(image::ImageOpKind::Rotate);
    rotate.max_angle = 10.0f;
    image::ImageAugmentation steps;
    steps.ops = {crop, Op(image::ImageOpKind::Grayscale), flip, rotate};
    steps.normalize = true;  // left out of the pictures
    steps.mean = 0.5f;
    steps.std_dev = 0.5f;

    const ImagePipelinePreview p = PreviewImagePipeline(file, spec, steps, 3, 7);
    Check(p.error.empty(), "no error: " + p.error);

    // Resize: the decode training uses, alpha opaque.
    std::vector<uint8_t> decoded(24 * 16 * 3);
    Check(DecodePreparedImage(file, spec, decoded.data()), "decoded");
    Check(p.decoded.width == 24 && p.decoded.height == 16 && p.decoded.rgba.size() == 24 * 16 * 4, "24 x 16");
    bool same = true;
    for (size_t i = 0; i < 24 * 16; ++i)
        for (size_t c = 0; c < 3; ++c) same = same && p.decoded.rgba[i * 4 + c] == decoded[i * 3 + c];
    Check(same && p.decoded.rgba[3] == 255, "the decoded picture is the training decode, not normalized");
    Check(p.decoded.title == "Resize 24 \xC3\x97 16" && p.decoded.detail == "exact, once", "decoded title");

    // Fixed: the crop, then grey.
    Check(p.fixed.size() == 2, "two fixed steps");
    Check(p.fixed[0].title == "Center Crop 12 \xC3\x97 8" && p.fixed[0].detail == "12 \xC3\x97 8, once" &&
              p.fixed[0].width == 12 && p.fixed[0].height == 8,
          "center crop");
    bool middle = true;  // the crop is the middle of the decoded picture
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 12; ++x)
            for (int c = 0; c < 3; ++c)
                middle = middle && p.fixed[0].rgba[(y * 12 + x) * 4 + c] == decoded[((y + 4) * 24 + (x + 6)) * 3 + c];
    Check(middle, "the crop keeps the middle");
    bool grey = true;
    for (size_t i = 0; i < 12 * 8; ++i)
        grey = grey && p.fixed[1].rgba[i * 4] == p.fixed[1].rgba[i * 4 + 1] && p.fixed[1].rgba[i * 4] == p.fixed[1].rgba[i * 4 + 2];
    Check(p.fixed[1].title == "Grayscale" && grey, "grayscale");

    // Random: listed, three draws, all flipped (p 1), repeatable from the seed.
    Check(p.random_ops == std::vector<std::string>({"Horizontal Flip (p 1)", "Rotate up to 10 degrees"}), "random ops");
    Check(p.draws.size() == 3 && p.draws[0].title == "Draw 1" && p.draws[0].width == 12, "three draws");
    for (const auto& d : p.draws) Check(d.detail.rfind("flipped, ", 0) == 0, "each draw is flipped: " + d.detail);
    const ImagePipelinePreview again = PreviewImagePipeline(file, spec, steps, 3, 7);
    Check(again.draws[1].rgba == p.draws[1].rgba && again.draws[1].detail == p.draws[1].detail, "same seed, same draws");

    // A flip alone mirrors the fixed result exactly.
    steps.ops = {crop, flip};
    const ImagePipelinePreview flipped = PreviewImagePipeline(file, spec, steps, 1, 3);
    bool mirrored = true;
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 12; ++x)
            for (int c = 0; c < 4; ++c)
                mirrored = mirrored && flipped.draws[0].rgba[(y * 12 + x) * 4 + c] == flipped.fixed[0].rgba[(y * 12 + (11 - x)) * 4 + c];
    Check(mirrored, "the flip mirrors the crop");

    // No random steps: no draws. No Resize, or an unreadable file: an error.
    steps.ops = {crop};
    const ImagePipelinePreview fixed_only = PreviewImagePipeline(file, spec, steps, 4, 1);
    Check(fixed_only.error.empty() && fixed_only.draws.empty() && fixed_only.random_ops.empty(), "fixed only: no draws");
    Check(PreviewImagePipeline(file, {0, 0, ResizeMode::Exact}, steps, 1, 1).error == "The graph has no Resize node.", "no Resize");
    {
        std::ofstream broken(dir / "broken.png");
        broken << "not an image";
    }
    Check(PreviewImagePipeline((dir / "broken.png").string(), spec, steps, 1, 1).error == "This file cannot be read as an image.",
          "unreadable");
    image::ImageOp too_big = crop;
    too_big.width = 99;
    steps.ops = {too_big};
    Check(PreviewImagePipeline(file, spec, steps, 1, 1).error.rfind("Center Crop 99 \xC3\x97 8: ", 0) == 0, "an invalid op says why");

    fs::remove_all(dir, ec);
    std::cout << "test_image_pipeline_preview: all checks passed\n";
    return 0;
}
