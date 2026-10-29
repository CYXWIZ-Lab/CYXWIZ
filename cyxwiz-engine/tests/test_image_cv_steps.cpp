// The OpenCV image steps (TOFIX178 1.3f): each step on uint8 images against
// OpenCV called directly, their checks and shapes, and their place among
// the device transforms.

#include "../src/core/image_cv_steps.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cyxwiz;

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << "\n";
        std::exit(1);
    }
}

constexpr int kW = 40, kH = 32;
const image::ImageShape kRgb{kH, kW, 3};

// A picture with edges, gradients and noise, so every step changes it.
cv::Mat Picture() {
    cv::Mat m(kH, kW, CV_8UC3);
    cv::RNG rng(7);
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) {
            const int base = (x > 14 && x < 30 && y > 8 && y < 24) ? 200 : 40 + x * 2;
            m.at<cv::Vec3b>(y, x) = cv::Vec3b(cv::saturate_cast<uchar>(base + rng.uniform(-20, 20)),
                                              cv::saturate_cast<uchar>(base / 2 + y * 3),
                                              cv::saturate_cast<uchar>(255 - base));
        }
    return m;
}

std::vector<uint8_t> Run(const CvImageStep& s, const cv::Mat& in, const image::ImageShape& shape = kRgb) {
    std::vector<uint8_t> out(CvImageShapeAfter(s, shape).Size());
    std::string error;
    Check(ApplyCvImageStep(s, shape, in.data, out.data(), &error), CvImageStepText(s) + " runs: " + error);
    return out;
}

int Worst(const std::vector<uint8_t>& got, const cv::Mat& expected) {
    Check(expected.isContinuous() && got.size() == expected.total() * expected.elemSize(),
          "sizes agree (" + std::to_string(got.size()) + " vs " + std::to_string(expected.total() * expected.elemSize()) + ")");
    int worst = 0;
    for (size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::abs(int(got[i]) - int(expected.data[i])));
    return worst;
}

CvImageStep Step(CvImageStepKind kind) {
    CvImageStep s;
    s.kind = kind;
    return s;
}

// OpenCV on float [0, 1], as the classes run, rounded once.
cv::Mat OnFloat(const cv::Mat& in, const std::function<void(const cv::Mat&, cv::Mat&)>& f) {
    cv::Mat x, y, out;
    in.convertTo(x, CV_32F, 1.0 / 255.0);
    f(x, y);
    y.convertTo(out, CV_8U, 255.0);
    return out;
}

}  // namespace

int main() {
    const cv::Mat rgb = Picture();

    // Contrast: CLAHE on Lab L; each channel: on R, G and B alone.
    {
        CvImageStep s = Step(CvImageStepKind::Contrast);
        cv::Mat lab, expected;
        cv::cvtColor(rgb, lab, cv::COLOR_RGB2Lab);
        std::vector<cv::Mat> planes;
        cv::split(lab, planes);
        cv::createCLAHE(2.0, cv::Size(8, 8))->apply(planes[0], planes[0]);
        cv::merge(planes, lab);
        cv::cvtColor(lab, expected, cv::COLOR_Lab2RGB);
        Check(Worst(Run(s, rgb), expected) == 0, "CLAHE on lightness is OpenCV's");
        s.colour = CvContrastColour::EachChannel;
        cv::split(rgb, planes);
        for (auto& p : planes) cv::createCLAHE(2.0, cv::Size(8, 8))->apply(p, p);
        cv::merge(planes, expected);
        Check(Worst(Run(s, rgb), expected) == 0, "CLAHE on each channel");
        Check(CvImageStepText(Step(CvImageStepKind::Contrast)) == "CLAHE 2.0 / 8", "contrast text");
    }
    // Denoise: non-local means on colour.
    {
        cv::Mat expected;
        cv::fastNlMeansDenoisingColored(rgb, expected, 10, 10, 7, 21);
        Check(Worst(Run(Step(CvImageStepKind::Denoise), rgb), expected) == 0, "denoise is OpenCV's");
    }
    // Sharpen, also with a threshold on a colour image (it used to fail).
    {
        CvImageStep s = Step(CvImageStepKind::Sharpen);
        const cv::Mat expected = OnFloat(rgb, [](const cv::Mat& x, cv::Mat& y) {
            cv::Mat blurred;
            cv::GaussianBlur(x, blurred, cv::Size(7, 7), 1.0);
            y = cv::max(0.0, cv::min(1.0, x + 1.0 * (x - blurred)));
        });
        Check(Worst(Run(s, rgb), expected) <= 1, "sharpen is the unsharp mask");
        s.threshold = 0.05f;
        const auto thresholded = Run(s, rgb);
        Check(thresholded.size() == kRgb.Size(), "sharpen with a threshold on RGB");
    }
    // Edges: one channel out.
    {
        CvImageStep s = Step(CvImageStepKind::Edges);
        Check(CvImageShapeAfter(s, kRgb) == image::ImageShape{kH, kW, 1}, "edges give one channel");
        cv::Mat grey, expected;
        cv::cvtColor(rgb, grey, cv::COLOR_RGB2GRAY);
        cv::Canny(grey, expected, 50, 150);
        Check(Worst(Run(s, rgb), expected) == 0, "Canny is OpenCV's");
        s.edge_method = CvEdgeMethod::Sobel;
        cv::Mat gx, gy, ax, ay;
        cv::Sobel(grey, gx, CV_16S, 1, 0, 3);
        cv::Sobel(grey, gy, CV_16S, 0, 1, 3);
        cv::convertScaleAbs(gx, ax);
        cv::convertScaleAbs(gy, ay);
        cv::addWeighted(ax, 0.5, ay, 0.5, 0, expected);
        Check(Worst(Run(s, rgb), expected) == 0, "Sobel magnitude");
        Check(CvImageStepText(s) == "Edges (Sobel 3)", "edges text");
    }
    // Smooth: median on uint8, box and bilateral as the classes run them.
    {
        CvImageStep s = Step(CvImageStepKind::Smooth);
        cv::Mat expected;
        cv::medianBlur(rgb, expected, 5);
        Check(Worst(Run(s, rgb), expected) == 0, "median is OpenCV's");
        s.smooth_method = CvSmoothMethod::Box;
        expected = OnFloat(rgb, [](const cv::Mat& x, cv::Mat& y) { cv::blur(x, y, cv::Size(5, 5)); });
        Check(Worst(Run(s, rgb), expected) <= 1, "box");
        s.smooth_method = CvSmoothMethod::Bilateral;
        Check(Run(s, rgb).size() == kRgb.Size(), "bilateral keeps the shape");
    }
    // Warp: identity corners keep the image; a typed size changes the shape.
    {
        CvImageStep s = Step(CvImageStepKind::Warp);
        Check(Worst(Run(s, rgb), rgb) <= 1, "identity corners keep the image");
        s.target_points = {0.1f, 0.1f, 0.9f, 0.0f, 1.0f, 1.0f, 0.0f, 0.9f};
        s.out_width = 20;
        s.out_height = 16;
        Check(CvImageShapeAfter(s, kRgb) == image::ImageShape{16, 20, 3}, "typed output size");
        const cv::Mat expected = OnFloat(rgb, [&](const cv::Mat& x, cv::Mat& y) {
            std::vector<cv::Point2f> from, to;
            for (int i = 0; i < 4; ++i) {
                from.emplace_back(s.source_points[2 * i] * kW, s.source_points[2 * i + 1] * kH);
                to.emplace_back(s.target_points[2 * i] * 20, s.target_points[2 * i + 1] * 16);
            }
            cv::warpPerspective(x, y, cv::getPerspectiveTransform(from, to), cv::Size(20, 16), cv::INTER_LINEAR);
        });
        Check(Worst(Run(s, rgb), expected) <= 1, "four corners is warpPerspective");
        s.target_points[4] = 1.2f;
        Check(ValidateCvImageStep(s, kRgb) ==
                  "Corner 3 of the target is outside the image: points are fractions from 0 to 1",
              "a corner outside the image: " + ValidateCvImageStep(s, kRgb));
        CvImageStep polar = Step(CvImageStepKind::Warp);
        polar.warp_method = CvWarpMethod::LogPolar;
        Check(Run(polar, rgb).size() == kRgb.Size(), "log-polar keeps the size");
        polar.out_width = 10;
        Check(!ValidateCvImageStep(polar, kRgb).empty(), "a polar warp refuses an output size");
    }
    // Pyramid: down halves (rounding up), detail is mid-grey where flat.
    {
        CvImageStep s = Step(CvImageStepKind::Pyramid);
        s.levels = 2;
        Check(CvImageShapeAfter(s, kRgb) == image::ImageShape{8, 10, 3}, "two levels down");
        cv::Mat expected = OnFloat(rgb, [](const cv::Mat& x, cv::Mat& y) {
            cv::Mat half;
            cv::pyrDown(x, half);
            cv::pyrDown(half, y);
        });
        Check(Worst(Run(s, rgb), expected) <= 1, "pyrDown twice");
        const image::ImageShape side56{56, 56, 3};
        s.levels = 4;
        Check(ValidateCvImageStep(s, side56) == "4 levels down from 56 \xC3\x97 56 gives 4 \xC3\x97 4: at most 3 levels here",
              "too many levels: " + ValidateCvImageStep(s, side56));
        s.levels = 3;
        Check(ValidateCvImageStep(s, side56).empty(), "three levels from 56");
        CvImageStep detail = Step(CvImageStepKind::Pyramid);
        detail.pyramid = CvPyramidOperation::Detail;
        Check(CvImageShapeAfter(detail, kRgb) == kRgb, "detail at level 1 keeps the size");
        const cv::Mat flat(kH, kW, CV_8UC3, cv::Scalar(90, 90, 90));
        const auto flat_detail = Run(detail, flat);
        Check(std::all_of(flat_detail.begin(), flat_detail.end(), [](uint8_t v) { return v == 128; }),
              "no detail is mid-grey");
        Check(Run(detail, rgb) != flat_detail, "an image with edges has detail");
    }
    // Grey input works for every step.
    {
        cv::Mat grey;
        cv::cvtColor(rgb, grey, cv::COLOR_RGB2GRAY);
        const image::ImageShape g{kH, kW, 1};
        for (auto kind : {CvImageStepKind::Contrast, CvImageStepKind::Denoise, CvImageStepKind::Sharpen,
                          CvImageStepKind::Edges, CvImageStepKind::Smooth, CvImageStepKind::Warp,
                          CvImageStepKind::Pyramid})
            Check(Run(Step(kind), grey, g).size() == CvImageShapeAfter(Step(kind), g).Size(), "grey input");
    }
    // Checks.
    {
        CvImageStep s = Step(CvImageStepKind::Contrast);
        s.tile_grid = 64;
        Check(ValidateCvImageStep(s, kRgb) == "tile grid must be from 1 to 32 (the image's shorter side)", "tile grid");
        s = Step(CvImageStepKind::Denoise);
        s.patch_size = 4;
        Check(!ValidateCvImageStep(s, kRgb).empty(), "even patch");
        s = Step(CvImageStepKind::Smooth);
        s.kernel_size = 4;
        Check(!ValidateCvImageStep(s, kRgb).empty(), "even kernel");
        Check(!ValidateCvImageStep(Step(CvImageStepKind::Sharpen), image::ImageShape{kH, kW, 4}).empty(),
              "four channels refused");
    }
    // Identity: every setting is in it.
    {
        CvImageStep a = Step(CvImageStepKind::Contrast), b = a;
        b.clip_limit = 3.0f;
        Check(CvImageStepIdentity(a) != CvImageStepIdentity(b), "another clip limit is another set");
        b = a;
        b.node_id = 99;
        b.position = 4;
        Check(CvImageStepIdentity(a) == CvImageStepIdentity(b), "node and position are not settings");
        Check(CvImageStepIdentity(a).value("text", "") == "CLAHE 2.0 / 8", "identity carries the text");
    }
    // Order among the device transforms, and the compiled shape.
    {
        image::ImageOp crop, flip;
        crop.kind = image::ImageOpKind::CenterCrop;
        crop.width = 20;
        crop.height = 16;
        flip.kind = image::ImageOpKind::HorizontalFlip;
        image::ImageAugmentation compiled;
        compiled.ops = {crop, flip};
        CvImageStep edges = Step(CvImageStepKind::Edges);
        edges.position = 1;
        CvImageStep clahe = Step(CvImageStepKind::Contrast);
        clahe.position = 0;
        const auto merged = MergeImageSteps(compiled, {clahe, edges});
        Check(merged.size() == 4 && std::holds_alternative<CvImageStep>(merged[0]) &&
                  std::holds_alternative<image::ImageOp>(merged[1]) &&
                  std::get<CvImageStep>(merged[2]).kind == CvImageStepKind::Edges &&
                  std::get<image::ImageOp>(merged[3]).kind == image::ImageOpKind::HorizontalFlip,
              "CLAHE, Center Crop, Edges, Flip in canvas order");
        Check(CompiledImageShape(compiled, {clahe, edges}, kRgb) == image::ImageShape{16, 20, 1},
              "the model sees 16 x 20 x 1");
        Check(FixedImageStepText(merged[1]) == "Center Crop 20 \xC3\x97 16" &&
                  FixedImageStepText(merged[2]) == "Edges (Canny 50 / 150)",
              "step texts");
    }

    std::cout << "test_image_cv_steps: all checks passed\n";
    return 0;
}
