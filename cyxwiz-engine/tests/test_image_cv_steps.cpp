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

        // One direction, and the angle (1.10b).
        const nlohmann::json magnitude_key = CvImageStepIdentity(s);
        Check(!magnitude_key.contains("direction") && !magnitude_key.contains("sigma"),
              "a set prepared before the direction setting keeps its key");
        s.edge_direction = CvEdgeDirection::X;
        Check(Worst(Run(s, rgb), ax) == 0, "Sobel x is |gx|");
        Check(CvImageStepText(s) == "Edges (Sobel 3, x)", "direction in the text");
        Check(CvImageStepIdentity(s) != magnitude_key, "another direction is another set");
        s.edge_direction = CvEdgeDirection::Y;
        Check(Worst(Run(s, rgb), ay) == 0, "Sobel y is |gy|");
        s.edge_direction = CvEdgeDirection::Angle;
        cv::Mat fx, fy, angle;
        gx.convertTo(fx, CV_32F);
        gy.convertTo(fy, CV_32F);
        cv::phase(fx, fy, angle, true);
        angle.convertTo(expected, CV_8U, 255.0 / 360.0);
        Check(Worst(Run(s, rgb), expected) == 0, "Sobel angle: 0 to 360 degrees as 0 to 255");
        {
            // Brighter to the right: the gradient points along +x, angle 0; brighter downwards: 90 degrees = 64.
            cv::Mat ramp_x(kH, kW, CV_8UC3), ramp_y(kH, kW, CV_8UC3);
            for (int y = 0; y < kH; ++y)
                for (int x = 0; x < kW; ++x) {
                    ramp_x.at<cv::Vec3b>(y, x) = cv::Vec3b::all(static_cast<uchar>(x * 5));
                    ramp_y.at<cv::Vec3b>(y, x) = cv::Vec3b::all(static_cast<uchar>(y * 5));
                }
            const auto right = Run(s, ramp_x), down = Run(s, ramp_y);
            const size_t middle = static_cast<size_t>(kH / 2) * kW + kW / 2;
            Check(right[middle] == 0 && down[middle] == 64, "the angle of a ramp to the right is 0, downwards 90 degrees");
        }
        {
            // A 7 x 7 Sobel on a steep ramp passes 16 bits: the angle is read from unclipped gradients.
            cv::Mat steep(kH, kW, CV_8UC3), steep_grey;
            for (int y = 0; y < kH; ++y)
                for (int x = 0; x < kW; ++x)
                    steep.at<cv::Vec3b>(y, x) = cv::Vec3b::all(static_cast<uchar>(std::min(255, x * 20 + y * 10)));
            cv::cvtColor(steep, steep_grey, cv::COLOR_RGB2GRAY);
            CvImageStep wide = s;
            wide.edge_kernel = 7;
            cv::Mat wx, wy, wide_angle, wide_expected;
            cv::Sobel(steep_grey, wx, CV_32F, 1, 0, 7);
            cv::Sobel(steep_grey, wy, CV_32F, 0, 1, 7);
            cv::phase(wx, wy, wide_angle, true);
            wide_angle.convertTo(wide_expected, CV_8U, 255.0 / 360.0);
            Check(Worst(Run(wide, steep), wide_expected) == 0, "Sobel 7 angle on a steep ramp is not clipped");
        }
        s.edge_method = CvEdgeMethod::Scharr;
        s.edge_direction = CvEdgeDirection::Y;
        cv::Scharr(grey, gy, CV_16S, 0, 1);
        cv::convertScaleAbs(gy, expected);
        Check(Worst(Run(s, rgb), expected) == 0, "Scharr y");
        Check(CvImageStepText(s) == "Edges (Scharr, y)", "Scharr text");
        s.edge_method = CvEdgeMethod::Laplacian;
        CvImageStep plain = Step(CvImageStepKind::Edges);
        plain.edge_method = CvEdgeMethod::Laplacian;
        Check(CvImageStepIdentity(s) == CvImageStepIdentity(plain) && CvImageStepText(s) == "Edges (Laplacian 3)",
              "the Laplacian has no direction");

        // LoG: smooth, then the Laplacian; DoG: one blur minus a wider one.
        s = Step(CvImageStepKind::Edges);
        s.edge_method = CvEdgeMethod::LoG;
        s.edge_sigma = 1.4f;
        s.edge_kernel = 5;
        cv::Mat smooth, second;
        cv::GaussianBlur(grey, smooth, cv::Size(0, 0), 1.4f);
        cv::Laplacian(smooth, second, CV_16S, 5);
        cv::convertScaleAbs(second, expected);
        Check(Worst(Run(s, rgb), expected) == 0, "LoG is GaussianBlur then Laplacian");
        Check(CvImageStepText(s) == "Edges (LoG 5, sigma 1.4)", "LoG text");
        CvImageStep other = s;
        other.edge_sigma = 2.0f;
        Check(CvImageStepIdentity(s) != CvImageStepIdentity(other), "another sigma is another set");
        s.edge_sigma = 0.0f;
        Check(ValidateCvImageStep(s, kRgb) == "sigma must be greater than 0", "LoG sigma 0");
        s = Step(CvImageStepKind::Edges);
        s.edge_method = CvEdgeMethod::DoG;
        cv::Mat grey_float, near_blur, far_blur;
        grey.convertTo(grey_float, CV_32F);
        cv::GaussianBlur(grey_float, near_blur, cv::Size(0, 0), 1.0);
        cv::GaussianBlur(grey_float, far_blur, cv::Size(0, 0), 1.6f);
        cv::convertScaleAbs(near_blur - far_blur, expected);
        Check(Worst(Run(s, rgb), expected) == 0, "DoG is blur 1 minus blur 1.6");
        Check(CvImageStepText(s) == "Edges (DoG 1 / 1.6)", "DoG text");
        const cv::Mat flat(kH, kW, CV_8UC3, cv::Scalar(90, 90, 90));
        const auto none = Run(s, flat);
        Check(std::all_of(none.begin(), none.end(), [](uint8_t v) { return v == 0; }), "a flat image has no DoG");
        s.edge_sigma2 = 0.8f;
        Check(ValidateCvImageStep(s, kRgb) == "the second sigma must be greater than the first (1)", "DoG sigmas in order");
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
    // Filter: a kernel over every channel, as cv::filter2D gives it.
    {
        const auto by_opencv = [&](const std::vector<float>& values, int rows, int cols, int border, double add,
                                   bool absolute) {
            return OnFloat(rgb, [&](const cv::Mat& x, cv::Mat& y) {
                cv::filter2D(x, y, CV_32F, cv::Mat(rows, cols, CV_32F, const_cast<float*>(values.data())),
                             cv::Point(-1, -1), add, border);
                if (absolute) y = cv::abs(y);
            });
        };
        CvImageStep s = Step(CvImageStepKind::Filter);  // sharpen 3 x 3
        Check(CvImageShapeAfter(s, kRgb) == kRgb, "a filter keeps the shape");
        Check(Worst(Run(s, rgb), by_opencv({0, -1, 0, -1, 5, -1, 0, -1, 0}, 3, 3, cv::BORDER_REFLECT_101, 0, false)) == 0,
              "sharpen is filter2D with its kernel");
        Check(CvImageStepText(s) == "Filter (Sharpen 3 \xC3\x97 3)", "filter text");

        // Identity changes nothing; a typed identity is the same set.
        s.filter_preset = CvFilterPreset::Identity;
        const auto same = Run(s, rgb);
        Check(std::equal(same.begin(), same.end(), rgb.data), "identity keeps every pixel");
        CvImageStep typed = s;
        typed.filter_preset = CvFilterPreset::Custom;
        typed.filter_kernel = "0 0 0; 0, 1, 0\n0 0 0;";
        Check(Run(typed, rgb) == same, "rows by ';' or a new line, numbers by spaces or commas");

        // Prewitt x, absolute, zero border.
        s = Step(CvImageStepKind::Filter);
        s.filter_preset = CvFilterPreset::PrewittX;
        s.filter_output = CvFilterOutput::Absolute;
        s.filter_border = CvFilterBorder::Zero;
        Check(Worst(Run(s, rgb), by_opencv({-1, 0, 1, -1, 0, 1, -1, 0, 1}, 3, 3, cv::BORDER_CONSTANT, 0, true)) == 0,
              "Prewitt x, its size, zeros past the edge");
        Check(CvImageStepText(s) == "Filter (Prewitt x 3 \xC3\x97 3, absolute)", "absolute in the text");

        // Roberts (2 x 2) around mid-grey, the edge repeated.
        s = Step(CvImageStepKind::Filter);
        s.filter_preset = CvFilterPreset::RobertsY;
        s.filter_output = CvFilterOutput::Offset;
        s.filter_border = CvFilterBorder::Replicate;
        Check(Worst(Run(s, rgb), by_opencv({0, 1, -1, 0}, 2, 2, cv::BORDER_REPLICATE, 128.0 / 255.0, false)) == 0,
              "Roberts y around mid-grey");
        const cv::Mat flat(kH, kW, CV_8UC3, cv::Scalar(90, 90, 90));
        const auto none = Run(s, flat);
        Check(std::all_of(none.begin(), none.end(), [](uint8_t v) { return v == 128; }), "no response is mid-grey");

        // The compass kernels turn clockwise from north.
        s = Step(CvImageStepKind::Filter);
        s.filter_preset = CvFilterPreset::Kirsch;
        FilterKernel k;
        Check(ResolveFilterKernel(s, k) && k.values == std::vector<float>({5, 5, 5, -3, 0, -3, -3, -3, -3}), "Kirsch north");
        s.filter_direction = CvFilterDirection::NE;
        Check(ResolveFilterKernel(s, k) && k.values == std::vector<float>({-3, 5, 5, -3, 0, 5, -3, -3, -3}), "Kirsch north-east");
        Check(CvImageStepText(s) == "Filter (Kirsch NE)", "compass text");
        s.filter_preset = CvFilterPreset::Robinson;
        s.filter_direction = CvFilterDirection::E;
        Check(ResolveFilterKernel(s, k) && k.values == std::vector<float>({-1, 0, 1, -2, 0, 2, -1, 0, 1}), "Robinson east");
        s.filter_direction = CvFilterDirection::S;
        Check(ResolveFilterKernel(s, k) && k.values == std::vector<float>({-1, -2, -1, 0, 0, 0, 1, 2, 1}), "Robinson south");

        // Normalize: a typed box of ones is the box preset; a kernel summing to 0 stays.
        s = Step(CvImageStepKind::Filter);
        s.filter_preset = CvFilterPreset::Custom;
        s.filter_kernel = "1 1 1; 1 1 1; 1 1 1";
        s.filter_normalize = true;
        CvImageStep box = Step(CvImageStepKind::Filter);
        box.filter_preset = CvFilterPreset::Box;
        Check(Run(s, rgb) == Run(box, rgb), "ones normalized are the box");
        Check(Worst(Run(box, rgb), OnFloat(rgb, [](const cv::Mat& x, cv::Mat& y) { cv::blur(x, y, cv::Size(3, 3)); })) <= 1,
              "the box is cv::blur 3 x 3");
        s.filter_kernel = "-1 0 1";
        Check(ResolveFilterKernel(s, k) && k.rows == 1 && k.cols == 3 && k.values == std::vector<float>({-1, 0, 1}),
              "a kernel that sums to 0 is not divided; one row is a kernel");

        // What cannot be read says why.
        s.filter_kernel = "1 2; 3";
        Check(ValidateCvImageStep(s, kRgb) == "the kernel's rows differ in length: row 2 has 1, the rows before it 2", "ragged rows");
        s.filter_kernel = "1 x 1";
        Check(ValidateCvImageStep(s, kRgb) == "the kernel has something that is not a number in row 1", "not a number");
        s.filter_kernel = "1 2 3 -";
        Check(ValidateCvImageStep(s, kRgb) == "the kernel has something that is not a number in row 1", "a lone sign is not a number");
        s.filter_kernel = "0 -1 0; -1 5 2e; 0 -1 0";
        Check(ValidateCvImageStep(s, kRgb) == "the kernel has something that is not a number in row 2", "half a number is not a number");
        s.filter_kernel = "1 1e99 1";
        Check(!ValidateCvImageStep(s, kRgb).empty(), "a number past float's range is refused");
        s.filter_kernel = " ; ";
        Check(!ValidateCvImageStep(s, kRgb).empty(), "an empty kernel");
        s.filter_kernel = "1 1 1 1 1";
        Check(ValidateCvImageStep(s, image::ImageShape{4, 4, 1}) ==
                  "the kernel is 5 \xC3\x97 1: at most 4 each way here (the image's shorter side)",
              "a kernel wider than the image");

        // The set's key: the numbers applied, the border and the output.
        CvImageStep a = Step(CvImageStepKind::Filter), b = a;
        b.filter_border = CvFilterBorder::Zero;
        Check(CvImageStepIdentity(a) != CvImageStepIdentity(b), "another border is another set");
        b = a;
        b.filter_output = CvFilterOutput::Offset;
        Check(CvImageStepIdentity(a) != CvImageStepIdentity(b), "another output is another set");
        b = a;
        b.filter_preset = CvFilterPreset::Kirsch;
        CvImageStep turned = b;
        turned.filter_direction = CvFilterDirection::W;
        Check(CvImageStepIdentity(b) != CvImageStepIdentity(turned), "another direction is another set");
        b = a;
        b.filter_kernel = "9 9 9";  // not read by a preset
        Check(CvImageStepIdentity(a) == CvImageStepIdentity(b), "a preset ignores the typed kernel");
    }
    // Grey input works for every step.
    {
        cv::Mat grey;
        cv::cvtColor(rgb, grey, cv::COLOR_RGB2GRAY);
        const image::ImageShape g{kH, kW, 1};
        for (auto kind : {CvImageStepKind::Contrast, CvImageStepKind::Denoise, CvImageStepKind::Sharpen,
                          CvImageStepKind::Edges, CvImageStepKind::Smooth, CvImageStepKind::Warp,
                          CvImageStepKind::Pyramid, CvImageStepKind::Filter})
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
        Check(ValidateCvImageStep(Step(CvImageStepKind::Edges), image::ImageShape{kH, kW, 4}) ==
                  "needs grey or RGB images: these have 4 channels. Put it before the step that adds channels",
              "Edges needs grey or RGB");
    }
    // More than three channels: every other step runs on each channel alone.
    {
        const image::ImageShape four{kH, kW, 4};
        cv::Mat grey;
        cv::cvtColor(rgb, grey, cv::COLOR_RGB2GRAY);
        std::vector<uint8_t> stacked(four.Size());  // R, G, B and the grey image as a 4th channel
        for (size_t i = 0; i < static_cast<size_t>(kH) * kW; ++i) {
            for (int k = 0; k < 3; ++k) stacked[i * 4 + k] = rgb.data[i * 3 + k];
            stacked[i * 4 + 3] = grey.data[i];
        }
        const image::ImageShape g{kH, kW, 1};
        for (auto kind : {CvImageStepKind::Contrast, CvImageStepKind::Denoise, CvImageStepKind::Sharpen,
                          CvImageStepKind::Smooth, CvImageStepKind::Warp, CvImageStepKind::Pyramid,
                          CvImageStepKind::Filter}) {
            const CvImageStep s = Step(kind);
            const std::string name = CvImageStepText(s);
            Check(ValidateCvImageStep(s, four).empty(), name + " accepts four channels");
            const image::ImageShape after = CvImageShapeAfter(s, four);
            Check(after.channels == 4, name + " keeps four channels");
            std::vector<uint8_t> out(after.Size());
            std::string error;
            Check(ApplyCvImageStep(s, four, stacked.data(), out.data(), &error), name + " runs on four channels: " + error);
            const size_t pixels = after.height * after.width;
            for (int k = 0; k < 4; ++k) {
                std::vector<uint8_t> plane(static_cast<size_t>(kH) * kW), expected(pixels);
                for (size_t i = 0; i < plane.size(); ++i) plane[i] = stacked[i * 4 + k];
                Check(ApplyCvImageStep(s, g, plane.data(), expected.data(), &error), name + " runs on the plane");
                bool same = true;
                for (size_t i = 0; i < pixels; ++i) same = same && out[i * 4 + k] == expected[i];
                Check(same, name + ": channel " + std::to_string(k + 1) + " is the step on that channel alone");
            }
        }
    }
    // Image Edges with the image kept: the image's channels, then the edge map.
    {
        CvImageStep s = Step(CvImageStepKind::Edges);
        const auto map = Run(s, rgb);
        s.edge_keep_image = true;
        Check(CvImageShapeAfter(s, kRgb) == image::ImageShape{kH, kW, 4}, "RGB + edges is four channels");
        Check(CvImageStepText(s) == "Edges (Canny 50 / 150), image kept", "kept in the text");
        const auto kept = Run(s, rgb);
        bool image_same = true, map_same = true;
        for (size_t i = 0; i < static_cast<size_t>(kH) * kW; ++i) {
            for (int k = 0; k < 3; ++k) image_same = image_same && kept[i * 4 + k] == rgb.data[i * 3 + k];
            map_same = map_same && kept[i * 4 + 3] == map[i];
        }
        Check(image_same, "the image's pixels are untouched");
        Check(map_same, "the fourth channel is the edge map");
        CvImageStep plain = Step(CvImageStepKind::Edges);
        Check(CvImageStepIdentity(s) != CvImageStepIdentity(plain) && !CvImageStepIdentity(plain).contains("keep_image"),
              "kept is another set; sets prepared before keep their key");
        cv::Mat grey;
        cv::cvtColor(rgb, grey, cv::COLOR_RGB2GRAY);
        const image::ImageShape g{kH, kW, 1};
        Check(CvImageShapeAfter(s, g) == image::ImageShape{kH, kW, 2} && Run(s, grey, g).size() == g.Size() * 2,
              "grey + edges is two channels");
        Check(CompiledImageChannels({}, {s}, kRgb) == ImageChannelLayout{{"R", "G", "B", "edges"}, 3},
              "RGB + edges: the picture and one named channel");
        CvImageStep sharpen = Step(CvImageStepKind::Sharpen);
        sharpen.position = 0;
        Check(CompiledImageChannels({}, {s, sharpen}, kRgb).names.back() == "edges" &&
                  CompiledImageShape({}, {s, sharpen}, kRgb).channels == 4,
              "a step after it keeps the four channels and their names");
        Check(!ValidateCvImageStep(s, image::ImageShape{kH, kW, 4}).empty(), "a second Image Edges on four channels is refused");
    }
    // What the channels are, through the steps.
    {
        Check(DecodedChannelLayout(3) == ImageChannelLayout{{"R", "G", "B"}, 3}, "a decoded image is RGB");
        Check(DecodedChannelLayout(1) == ImageChannelLayout{{"grey"}, 0}, "or grey");
        Check(DecodedChannelLayout(5).names.back() == "channel 5" && DecodedChannelLayout(5).colour == 0, "or numbered");
        image::ImageAugmentation compiled;
        image::ImageOp crop;
        crop.kind = image::ImageOpKind::CenterCrop;
        crop.width = 20;
        crop.height = 16;
        compiled.ops = {crop};
        CvImageStep sharpen = Step(CvImageStepKind::Sharpen);
        Check(CompiledImageChannels(compiled, {sharpen}, kRgb) == DecodedChannelLayout(3),
              "steps that keep the count keep the picture");
        CvImageStep edges = Step(CvImageStepKind::Edges);
        edges.position = 1;
        Check(CompiledImageChannels(compiled, {sharpen, edges}, kRgb) == ImageChannelLayout{{"edges"}, 0},
              "Image Edges leaves one channel named edges");
        CvImageStep on_grey = Step(CvImageStepKind::Edges);
        Check(CompiledImageChannels({}, {on_grey}, image::ImageShape{kH, kW, 1}) == ImageChannelLayout{{"edges"}, 0},
              "Image Edges on a grey image names its channel edges, not grey");
        on_grey.edge_keep_image = true;
        Check(CompiledImageChannels({}, {on_grey}, image::ImageShape{kH, kW, 1}) == ImageChannelLayout{{"grey", "edges"}, 0},
              "a grey image kept beside its edge map");
        image::ImageOp gray;
        gray.kind = image::ImageOpKind::Grayscale;
        compiled.ops = {gray};
        const auto after_gray = CompiledImageChannels(compiled, {}, kRgb);
        Check(after_gray.names.size() == compiled.ShapeAfter(kRgb).channels && after_gray.colour == (after_gray.names.size() == 3 ? 3u : 0u),
              "Grayscale: one name per channel it leaves");
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

    // Image Gabor: a bank of striped kernels over the grey image, one channel each.
    {
        CvImageStep s = Step(CvImageStepKind::Gabor);
        Check(s.gabor_orientations == 4 && s.gabor_wavelengths == "3, 5" && ValidateCvImageStep(s, image::ImageShape{28, 28, 1}).empty(),
              "the default bank fits a 28 x 28 image");
        s.gabor_wavelengths = "3, 6";
        std::vector<GaborFilter> bank;
        Check(ResolveGaborBank(s, bank) && bank.size() == 8, "four orientations at two wavelengths are eight filters");
        Check(bank[0].degrees == 0.0f && bank[1].degrees == 45.0f && bank[3].degrees == 135.0f &&
                  bank[0].wavelength == 3.0f && bank[4].wavelength == 6.0f,
              "each wavelength, its orientations in turn");
        // The kernels are OpenCV's, then zero mean and absolute values that sum to 1.
        const float pi = 3.14159265358979f;
        for (const size_t g : {size_t{1}, size_t{6}}) {
            const float sigma = bank[g].wavelength / pi * std::sqrt(std::log(2.0f) / 2.0f) * 3.0f;  // 1 octave
            const int side = bank[g].even.rows;
            Check(side == bank[g].side && side == 2 * static_cast<int>(std::ceil(2.0f * sigma / 0.5f)) + 1 &&
                      bank[g].even.cols == side,
                  "the kernel reaches two of the longer sigma each way");
            for (const bool odd : {false, true}) {
                cv::Mat cvk = cv::getGaborKernel(cv::Size(side, side), sigma, bank[g].degrees * pi / 180.0f,
                                                 bank[g].wavelength, 0.5, odd ? -CV_PI / 2 : 0.0, CV_32F);
                cvk -= cv::mean(cvk)[0];
                cvk /= cv::norm(cvk, cv::NORM_L1);
                const FilterKernel& ours = odd ? bank[g].odd : bank[g].even;
                float worst = 0.0f, total = 0.0f, size = 0.0f;
                for (int i = 0; i < side * side; ++i) {
                    worst = std::max(worst, std::fabs(ours.values[static_cast<size_t>(i)] - cvk.at<float>(i / side, i % side)));
                    total += ours.values[static_cast<size_t>(i)];
                    size += std::fabs(ours.values[static_cast<size_t>(i)]);
                }
                Check(worst < 1e-5f, std::string(odd ? "odd" : "even") + " kernel is cv::getGaborKernel's, worst " + std::to_string(worst));
                Check(std::fabs(total) < 1e-4f && std::fabs(size - 1.0f) < 1e-4f, "zero mean, absolute values sum to 1");
            }
        }

        Check(bank[0].side == 15 && bank[4].side == 29 && ValidateCvImageStep(s, kRgb).empty(), "15 x 15 and 29 x 29 kernels fit 40 x 32");
        Check(CvImageShapeAfter(s, kRgb) == image::ImageShape{kH, kW, 8}, "eight channels");
        Check(CvImageStepText(s) == "Gabor (4 orientations, wavelengths 3 / 6)", "text: " + CvImageStepText(s));
        Check(GaborChannelNames(s) == std::vector<std::string>({"gabor 0 deg 3 px", "gabor 45 deg 3 px", "gabor 90 deg 3 px",
                                                                "gabor 135 deg 3 px", "gabor 0 deg 6 px", "gabor 45 deg 6 px",
                                                                "gabor 90 deg 6 px", "gabor 135 deg 6 px"}),
              "channel names");
        ImageChannelLayout expected_layout;
        expected_layout.names = GaborChannelNames(s);
        Check(CompiledImageChannels({}, {s}, kRgb) == expected_layout, "the bank's channels are named, with no colour picture");

        // Energy equals sqrt(even^2 + odd^2) of cv::filter2D on the grey image.
        const std::vector<uint8_t> out = Run(s, rgb);
        cv::Mat grey_f;
        {
            cv::Mat f;
            rgb.convertTo(f, CV_32F, 1.0 / 255.0);
            std::vector<cv::Mat> planes;
            cv::split(f, planes);
            grey_f = 0.299f * planes[0] + 0.587f * planes[1] + 0.114f * planes[2];
        }
        int worst_level = 0;
        for (size_t g = 0; g < bank.size(); ++g) {
            const int side = bank[g].even.rows;
            const cv::Mat ke(side, side, CV_32F, const_cast<float*>(bank[g].even.values.data()));
            const cv::Mat ko(side, side, CV_32F, const_cast<float*>(bank[g].odd.values.data()));
            cv::Mat e, o, energy;
            cv::filter2D(grey_f, e, CV_32F, ke);
            cv::filter2D(grey_f, o, CV_32F, ko);
            cv::magnitude(e, o, energy);
            for (int i = 0; i < kH * kW; ++i) {
                const int want = static_cast<int>(
                    std::lround(std::clamp(1.41421356f * energy.at<float>(i / kW, i % kW), 0.0f, 1.0f) * 255.0f));
                worst_level = std::max(worst_level, std::abs(want - int(out[static_cast<size_t>(i) * 8 + g])));
            }
        }
        Check(worst_level <= 1, "energy is both phases' size, scaled to the range, worst " + std::to_string(worst_level) + " level");

        // Vertical stripes 6 px apart answer in the 0 degree, 6 px channel; a flat image answers nowhere.
        cv::Mat stripes(kH, kW, CV_8UC3), flat(kH, kW, CV_8UC3, cv::Scalar::all(120));
        for (int y = 0; y < kH; ++y)
            for (int x = 0; x < kW; ++x)
                stripes.at<cv::Vec3b>(y, x) = cv::Vec3b::all(static_cast<uchar>(127.5 + 127.5 * std::cos(2.0 * CV_PI * x / 6.0)));
        const auto striped = Run(s, stripes), even_field = Run(s, flat);
        const size_t middle = (static_cast<size_t>(kH / 2) * kW + kW / 2) * 8;
        for (size_t g = 0; g < 8; ++g)
            if (g != 4)
                Check(striped[middle + 4] > striped[middle + g] + 20,
                      "the matching filter answers most (channel " + std::to_string(g) + ": " +
                          std::to_string(striped[middle + g]) + " against " + std::to_string(striped[middle + 4]) + ")");
        Check(*std::max_element(even_field.begin(), even_field.end()) == 0, "a flat image has no energy");

        // One phase is stored around mid-grey.
        CvImageStep even = s;
        even.gabor_response = CvGaborResponse::Even;
        const auto flat_even = Run(even, flat);
        Check(*std::min_element(flat_even.begin(), flat_even.end()) == 128 &&
                  *std::max_element(flat_even.begin(), flat_even.end()) == 128,
              "even: no response is mid-grey");
        Check(CvImageStepText(even) == "Gabor (4 orientations, wavelengths 3 / 6, even)", "response in the text");
        Check(CvImageStepIdentity(even) != CvImageStepIdentity(s), "another response is another set");

        // The image kept: its channels untouched, then the bank's.
        CvImageStep kept = s;
        kept.gabor_keep_image = true;
        Check(CvImageShapeAfter(kept, kRgb) == image::ImageShape{kH, kW, 11}, "RGB + eight channels");
        const auto with_image = Run(kept, rgb);
        bool same = true;
        for (size_t i = 0; i < static_cast<size_t>(kH) * kW; ++i) {
            for (size_t k = 0; k < 3; ++k) same = same && with_image[i * 11 + k] == rgb.data[i * 3 + k];
            for (size_t g = 0; g < 8; ++g) same = same && with_image[i * 11 + 3 + g] == out[i * 8 + g];
        }
        Check(same, "the kept image is untouched and the bank's channels are the bank's");
        const auto kept_layout = CompiledImageChannels({}, {kept}, kRgb);
        Check(kept_layout.colour == 3 && kept_layout.names.size() == 11 && kept_layout.names[3] == "gabor 0 deg 3 px",
              "R, G, B then the bank, the colour picture kept");
        Check(CvImageStepText(kept) == "Gabor (4 orientations, wavelengths 3 / 6), image kept", "kept in the text");

        // The same filters typed another way are the same set; a setting that changes them is another.
        CvImageStep spaced = s;
        spaced.gabor_wavelengths = "3  6";
        nlohmann::json a = CvImageStepIdentity(s), b = CvImageStepIdentity(spaced);
        Check(a == b, "\"3, 6\" and \"3  6\" are one prepared set");
        CvImageStep wider = s;
        wider.gabor_bandwidth = 1.5f;
        Check(CvImageStepIdentity(wider) != CvImageStepIdentity(s), "another bandwidth is another set");

        // What cannot run says why.
        CvImageStep bad = s;
        bad.gabor_wavelengths = "3, x";
        Check(ValidateCvImageStep(bad, kRgb) == "the wavelengths have something that is not a number: x", "not a number");
        bad.gabor_wavelengths = "1";
        Check(ValidateCvImageStep(bad, kRgb) == "a wavelength must be 2 pixels or more (a stripe and its gap)", "too short");
        bad.gabor_wavelengths = "";
        Check(ValidateCvImageStep(bad, kRgb) == "give at least one wavelength, in pixels", "none");
        bad.gabor_wavelengths = "16";
        Check(ValidateCvImageStep(bad, kRgb).find("wavelength 16 needs a 73 \xC3\x97 73 kernel: at most 32") == 0,
              "a kernel larger than the image: " + ValidateCvImageStep(bad, kRgb));
        // A wavelength whose kernel would be huge is refused before anything is built.
        for (const char* huge : {"2000", "1e30"}) {
            bad.gabor_wavelengths = huge;
            std::vector<GaborFilter> none_built;
            Check(ValidateCvImageStep(bad, kRgb).find("needs a kernel wider than 255 pixels") != std::string::npos &&
                      !ResolveGaborBank(bad, none_built) && none_built.empty(),
                  std::string("wavelength ") + huge + " is refused without building a kernel");
        }
        bad.gabor_wavelengths = "3, 6, 3";
        Check(ValidateCvImageStep(bad, kRgb) == "wavelength 3 is given twice", "a wavelength twice");
        // A stripe image at full contrast uses the stored range: the matching filter answers above half.
        Check(striped[middle + 4] > 128, "the matching filter's energy is " + std::to_string(striped[middle + 4]) + " of 255");
        // The text shows the wavelengths as they are read.
        CvImageStep decimal = s;
        decimal.gabor_wavelengths = "3.0,6";
        Check(CvImageStepText(decimal) == CvImageStepText(s), "3.0 reads as 3 in the text");
        bad = s;
        bad.gabor_orientations = 16;
        bad.gabor_wavelengths = "3, 4, 5";
        Check(ValidateCvImageStep(bad, kRgb).find("is 48 channels: at most 32") != std::string::npos, "too many channels");
        bad = s;
        bad.gabor_orientations = 0;
        Check(ValidateCvImageStep(bad, kRgb) == "orientations must be from 1 to 16", "no orientation");
        const image::ImageShape four{kH, kW, 4};
        Check(ValidateCvImageStep(s, four).find("needs grey or RGB images") == 0, "not on four channels");
        std::vector<uint8_t> none(static_cast<size_t>(kH) * kW * 8), four_in(four.Size(), 10);
        std::string error;
        Check(!ApplyCvImageStep(s, four, four_in.data(), none.data(), &error) && error.find("needs grey or RGB") != std::string::npos,
              "and it does not run there");
        // A grey image goes in as it is.
        cv::Mat grey8;
        cv::cvtColor(stripes, grey8, cv::COLOR_RGB2GRAY);
        const image::ImageShape grey_shape{kH, kW, 1};
        const auto from_grey = Run(s, grey8, grey_shape);
        Check(from_grey.size() == striped.size() && from_grey[middle + 4] == striped[middle + 4], "grey in, the same bank out");
    }

    std::cout << "test_image_cv_steps: all checks passed\n";
    return 0;
}
