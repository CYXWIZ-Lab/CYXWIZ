#include "image_cv_steps.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cyxwiz {

namespace {

const char* kTimes = " \xC3\x97 ";

std::string Number(float v) {
    char out[32];
    std::snprintf(out, sizeof(out), "%g", v);
    return out;
}

// "2.0" for whole numbers, so a setting reads as a setting ("CLAHE 2.0 / 8").
std::string Setting(float v) {
    char out[32];
    std::snprintf(out, sizeof(out), v == std::floor(v) ? "%.1f" : "%g", v);
    return out;
}

std::string Size(size_t width, size_t height) {
    return std::to_string(width) + kTimes + std::to_string(height);
}

bool Odd(int v) { return v > 0 && v % 2 == 1; }

// Twice the signed area of the triangle a, b, c (points as fractions).
float Area2(const std::array<float, 8>& p, int a, int b, int c) {
    return (p[2 * b] - p[2 * a]) * (p[2 * c + 1] - p[2 * a + 1]) -
           (p[2 * c] - p[2 * a]) * (p[2 * b + 1] - p[2 * a + 1]);
}

std::string CheckPoints(const std::array<float, 8>& p, int count, const char* what, const char* noun) {
    for (int i = 0; i < count; ++i) {
        if (!(p[2 * i] >= 0.0f && p[2 * i] <= 1.0f && p[2 * i + 1] >= 0.0f && p[2 * i + 1] <= 1.0f))
            return std::string(noun) + " " + std::to_string(i + 1) + " of the " + what +
                   " is outside the image: points are fractions from 0 to 1";
    }
    for (int a = 0; a < count; ++a)
        for (int b = a + 1; b < count; ++b)
            for (int c = b + 1; c < count; ++c)
                if (std::fabs(Area2(p, a, b, c)) < 1e-4f)
                    return std::string("three of the ") + what + " points lie on one line: the warp is undefined";
    return "";
}

size_t Half(size_t side) { return (side + 1) / 2; }  // cv::pyrDown

const char* EdgeName(CvEdgeMethod m) {
    switch (m) {
        case CvEdgeMethod::Canny: return "Canny";
        case CvEdgeMethod::Sobel: return "Sobel";
        case CvEdgeMethod::Laplacian: return "Laplacian";
        case CvEdgeMethod::Scharr: return "Scharr";
    }
    return "edges";
}

const char* WarpName(CvWarpMethod m) {
    switch (m) {
        case CvWarpMethod::FourCorners: return "four corners";
        case CvWarpMethod::ThreePoints: return "three points";
        case CvWarpMethod::LogPolar: return "log-polar";
        case CvWarpMethod::LinearPolar: return "linear-polar";
    }
    return "warp";
}

bool Polar(CvWarpMethod m) { return m == CvWarpMethod::LogPolar || m == CvWarpMethod::LinearPolar; }

}  // namespace

std::string ValidateCvImageStep(const CvImageStep& s, const image::ImageShape& in) {
    if (in.height == 0 || in.width == 0) return "the image has no pixels";
    if (in.channels != 1 && in.channels != 3) return "needs grey or RGB images (1 or 3 channels)";
    const size_t side = std::min(in.height, in.width);
    switch (s.kind) {
        case CvImageStepKind::Contrast:
            if (!(s.clip_limit > 0.0f)) return "clip limit must be greater than 0";
            if (s.tile_grid < 1 || static_cast<size_t>(s.tile_grid) > side)
                return "tile grid must be from 1 to " + std::to_string(side) + " (the image's shorter side)";
            return "";
        case CvImageStepKind::Denoise:
            if (!(s.strength > 0.0f)) return "strength must be greater than 0";
            if (!(s.colour_strength >= 0.0f)) return "colour strength must be 0 or more";
            if (!Odd(s.patch_size)) return "patch size must be an odd number of pixels";
            if (!Odd(s.search_size) || s.search_size < s.patch_size)
                return "search size must be odd and at least the patch size";
            return "";
        case CvImageStepKind::Sharpen:
            if (!(s.amount >= 0.0f)) return "amount must be 0 or more";
            if (!(s.radius > 0.0f)) return "radius must be greater than 0";
            if (!(s.threshold >= 0.0f && s.threshold <= 1.0f)) return "threshold must be from 0 to 1";
            return "";
        case CvImageStepKind::Edges:
            if (s.edge_method == CvEdgeMethod::Canny) {
                if (!(s.threshold1 >= 0.0f) || !(s.threshold2 >= s.threshold1))
                    return "thresholds must be 0 or more, the second at least the first";
            } else if (s.edge_method == CvEdgeMethod::Sobel) {
                if (s.edge_kernel != 1 && s.edge_kernel != 3 && s.edge_kernel != 5 && s.edge_kernel != 7)
                    return "kernel size must be 1, 3, 5 or 7 for Sobel";
            } else if (s.edge_method == CvEdgeMethod::Laplacian) {
                if (!Odd(s.edge_kernel) || s.edge_kernel > 31) return "kernel size must be odd, at most 31, for Laplacian";
            }
            return "";
        case CvImageStepKind::Smooth:
            if (!Odd(s.kernel_size)) return "kernel size must be an odd number of pixels";
            if (static_cast<size_t>(s.kernel_size) > side)
                return "kernel size must be at most " + std::to_string(side) + " (the image's shorter side)";
            if (s.smooth_method == CvSmoothMethod::Bilateral && !(s.sigma_color > 0.0f && s.sigma_space > 0.0f))
                return "colour and space sigmas must be greater than 0";
            return "";
        case CvImageStepKind::Warp:
            if (s.out_width < 0 || s.out_height < 0) return "output size must be 0 (the input's) or more";
            if (Polar(s.warp_method)) {
                if (s.out_width > 0 || s.out_height > 0)
                    return "a polar warp keeps the input's size: leave the output size at 0";
                if (!(s.center_x >= 0.0f && s.center_x <= 1.0f && s.center_y >= 0.0f && s.center_y <= 1.0f))
                    return "the centre is outside the image: it is a fraction from 0 to 1";
                if (!(s.max_radius > 0.0f)) return "radius must be greater than 0";
                return "";
            } else {
                const int count = s.warp_method == CvWarpMethod::FourCorners ? 4 : 3;
                const char* noun = count == 4 ? "Corner" : "Point";
                if (auto why = CheckPoints(s.source_points, count, "source", noun); !why.empty()) return why;
                return CheckPoints(s.target_points, count, "target", noun);
            }
        case CvImageStepKind::Pyramid: {
            if (s.levels < 1) return "levels must be 1 or more";
            if (s.pyramid == CvPyramidOperation::Up) {
                const size_t longest = std::max(in.height, in.width) << std::min(s.levels, 16);
                if (s.levels > 16 || longest > 4096)
                    return std::to_string(s.levels) + " levels up from " + Size(in.width, in.height) +
                           " is over 4096 pixels a side";
                return "";
            }
            // Down keeps sides of at least 5; Detail at level n subtracts level
            // n + 1, which needs sides of at least 2.
            const size_t smallest = s.pyramid == CvPyramidOperation::Down ? 5 : 2;
            size_t h = in.height, w = in.width;
            int ok = 0;
            for (int i = 0; i < s.levels; ++i) {
                h = Half(h);
                w = Half(w);
                if (std::min(h, w) >= smallest) ok = i + 1;
            }
            if (ok == 0) {
                const size_t h1 = Half(in.height), w1 = Half(in.width);
                return std::string(s.pyramid == CvPyramidOperation::Down ? "1 level down" : "1 level of detail") +
                       " from " + Size(in.width, in.height) + " gives " + Size(w1, h1) + ", under " +
                       std::to_string(smallest) + " pixels a side: use a larger Resize";
            }
            if (ok < s.levels)
                return std::to_string(s.levels) + " levels " +
                       (s.pyramid == CvPyramidOperation::Down ? "down" : "of detail") + " from " +
                       Size(in.width, in.height) + " gives " + Size(w, h) + ": at most " + std::to_string(ok) +
                       " level" + (ok == 1 ? "" : "s") + " here";
            return "";
        }
    }
    return "unknown OpenCV step";
}

image::ImageShape CvImageShapeAfter(const CvImageStep& s, const image::ImageShape& in) {
    image::ImageShape out = in;
    switch (s.kind) {
        case CvImageStepKind::Edges:
            out.channels = 1;
            break;
        case CvImageStepKind::Warp:
            if (s.out_width > 0) out.width = static_cast<size_t>(s.out_width);
            if (s.out_height > 0) out.height = static_cast<size_t>(s.out_height);
            break;
        case CvImageStepKind::Pyramid: {
            const int halvings = s.pyramid == CvPyramidOperation::Down ? s.levels
                                 : s.pyramid == CvPyramidOperation::Detail ? s.levels - 1 : 0;
            for (int i = 0; i < halvings; ++i) {
                out.height = Half(out.height);
                out.width = Half(out.width);
            }
            if (s.pyramid == CvPyramidOperation::Up) {
                out.height <<= s.levels;
                out.width <<= s.levels;
            }
            break;
        }
        default:
            break;
    }
    return out;
}

std::string CvImageStepText(const CvImageStep& s) {
    switch (s.kind) {
        case CvImageStepKind::Contrast:
            return "CLAHE " + Setting(s.clip_limit) + " / " + std::to_string(s.tile_grid) +
                   (s.colour == CvContrastColour::EachChannel ? ", each channel" : "");
        case CvImageStepKind::Denoise:
            return "Denoise " + Number(s.strength) + " / " + Number(s.colour_strength) + ", patch " +
                   std::to_string(s.patch_size) + ", search " + std::to_string(s.search_size);
        case CvImageStepKind::Sharpen:
            return "Sharpen " + Setting(s.amount) + ", radius " + Setting(s.radius) +
                   (s.threshold > 0.0f ? ", threshold " + Number(s.threshold) : "");
        case CvImageStepKind::Edges:
            if (s.edge_method == CvEdgeMethod::Canny)
                return "Edges (Canny " + Number(s.threshold1) + " / " + Number(s.threshold2) + ")";
            if (s.edge_method == CvEdgeMethod::Scharr) return "Edges (Scharr)";
            return std::string("Edges (") + EdgeName(s.edge_method) + " " + std::to_string(s.edge_kernel) + ")";
        case CvImageStepKind::Smooth: {
            const std::string k = std::to_string(s.kernel_size) + kTimes + std::to_string(s.kernel_size);
            if (s.smooth_method == CvSmoothMethod::Box) return "Box " + k;
            if (s.smooth_method == CvSmoothMethod::Median) return "Median " + k;
            return "Bilateral " + std::to_string(s.kernel_size) + ", sigmas " + Number(s.sigma_color) + " / " +
                   Number(s.sigma_space);
        }
        case CvImageStepKind::Warp: {
            std::string text = std::string("Warp (") + WarpName(s.warp_method) + (s.inverse && Polar(s.warp_method) ? ", inverse" : "") + ")";
            if (s.out_width > 0 || s.out_height > 0)
                text += " to " + (s.out_width > 0 ? std::to_string(s.out_width) : std::string("same")) + kTimes +
                        (s.out_height > 0 ? std::to_string(s.out_height) : std::string("same"));
            return text;
        }
        case CvImageStepKind::Pyramid: {
            const char* op = s.pyramid == CvPyramidOperation::Down ? "down"
                             : s.pyramid == CvPyramidOperation::Up ? "up" : "detail";
            return std::string("Pyramid ") + op + " " + std::to_string(s.levels);
        }
    }
    return "OpenCV step";
}

nlohmann::json CvImageStepIdentity(const CvImageStep& s) {
    nlohmann::json j = {{"cv", static_cast<int>(s.kind)}};
    const auto f = [](float v) { return Number(v); };  // text: the same float is the same key everywhere
    switch (s.kind) {
        case CvImageStepKind::Contrast:
            j["clip_limit"] = f(s.clip_limit);
            j["tile_grid"] = s.tile_grid;
            j["colour"] = static_cast<int>(s.colour);
            break;
        case CvImageStepKind::Denoise:
            j["strength"] = f(s.strength);
            j["colour_strength"] = f(s.colour_strength);
            j["patch_size"] = s.patch_size;
            j["search_size"] = s.search_size;
            break;
        case CvImageStepKind::Sharpen:
            j["amount"] = f(s.amount);
            j["radius"] = f(s.radius);
            j["threshold"] = f(s.threshold);
            break;
        case CvImageStepKind::Edges:
            j["method"] = static_cast<int>(s.edge_method);
            j["threshold1"] = f(s.threshold1);
            j["threshold2"] = f(s.threshold2);
            j["kernel"] = s.edge_kernel;
            break;
        case CvImageStepKind::Smooth:
            j["method"] = static_cast<int>(s.smooth_method);
            j["kernel_size"] = s.kernel_size;
            j["sigma_color"] = f(s.sigma_color);
            j["sigma_space"] = f(s.sigma_space);
            break;
        case CvImageStepKind::Warp: {
            j["method"] = static_cast<int>(s.warp_method);
            j["interpolation"] = static_cast<int>(s.interpolation);
            nlohmann::json source = nlohmann::json::array(), target = nlohmann::json::array();
            for (int i = 0; i < 8; ++i) {
                source.push_back(f(s.source_points[i]));
                target.push_back(f(s.target_points[i]));
            }
            j["source"] = source;
            j["target"] = target;
            j["center"] = {f(s.center_x), f(s.center_y)};
            j["max_radius"] = f(s.max_radius);
            j["inverse"] = s.inverse;
            j["out"] = {s.out_width, s.out_height};
            break;
        }
        case CvImageStepKind::Pyramid:
            j["operation"] = static_cast<int>(s.pyramid);
            j["levels"] = s.levels;
            break;
    }
    j["text"] = CvImageStepText(s);
    return j;
}

std::string ImageOpText(const image::ImageOp& op) {
    using K = image::ImageOpKind;
    static const char* kMorphology[] = {"Erode", "Dilate", "Open", "Close", "Gradient", "Top hat", "Black hat"};
    const std::string size = std::to_string(op.width) + " \xC3\x97 " + std::to_string(op.height);
    const std::string kernel = std::to_string(op.kernel_size) + " \xC3\x97 " + std::to_string(op.kernel_size);
    switch (op.kind) {
        case K::CenterCrop: return "Center Crop " + size;
        case K::RandomCrop: return "Random Crop " + size;
        case K::HorizontalFlip: return "Horizontal Flip (p " + Number(op.probability) + ")";
        case K::VerticalFlip: return "Vertical Flip (p " + Number(op.probability) + ")";
        case K::Rotate: return "Rotate up to " + Number(op.max_angle) + " degrees";
        case K::ColorJitter: return "Color Jitter";
        case K::GaussianBlur: return "Gaussian Blur " + kernel + ", sigma " + Number(op.sigma);
        case K::Grayscale: return "Grayscale";
        case K::Morphology: {
            const int m = static_cast<int>(op.morphology);
            return std::string(m >= 0 && m < 7 ? kMorphology[m] : "Morphology") + " " + kernel;
        }
        case K::Erase: return op.erase_method == image::EraseMethod::Cutout ? "Cutout" : "Random Erasing";
        case K::RandAugment: return "RandAugment (" + std::to_string(op.num_ops) + " ops, magnitude " + std::to_string(op.magnitude) + ")";
    }
    return "transform";
}

std::string FixedImageStepText(const FixedImageStep& step) {
    if (const auto* cv_step = std::get_if<CvImageStep>(&step)) return CvImageStepText(*cv_step);
    return ImageOpText(std::get<image::ImageOp>(step));
}

std::vector<FixedImageStep> MergeImageSteps(const image::ImageAugmentation& compiled,
                                            const std::vector<CvImageStep>& cv_steps) {
    std::vector<FixedImageStep> steps;
    size_t next = 0;
    for (size_t i = 0; i <= compiled.ops.size(); ++i) {
        for (; next < cv_steps.size() && cv_steps[next].position <= i; ++next) steps.emplace_back(cv_steps[next]);
        if (i < compiled.ops.size()) steps.emplace_back(compiled.ops[i]);
    }
    for (; next < cv_steps.size(); ++next) steps.emplace_back(cv_steps[next]);
    return steps;
}

image::ImageShape CompiledImageShape(const image::ImageAugmentation& compiled,
                                     const std::vector<CvImageStep>& cv_steps, const image::ImageShape& decoded) {
    image::ImageShape shape = decoded;
    for (const auto& step : MergeImageSteps(compiled, cv_steps)) {
        if (const auto* op = std::get_if<image::ImageOp>(&step)) shape = image::ImageShapeAfter(*op, shape);
        else shape = CvImageShapeAfter(std::get<CvImageStep>(step), shape);
    }
    return shape;
}

}  // namespace cyxwiz
