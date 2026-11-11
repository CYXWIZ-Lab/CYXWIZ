// ApplyCvImageStep: the Engine's OpenCV classes (ImageEnhancer incl. its kernel filter,
// BlurTransform, PerspectiveTransform, PyramidTransform) on one uint8 image.
// They work on float [0, 1]; uint8 in and rounding once at the end keeps
// the prepared set's values what the classes give.

#include "image_cv_steps.h"

#include "../preprocessing/blur_transform.h"
#include "../preprocessing/perspective_transform.h"
#include "../preprocessing/pyramid_transform.h"
#include "../utils/image_enhancer.h"

#include <algorithm>
#include <cmath>

namespace cyxwiz {

namespace {

WarpInterpolation ToWarp(CvWarpInterpolation i) {
    switch (i) {
        case CvWarpInterpolation::Nearest: return WarpInterpolation::Nearest;
        case CvWarpInterpolation::Linear: return WarpInterpolation::Linear;
        case CvWarpInterpolation::Cubic: return WarpInterpolation::Cubic;
        case CvWarpInterpolation::Lanczos: return WarpInterpolation::Lanczos;
    }
    return WarpInterpolation::Linear;
}

ImageEnhancer::EdgeDetectorType ToEdge(CvEdgeMethod m) {
    switch (m) {
        case CvEdgeMethod::Canny: return ImageEnhancer::EdgeDetectorType::Canny;
        case CvEdgeMethod::Sobel: return ImageEnhancer::EdgeDetectorType::Sobel;
        case CvEdgeMethod::Laplacian: return ImageEnhancer::EdgeDetectorType::Laplacian;
        case CvEdgeMethod::Scharr: return ImageEnhancer::EdgeDetectorType::Scharr;
        case CvEdgeMethod::LoG: return ImageEnhancer::EdgeDetectorType::LoG;
        case CvEdgeMethod::DoG: return ImageEnhancer::EdgeDetectorType::DoG;
    }
    return ImageEnhancer::EdgeDetectorType::Canny;
}

ImageEnhancer::EdgeDirection ToDirection(CvEdgeDirection d) {
    switch (d) {
        case CvEdgeDirection::Magnitude: return ImageEnhancer::EdgeDirection::Magnitude;
        case CvEdgeDirection::X: return ImageEnhancer::EdgeDirection::X;
        case CvEdgeDirection::Y: return ImageEnhancer::EdgeDirection::Y;
        case CvEdgeDirection::Angle: return ImageEnhancer::EdgeDirection::Angle;
    }
    return ImageEnhancer::EdgeDirection::Magnitude;
}

// CLAHE on each channel alone, as a grey image.
bool ContrastEachChannel(std::vector<float>& x, int w, int h, int c, float clip, int tiles) {
    const size_t n = static_cast<size_t>(w) * h;
    std::vector<float> plane(n);
    for (int k = 0; k < c; ++k) {
        for (size_t i = 0; i < n; ++i) plane[i] = x[i * c + k];
        if (!ImageEnhancer::ApplyCLAHE(plane, w, h, 1, clip, tiles)) return false;
        for (size_t i = 0; i < n; ++i) x[i * c + k] = plane[i];
    }
    return true;
}

}  // namespace

bool ApplyCvImageStep(const CvImageStep& s, const image::ImageShape& input, const uint8_t* in, uint8_t* out,
                      std::string* error) {
    const auto fail = [&](const std::string& why) {
        if (error) *error = CvImageStepText(s) + ": " + why;
        return false;
    };
    int w = static_cast<int>(input.width), h = static_cast<int>(input.height);
    const int c = static_cast<int>(input.channels);
    // Neither grey nor RGB: the step on each channel alone, as a grey image.
    // (Image Filter's kernel already runs per channel for any count.)
    if (c != 1 && c != 3 && s.kind != CvImageStepKind::Filter) {
        const image::ImageShape plane_in{input.height, input.width, 1};
        const image::ImageShape plane_out = CvImageShapeAfter(s, plane_in);
        if (s.kind == CvImageStepKind::Edges || plane_out.channels != 1) return fail("needs grey or RGB images");
        const size_t pixels_in = input.height * input.width, pixels_out = plane_out.height * plane_out.width;
        std::vector<uint8_t> one(pixels_in), result(pixels_out);
        for (int k = 0; k < c; ++k) {
            for (size_t i = 0; i < pixels_in; ++i) one[i] = in[i * c + k];
            if (!ApplyCvImageStep(s, plane_in, one.data(), result.data(), error)) return false;
            for (size_t i = 0; i < pixels_out; ++i) out[i * c + k] = result[i];
        }
        return true;
    }
    std::vector<float> x(input.Size());
    for (size_t i = 0; i < x.size(); ++i) x[i] = in[i] / 255.0f;
    const std::vector<float> original = s.kind == CvImageStepKind::Edges && s.edge_keep_image ? x : std::vector<float>();
    bool ok = false;
    switch (s.kind) {
        case CvImageStepKind::Contrast:
            ok = s.colour == CvContrastColour::EachChannel && c > 1
                     ? ContrastEachChannel(x, w, h, c, s.clip_limit, s.tile_grid)
                     : ImageEnhancer::ApplyCLAHE(x, w, h, c, s.clip_limit, s.tile_grid);
            break;
        case CvImageStepKind::Denoise:
            ok = ImageEnhancer::ApplyDenoise(x, w, h, c, s.strength, s.colour_strength, s.patch_size, s.search_size);
            break;
        case CvImageStepKind::Sharpen:
            ok = ImageEnhancer::ApplySharpen(x, w, h, c, s.amount, s.radius, s.threshold);
            break;
        case CvImageStepKind::Edges:
            ok = ImageEnhancer::ApplyEdgeDetection(x, w, h, c, ToEdge(s.edge_method), s.threshold1, s.threshold2,
                                                   s.edge_kernel, ToDirection(s.edge_direction),
                                                   s.edge_sigma, s.edge_sigma2);
            if (ok && c > 1) {  // the class repeats the edge map in each channel: keep one
                for (size_t i = 0, n = static_cast<size_t>(w) * h; i < n; ++i) x[i] = x[i * c];
                x.resize(static_cast<size_t>(w) * h);
            }
            if (ok && s.edge_keep_image) {  // the image's channels, then the edge map
                const size_t pixels = static_cast<size_t>(w) * h, width = static_cast<size_t>(c) + 1;
                std::vector<float> joined(pixels * width);
                for (size_t i = 0; i < pixels; ++i) {
                    for (int k = 0; k < c; ++k) joined[i * width + k] = original[i * c + k];
                    joined[i * width + c] = x[i];
                }
                x = std::move(joined);
            }
            break;
        case CvImageStepKind::Smooth: {
            BlurConfig config;
            config.enabled = true;
            config.type = s.smooth_method == CvSmoothMethod::Box      ? BlurType::Box
                          : s.smooth_method == CvSmoothMethod::Median ? BlurType::Median
                                                                       : BlurType::Bilateral;
            config.kernel_size = s.kernel_size;
            config.sigma_color = s.sigma_color;
            config.sigma_space = s.sigma_space;
            ok = BlurTransform::Apply(x, w, h, c, config);
            break;
        }
        case CvImageStepKind::Warp: {
            PerspectiveConfig config;
            config.enabled = true;
            config.type = s.warp_method == CvWarpMethod::FourCorners   ? CoordinateTransform::Perspective
                          : s.warp_method == CvWarpMethod::ThreePoints ? CoordinateTransform::Affine
                          : s.warp_method == CvWarpMethod::LogPolar    ? CoordinateTransform::LogPolar
                                                                       : CoordinateTransform::LinearPolar;
            config.interpolation = ToWarp(s.interpolation);
            config.src_points = s.source_points;
            config.dst_points = s.target_points;
            config.center_x = s.center_x;
            config.center_y = s.center_y;
            config.max_radius = s.max_radius;
            config.inverse_polar = s.inverse;
            config.output_width = s.out_width;
            config.output_height = s.out_height;
            ok = PerspectiveTransform::Apply(x, w, h, c, config);
            break;
        }
        case CvImageStepKind::Filter: {
            FilterKernel k;
            std::string why;
            if (!ResolveFilterKernel(s, k, &why)) return fail(why);
            const auto border = s.filter_border == CvFilterBorder::Reflect     ? ImageEnhancer::KernelBorder::Reflect
                                : s.filter_border == CvFilterBorder::Replicate ? ImageEnhancer::KernelBorder::Replicate
                                : s.filter_border == CvFilterBorder::Zero      ? ImageEnhancer::KernelBorder::Zero
                                                                               : ImageEnhancer::KernelBorder::Reflect101;
            ok = ImageEnhancer::ApplyKernel(x, w, h, c, k.values, k.rows, k.cols, border);
            // Stored 0 to 255: the response's size, or around mid-grey (128 = no response).
            if (ok && s.filter_output == CvFilterOutput::Absolute)
                for (float& v : x) v = std::fabs(v);
            if (ok && s.filter_output == CvFilterOutput::Offset)
                for (float& v : x) v += 128.0f / 255.0f;
            break;
        }
        case CvImageStepKind::Pyramid:
            if (s.pyramid == CvPyramidOperation::Down) ok = PyramidTransform::PyrDownMultiple(x, w, h, c, s.levels);
            else if (s.pyramid == CvPyramidOperation::Up) ok = PyramidTransform::PyrUpMultiple(x, w, h, c, s.levels);
            else ok = PyramidTransform::BuildLaplacianLevel(x, w, h, c, s.levels - 1);
            break;
    }
    if (!ok) return fail("OpenCV failed (see the log)");
    const image::ImageShape expected = CvImageShapeAfter(s, input);
    if (static_cast<size_t>(w) != expected.width || static_cast<size_t>(h) != expected.height ||
        x.size() != expected.Size())
        return fail("gave " + std::to_string(w) + " x " + std::to_string(h) + ", expected " +
                    std::to_string(expected.width) + " x " + std::to_string(expected.height));
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = static_cast<uint8_t>(std::lround(std::clamp(x[i], 0.0f, 1.0f) * 255.0f));
    return true;
}

}  // namespace cyxwiz
