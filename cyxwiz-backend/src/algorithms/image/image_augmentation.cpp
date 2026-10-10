#include "cyxwiz/image_augmentation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz::image {

namespace {

constexpr double kPi = 3.14159265358979323846;

std::string ShapeText(size_t height, size_t width) {
    return std::to_string(height) + " x " + std::to_string(width);
}

// Python's round(): halves go to the even neighbour (torchvision center_crop).
int RoundHalfEven(double value) {
    return static_cast<int>(std::nearbyint(value));
}

int CenterOffset(size_t size, int crop) {
    return RoundHalfEven((static_cast<double>(size) - crop) / 2.0);
}

const char* KindName(ImageOpKind kind) {
    switch (kind) {
        case ImageOpKind::CenterCrop: return "Center Crop";
        case ImageOpKind::RandomCrop: return "Random Crop";
        case ImageOpKind::HorizontalFlip: return "Horizontal Flip";
        case ImageOpKind::VerticalFlip: return "Vertical Flip";
        case ImageOpKind::Rotate: return "Image Rotate";
        case ImageOpKind::ColorJitter: return "Color Jitter";
        case ImageOpKind::GaussianBlur: return "Image Gaussian Blur";
        case ImageOpKind::Grayscale: return "Grayscale";
        case ImageOpKind::Morphology: return "Morphology Transform";
        case ImageOpKind::Erase: return "Advanced Augment";
        case ImageOpKind::RandAugment: return "Advanced Augment";
    }
    return "image transform";
}

// torchvision _get_inverse_affine_matrix (inverted=True), in double.
std::array<double, 6> InverseAffineMatrix(double center_x, double center_y, double angle_degrees,
                                          double translate_x, double translate_y, double scale,
                                          double shear_x_degrees, double shear_y_degrees) {
    const double rot = angle_degrees * kPi / 180.0;
    const double sx = shear_x_degrees * kPi / 180.0, sy = shear_y_degrees * kPi / 180.0;
    const double a = std::cos(rot - sy) / std::cos(sy);
    const double b = -std::cos(rot - sy) * std::tan(sx) / std::cos(sy) - std::sin(rot);
    const double c = std::sin(rot - sy) / std::cos(sy);
    const double d = -std::sin(rot - sy) * std::tan(sx) / std::cos(sy) + std::cos(rot);
    std::array<double, 6> m = {d / scale, -b / scale, 0.0, -c / scale, a / scale, 0.0};
    m[2] += m[0] * (-center_x - translate_x) + m[1] * (-center_y - translate_y);
    m[5] += m[3] * (-center_x - translate_x) + m[4] * (-center_y - translate_y);
    m[2] += center_x;
    m[5] += center_y;
    return m;
}

// torch.linspace(start, end, steps)[index] in float32.
float LinspaceAt(float start, float end, int steps, int index) {
    const float step = (end - start) / static_cast<float>(steps - 1);
    return index < steps / 2 ? start + step * static_cast<float>(index)
                             : end - step * static_cast<float>(steps - 1 - index);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE

// Rows [N, H*W*C] (HWC per row) <-> the device layout [H, W, C, N].
af::array RowsToImages(const af::array& rows, const ImageShape& shape) {
    const dim_t n = rows.dims(0);
    return af::reorder(af::moddims(rows, n, static_cast<dim_t>(shape.channels),
                                   static_cast<dim_t>(shape.width), static_cast<dim_t>(shape.height)),
                       3, 2, 1, 0);
}

af::array ImagesToRows(const af::array& images) {
    const dim_t n = images.dims(3);
    return af::moddims(af::reorder(images, 3, 2, 1, 0), n, images.dims(0) * images.dims(1) * images.dims(2));
}

std::vector<float> ToFloat(const std::vector<int>& values) {
    std::vector<float> out(values.size());
    std::transform(values.begin(), values.end(), out.begin(), [](int v) { return static_cast<float>(v); });
    return out;
}

std::vector<int> Shifted(std::vector<int> values, int by) {
    for (int& v : values) v += by;
    return values;
}

// A per-sample value [1, 1, 1, N] repeated over [H, W, C, N].
af::array PerSample(const std::vector<float>& values, const af::dim4& dims) {
    const af::array column(af::dim4(1, 1, 1, static_cast<dim_t>(values.size())), values.data());
    return af::tile(column, static_cast<unsigned>(dims[0]), static_cast<unsigned>(dims[1]),
                    static_cast<unsigned>(dims[2]), 1);
}

af::array PerSampleMask(const std::vector<int>& apply, const af::dim4& dims) {
    return PerSample(ToFloat(apply), dims) > 0.5f;
}

// out(i, j, c, n) = images(sy(i, j, n), sx(i, j, n), c, n), 0 outside the image.
// sy and sx are integer-valued [Ho, Wo, 1, N].
af::array GatherPixels(const af::array& images, const af::array& sy, const af::array& sx) {
    const dim_t h = images.dims(0), w = images.dims(1), c = images.dims(2), n = images.dims(3);
    const dim_t ho = sy.dims(0), wo = sy.dims(1);
    const af::array valid = (sy >= 0) && (sy <= static_cast<double>(h - 1)) &&
                            (sx >= 0) && (sx <= static_cast<double>(w - 1));
    const af::array cy = af::clamp(sy, 0.0, static_cast<double>(h - 1)).as(s64);
    const af::array cx = af::clamp(sx, 0.0, static_cast<double>(w - 1)).as(s64);
    const af::array sample = af::iota(af::dim4(1, 1, 1, n), af::dim4(ho, wo, 1, 1), s64);
    const af::array base = cy + cx * h + sample * (h * w * c);
    const af::array channel = af::iota(af::dim4(1, 1, c, 1), af::dim4(ho, wo, 1, n), s64) * (h * w);
    const af::array index = af::tile(base, 1, 1, static_cast<unsigned>(c), 1) + channel;
    const af::array picked = af::moddims(af::lookup(af::flat(images), af::flat(index), 0), ho, wo, c, n);
    return picked * af::tile(valid, 1, 1, static_cast<unsigned>(c), 1).as(images.type());
}

af::array Crop(const af::array& images, const std::vector<int>& top, const std::vector<int>& left,
               int height, int width) {
    const dim_t n = images.dims(3);
    const af::dim4 out(height, width, 1, n);
    const af::array rows = af::iota(af::dim4(height, 1, 1, 1), af::dim4(1, width, 1, n), f32);
    const af::array cols = af::iota(af::dim4(1, width, 1, 1), af::dim4(height, 1, 1, n), f32);
    return GatherPixels(images, rows + PerSample(ToFloat(top), out), cols + PerSample(ToFloat(left), out));
}

af::array RoundHalfEven(const af::array& value) {
    const af::array low = af::floor(value);
    const af::array fraction = value - low;
    const af::array odd = (low - 2.0 * af::floor(low / 2.0)) > 0.5;
    return low + ((fraction > 0.5) || ((fraction == 0.5) && odd)).as(value.type());
}

// torchvision's affine grid sampling with each sample's inverse matrix: in
// float32, step for step, the matrix is scaled by (0.5 W, 0.5 H) and applied
// to pixel centres relative to the middle (_affine_grid), then grid_sample
// (zero padding, align_corners=False) unnormalises ((g + 1) * size - 1) / 2.
// Half-pixel ties (90 degrees on an even side) round as torchvision's do.
af::array AffineSample(const af::array& images, const std::vector<std::array<double, 6>>& matrices,
                       Interpolation interpolation) {
    const dim_t h = images.dims(0), w = images.dims(1), n = images.dims(3);
    const af::dim4 grid(h, w, 1, n);
    const float half_w = 0.5f * static_cast<float>(w), half_h = 0.5f * static_cast<float>(h);
    std::vector<float> ax(matrices.size()), bx(matrices.size()), cx(matrices.size());
    std::vector<float> ay(matrices.size()), by(matrices.size()), cy(matrices.size());
    for (size_t i = 0; i < matrices.size(); ++i) {
        const auto& m = matrices[i];
        ax[i] = static_cast<float>(m[0]) / half_w;
        bx[i] = static_cast<float>(m[1]) / half_w;
        cx[i] = static_cast<float>(m[2]) / half_w;
        ay[i] = static_cast<float>(m[3]) / half_h;
        by[i] = static_cast<float>(m[4]) / half_h;
        cy[i] = static_cast<float>(m[5]) / half_h;
    }
    const af::array x = af::iota(af::dim4(1, w, 1, 1), af::dim4(h, 1, 1, n), f32) - (half_w - 0.5f);
    const af::array y = af::iota(af::dim4(h, 1, 1, 1), af::dim4(1, w, 1, n), f32) - (half_h - 0.5f);
    const af::array gx = x * PerSample(ax, grid) + y * PerSample(bx, grid) + PerSample(cx, grid);
    const af::array gy = x * PerSample(ay, grid) + y * PerSample(by, grid) + PerSample(cy, grid);
    const af::array sx = ((gx + 1.0f) * static_cast<float>(w) - 1.0f) / 2.0f;
    const af::array sy = ((gy + 1.0f) * static_cast<float>(h) - 1.0f) / 2.0f;
    if (interpolation == Interpolation::Nearest) {
        return GatherPixels(images, RoundHalfEven(sy), RoundHalfEven(sx));
    }
    const af::array x0 = af::floor(sx), y0 = af::floor(sy);
    const af::array wx = sx - x0, wy = sy - y0;
    const unsigned c = static_cast<unsigned>(images.dims(2));
    const auto weight = [c](const af::array& v) { return af::tile(v, 1, 1, c, 1); };
    return GatherPixels(images, y0, x0) * weight((1 - wy) * (1 - wx)) +
           GatherPixels(images, y0, x0 + 1) * weight((1 - wy) * wx) +
           GatherPixels(images, y0 + 1, x0) * weight(wy * (1 - wx)) +
           GatherPixels(images, y0 + 1, x0 + 1) * weight(wy * wx);
}

// torchvision F.rotate: the inverse affine of -angle about the image centre.
af::array Rotate(const af::array& images, const std::vector<float>& degrees, Interpolation interpolation) {
    std::vector<std::array<double, 6>> matrices(degrees.size());
    for (size_t i = 0; i < degrees.size(); ++i) {
        matrices[i] = InverseAffineMatrix(0.0, 0.0, -static_cast<double>(degrees[i]), 0.0, 0.0, 1.0, 0.0, 0.0);
    }
    return AffineSample(images, matrices, interpolation);
}

af::array Luminance(const af::array& images) {
    return 0.2989f * images(af::span, af::span, 0, af::span) +
           0.587f * images(af::span, af::span, 1, af::span) +
           0.114f * images(af::span, af::span, 2, af::span);
}

af::array Blend(const af::array& image, const af::array& other, const af::array& factor) {
    return af::clamp(factor * image + (1.0f - factor) * other, 0.0, 1.0);
}

af::array AdjustBrightness(const af::array& images, const af::array& factor) {
    return af::clamp(factor * images, 0.0, 1.0);
}

af::array AdjustContrast(const af::array& images, const af::array& factor) {
    const af::array gray = images.dims(2) == 3 ? Luminance(images) : images;
    const dim_t pixels = gray.dims(0) * gray.dims(1) * gray.dims(2);
    const af::array mean = af::sum(af::moddims(gray, pixels, 1, 1, gray.dims(3)), 0) / static_cast<float>(pixels);
    const af::array means = af::tile(mean, static_cast<unsigned>(images.dims(0)),
                                     static_cast<unsigned>(images.dims(1)),
                                     static_cast<unsigned>(images.dims(2)), 1);
    return Blend(images, means, factor);
}

af::array AdjustSaturation(const af::array& images, const af::array& factor) {
    if (images.dims(2) != 3) return images;
    return Blend(images, af::tile(Luminance(images), 1, 1, 3, 1), factor);
}

// torchvision _rgb2hsv / _hsv2rgb.
af::array AdjustHue(const af::array& images, const af::array& shift) {
    if (images.dims(2) != 3) return images;
    const af::array r = images(af::span, af::span, 0, af::span);
    const af::array g = images(af::span, af::span, 1, af::span);
    const af::array b = images(af::span, af::span, 2, af::span);
    const af::array maxc = (af::max)((af::max)(r, g), b);
    const af::array minc = (af::min)((af::min)(r, g), b);
    const af::array equal = maxc == minc;
    const af::array range = maxc - minc;
    const af::array ones = af::constant(1.0f, maxc.dims());
    const af::array s = range / af::select(equal, ones, maxc);
    const af::array divisor = af::select(equal, ones, range);
    const af::array rc = (maxc - r) / divisor, gc = (maxc - g) / divisor, bc = (maxc - b) / divisor;
    const af::array is_r = maxc == r, is_g = (maxc == g) && !is_r;
    const af::array hue = is_r.as(f32) * (bc - gc) + is_g.as(f32) * (2.0f + rc - bc) +
                          (!is_g && !is_r).as(f32) * (4.0f + gc - rc);
    af::array h = hue / 6.0f + 1.0f;
    h = h - af::floor(h);
    const af::array shift1 = shift(af::span, af::span, 0, af::span);
    h = h + shift1;
    h = h - af::floor(h);

    const af::array v = maxc;
    const af::array sector = af::floor(h * 6.0f);
    const af::array f = h * 6.0f - sector;
    const af::array p = af::clamp(v * (1.0f - s), 0.0, 1.0);
    const af::array q = af::clamp(v * (1.0f - s * f), 0.0, 1.0);
    const af::array t = af::clamp(v * (1.0f - s * (1.0f - f)), 0.0, 1.0);
    const af::array i = sector - 6.0f * af::floor(sector / 6.0f);
    const auto pick = [&i](const af::array (&by_sector)[6]) {
        af::array out = by_sector[0];
        for (int k = 1; k < 6; ++k) out = af::select(i == static_cast<float>(k), by_sector[k], out);
        return out;
    };
    const af::array red[6] = {v, q, p, p, t, v};
    const af::array green[6] = {t, v, v, q, p, p};
    const af::array blue[6] = {p, p, t, v, v, q};
    return af::join(2, pick(red), pick(green), pick(blue));
}

af::array ColorJitter(af::array images, const ImageOpDraws& draws) {
    const size_t n = draws.order.size();
    const size_t steps = n == 0 ? 0 : draws.order.front().size();
    const af::dim4 dims = images.dims();
    for (size_t step = 0; step < steps; ++step) {
        for (int op = 0; op < 4; ++op) {
            std::vector<int> chosen(n);
            std::vector<float> factor(n);
            bool any = false;
            for (size_t i = 0; i < n; ++i) {
                chosen[i] = draws.order[i][step] == op ? 1 : 0;
                factor[i] = draws.factors[i][op];
                any = any || chosen[i];
            }
            if (!any) continue;
            const af::array f = PerSample(factor, dims);
            af::array adjusted;
            switch (op) {
                case 0: adjusted = AdjustBrightness(images, f); break;
                case 1: adjusted = AdjustContrast(images, f); break;
                case 2: adjusted = AdjustSaturation(images, f); break;
                default: adjusted = AdjustHue(images, f); break;
            }
            images = af::select(PerSampleMask(chosen, dims), adjusted, images);
        }
    }
    return images;
}

// The [size, size] operator of a 1-D Gaussian with reflect padding
// (torchvision gaussian_blur: kernel linspace(-(k-1)/2, (k-1)/2), reflect pad k/2).
af::array BlurOperator(size_t size, int kernel_size, float sigma) {
    const int radius = kernel_size / 2;
    std::vector<double> kernel(static_cast<size_t>(kernel_size));
    double total = 0.0;
    for (int t = 0; t < kernel_size; ++t) {
        const double x = (t - (kernel_size - 1) * 0.5) / sigma;
        kernel[t] = std::exp(-0.5 * x * x);
        total += kernel[t];
    }
    const int n = static_cast<int>(size);
    std::vector<float> matrix(size * size, 0.0f);  // column-major [out, in]
    for (int out = 0; out < n; ++out) {
        for (int t = -radius; t <= radius; ++t) {
            int in = out + t;
            if (in < 0) in = -in;
            if (in >= n) in = 2 * (n - 1) - in;
            matrix[static_cast<size_t>(in) * size + out] += static_cast<float>(kernel[t + radius] / total);
        }
    }
    return af::array(static_cast<dim_t>(size), static_cast<dim_t>(size), matrix.data());
}

af::array GaussianBlur(const af::array& images, int kernel_size, float sigma) {
    const dim_t h = images.dims(0), w = images.dims(1), c = images.dims(2), n = images.dims(3);
    const af::array along_h = af::moddims(
        af::matmul(BlurOperator(static_cast<size_t>(h), kernel_size, sigma), af::moddims(images, h, w * c * n)),
        h, w, c, n);
    const af::array by_width = af::moddims(af::reorder(along_h, 1, 0, 2, 3), w, h * c * n);
    const af::array along_w = af::moddims(
        af::matmul(BlurOperator(static_cast<size_t>(w), kernel_size, sigma), by_width), w, h, c, n);
    return af::reorder(along_w, 1, 0, 2, 3);
}

// Max over a (2r+1)-wide window along one axis (0 = H, 1 = W); pixels
// outside the image are ignored, as torch max_pool2d's implicit padding.
af::array WindowMax(const af::array& images, int radius, int axis) {
    if (radius == 0) return images;
    af::dim4 dims = images.dims();
    const dim_t size = dims[axis];
    af::dim4 padded_dims = dims;
    padded_dims[axis] = size + 2 * radius;
    af::array padded = af::constant(-std::numeric_limits<float>::infinity(), padded_dims, images.type());
    const af::seq inner(radius, radius + static_cast<double>(size) - 1);
    if (axis == 0) {
        padded(inner, af::span, af::span, af::span) = images;
    } else {
        padded(af::span, inner, af::span, af::span) = images;
    }
    af::array out;
    for (int d = 0; d <= 2 * radius; ++d) {
        const af::seq window(d, d + static_cast<double>(size) - 1);
        const af::array slice = axis == 0 ? padded(window, af::span, af::span, af::span)
                                          : padded(af::span, window, af::span, af::span);
        out = d == 0 ? slice : (af::max)(out, slice);
    }
    return out;
}

af::array Dilate(const af::array& images, int kernel_size) {
    const int radius = kernel_size / 2;
    return WindowMax(WindowMax(images, radius, 0), radius, 1);
}

af::array Erode(const af::array& images, int kernel_size) {
    return -Dilate(-images, kernel_size);
}

af::array Morphology(const af::array& images, MorphologyOp operation, int kernel_size) {
    switch (operation) {
        case MorphologyOp::Erode: return Erode(images, kernel_size);
        case MorphologyOp::Dilate: return Dilate(images, kernel_size);
        case MorphologyOp::Open: return Dilate(Erode(images, kernel_size), kernel_size);
        case MorphologyOp::Close: return Erode(Dilate(images, kernel_size), kernel_size);
        case MorphologyOp::Gradient: return Dilate(images, kernel_size) - Erode(images, kernel_size);
        case MorphologyOp::TopHat: return images - Dilate(Erode(images, kernel_size), kernel_size);
        case MorphologyOp::BlackHat: return Erode(Dilate(images, kernel_size), kernel_size) - images;
    }
    return images;
}

// torchvision adjust_sharpness: the interior blends towards the 3x3 smoothing
// kernel [1 1 1; 1 5 1; 1 1 1] / 13 with weight 1 - factor; borders keep their
// pixels; images 2 pixels or less on a side are unchanged.
af::array AdjustSharpness(const af::array& images, const af::array& factor) {
    const dim_t h = images.dims(0), w = images.dims(1);
    if (h <= 2 || w <= 2) return images;
    const af::seq inner_r(1, static_cast<double>(h) - 2), inner_c(1, static_cast<double>(w) - 2);
    af::array window_sum = af::constant(0.0f, af::dim4(h - 2, w - 2, images.dims(2), images.dims(3)));
    for (int dr = 0; dr < 3; ++dr) {
        for (int dc = 0; dc < 3; ++dc) {
            window_sum += images(af::seq(dr, static_cast<double>(h) - 3 + dr),
                                 af::seq(dc, static_cast<double>(w) - 3 + dc), af::span, af::span);
        }
    }
    const af::array center = images(inner_r, inner_c, af::span, af::span);
    const af::array blurred = (window_sum + 4.0f * center) / 13.0f;
    af::array out = images.copy();
    out(inner_r, inner_c, af::span, af::span) =
        center + (blurred - center) * (1.0f - factor(inner_r, inner_c, af::span, af::span));
    return af::clamp(out, 0.0, 1.0);
}

// torchvision posterize on floats: floor(x * 2^bits) clamped, / 2^bits.
af::array Posterize(const af::array& images, const af::array& levels) {
    return af::clamp(af::floor(images * levels), 0.0f, levels - 1.0f) / levels;
}

af::array Solarize(const af::array& images, const af::array& threshold) {
    return af::select(images >= threshold, 1.0f - images, images);
}

// torchvision autocontrast: each channel stretched from its min..max to 0..1
// (unchanged where min == max).
af::array AutoContrast(const af::array& images) {
    const dim_t h = images.dims(0), w = images.dims(1), c = images.dims(2), n = images.dims(3);
    const af::array flat = af::moddims(images, h * w, 1, c, n);
    af::array low = (af::min)(flat, 0), high = (af::max)(flat, 0);
    const af::array equal = high == low;
    af::array scale = high - low;
    low = af::select(equal, af::constant(0.0f, low.dims()), low);
    scale = af::select(equal, af::constant(1.0f, scale.dims()), scale);
    const auto spread = [h, w](const af::array& v) {
        return af::tile(v, static_cast<unsigned>(h), static_cast<unsigned>(w), 1, 1);
    };
    return af::clamp((images - spread(low)) / spread(scale), 0.0, 1.0);
}

// torchvision equalize on floats: to uint8 as trunc(x * 255.999), a 256-bin
// histogram per image and channel, the PIL step lookup, back to x / 255.
af::array Equalize(const af::array& images) {
    const dim_t h = images.dims(0), w = images.dims(1), c = images.dims(2), n = images.dims(3);
    const dim_t pixels = h * w, slices = c * n;
    const af::array bytes = af::floor(images * (255.0f + 1.0f - 1e-3f));
    const af::array flat = af::moddims(bytes, pixels, slices);  // [pixels, slices]
    const af::array slice = af::iota(af::dim4(1, slices), af::dim4(pixels, 1), f32) * 256.0f;
    const af::array global = af::flat(flat + slice);
    const af::array hist = af::moddims(
        af::histogram(global, static_cast<unsigned>(256 * slices), 0.0, 256.0 * static_cast<double>(slices)).as(f32),
        256, slices);
    const af::array cum = af::accum(hist, 0);
    const af::array total = cum(255, af::span);
    // argmax of the cumulative histogram = the first bin where it reaches the total.
    const af::array reached = (cum >= af::tile(total, 256, 1)).as(f32);
    const af::array first = 256.0f - af::sum(reached, 0);
    const af::array bins = af::iota(af::dim4(256, 1), af::dim4(1, slices), f32);
    const af::array at_first = af::sum(hist * (bins == af::tile(first, 256, 1)).as(f32), 0);
    const af::array step = af::floor((static_cast<float>(pixels) - at_first) / 255.0f);
    const af::array divisor = (af::max)(step, 1.0f);
    af::array lut = af::clamp(af::floor((cum + af::floor(step / 2.0f)) / af::tile(divisor, 256, 1)), 0.0, 255.0);
    // lut[v] uses the cumulative count below v: shift by one, lut[0] = 0.
    lut = af::join(0, af::constant(0.0f, af::dim4(1, slices)), lut(af::seq(0, 254), af::span));
    const af::array equalized = af::moddims(
        af::lookup(af::flat(lut), af::flat(flat + slice), 0), pixels, slices);
    const af::array valid = af::tile(step != 0.0f, static_cast<unsigned>(pixels), 1);
    return af::moddims(af::select(valid, equalized, flat), h, w, c, n) / 255.0f;
}

af::array RandAugmentStep(const af::array& images, const std::vector<int>& ops, const std::vector<float>& magnitudes,
                          int op_index, Interpolation interpolation) {
    const af::dim4 dims = images.dims();
    const size_t n = ops.size();
    const RandAugmentOp op = static_cast<RandAugmentOp>(op_index);
    const auto per_sample = [&](float neutral, auto value_of) {
        std::vector<float> values(n, neutral);
        for (size_t i = 0; i < n; ++i) {
            if (ops[i] == op_index) values[i] = value_of(magnitudes[i]);
        }
        return PerSample(values, dims);
    };
    const auto affine = [&](auto matrix_of) {
        std::vector<std::array<double, 6>> matrices(n, std::array<double, 6>{1, 0, 0, 0, 1, 0});
        for (size_t i = 0; i < n; ++i) {
            if (ops[i] == op_index) matrices[i] = matrix_of(static_cast<double>(magnitudes[i]));
        }
        return AffineSample(images, matrices, interpolation);
    };
    const double w = static_cast<double>(dims[1]), h = static_cast<double>(dims[0]);
    switch (op) {
        case RandAugmentOp::Identity:
            return images;
        case RandAugmentOp::ShearX:  // centre [0, 0] = the top-left corner
            return affine([&](double m) {
                return InverseAffineMatrix(-0.5 * w, -0.5 * h, 0.0, 0.0, 0.0, 1.0, std::atan(m) * 180.0 / kPi, 0.0);
            });
        case RandAugmentOp::ShearY:
            return affine([&](double m) {
                return InverseAffineMatrix(-0.5 * w, -0.5 * h, 0.0, 0.0, 0.0, 1.0, 0.0, std::atan(m) * 180.0 / kPi);
            });
        case RandAugmentOp::TranslateX:
            return affine([](double m) {
                return InverseAffineMatrix(0.0, 0.0, 0.0, static_cast<double>(static_cast<int>(m)), 0.0, 1.0, 0.0, 0.0);
            });
        case RandAugmentOp::TranslateY:
            return affine([](double m) {
                return InverseAffineMatrix(0.0, 0.0, 0.0, 0.0, static_cast<double>(static_cast<int>(m)), 1.0, 0.0, 0.0);
            });
        case RandAugmentOp::Rotate:
            return affine([](double m) { return InverseAffineMatrix(0.0, 0.0, -m, 0.0, 0.0, 1.0, 0.0, 0.0); });
        case RandAugmentOp::Brightness:
            return AdjustBrightness(images, per_sample(1.0f, [](float m) { return 1.0f + m; }));
        case RandAugmentOp::Color:
            return AdjustSaturation(images, per_sample(1.0f, [](float m) { return 1.0f + m; }));
        case RandAugmentOp::Contrast:
            return AdjustContrast(images, per_sample(1.0f, [](float m) { return 1.0f + m; }));
        case RandAugmentOp::Sharpness:
            return AdjustSharpness(images, per_sample(1.0f, [](float m) { return 1.0f + m; }));
        case RandAugmentOp::Posterize:
            return Posterize(images, per_sample(256.0f, [](float m) {
                return static_cast<float>(1 << static_cast<int>(m));
            }));
        case RandAugmentOp::Solarize:
            return Solarize(images, per_sample(2.0f, [](float m) { return m; }));
        case RandAugmentOp::AutoContrast:
            return AutoContrast(images);
        case RandAugmentOp::Equalize:
            return Equalize(images);
    }
    return images;
}

af::array RandAugment(af::array images, const ImageOpDraws& draws, Interpolation interpolation) {
    const size_t n = draws.order.size();
    const size_t steps = n == 0 ? 0 : draws.order.front().size();
    const af::dim4 dims = images.dims();
    for (size_t step = 0; step < steps; ++step) {
        std::vector<int> ops(n);
        std::vector<float> magnitudes(n);
        for (size_t i = 0; i < n; ++i) {
            ops[i] = draws.order[i][step];
            magnitudes[i] = draws.factors[i][step];
        }
        for (int op = 1; op < kRandAugmentOps; ++op) {
            std::vector<int> chosen(n);
            bool any = false;
            for (size_t i = 0; i < n; ++i) {
                chosen[i] = ops[i] == op ? 1 : 0;
                any = any || chosen[i];
            }
            if (!any) continue;
            images = af::select(PerSampleMask(chosen, dims),
                                RandAugmentStep(images, ops, magnitudes, op, interpolation), images);
        }
    }
    return images;
}

// Sets each sample's box [top, top + h) x [left, left + w) to value
// (torchvision F.erase); a 0-sized box leaves the sample unchanged.
af::array Erase(const af::array& images, const ImageOpDraws& draws, float value) {
    const af::dim4 dims = images.dims();
    const af::dim4 grid(dims[0], dims[1], 1, dims[3]);
    const af::array rows = af::iota(af::dim4(dims[0], 1, 1, 1), af::dim4(1, dims[1], 1, dims[3]), f32);
    const af::array cols = af::iota(af::dim4(1, dims[1], 1, 1), af::dim4(dims[0], 1, 1, dims[3]), f32);
    const af::array top = PerSample(ToFloat(draws.top), grid);
    const af::array left = PerSample(ToFloat(draws.left), grid);
    const af::array inside = (rows >= top) && (rows < top + PerSample(ToFloat(draws.box_height), grid)) &&
                             (cols >= left) && (cols < left + PerSample(ToFloat(draws.box_width), grid));
    return af::select(af::tile(inside, 1, 1, static_cast<unsigned>(dims[2]), 1),
                      af::constant(value, dims, images.type()), images);
}

// torchvision v2: mixed = rolled * (1 - lambda) + batch * lambda (MixUp), or
// the box pasted from the batch rolled by one (CutMix); roll(1, 0) means
// sample i takes sample i - 1, so sample 0 takes the last one.
af::array MixImages(BatchMix method, const BatchMixDraw& draw, const af::array& images) {
    const af::array rolled = af::shift(images, 0, 0, 0, 1);
    if (method == BatchMix::MixUp) {
        return rolled * (1.0f - draw.lambda) + images * draw.lambda;
    }
    af::array mixed = images.copy();
    if (draw.height > 0 && draw.width > 0) {
        const af::seq rows(draw.top, draw.top + draw.height - 1), cols(draw.left, draw.left + draw.width - 1);
        mixed(rows, cols, af::span, af::span) = rolled(rows, cols, af::span, af::span);
    }
    return mixed;
}

af::array MixLabels(const BatchMixDraw& draw, const af::array& targets) {  // [N, C]
    return af::shift(targets, 1) * (1.0f - draw.lambda) + targets * draw.lambda;
}

af::array ApplyOnDevice(const ImageOp& op, const ImageOpDraws& draws, const ImageShape& shape,
                        const af::array& images) {
    const af::dim4 dims = images.dims();
    switch (op.kind) {
        case ImageOpKind::CenterCrop: {
            const int top = CenterOffset(shape.height, op.height);
            const int left = CenterOffset(shape.width, op.width);
            return images(af::seq(top, top + op.height - 1), af::seq(left, left + op.width - 1),
                          af::span, af::span);
        }
        case ImageOpKind::RandomCrop:
            if (draws.top.size() != static_cast<size_t>(dims[3])) {
                throw std::invalid_argument("Random Crop needs one position per sample");
            }
            return Crop(images, Shifted(draws.top, -op.padding), Shifted(draws.left, -op.padding),
                        op.height, op.width);
        case ImageOpKind::HorizontalFlip:
        case ImageOpKind::VerticalFlip:
            if (draws.apply.empty()) return images;
            return af::select(PerSampleMask(draws.apply, dims),
                              af::flip(images, op.kind == ImageOpKind::HorizontalFlip ? 1 : 0), images);
        case ImageOpKind::Rotate:
            if (draws.angle.empty()) return images;
            return Rotate(images, draws.angle, op.interpolation);
        case ImageOpKind::ColorJitter:
            if (draws.order.empty()) return images;
            return ColorJitter(images, draws);
        case ImageOpKind::GaussianBlur:
            return GaussianBlur(images, op.kernel_size, op.sigma);
        case ImageOpKind::Grayscale:
            return shape.channels == 3 ? Luminance(images) : images;
        case ImageOpKind::Morphology:
            return Morphology(images, op.morphology, op.kernel_size);
        case ImageOpKind::Erase:
            if (draws.box_height.empty()) return images;
            return Erase(images, draws, op.value);
        case ImageOpKind::RandAugment:
            if (draws.order.empty()) return images;
            return RandAugment(images, draws, op.interpolation);
    }
    return images;
}

#endif  // CYXWIZ_HAS_ARRAYFIRE

}  // namespace

bool IsRandomImageOp(ImageOpKind kind) {
    switch (kind) {
        case ImageOpKind::RandomCrop:
        case ImageOpKind::HorizontalFlip:
        case ImageOpKind::VerticalFlip:
        case ImageOpKind::Rotate:
        case ImageOpKind::ColorJitter:
        case ImageOpKind::Erase:
        case ImageOpKind::RandAugment:
            return true;
        default:
            return false;
    }
}

std::string ValidateImageOp(const ImageOp& op, const ImageShape& input) {
    std::ostringstream reason;
    switch (op.kind) {
        case ImageOpKind::CenterCrop:
        case ImageOpKind::RandomCrop: {
            if (op.height <= 0 || op.width <= 0) {
                return std::string(KindName(op.kind)) + " needs a positive width and height";
            }
            if (op.padding < 0 || (op.kind == ImageOpKind::CenterCrop && op.padding != 0)) {
                return std::string(KindName(op.kind)) + " padding must be 0 or more";
            }
            const size_t padded_h = input.height + 2 * static_cast<size_t>(op.padding);
            const size_t padded_w = input.width + 2 * static_cast<size_t>(op.padding);
            if (static_cast<size_t>(op.height) > padded_h || static_cast<size_t>(op.width) > padded_w) {
                return std::string(KindName(op.kind)) + " " + ShapeText(op.height, op.width) +
                       " (height x width) is larger than the " + ShapeText(padded_h, padded_w) +
                       (op.padding > 0 ? " padded image" : " image") + "; use a smaller crop or a larger Resize";
            }
            return {};
        }
        case ImageOpKind::HorizontalFlip:
        case ImageOpKind::VerticalFlip:
        case ImageOpKind::Rotate:
            if (!(op.probability >= 0.0f && op.probability <= 1.0f)) {
                return std::string(KindName(op.kind)) + " probability must be between 0 and 1";
            }
            if (op.kind == ImageOpKind::Rotate && !(op.max_angle >= 0.0f && op.max_angle <= 180.0f)) {
                return "Image Rotate max_angle must be between 0 and 180 degrees";
            }
            return {};
        case ImageOpKind::ColorJitter:
            if (!(op.brightness >= 0.0f && op.contrast >= 0.0f && op.saturation >= 0.0f)) {
                return "Color Jitter brightness, contrast and saturation must be 0 or more";
            }
            if (!(op.hue >= 0.0f && op.hue <= 0.5f)) {
                return "Color Jitter hue must be between 0 and 0.5";
            }
            if (input.channels != 1 && input.channels != 3) {
                return "Color Jitter needs a 1- or 3-channel image";
            }
            return {};
        case ImageOpKind::GaussianBlur:
            if (op.kernel_size <= 0 || op.kernel_size % 2 == 0) {
                return "Image Gaussian Blur kernel_size must be a positive odd number";
            }
            if (!(op.sigma > 0.0f)) {
                return "Image Gaussian Blur sigma must be greater than 0";
            }
            if (static_cast<size_t>(op.kernel_size / 2) >= (std::min)(input.height, input.width)) {
                return "Image Gaussian Blur kernel_size " + std::to_string(op.kernel_size) +
                       " is too large for the " + ShapeText(input.height, input.width) +
                       " image (reflect padding needs kernel_size / 2 below both sides)";
            }
            return {};
        case ImageOpKind::Grayscale:
            if (input.channels != 1 && input.channels != 3) {
                return "Grayscale needs a 1- or 3-channel image";
            }
            return {};
        case ImageOpKind::Morphology:
            if (op.kernel_size <= 0 || op.kernel_size % 2 == 0) {
                return "Morphology Transform kernel_size must be a positive odd number";
            }
            return {};
        case ImageOpKind::RandAugment:
            if (!(op.probability >= 0.0f && op.probability <= 1.0f)) {
                return "Advanced Augment probability must be between 0 and 1";
            }
            if (op.num_ops < 0 || op.num_ops > 10) {
                return "Advanced Augment num_ops must be between 0 and 10";
            }
            if (op.magnitude < 0 || op.magnitude >= kRandAugmentBins) {
                return "Advanced Augment magnitude must be between 0 and 30";
            }
            if (input.channels != 1 && input.channels != 3) {
                return "RandAugment needs a 1- or 3-channel image";
            }
            return {};
        case ImageOpKind::Erase:
            if (!(op.probability >= 0.0f && op.probability <= 1.0f)) {
                return "Advanced Augment probability must be between 0 and 1";
            }
            if (!(op.value >= 0.0f && op.value <= 1.0f)) {
                return "Advanced Augment value is a pixel value between 0 and 1";
            }
            if (op.erase_method == EraseMethod::Cutout && op.cutout_size <= 0) {
                return "Advanced Augment cutout_size must be a positive number of pixels";
            }
            if (op.erase_method == EraseMethod::RandomErasing &&
                !(op.scale_min > 0.0f && op.scale_min <= op.scale_max && op.scale_max <= 1.0f)) {
                return "Advanced Augment needs 0 < scale_min <= scale_max <= 1";
            }
            if (op.erase_method == EraseMethod::RandomErasing &&
                !(op.ratio_min > 0.0f && op.ratio_min <= op.ratio_max)) {
                return "Advanced Augment needs 0 < ratio_min <= ratio_max";
            }
            return {};
    }
    return {};
}

ImageShape ImageShapeAfter(const ImageOp& op, const ImageShape& input) {
    const std::string reason = ValidateImageOp(op, input);
    if (!reason.empty()) throw std::invalid_argument(reason);
    ImageShape out = input;
    if (op.kind == ImageOpKind::CenterCrop || op.kind == ImageOpKind::RandomCrop) {
        out.height = static_cast<size_t>(op.height);
        out.width = static_cast<size_t>(op.width);
    } else if (op.kind == ImageOpKind::Grayscale) {
        out.channels = 1;
    }
    return out;
}

ImageOpDraws DrawImageOp(const ImageOp& op, const ImageShape& input, size_t batch, bool training,
                         std::mt19937& rng) {
    ImageOpDraws draws;
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    switch (op.kind) {
        case ImageOpKind::RandomCrop: {
            draws.top.resize(batch);
            draws.left.resize(batch);
            for (size_t i = 0; i < batch; ++i) {
                const size_t padded_h = input.height + 2 * static_cast<size_t>(op.padding);
                const size_t padded_w = input.width + 2 * static_cast<size_t>(op.padding);
                if (training) {
                    draws.top[i] = std::uniform_int_distribution<int>(
                        0, static_cast<int>(padded_h) - op.height)(rng);
                    draws.left[i] = std::uniform_int_distribution<int>(
                        0, static_cast<int>(padded_w) - op.width)(rng);
                } else {
                    draws.top[i] = CenterOffset(padded_h, op.height);
                    draws.left[i] = CenterOffset(padded_w, op.width);
                }
            }
            break;
        }
        case ImageOpKind::HorizontalFlip:
        case ImageOpKind::VerticalFlip:
            if (!training) break;
            draws.apply.resize(batch);
            for (auto& apply : draws.apply) apply = unit(rng) < op.probability ? 1 : 0;
            break;
        case ImageOpKind::Rotate:
            if (!training) break;
            draws.angle.resize(batch);
            for (auto& angle : draws.angle) {
                const bool rotate = unit(rng) < op.probability;
                const float a = std::uniform_real_distribution<float>(-op.max_angle, op.max_angle)(rng);
                angle = rotate ? a : 0.0f;
            }
            break;
        case ImageOpKind::ColorJitter: {
            if (!training) break;
            const bool active[4] = {op.brightness > 0.0f, op.contrast > 0.0f, op.saturation > 0.0f, op.hue > 0.0f};
            if (!(active[0] || active[1] || active[2] || active[3])) break;
            draws.factors.resize(batch);
            draws.order.resize(batch);
            for (size_t i = 0; i < batch; ++i) {
                std::vector<int> order(4);
                std::iota(order.begin(), order.end(), 0);
                std::shuffle(order.begin(), order.end(), rng);
                const auto factor = [&rng](float spread) {
                    return std::uniform_real_distribution<float>((std::max)(0.0f, 1.0f - spread), 1.0f + spread)(rng);
                };
                draws.factors[i] = {
                    active[0] ? factor(op.brightness) : 1.0f,
                    active[1] ? factor(op.contrast) : 1.0f,
                    active[2] ? factor(op.saturation) : 1.0f,
                    active[3] ? std::uniform_real_distribution<float>(-op.hue, op.hue)(rng) : 0.0f,
                };
                for (int step : order) {
                    if (active[step]) draws.order[i].push_back(step);
                }
            }
            break;
        }
        case ImageOpKind::RandAugment: {
            if (!training) break;
            draws.order.assign(batch, std::vector<int>(static_cast<size_t>(op.num_ops), 0));
            draws.factors.assign(batch, std::vector<float>(static_cast<size_t>(op.num_ops), 0.0f));
            for (size_t i = 0; i < batch; ++i) {
                if (!(unit(rng) < op.probability)) continue;  // Identity picks
                for (int pick = 0; pick < op.num_ops; ++pick) {
                    const int id = std::uniform_int_distribution<int>(0, kRandAugmentOps - 1)(rng);
                    const auto chosen = static_cast<RandAugmentOp>(id);
                    float m = RandAugmentMagnitude(chosen, op.magnitude, input.height, input.width);
                    if (RandAugmentSigned(chosen) && unit(rng) <= 0.5f) m = -m;
                    draws.order[i][static_cast<size_t>(pick)] = id;
                    draws.factors[i][static_cast<size_t>(pick)] = m;
                }
            }
            break;
        }
        case ImageOpKind::Erase: {
            if (!training) break;
            draws.top.assign(batch, 0);
            draws.left.assign(batch, 0);
            draws.box_height.assign(batch, 0);
            draws.box_width.assign(batch, 0);
            const int h = static_cast<int>(input.height), w = static_cast<int>(input.width);
            for (size_t i = 0; i < batch; ++i) {
                if (!(unit(rng) < op.probability)) continue;
                if (op.erase_method == EraseMethod::Cutout) {
                    // DeVries & Taylor: centre anywhere, square clipped to the image.
                    const int cy = std::uniform_int_distribution<int>(0, h - 1)(rng);
                    const int cx = std::uniform_int_distribution<int>(0, w - 1)(rng);
                    const int half = op.cutout_size / 2;
                    const int y1 = std::clamp(cy - half, 0, h), y2 = std::clamp(cy + half, 0, h);
                    const int x1 = std::clamp(cx - half, 0, w), x2 = std::clamp(cx + half, 0, w);
                    draws.top[i] = y1;
                    draws.left[i] = x1;
                    draws.box_height[i] = y2 - y1;
                    draws.box_width[i] = x2 - x1;
                    continue;
                }
                // torchvision RandomErasing.get_params: ten attempts, else unchanged.
                const double area = static_cast<double>(h) * w;
                const double log_min = std::log(op.ratio_min), log_max = std::log(op.ratio_max);
                for (int attempt = 0; attempt < 10; ++attempt) {
                    const double erase_area =
                        area * std::uniform_real_distribution<double>(op.scale_min, op.scale_max)(rng);
                    const double aspect = std::exp(std::uniform_real_distribution<double>(log_min, log_max)(rng));
                    const int eh = RoundHalfEven(std::sqrt(erase_area * aspect));
                    const int ew = RoundHalfEven(std::sqrt(erase_area / aspect));
                    if (!(eh < h && ew < w)) continue;
                    draws.top[i] = std::uniform_int_distribution<int>(0, h - eh)(rng);
                    draws.left[i] = std::uniform_int_distribution<int>(0, w - ew)(rng);
                    draws.box_height[i] = eh;
                    draws.box_width[i] = ew;
                    break;
                }
            }
            break;
        }
        default:
            break;
    }
    return draws;
}

Tensor ApplyImageOp(const ImageOp& op, const ImageOpDraws& draws, const ImageShape& input, const Tensor& rows) {
    ImageShapeAfter(op, input);  // throws for an invalid op
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const af::array images = RowsToImages(rows.GetArrayRowMajor2D(), input);
    return Tensor::FromArrayRowMajor2D(ImagesToRows(ApplyOnDevice(op, draws, input, images)));
#else
    (void)draws;
    (void)rows;
    throw std::runtime_error("image transforms need the ArrayFire build");
#endif
}

float RandAugmentMagnitude(RandAugmentOp op, int magnitude, size_t height, size_t width) {
    const int bins = kRandAugmentBins;
    switch (op) {
        case RandAugmentOp::ShearX:
        case RandAugmentOp::ShearY:
            return LinspaceAt(0.0f, 0.3f, bins, magnitude);
        case RandAugmentOp::TranslateX:
            return LinspaceAt(0.0f, 150.0f / 331.0f * static_cast<float>(width), bins, magnitude);
        case RandAugmentOp::TranslateY:
            return LinspaceAt(0.0f, 150.0f / 331.0f * static_cast<float>(height), bins, magnitude);
        case RandAugmentOp::Rotate:
            return LinspaceAt(0.0f, 30.0f, bins, magnitude);
        case RandAugmentOp::Brightness:
        case RandAugmentOp::Color:
        case RandAugmentOp::Contrast:
        case RandAugmentOp::Sharpness:
            return LinspaceAt(0.0f, 0.9f, bins, magnitude);
        case RandAugmentOp::Posterize:  // 8 - round(m / ((bins - 1) / 4)) bits
            return static_cast<float>(8 - static_cast<int>(std::nearbyint(
                static_cast<float>(magnitude) / (static_cast<float>(bins - 1) / 4.0f))));
        case RandAugmentOp::Solarize:
            return LinspaceAt(1.0f, 0.0f, bins, magnitude);
        default:
            return 0.0f;
    }
}

bool RandAugmentSigned(RandAugmentOp op) {
    switch (op) {
        case RandAugmentOp::ShearX:
        case RandAugmentOp::ShearY:
        case RandAugmentOp::TranslateX:
        case RandAugmentOp::TranslateY:
        case RandAugmentOp::Rotate:
        case RandAugmentOp::Brightness:
        case RandAugmentOp::Color:
        case RandAugmentOp::Contrast:
        case RandAugmentOp::Sharpness:
            return true;
        default:
            return false;
    }
}

BatchMixDraw DrawBatchMix(BatchMix method, float alpha, float probability, const ImageShape& shape,
                          std::mt19937& rng) {
    BatchMixDraw draw;
    if (method == BatchMix::None || !(std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < probability)) {
        return draw;
    }
    std::gamma_distribution<double> gamma(alpha, 1.0);
    const double x = gamma(rng), y = gamma(rng);
    const double lambda = x + y > 0.0 ? x / (x + y) : 0.5;  // Beta(alpha, alpha)
    draw.apply = true;
    draw.lambda = static_cast<float>(lambda);
    if (method == BatchMix::CutMix) {
        // torchvision v2 CutMix: centre anywhere, half sides r * size with
        // r = 0.5 sqrt(1 - lambda), clipped; lambda becomes 1 - box area.
        const int h = static_cast<int>(shape.height), w = static_cast<int>(shape.width);
        const int rx = std::uniform_int_distribution<int>(0, w - 1)(rng);
        const int ry = std::uniform_int_distribution<int>(0, h - 1)(rng);
        const double r = 0.5 * std::sqrt(1.0 - lambda);
        const int half_w = static_cast<int>(r * w), half_h = static_cast<int>(r * h);
        const int x1 = (std::max)(rx - half_w, 0), y1 = (std::max)(ry - half_h, 0);
        const int x2 = (std::min)(rx + half_w, w), y2 = (std::min)(ry + half_h, h);
        draw.top = y1;
        draw.left = x1;
        draw.height = y2 - y1;
        draw.width = x2 - x1;
        draw.lambda = static_cast<float>(1.0 - static_cast<double>(draw.height) * draw.width / (static_cast<double>(w) * h));
    }
    return draw;
}

void ApplyBatchMix(BatchMix method, const BatchMixDraw& draw, const ImageShape& shape, Tensor& rows,
                   Tensor& labels) {
    if (!draw.apply || method == BatchMix::None) return;
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const af::array images = RowsToImages(rows.GetArrayRowMajor2D(), shape);
    rows = Tensor::FromArrayRowMajor2D(ImagesToRows(MixImages(method, draw, images)));
    const af::array targets = labels.GetArrayRowMajor2D();
    labels = Tensor::FromArrayRowMajor2D(MixLabels(draw, targets));
#else
    (void)shape;
    (void)rows;
    (void)labels;
    throw std::runtime_error("image transforms need the ArrayFire build");
#endif
}

ImageShape ImageAugmentation::ShapeAfter(const ImageShape& input) const {
    ImageShape shape = input;
    for (const auto& op : ops) shape = ImageShapeAfter(op, shape);
    return shape;
}

Tensor ImageAugmentation::Apply(const Tensor& rows, const ImageShape& input, bool training,
                                std::mt19937& rng, Tensor* labels) const {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    af::array images = RowsToImages(rows.GetArrayRowMajor2D(), input);
    const size_t batch = static_cast<size_t>(images.dims(3));
    ImageShape shape = input;
    for (const auto& op : ops) {
        const ImageShape next = ImageShapeAfter(op, shape);
        images = ApplyOnDevice(op, DrawImageOp(op, shape, batch, training, rng), shape, images);
        shape = next;
    }
    if (training && labels != nullptr && mix != BatchMix::None) {
        const BatchMixDraw draw = DrawBatchMix(mix, mix_alpha, mix_probability, shape, rng);
        if (draw.apply) {
            images = MixImages(mix, draw, images);
            *labels = Tensor::FromArrayRowMajor2D(MixLabels(draw, labels->GetArrayRowMajor2D()));
        }
    }
    if (normalize) {
        images = (images - mean) / std_dev;
    }
    return Tensor::FromArrayRowMajor2D(ImagesToRows(images));
#else
    (void)rows;
    (void)input;
    (void)training;
    (void)rng;
    (void)labels;
    throw std::runtime_error("image transforms need the ArrayFire build");
#endif
}

}  // namespace cyxwiz::image
