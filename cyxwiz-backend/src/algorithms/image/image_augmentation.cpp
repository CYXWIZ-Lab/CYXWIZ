#include "cyxwiz/image_augmentation.h"

#include <algorithm>
#include <cmath>
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
    }
    return "image transform";
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

// torchvision F.rotate: inverse affine about ((W-1)/2, (H-1)/2), grid_sample
// with zero padding (align_corners=False), expand=False.
af::array Rotate(const af::array& images, const std::vector<float>& degrees, Interpolation interpolation) {
    const dim_t h = images.dims(0), w = images.dims(1), n = images.dims(3);
    const af::dim4 grid(h, w, 1, n);
    // torchvision's arithmetic in float32, step for step: the inverse matrix
    // [cos, sin; -sin, cos] of -angle is scaled by (0.5 W, 0.5 H), applied to
    // pixel centres relative to the middle, then grid_sample unnormalises
    // ((g + 1) * size - 1) / 2. Half-pixel ties (90 degrees on an even side)
    // then round the same way torchvision's do.
    const float half_w = 0.5f * static_cast<float>(w), half_h = 0.5f * static_cast<float>(h);
    std::vector<float> ax(degrees.size()), bx(degrees.size()), ay(degrees.size()), by(degrees.size());
    for (size_t i = 0; i < degrees.size(); ++i) {
        const double theta = -degrees[i] * kPi / 180.0;
        const float cosine = static_cast<float>(std::cos(theta));
        const float sine = static_cast<float>(std::sin(theta));
        ax[i] = cosine / half_w;
        bx[i] = sine / half_w;
        ay[i] = -sine / half_h;
        by[i] = cosine / half_h;
    }
    const af::array x = af::iota(af::dim4(1, w, 1, 1), af::dim4(h, 1, 1, n), f32) - (half_w - 0.5f);
    const af::array y = af::iota(af::dim4(h, 1, 1, 1), af::dim4(1, w, 1, n), f32) - (half_h - 0.5f);
    const af::array gx = x * PerSample(ax, grid) + y * PerSample(bx, grid);
    const af::array gy = x * PerSample(ay, grid) + y * PerSample(by, grid);
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
            return Crop(images, draws.top, draws.left, op.height, op.width);
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
            return true;
        default:
            return false;
    }
}

std::string ValidateImageOp(const ImageOp& op, const ImageShape& input) {
    std::ostringstream reason;
    switch (op.kind) {
        case ImageOpKind::CenterCrop:
        case ImageOpKind::RandomCrop:
            if (op.height <= 0 || op.width <= 0) {
                return std::string(KindName(op.kind)) + " needs a positive width and height";
            }
            if (static_cast<size_t>(op.height) > input.height || static_cast<size_t>(op.width) > input.width) {
                return std::string(KindName(op.kind)) + " " + ShapeText(op.height, op.width) +
                       " (height x width) is larger than the " + ShapeText(input.height, input.width) +
                       " image; use a smaller crop or a larger Resize";
            }
            return {};
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
                if (training) {
                    draws.top[i] = std::uniform_int_distribution<int>(
                        0, static_cast<int>(input.height) - op.height)(rng);
                    draws.left[i] = std::uniform_int_distribution<int>(
                        0, static_cast<int>(input.width) - op.width)(rng);
                } else {
                    draws.top[i] = CenterOffset(input.height, op.height);
                    draws.left[i] = CenterOffset(input.width, op.width);
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

ImageShape ImageAugmentation::ShapeAfter(const ImageShape& input) const {
    ImageShape shape = input;
    for (const auto& op : ops) shape = ImageShapeAfter(op, shape);
    return shape;
}

Tensor ImageAugmentation::Apply(const Tensor& rows, const ImageShape& input, bool training,
                                std::mt19937& rng) const {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    af::array images = RowsToImages(rows.GetArrayRowMajor2D(), input);
    const size_t batch = static_cast<size_t>(images.dims(3));
    ImageShape shape = input;
    for (const auto& op : ops) {
        const ImageShape next = ImageShapeAfter(op, shape);
        images = ApplyOnDevice(op, DrawImageOp(op, shape, batch, training, rng), shape, images);
        shape = next;
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
    throw std::runtime_error("image transforms need the ArrayFire build");
#endif
}

}  // namespace cyxwiz::image
