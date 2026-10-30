#include "cyxwiz/image_augmentation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <arrayfire.h>

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
        case ImageOpKind::RandomResizedCrop: return "Random Resized Crop";
        case ImageOpKind::RandomAffine: return "Random Affine";
        case ImageOpKind::RandomPerspective: return "Random Perspective";
        case ImageOpKind::GaussianNoise: return "Gaussian Noise";
        case ImageOpKind::RandomGrayscale: return "Random Grayscale";
        case ImageOpKind::AutoAugment: return "Advanced Augment";
        case ImageOpKind::TrivialAugment: return "Advanced Augment";
        case ImageOpKind::AugMix: return "Advanced Augment";
    }
    return "image transform";
}

// torchvision AutoAugmentPolicy sub-policies: (op, probability, magnitude
// bin of 10, -1 = none) twice, as torchvision.transforms.v2.AutoAugment lists them.
struct SubPolicy {
    AugmentOp op;
    float probability;
    int bin;
};
using Policy = std::array<SubPolicy, 2>;

const std::array<Policy, 25>& AutoAugmentPolicies(AutoAugmentPolicy policy) {
    using A = AugmentOp;
    static const std::array<Policy, 25> imagenet = {{
        {{{A::Posterize, 0.4f, 8}, {A::Rotate, 0.6f, 9}}},       {{{A::Solarize, 0.6f, 5}, {A::AutoContrast, 0.6f, -1}}},
        {{{A::Equalize, 0.8f, -1}, {A::Equalize, 0.6f, -1}}},    {{{A::Posterize, 0.6f, 7}, {A::Posterize, 0.6f, 6}}},
        {{{A::Equalize, 0.4f, -1}, {A::Solarize, 0.2f, 4}}},     {{{A::Equalize, 0.4f, -1}, {A::Rotate, 0.8f, 8}}},
        {{{A::Solarize, 0.6f, 3}, {A::Equalize, 0.6f, -1}}},     {{{A::Posterize, 0.8f, 5}, {A::Equalize, 1.0f, -1}}},
        {{{A::Rotate, 0.2f, 3}, {A::Solarize, 0.6f, 8}}},        {{{A::Equalize, 0.6f, -1}, {A::Posterize, 0.4f, 6}}},
        {{{A::Rotate, 0.8f, 8}, {A::Color, 0.4f, 0}}},           {{{A::Rotate, 0.4f, 9}, {A::Equalize, 0.6f, -1}}},
        {{{A::Equalize, 0.0f, -1}, {A::Equalize, 0.8f, -1}}},    {{{A::Invert, 0.6f, -1}, {A::Equalize, 1.0f, -1}}},
        {{{A::Color, 0.6f, 4}, {A::Contrast, 1.0f, 8}}},         {{{A::Rotate, 0.8f, 8}, {A::Color, 1.0f, 2}}},
        {{{A::Color, 0.8f, 8}, {A::Solarize, 0.8f, 7}}},         {{{A::Sharpness, 0.4f, 7}, {A::Invert, 0.6f, -1}}},
        {{{A::ShearX, 0.6f, 5}, {A::Equalize, 1.0f, -1}}},       {{{A::Color, 0.4f, 0}, {A::Equalize, 0.6f, -1}}},
        {{{A::Equalize, 0.4f, -1}, {A::Solarize, 0.2f, 4}}},     {{{A::Solarize, 0.6f, 5}, {A::AutoContrast, 0.6f, -1}}},
        {{{A::Invert, 0.6f, -1}, {A::Equalize, 1.0f, -1}}},      {{{A::Color, 0.6f, 4}, {A::Contrast, 1.0f, 8}}},
        {{{A::Equalize, 0.8f, -1}, {A::Equalize, 0.6f, -1}}},
    }};
    static const std::array<Policy, 25> cifar10 = {{
        {{{A::Invert, 0.1f, -1}, {A::Contrast, 0.2f, 6}}},       {{{A::Rotate, 0.7f, 2}, {A::TranslateX, 0.3f, 9}}},
        {{{A::Sharpness, 0.8f, 1}, {A::Sharpness, 0.9f, 3}}},    {{{A::ShearY, 0.5f, 8}, {A::TranslateY, 0.7f, 9}}},
        {{{A::AutoContrast, 0.5f, -1}, {A::Equalize, 0.9f, -1}}}, {{{A::ShearY, 0.2f, 7}, {A::Posterize, 0.3f, 7}}},
        {{{A::Color, 0.4f, 3}, {A::Brightness, 0.6f, 7}}},       {{{A::Sharpness, 0.3f, 9}, {A::Brightness, 0.7f, 9}}},
        {{{A::Equalize, 0.6f, -1}, {A::Equalize, 0.5f, -1}}},    {{{A::Contrast, 0.6f, 7}, {A::Sharpness, 0.6f, 5}}},
        {{{A::Color, 0.7f, 7}, {A::TranslateX, 0.5f, 8}}},       {{{A::Equalize, 0.3f, -1}, {A::AutoContrast, 0.4f, -1}}},
        {{{A::TranslateY, 0.4f, 3}, {A::Sharpness, 0.2f, 6}}},   {{{A::Brightness, 0.9f, 6}, {A::Color, 0.2f, 8}}},
        {{{A::Solarize, 0.5f, 2}, {A::Invert, 0.0f, -1}}},       {{{A::Equalize, 0.2f, -1}, {A::AutoContrast, 0.6f, -1}}},
        {{{A::Equalize, 0.2f, -1}, {A::Equalize, 0.6f, -1}}},    {{{A::Color, 0.9f, 9}, {A::Equalize, 0.6f, -1}}},
        {{{A::AutoContrast, 0.8f, -1}, {A::Solarize, 0.2f, 8}}}, {{{A::Brightness, 0.1f, 3}, {A::Color, 0.7f, 0}}},
        {{{A::Solarize, 0.4f, 5}, {A::AutoContrast, 0.9f, -1}}}, {{{A::TranslateY, 0.9f, 9}, {A::TranslateY, 0.7f, 9}}},
        {{{A::AutoContrast, 0.9f, -1}, {A::Solarize, 0.8f, 3}}}, {{{A::Equalize, 0.8f, -1}, {A::Invert, 0.1f, -1}}},
        {{{A::TranslateY, 0.7f, 9}, {A::AutoContrast, 0.9f, -1}}},
    }};
    static const std::array<Policy, 25> svhn = {{
        {{{A::ShearX, 0.9f, 4}, {A::Invert, 0.2f, -1}}},         {{{A::ShearY, 0.9f, 8}, {A::Invert, 0.7f, -1}}},
        {{{A::Equalize, 0.6f, -1}, {A::Solarize, 0.6f, 6}}},     {{{A::Invert, 0.9f, -1}, {A::Equalize, 0.6f, -1}}},
        {{{A::Equalize, 0.6f, -1}, {A::Rotate, 0.9f, 3}}},       {{{A::ShearX, 0.9f, 4}, {A::AutoContrast, 0.8f, -1}}},
        {{{A::ShearY, 0.9f, 8}, {A::Invert, 0.4f, -1}}},         {{{A::ShearY, 0.9f, 5}, {A::Solarize, 0.2f, 6}}},
        {{{A::Invert, 0.9f, -1}, {A::AutoContrast, 0.8f, -1}}},  {{{A::Equalize, 0.6f, -1}, {A::Rotate, 0.9f, 3}}},
        {{{A::ShearX, 0.9f, 4}, {A::Solarize, 0.3f, 3}}},        {{{A::ShearY, 0.8f, 8}, {A::Invert, 0.7f, -1}}},
        {{{A::Equalize, 0.9f, -1}, {A::TranslateY, 0.6f, 6}}},   {{{A::Invert, 0.9f, -1}, {A::Equalize, 0.6f, -1}}},
        {{{A::Contrast, 0.3f, 3}, {A::Rotate, 0.8f, 4}}},        {{{A::Invert, 0.8f, -1}, {A::TranslateY, 0.0f, 2}}},
        {{{A::ShearY, 0.7f, 6}, {A::Solarize, 0.4f, 8}}},        {{{A::Invert, 0.6f, -1}, {A::Rotate, 0.8f, 4}}},
        {{{A::ShearY, 0.3f, 7}, {A::TranslateX, 0.9f, 3}}},      {{{A::ShearX, 0.1f, 6}, {A::Invert, 0.6f, -1}}},
        {{{A::Solarize, 0.7f, 2}, {A::TranslateY, 0.6f, 7}}},    {{{A::ShearY, 0.8f, 4}, {A::Invert, 0.8f, -1}}},
        {{{A::ShearX, 0.7f, 9}, {A::TranslateY, 0.8f, 3}}},      {{{A::ShearY, 0.8f, 5}, {A::AutoContrast, 0.7f, -1}}},
        {{{A::ShearX, 0.7f, 2}, {A::Invert, 0.1f, -1}}},
    }};
    switch (policy) {
        case AutoAugmentPolicy::Cifar10: return cifar10;
        case AutoAugmentPolicy::Svhn: return svhn;
        default: return imagenet;
    }
}

// AugMix's ops (all_ops=True), in torchvision's order.
constexpr std::array<AugmentOp, 13> kAugMixOps = {
    AugmentOp::ShearX,    AugmentOp::ShearY,   AugmentOp::TranslateX,   AugmentOp::TranslateY, AugmentOp::Rotate,
    AugmentOp::Posterize, AugmentOp::Solarize, AugmentOp::AutoContrast, AugmentOp::Equalize,   AugmentOp::Brightness,
    AugmentOp::Color,     AugmentOp::Contrast, AugmentOp::Sharpness,
};

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

// torchvision grid_sample (zero padding, align_corners=False) at grid
// coordinates gx, gy in [-1, 1] ([H, W, 1, N]), unnormalised as
// ((g + 1) * size - 1) / 2. With a fill (the transform classes: fill=0 by
// default) torchvision samples a ones mask alongside, then nearest takes the
// fill where the mask is below 0.5 and bilinear blends towards it,
// mask * (image - fill) + fill.
af::array GridSample(const af::array& images, const af::array& gx, const af::array& gy,
                     Interpolation interpolation, const float* fill) {
    const dim_t h = images.dims(0), w = images.dims(1);
    const af::array source = fill ? af::join(2, images, af::constant(1.0f, af::dim4(h, w, 1, images.dims(3))))
                                  : images;
    const af::array sx = ((gx + 1.0f) * static_cast<float>(w) - 1.0f) / 2.0f;
    const af::array sy = ((gy + 1.0f) * static_cast<float>(h) - 1.0f) / 2.0f;
    af::array sampled;
    if (interpolation == Interpolation::Nearest) {
        sampled = GatherPixels(source, RoundHalfEven(sy), RoundHalfEven(sx));
    } else {
        const af::array x0 = af::floor(sx), y0 = af::floor(sy);
        const af::array wx = sx - x0, wy = sy - y0;
        const unsigned c = static_cast<unsigned>(source.dims(2));
        const auto weight = [c](const af::array& v) { return af::tile(v, 1, 1, c, 1); };
        sampled = GatherPixels(source, y0, x0) * weight((1 - wy) * (1 - wx)) +
                  GatherPixels(source, y0, x0 + 1) * weight((1 - wy) * wx) +
                  GatherPixels(source, y0 + 1, x0) * weight(wy * (1 - wx)) +
                  GatherPixels(source, y0 + 1, x0 + 1) * weight(wy * wx);
    }
    if (!fill) return sampled;
    const dim_t c = images.dims(2);
    const af::array image = sampled(af::span, af::span, af::seq(0, static_cast<double>(c) - 1), af::span);
    const af::array mask =
        af::tile(sampled(af::span, af::span, static_cast<int>(c), af::span), 1, 1, static_cast<unsigned>(c), 1);
    if (interpolation == Interpolation::Nearest) {
        return af::select(mask < 0.5f, af::constant(*fill, image.dims(), image.type()), image);
    }
    return mask * (image - *fill) + *fill;
}

// torchvision's affine grid with each sample's inverse matrix: in float32,
// step for step, the matrix is scaled by (0.5 W, 0.5 H) and applied to pixel
// centres relative to the middle (_affine_grid), then GridSample. Half-pixel
// ties (90 degrees on an even side) round as torchvision's do.
af::array AffineSample(const af::array& images, const std::vector<std::array<double, 6>>& matrices,
                       Interpolation interpolation, const float* fill = nullptr) {
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
    return GridSample(images, gx, gy, interpolation, fill);
}

// torchvision RandomRotation (F.rotate with its fill 0): the inverse affine
// of -angle about the image centre.
af::array Rotate(const af::array& images, const std::vector<float>& degrees, Interpolation interpolation) {
    std::vector<std::array<double, 6>> matrices(degrees.size());
    for (size_t i = 0; i < degrees.size(); ++i) {
        matrices[i] = InverseAffineMatrix(0.0, 0.0, -static_cast<double>(degrees[i]), 0.0, 0.0, 1.0, 0.0, 0.0);
    }
    const float fill = 0.0f;
    return AffineSample(images, matrices, interpolation, &fill);
}

// torchvision RandomAffine (F.affine about the image centre): each sample's
// angle, translation (pixels), scale and shears from its factors.
af::array Affine(const af::array& images, const ImageOpDraws& draws, Interpolation interpolation, float fill) {
    std::vector<std::array<double, 6>> matrices(draws.factors.size());
    for (size_t i = 0; i < matrices.size(); ++i) {
        const auto& f = draws.factors[i];
        matrices[i] = InverseAffineMatrix(0.0, 0.0, f[0], f[1], f[2], f[3], f[4], f[5]);
    }
    return AffineSample(images, matrices, interpolation, &fill);
}

// torch interpolate's weights from `in_size` pixels to `out` along one axis,
// reading only [start, start + size) (the crop box): column-major [out,
// in_size]. Bilinear is the antialiased filter torchvision resize uses
// (antialias=True: a triangle widened by the scale when shrinking, the plain
// bilinear when growing); nearest takes floor(i * scale).
std::vector<float> ResampleWeights(int in_size, int start, int size, int out, Interpolation interpolation) {
    std::vector<float> matrix(static_cast<size_t>(out) * static_cast<size_t>(in_size), 0.0f);
    const auto at = [&](int source, int i) -> float& {
        return matrix[static_cast<size_t>(start + source) * static_cast<size_t>(out) + static_cast<size_t>(i)];
    };
    const float scale = static_cast<float>(size) / static_cast<float>(out);
    for (int i = 0; i < out; ++i) {
        if (interpolation == Interpolation::Nearest) {
            const int source = size == out ? i : (std::min)(static_cast<int>(std::floor(static_cast<float>(i) * scale)), size - 1);
            at(source, i) = 1.0f;
            continue;
        }
        const float support = scale >= 1.0f ? scale : 1.0f;
        const float invscale = scale >= 1.0f ? 1.0f / scale : 1.0f;
        const float center = scale * (static_cast<float>(i) + 0.5f);
        const int first = (std::max)(static_cast<int>(center - support + 0.5f), 0);
        const int count = (std::min)(static_cast<int>(center + support + 0.5f), size) - first;
        std::vector<float> weights(static_cast<size_t>((std::max)(count, 0)));
        float total = 0.0f;
        for (int j = 0; j < count; ++j) {
            const float x = std::fabs((static_cast<float>(j + first) - center + 0.5f) * invscale);
            weights[static_cast<size_t>(j)] = x < 1.0f ? 1.0f - x : 0.0f;
            total += weights[static_cast<size_t>(j)];
        }
        for (int j = 0; j < count; ++j) at(first + j, i) = total != 0.0f ? weights[static_cast<size_t>(j)] / total : 0.0f;
    }
    return matrix;
}

// torchvision F.resized_crop: each sample's box resized to height x width,
// as two batched matrix products (rows, then columns).
af::array ResizedCrop(const af::array& images, const ImageOpDraws& draws, int height, int width,
                      Interpolation interpolation) {
    const dim_t h = images.dims(0), w = images.dims(1), c = images.dims(2), n = images.dims(3);
    std::vector<float> rows_op, cols_op;
    for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
        const auto r = ResampleWeights(static_cast<int>(h), draws.top[i], draws.box_height[i], height, interpolation);
        const auto k = ResampleWeights(static_cast<int>(w), draws.left[i], draws.box_width[i], width, interpolation);
        rows_op.insert(rows_op.end(), r.begin(), r.end());
        cols_op.insert(cols_op.end(), k.begin(), k.end());
    }
    const af::array along_h = af::moddims(
        af::matmul(af::array(height, h, 1, n, rows_op.data()), af::moddims(images, h, w * c, 1, n)), height, w, c, n);
    const af::array by_width = af::moddims(af::reorder(along_h, 1, 0, 2, 3), w, height * c, 1, n);
    const af::array along_w =
        af::moddims(af::matmul(af::array(width, w, 1, n, cols_op.data()), by_width), width, height, c, n);
    return af::reorder(along_w, 1, 0, 2, 3);
}

// torchvision _get_perspective_coeffs: the a..h mapping an output pixel
// (x, y) to ((ax + by + c), (dx + ey + f)) / (gx + hy + 1) in the input, from
// the drawn end corners back to the image's own; the exact 8 x 8 solve in
// double, then float32. False when the corners give no mapping (three or
// more on one line, possible on tiny images with a large distortion).
bool SolvePerspective(const std::vector<int>& corners, int height, int width, std::array<float, 8>& out) {
    if (corners.size() != 8) return false;
    const double start[4][2] = {{0.0, 0.0}, {width - 1.0, 0.0}, {width - 1.0, height - 1.0}, {0.0, height - 1.0}};
    std::array<std::array<double, 9>, 8> a{};
    for (int i = 0; i < 4; ++i) {
        const double ex = corners[static_cast<size_t>(2 * i)], ey = corners[static_cast<size_t>(2 * i + 1)];
        const double sx = start[i][0], sy = start[i][1];
        a[static_cast<size_t>(2 * i)] = {ex, ey, 1.0, 0.0, 0.0, 0.0, -sx * ex, -sx * ey, sx};
        a[static_cast<size_t>(2 * i + 1)] = {0.0, 0.0, 0.0, ex, ey, 1.0, -sy * ex, -sy * ey, sy};
    }
    for (size_t col = 0; col < 8; ++col) {  // Gauss-Jordan with partial pivoting
        size_t pivot = col;
        for (size_t r = col + 1; r < 8; ++r)
            if (std::fabs(a[r][col]) > std::fabs(a[pivot][col])) pivot = r;
        if (std::fabs(a[pivot][col]) < 1e-9) return false;
        std::swap(a[col], a[pivot]);
        for (size_t r = 0; r < 8; ++r) {
            if (r == col || a[r][col] == 0.0) continue;
            const double f = a[r][col] / a[col][col];
            for (size_t k = col; k < 9; ++k) a[r][k] -= f * a[col][k];
        }
    }
    for (size_t i = 0; i < 8; ++i) out[i] = static_cast<float>(a[i][8] / a[i][i]);
    return true;
}

// torchvision F.perspective (_perspective_grid in float32, then GridSample)
// on the samples whose apply is set.
af::array Perspective(const af::array& images, const ImageOpDraws& draws, Interpolation interpolation, float fill) {
    const dim_t h = images.dims(0), w = images.dims(1), n = images.dims(3);
    const af::dim4 grid(h, w, 1, n);
    const float half_w = 0.5f * static_cast<float>(w), half_h = 0.5f * static_cast<float>(h);
    std::array<std::vector<float>, 8> c;
    for (auto& v : c) v.assign(static_cast<size_t>(n), 0.0f);
    for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
        std::array<float, 8> k{1, 0, 0, 0, 1, 0, 0, 0};
        if (draws.apply[i] && !SolvePerspective(draws.corners[i], static_cast<int>(h), static_cast<int>(w), k)) {
            throw std::invalid_argument("Random Perspective needs 8 corner values per sample that are not on one line");
        }
        for (size_t j = 0; j < 8; ++j) c[j][i] = k[j];
    }
    const auto scaled = [&](const std::vector<float>& v, float by) {
        std::vector<float> out(v.size());
        for (size_t i = 0; i < v.size(); ++i) out[i] = v[i] / by;
        return PerSample(out, grid);
    };
    const af::array x = af::iota(af::dim4(1, w, 1, 1), af::dim4(h, 1, 1, n), f32) + 0.5f;
    const af::array y = af::iota(af::dim4(h, 1, 1, 1), af::dim4(1, w, 1, n), f32) + 0.5f;
    const af::array den = x * PerSample(c[6], grid) + y * PerSample(c[7], grid) + 1.0f;
    const af::array gx = (x * scaled(c[0], half_w) + y * scaled(c[1], half_w) + scaled(c[2], half_w)) / den - 1.0f;
    const af::array gy = (x * scaled(c[3], half_h) + y * scaled(c[4], half_h) + scaled(c[5], half_h)) / den - 1.0f;
    return af::select(PerSampleMask(draws.apply, images.dims()), GridSample(images, gx, gy, interpolation, &fill),
                      images);
}

// torchvision v2 gaussian_noise on floats: image + (mean + N(0, 1) * sigma),
// clamped to [0, 1] when clip. The noise is the draws' own (as recorded) or
// drawn on the device from their seed.
af::array GaussianNoise(const af::array& images, const ImageOpDraws& draws, const ImageOp& op, const ImageShape& shape) {
    af::array noise;
    if (!draws.noise.empty()) {
        const dim_t n = images.dims(3);
        if (draws.noise.size() != shape.Size() * static_cast<size_t>(n)) {
            throw std::invalid_argument("Gaussian Noise needs one noise value per pixel of every sample");
        }
        noise = RowsToImages(af::transpose(af::array(static_cast<dim_t>(shape.Size()), n, draws.noise.data())), shape);
    } else {
        af::randomEngine engine(AF_RANDOM_ENGINE_DEFAULT, draws.noise_seed);
        noise = af::randn(images.dims(), f32, engine);
    }
    const af::array out = images + (op.noise_mean + noise * op.sigma);
    return op.clip ? af::clamp(out, 0.0, 1.0) : out;
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

af::array PolicyStep(const af::array& images, const std::vector<int>& ops, const std::vector<float>& magnitudes,
                     int op_index, Interpolation interpolation) {
    const af::dim4 dims = images.dims();
    const size_t n = ops.size();
    const AugmentOp op = static_cast<AugmentOp>(op_index);
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
        case AugmentOp::Identity:
            return images;
        case AugmentOp::ShearX:  // centre [0, 0] = the top-left corner
            return affine([&](double m) {
                return InverseAffineMatrix(-0.5 * w, -0.5 * h, 0.0, 0.0, 0.0, 1.0, std::atan(m) * 180.0 / kPi, 0.0);
            });
        case AugmentOp::ShearY:
            return affine([&](double m) {
                return InverseAffineMatrix(-0.5 * w, -0.5 * h, 0.0, 0.0, 0.0, 1.0, 0.0, std::atan(m) * 180.0 / kPi);
            });
        case AugmentOp::TranslateX:
            return affine([](double m) {
                return InverseAffineMatrix(0.0, 0.0, 0.0, static_cast<double>(static_cast<int>(m)), 0.0, 1.0, 0.0, 0.0);
            });
        case AugmentOp::TranslateY:
            return affine([](double m) {
                return InverseAffineMatrix(0.0, 0.0, 0.0, 0.0, static_cast<double>(static_cast<int>(m)), 1.0, 0.0, 0.0);
            });
        case AugmentOp::Rotate:
            return affine([](double m) { return InverseAffineMatrix(0.0, 0.0, -m, 0.0, 0.0, 1.0, 0.0, 0.0); });
        case AugmentOp::Brightness:
            return AdjustBrightness(images, per_sample(1.0f, [](float m) { return 1.0f + m; }));
        case AugmentOp::Color:
            return AdjustSaturation(images, per_sample(1.0f, [](float m) { return 1.0f + m; }));
        case AugmentOp::Contrast:
            return AdjustContrast(images, per_sample(1.0f, [](float m) { return 1.0f + m; }));
        case AugmentOp::Sharpness:
            return AdjustSharpness(images, per_sample(1.0f, [](float m) { return 1.0f + m; }));
        case AugmentOp::Posterize:
            return Posterize(images, per_sample(256.0f, [](float m) {
                return static_cast<float>(1 << static_cast<int>(m));
            }));
        case AugmentOp::Solarize:
            return Solarize(images, per_sample(2.0f, [](float m) { return m; }));
        case AugmentOp::AutoContrast:
            return AutoContrast(images);
        case AugmentOp::Equalize:
            return Equalize(images);
        case AugmentOp::Invert:
            return 1.0f - images;
    }
    return images;
}

// One pick per sample: draws' order[i][step] with its magnitude.
af::array PolicyPicks(af::array images, const ImageOpDraws& draws, size_t step, Interpolation interpolation) {
    const size_t n = draws.order.size();
    const af::dim4 dims = images.dims();
    std::vector<int> ops(n);
    std::vector<float> magnitudes(n);
    for (size_t i = 0; i < n; ++i) {
        ops[i] = draws.order[i][step];
        magnitudes[i] = draws.factors[i][step];
    }
    for (int op = 1; op < kAugmentOps; ++op) {
        std::vector<int> chosen(n);
        bool any = false;
        for (size_t i = 0; i < n; ++i) {
            chosen[i] = ops[i] == op ? 1 : 0;
            any = any || chosen[i];
        }
        if (!any) continue;
        images = af::select(PerSampleMask(chosen, dims), PolicyStep(images, ops, magnitudes, op, interpolation), images);
    }
    return images;
}

// RandAugment, AutoAugment, TrivialAugment: each sample's picks in order.
af::array PolicyAugment(af::array images, const ImageOpDraws& draws, Interpolation interpolation) {
    const size_t steps = draws.order.empty() ? 0 : draws.order.front().size();
    for (size_t step = 0; step < steps; ++step) images = PolicyPicks(images, draws, step, interpolation);
    return images;
}

// torchvision v2 AugMix: the original weighted by weights[i][0] plus each
// chain's result weighted by its weight; a chain's picks are its slots of
// order, Identity past its depth.
af::array AugMix(const af::array& images, const ImageOpDraws& draws, int mixture_width, Interpolation interpolation) {
    const size_t n = draws.order.size();
    const size_t slots = n == 0 ? 0 : draws.order.front().size() / static_cast<size_t>(mixture_width);
    const auto weight = [&](size_t k) {
        std::vector<float> values(n);
        for (size_t i = 0; i < n; ++i) values[i] = draws.weights[i][k];
        return PerSample(values, images.dims());
    };
    af::array mix = weight(0) * images;
    for (size_t chain = 0; chain < static_cast<size_t>(mixture_width); ++chain) {
        af::array augmented = images;
        for (size_t s = 0; s < slots; ++s) augmented = PolicyPicks(augmented, draws, chain * slots + s, interpolation);
        mix = mix + weight(chain + 1) * augmented;
    }
    return mix;
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
        case ImageOpKind::AutoAugment:
        case ImageOpKind::TrivialAugment:
            if (draws.order.empty()) return images;
            return PolicyAugment(images, draws, op.interpolation);
        case ImageOpKind::AugMix:
            if (draws.order.empty()) return images;
            return AugMix(images, draws, op.mixture_width, op.interpolation);
        case ImageOpKind::RandomResizedCrop:
            if (draws.top.size() != static_cast<size_t>(dims[3]) || draws.left.size() != draws.top.size() ||
                draws.box_height.size() != draws.top.size() || draws.box_width.size() != draws.top.size()) {
                throw std::invalid_argument("Random Resized Crop needs one box per sample");
            }
            for (size_t i = 0; i < draws.top.size(); ++i) {
                if (draws.box_height[i] < 1 || draws.box_width[i] < 1 || draws.top[i] < 0 || draws.left[i] < 0 ||
                    static_cast<dim_t>(draws.top[i] + draws.box_height[i]) > dims[0] ||
                    static_cast<dim_t>(draws.left[i] + draws.box_width[i]) > dims[1]) {
                    throw std::invalid_argument("Random Resized Crop boxes must lie inside the image");
                }
            }
            return ResizedCrop(images, draws, op.height, op.width, op.interpolation);
        case ImageOpKind::RandomAffine:
            if (draws.factors.empty()) return images;
            return Affine(images, draws, op.interpolation, op.fill);
        case ImageOpKind::RandomPerspective:
            if (draws.apply.empty()) return images;
            return Perspective(images, draws, op.interpolation, op.fill);
        case ImageOpKind::GaussianNoise:
            if (draws.apply.empty()) return images;  // outside training
            return GaussianNoise(images, draws, op, shape);
        case ImageOpKind::RandomGrayscale:
            if (draws.apply.empty() || shape.channels != 3) return images;
            return af::select(PerSampleMask(draws.apply, dims), af::tile(Luminance(images), 1, 1, 3, 1), images);
    }
    return images;
}

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
        case ImageOpKind::RandomResizedCrop:
        case ImageOpKind::RandomAffine:
        case ImageOpKind::RandomPerspective:
        case ImageOpKind::GaussianNoise:
        case ImageOpKind::RandomGrayscale:
        case ImageOpKind::AutoAugment:
        case ImageOpKind::TrivialAugment:
        case ImageOpKind::AugMix:
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
        case ImageOpKind::RandomResizedCrop:
            if (op.height <= 0 || op.width <= 0) return "Random Resized Crop needs a positive width and height";
            if (!(op.scale_min > 0.0f && op.scale_min <= op.scale_max && op.scale_max <= 1.0f)) {
                return "Random Resized Crop needs 0 < scale_min <= scale_max <= 1 (the box's share of the image area)";
            }
            if (!(op.ratio_min > 0.0f && op.ratio_min <= op.ratio_max)) {
                return "Random Resized Crop needs 0 < ratio_min <= ratio_max (the box's width / height)";
            }
            return {};
        case ImageOpKind::RandomAffine:
            if (!(op.max_angle >= 0.0f && op.max_angle <= 180.0f)) {
                return "Random Affine degrees must be between 0 and 180";
            }
            if (!(op.translate_x >= 0.0f && op.translate_x <= 1.0f && op.translate_y >= 0.0f && op.translate_y <= 1.0f)) {
                return "Random Affine translate is a fraction of the image between 0 and 1";
            }
            if (!(op.zoom_min > 0.0f && op.zoom_min <= op.zoom_max)) return "Random Affine needs 0 < scale_min <= scale_max";
            if (!(op.shear_x_min <= op.shear_x_max && op.shear_y_min <= op.shear_y_max && op.shear_x_min > -90.0f &&
                  op.shear_x_max < 90.0f && op.shear_y_min > -90.0f && op.shear_y_max < 90.0f)) {
                return "Random Affine shears need min <= max, inside (-90, 90) degrees";
            }
            if (!(op.fill >= 0.0f && op.fill <= 1.0f)) return "Random Affine fill is a pixel value between 0 and 1";
            return {};
        case ImageOpKind::RandomPerspective:
            if (!(op.distortion >= 0.0f && op.distortion <= 1.0f)) {
                return "Random Perspective distortion must be between 0 and 1";
            }
            if (!(op.probability >= 0.0f && op.probability <= 1.0f)) {
                return "Random Perspective probability must be between 0 and 1";
            }
            if (!(op.fill >= 0.0f && op.fill <= 1.0f)) return "Random Perspective fill is a pixel value between 0 and 1";
            if (input.height < 2 || input.width < 2) return "Random Perspective needs an image of 2 x 2 or more";
            return {};
        case ImageOpKind::GaussianNoise:
            if (!(op.sigma >= 0.0f && std::isfinite(op.sigma))) return "Gaussian Noise sigma must be 0 or more";
            if (!std::isfinite(op.noise_mean)) return "Gaussian Noise mean must be a number";
            return {};
        case ImageOpKind::RandomGrayscale:
            if (!(op.probability >= 0.0f && op.probability <= 1.0f)) {
                return "Random Grayscale probability must be between 0 and 1";
            }
            if (input.channels != 1 && input.channels != 3) return "Random Grayscale needs a 1- or 3-channel image";
            return {};
        case ImageOpKind::AutoAugment:
        case ImageOpKind::TrivialAugment:
        case ImageOpKind::AugMix:
            if (!(op.probability >= 0.0f && op.probability <= 1.0f)) {
                return "Advanced Augment probability must be between 0 and 1";
            }
            if (input.channels != 1 && input.channels != 3) {
                return "Advanced Augment's policies need a 1- or 3-channel image";
            }
            if (op.kind == ImageOpKind::TrivialAugment && (op.magnitude_bins < 2 || op.magnitude_bins > 100)) {
                return "Advanced Augment magnitude_bins must be between 2 and 100";
            }
            if (op.kind == ImageOpKind::AugMix) {
                if (op.severity < 1 || op.severity > 10) return "Advanced Augment severity must be between 1 and 10";
                if (op.mixture_width < 1 || op.mixture_width > 10) {
                    return "Advanced Augment mixture_width must be between 1 and 10";
                }
                if (op.chain_depth == 0 || op.chain_depth < -1 || op.chain_depth > 10) {
                    return "Advanced Augment chain_depth is 1 to 10, or -1 for 1 to 3 at random";
                }
                if (!(op.alpha > 0.0f)) return "Advanced Augment alpha must be greater than 0";
            }
            return {};
    }
    return {};
}

ImageShape ImageShapeAfter(const ImageOp& op, const ImageShape& input) {
    const std::string reason = ValidateImageOp(op, input);
    if (!reason.empty()) throw std::invalid_argument(reason);
    ImageShape out = input;
    if (op.kind == ImageOpKind::CenterCrop || op.kind == ImageOpKind::RandomCrop ||
        op.kind == ImageOpKind::RandomResizedCrop) {
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
                    const auto chosen = static_cast<AugmentOp>(id);
                    float m = AugmentMagnitude(AugmentSpace::Standard, chosen, kRandAugmentBins, op.magnitude,
                                               input.height, input.width);
                    if (AugmentSigned(chosen) && unit(rng) <= 0.5f) m = -m;
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
        case ImageOpKind::RandomResizedCrop: {
            // torchvision RandomResizedCrop.make_params: ten tries at a box of
            // random area and aspect (w / h); else, and outside training, the
            // centred box of the nearest allowed aspect.
            draws.top.assign(batch, 0);
            draws.left.assign(batch, 0);
            draws.box_height.assign(batch, 0);
            draws.box_width.assign(batch, 0);
            const int h = static_cast<int>(input.height), w = static_cast<int>(input.width);
            const double area = static_cast<double>(h) * w;
            const double log_min = std::log(op.ratio_min), log_max = std::log(op.ratio_max);
            for (size_t i = 0; i < batch; ++i) {
                bool found = false;
                for (int attempt = 0; training && attempt < 10 && !found; ++attempt) {
                    const double target = area * std::uniform_real_distribution<double>(op.scale_min, op.scale_max)(rng);
                    const double aspect = std::exp(std::uniform_real_distribution<double>(log_min, log_max)(rng));
                    const int bw = RoundHalfEven(std::sqrt(target * aspect));
                    const int bh = RoundHalfEven(std::sqrt(target / aspect));
                    if (!(bw > 0 && bw <= w && bh > 0 && bh <= h)) continue;
                    draws.top[i] = std::uniform_int_distribution<int>(0, h - bh)(rng);
                    draws.left[i] = std::uniform_int_distribution<int>(0, w - bw)(rng);
                    draws.box_height[i] = bh;
                    draws.box_width[i] = bw;
                    found = true;
                }
                if (found) continue;
                const double in_ratio = static_cast<double>(w) / h;
                int bw = w, bh = h;
                if (in_ratio < op.ratio_min) {
                    bh = (std::max)(RoundHalfEven(w / static_cast<double>(op.ratio_min)), 1);
                } else if (in_ratio > op.ratio_max) {
                    bw = (std::max)(RoundHalfEven(h * static_cast<double>(op.ratio_max)), 1);
                }
                draws.top[i] = (h - bh) / 2;
                draws.left[i] = (w - bw) / 2;
                draws.box_height[i] = bh;
                draws.box_width[i] = bw;
            }
            break;
        }
        case ImageOpKind::RandomAffine: {
            // torchvision RandomAffine.make_params; the shift rounds to whole pixels.
            if (!training) break;
            draws.factors.resize(batch);
            const double max_dx = op.translate_x * static_cast<double>(input.width);
            const double max_dy = op.translate_y * static_cast<double>(input.height);
            for (auto& f : draws.factors) {
                const double angle = std::uniform_real_distribution<double>(-op.max_angle, op.max_angle)(rng);
                const double tx = RoundHalfEven(std::uniform_real_distribution<double>(-max_dx, max_dx)(rng));
                const double ty = RoundHalfEven(std::uniform_real_distribution<double>(-max_dy, max_dy)(rng));
                const double scale = std::uniform_real_distribution<double>(op.zoom_min, op.zoom_max)(rng);
                const double shear_x = std::uniform_real_distribution<double>(op.shear_x_min, op.shear_x_max)(rng);
                const double shear_y = std::uniform_real_distribution<double>(op.shear_y_min, op.shear_y_max)(rng);
                f = {static_cast<float>(angle), static_cast<float>(tx), static_cast<float>(ty),
                     static_cast<float>(scale), static_cast<float>(shear_x), static_cast<float>(shear_y)};
            }
            break;
        }
        case ImageOpKind::RandomPerspective: {
            // torchvision RandomPerspective.make_params: each corner moves
            // inwards by up to distortion of half the side.
            if (!training) break;
            draws.apply.assign(batch, 0);
            draws.corners.assign(batch, std::vector<int>(8, 0));
            const int h = static_cast<int>(input.height), w = static_cast<int>(input.width);
            const int bound_h = static_cast<int>(op.distortion * static_cast<float>(h / 2)) + 1;
            const int bound_w = static_cast<int>(op.distortion * static_cast<float>(w / 2)) + 1;
            const auto pick = [&rng](int low, int high) { return std::uniform_int_distribution<int>(low, high - 1)(rng); };
            for (size_t i = 0; i < batch; ++i) {
                if (!(unit(rng) < op.probability)) continue;
                draws.corners[i] = {pick(0, bound_w),     pick(0, bound_h),     pick(w - bound_w, w), pick(0, bound_h),
                                    pick(w - bound_w, w), pick(h - bound_h, h), pick(0, bound_w),     pick(h - bound_h, h)};
                // Corners with no mapping (torchvision's lstsq would return
                // a meaningless one) leave the image as it is.
                std::array<float, 8> unused{};
                draws.apply[i] = SolvePerspective(draws.corners[i], h, w, unused) ? 1 : 0;
            }
            break;
        }
        case ImageOpKind::GaussianNoise:
            if (!training) break;
            draws.apply.assign(batch, 1);
            draws.noise_seed = static_cast<uint32_t>(rng());
            break;
        case ImageOpKind::RandomGrayscale:
            if (!training) break;
            draws.apply.resize(batch);
            for (auto& apply : draws.apply) apply = unit(rng) < op.probability ? 1 : 0;
            break;
        case ImageOpKind::AutoAugment: {
            // torchvision AutoAugment.forward: a random sub-policy, each of its
            // two ops with its own probability (else Identity).
            if (!training) break;
            draws.order.assign(batch, std::vector<int>(2, 0));
            draws.factors.assign(batch, std::vector<float>(2, 0.0f));
            const auto& policies = AutoAugmentPolicies(op.policy);
            for (size_t i = 0; i < batch; ++i) {
                if (!(unit(rng) < op.probability)) continue;
                const Policy& policy = policies[static_cast<size_t>(
                    std::uniform_int_distribution<int>(0, static_cast<int>(policies.size()) - 1)(rng))];
                for (size_t s = 0; s < 2; ++s) {
                    if (!(unit(rng) <= policy[s].probability)) continue;
                    float m = policy[s].bin < 0 ? 0.0f
                                                : AugmentMagnitude(AugmentSpace::Standard, policy[s].op, 10, policy[s].bin,
                                                                   input.height, input.width);
                    if (policy[s].bin >= 0 && AugmentSigned(policy[s].op) && unit(rng) <= 0.5f) m = -m;
                    draws.order[i][s] = static_cast<int>(policy[s].op);
                    draws.factors[i][s] = m;
                }
            }
            break;
        }
        case ImageOpKind::TrivialAugment:
            // torchvision TrivialAugmentWide.forward: one op of 14, one bin.
            if (!training) break;
            draws.order.assign(batch, std::vector<int>(1, 0));
            draws.factors.assign(batch, std::vector<float>(1, 0.0f));
            for (size_t i = 0; i < batch; ++i) {
                if (!(unit(rng) < op.probability)) continue;
                const auto chosen = static_cast<AugmentOp>(std::uniform_int_distribution<int>(0, kRandAugmentOps - 1)(rng));
                const int bin = std::uniform_int_distribution<int>(0, op.magnitude_bins - 1)(rng);
                float m = AugmentMagnitude(AugmentSpace::Wide, chosen, op.magnitude_bins, bin, input.height, input.width);
                if (AugmentSigned(chosen) && unit(rng) <= 0.5f) m = -m;
                draws.order[i][0] = static_cast<int>(chosen);
                draws.factors[i][0] = m;
            }
            break;
        case ImageOpKind::AugMix: {
            // torchvision AugMix.forward: Beta (Dirichlet of two) weights for the
            // original and the mix, Dirichlet weights across the chains, each
            // chain chain_depth (or 1 to 3) ops of 13 at a bin below severity.
            if (!training) break;
            const size_t chains = static_cast<size_t>(op.mixture_width);
            const size_t slots = op.chain_depth > 0 ? static_cast<size_t>(op.chain_depth) : 3;
            draws.order.assign(batch, std::vector<int>(chains * slots, 0));
            draws.factors.assign(batch, std::vector<float>(chains * slots, 0.0f));
            draws.weights.assign(batch, std::vector<float>(chains + 1, 0.0f));
            std::gamma_distribution<double> gamma(op.alpha, 1.0);
            const auto dirichlet = [&](size_t k) {
                std::vector<double> g(k);
                double total = 0.0;
                for (double& v : g) total += (v = gamma(rng));
                for (double& v : g) v = total > 0.0 ? v / total : 1.0 / static_cast<double>(k);
                return g;
            };
            for (size_t i = 0; i < batch; ++i) {
                draws.weights[i][0] = 1.0f;
                if (!(unit(rng) < op.probability)) continue;
                const std::vector<double> m = dirichlet(2), mixing = dirichlet(chains);
                draws.weights[i][0] = static_cast<float>(m[0]);
                for (size_t c = 0; c < chains; ++c) {
                    draws.weights[i][c + 1] = static_cast<float>(mixing[c] * m[1]);
                    const size_t depth = op.chain_depth > 0 ? slots
                                                            : static_cast<size_t>(std::uniform_int_distribution<int>(1, 3)(rng));
                    for (size_t s = 0; s < depth; ++s) {
                        const AugmentOp chosen = kAugMixOps[static_cast<size_t>(
                            std::uniform_int_distribution<int>(0, static_cast<int>(kAugMixOps.size()) - 1)(rng))];
                        const int bin = std::uniform_int_distribution<int>(0, op.severity - 1)(rng);
                        float mag = AugmentMagnitude(AugmentSpace::AugMix, chosen, 10, bin, input.height, input.width);
                        if (AugmentSigned(chosen) && unit(rng) <= 0.5f) mag = -mag;
                        draws.order[i][c * slots + s] = static_cast<int>(chosen);
                        draws.factors[i][c * slots + s] = mag;
                    }
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
    const af::array images = RowsToImages(rows.GetArrayRowMajor2D(), input);
    return Tensor::FromArrayRowMajor2D(ImagesToRows(ApplyOnDevice(op, draws, input, images)));
}

float AugmentMagnitude(AugmentSpace space, AugmentOp op, int bins, int index, size_t height, size_t width) {
    const bool wide = space == AugmentSpace::Wide;
    switch (op) {
        case AugmentOp::ShearX:
        case AugmentOp::ShearY:
            return LinspaceAt(0.0f, wide ? 0.99f : 0.3f, bins, index);
        case AugmentOp::TranslateX:
        case AugmentOp::TranslateY: {
            const size_t side = op == AugmentOp::TranslateX ? width : height;
            const float end = wide                            ? 32.0f
                              : space == AugmentSpace::AugMix ? static_cast<float>(static_cast<double>(side) / 3.0)
                                                              : static_cast<float>(150.0 / 331.0 * static_cast<double>(side));
            return LinspaceAt(0.0f, end, bins, index);
        }
        case AugmentOp::Rotate:
            return LinspaceAt(0.0f, wide ? 135.0f : 30.0f, bins, index);
        case AugmentOp::Brightness:
        case AugmentOp::Color:
        case AugmentOp::Contrast:
        case AugmentOp::Sharpness:
            return LinspaceAt(0.0f, wide ? 0.99f : 0.9f, bins, index);
        case AugmentOp::Posterize: {  // base - round(i / ((bins - 1) / steps)) bits
            const int base = space == AugmentSpace::AugMix ? 4 : 8;
            const float steps = wide ? 6.0f : 4.0f;
            return static_cast<float>(base - static_cast<int>(std::nearbyint(
                static_cast<float>(index) / (static_cast<float>(bins - 1) / steps))));
        }
        case AugmentOp::Solarize:
            return LinspaceAt(1.0f, 0.0f, bins, index);
        default:
            return 0.0f;
    }
}

bool AugmentSigned(AugmentOp op) {
    switch (op) {
        case AugmentOp::ShearX:
        case AugmentOp::ShearY:
        case AugmentOp::TranslateX:
        case AugmentOp::TranslateY:
        case AugmentOp::Rotate:
        case AugmentOp::Brightness:
        case AugmentOp::Color:
        case AugmentOp::Contrast:
        case AugmentOp::Sharpness:
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
    const af::array images = RowsToImages(rows.GetArrayRowMajor2D(), shape);
    rows = Tensor::FromArrayRowMajor2D(ImagesToRows(MixImages(method, draw, images)));
    const af::array targets = labels.GetArrayRowMajor2D();
    labels = Tensor::FromArrayRowMajor2D(MixLabels(draw, targets));
}

ImageShape ImageAugmentation::ShapeAfter(const ImageShape& input) const {
    ImageShape shape = input;
    for (const auto& op : ops) shape = ImageShapeAfter(op, shape);
    return shape;
}

Tensor ImageAugmentation::Apply(const Tensor& rows, const ImageShape& input, bool training,
                                std::mt19937& rng, Tensor* labels) const {
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
    if (normalize && !channel_mean.empty()) {
        // images are [H, W, C, N]: one mean and std per channel, tiled over the rest.
        const unsigned channels = static_cast<unsigned>(images.dims(2));
        if (channel_mean.size() != static_cast<size_t>(channels) || channel_std.size() != channel_mean.size())
            throw std::invalid_argument("Normalize has " + std::to_string(channel_mean.size()) +
                                        " channel means for images with " + std::to_string(channels) + " channels");
        const af::dim4 tiles(images.dims(0), images.dims(1), 1, images.dims(3));
        const af::array m = af::tile(af::array(1, 1, channels, 1, channel_mean.data()), tiles);
        const af::array s = af::tile(af::array(1, 1, channels, 1, channel_std.data()), tiles);
        images = (images - m) / s;
    } else if (normalize) {
        images = (images - mean) / std_dev;
    }
    return Tensor::FromArrayRowMajor2D(ImagesToRows(images));
}

}  // namespace cyxwiz::image
