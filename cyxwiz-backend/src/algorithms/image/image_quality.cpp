#include "cyxwiz/image_quality.h"

#include <algorithm>
#include <bit>
#include <stdexcept>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz::image {

namespace {

constexpr int kHashRows = 8;
constexpr int kHashCols = 9;

#ifdef CYXWIZ_HAS_ARRAYFIRE

// Luminance [H, W, N] of rows [N, H*W*C] (HWC per row), 0 to 255.
af::array Luminance(const af::array& rows, const ImageShape& shape) {
    const dim_t n = rows.dims(0);
    const dim_t h = static_cast<dim_t>(shape.height), w = static_cast<dim_t>(shape.width);
    af::array images = af::reorder(af::moddims(rows, n, static_cast<dim_t>(shape.channels), w, h), 3, 2, 1, 0);
    if (shape.channels == 3) {
        images = 0.299f * images(af::span, af::span, 0, af::span) +
                 0.587f * images(af::span, af::span, 1, af::span) +
                 0.114f * images(af::span, af::span, 2, af::span);
    }
    return af::moddims(images * 255.0f, h, w, n);
}

// Indices 0..size+1 of a reflect-101 padded axis: [1, 0, 1, ..., size-1, size-2].
af::array Reflect101(dim_t size) {
    std::vector<int> index(static_cast<size_t>(size) + 2);
    index.front() = 1;
    for (dim_t i = 0; i < size; ++i) index[static_cast<size_t>(i) + 1] = static_cast<int>(i);
    index.back() = static_cast<int>(size) - 2;
    return af::array(static_cast<dim_t>(index.size()), index.data());
}

// cv2.Laplacian(ksize=1): the 4-neighbour sum minus 4x the centre.
af::array Laplacian(const af::array& luma) {
    const dim_t h = luma.dims(0), w = luma.dims(1);
    const af::array p = luma(Reflect101(h), Reflect101(w), af::span);
    const af::seq rows(1, static_cast<double>(h)), cols(1, static_cast<double>(w));
    return p(af::seq(0, static_cast<double>(h - 1)), cols, af::span) +
           p(af::seq(2, static_cast<double>(h + 1)), cols, af::span) +
           p(rows, af::seq(0, static_cast<double>(w - 1)), af::span) +
           p(rows, af::seq(2, static_cast<double>(w + 1)), af::span) -
           4.0f * p(rows, cols, af::span);
}

// [out, in] weights of an area-average shrink (cv2 INTER_AREA): each output
// cell averages the input pixels it covers, partial pixels by their overlap.
af::array AreaWeights(size_t in, int out) {
    const double cell = static_cast<double>(in) / out;
    std::vector<float> w(static_cast<size_t>(out) * in, 0.0f);  // column-major [out, in]
    for (int o = 0; o < out; ++o) {
        const double begin = o * cell, end = begin + cell;
        for (size_t i = static_cast<size_t>(begin); i < in && static_cast<double>(i) < end; ++i) {
            const double overlap = (std::min)(end, i + 1.0) - (std::max)(begin, static_cast<double>(i));
            if (overlap > 0.0) w[i * out + o] = static_cast<float>(overlap / cell);
        }
    }
    return af::array(static_cast<dim_t>(out), static_cast<dim_t>(in), w.data());
}

// Difference-hash bits [8 (column), 8 (row), N] as 0/1.
af::array HashBits(const af::array& luma) {
    const dim_t h = luma.dims(0), w = luma.dims(1), n = luma.dims(2);
    const af::array rows = af::matmul(AreaWeights(static_cast<size_t>(h), kHashRows), af::moddims(luma, h, w * n));
    const af::array by_column = af::moddims(af::reorder(af::moddims(rows, kHashRows, w, n), 1, 0, 2), w, kHashRows * n);
    const af::array shrunk = af::moddims(af::matmul(AreaWeights(static_cast<size_t>(w), kHashCols), by_column),
                                        kHashCols, kHashRows, n);
    return shrunk(af::seq(1, kHashCols - 1), af::span, af::span) > shrunk(af::seq(0, kHashCols - 2), af::span, af::span);
}

#endif

}  // namespace

std::string ValidateImageQualityShape(const ImageShape& shape) {
    if (shape.channels != 1 && shape.channels != 3) {
        return "the Quality Analyzer measures 1- or 3-channel images, not " + std::to_string(shape.channels);
    }
    if (shape.height < static_cast<size_t>(kHashRows) || shape.width < static_cast<size_t>(kHashCols)) {
        return "the Quality Analyzer needs images of at least 8 x 9 pixels; set a larger Resize size";
    }
    return {};
}

std::vector<ImageQualityMetrics> MeasureImageQuality(const Tensor& rows, const ImageShape& shape) {
    if (const std::string why = ValidateImageQualityShape(shape); !why.empty()) {
        throw std::invalid_argument(why);
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const af::array luma = Luminance(rows.GetArrayRowMajor2D(), shape);
    const dim_t pixels = luma.dims(0) * luma.dims(1), n = luma.dims(2);
    const af::array flat = af::moddims(luma, pixels, n);
    const af::array lap = af::moddims(Laplacian(luma), pixels, n);

    std::vector<float> blur(static_cast<size_t>(n)), mean(static_cast<size_t>(n)), spread(static_cast<size_t>(n));
    af::var(lap, AF_VARIANCE_POPULATION, 0).host(blur.data());
    af::mean(flat, 0).host(mean.data());
    af::stdev(flat, AF_VARIANCE_POPULATION, 0).host(spread.data());
    std::vector<char> bits(static_cast<size_t>(n) * 64);
    HashBits(luma).as(b8).host(bits.data());

    std::vector<ImageQualityMetrics> out(static_cast<size_t>(n));
    for (size_t i = 0; i < out.size(); ++i) {
        out[i].blur = blur[i];
        out[i].brightness = mean[i];
        out[i].contrast = spread[i] / 255.0f;
        uint64_t hash = 0;
        for (size_t b = 0; b < 64; ++b) {
            if (bits[i * 64 + b]) hash |= uint64_t{1} << b;  // b = row * 8 + column
        }
        out[i].hash = hash;
    }
    return out;
#else
    (void)rows;
    throw std::runtime_error("the Quality Analyzer needs the ArrayFire build");
#endif
}

int HashDistance(uint64_t a, uint64_t b) {
    return std::popcount(a ^ b);
}

std::vector<int64_t> FindNearDuplicates(const std::vector<uint64_t>& hashes, int max_bits) {
    const size_t n = hashes.size();
    std::vector<int64_t> duplicate_of(n, -1);
    if (n < 2) return duplicate_of;
#ifdef CYXWIZ_HAS_ARRAYFIRE
    // Bits as +-1, [64, N]: the dot product of two hashes is 64 - 2 * distance.
    std::vector<float> signs(n * 64);
    for (size_t i = 0; i < n; ++i) {
        for (size_t b = 0; b < 64; ++b) signs[i * 64 + b] = ((hashes[i] >> b) & 1u) ? 1.0f : -1.0f;
    }
    const af::array all(64, static_cast<dim_t>(n), signs.data());
    const float closest = static_cast<float>(64 - 2 * max_bits);

    // earlier[j]: the images before j within max_bits, found tile by tile.
    std::vector<std::vector<int64_t>> earlier(n);
    const size_t tile = std::clamp<size_t>((size_t{1} << 24) / n, 1, n);
    for (size_t t0 = 0; t0 < n; t0 += tile) {
        const size_t rows = (std::min)(tile, n - t0);
        const af::array dots = af::matmul(all(af::span, af::seq(static_cast<double>(t0), static_cast<double>(t0 + rows - 1))),
                                          all, AF_MAT_TRANS, AF_MAT_NONE);
        const af::dim4 dims(static_cast<dim_t>(rows), static_cast<dim_t>(n));
        const af::array later = af::range(dims, 1) > (af::range(dims, 0) + static_cast<int>(t0));
        const af::array hits = af::where((dots >= closest) && later);
        if (hits.elements() == 0) continue;
        std::vector<unsigned> index(static_cast<size_t>(hits.elements()));
        hits.host(index.data());
        for (unsigned k : index) {
            earlier[k / rows].push_back(static_cast<int64_t>(t0 + k % rows));
        }
    }
    std::vector<bool> kept(n, true);
    for (size_t j = 0; j < n; ++j) {
        std::sort(earlier[j].begin(), earlier[j].end());
        for (int64_t i : earlier[j]) {
            if (kept[static_cast<size_t>(i)]) {
                duplicate_of[j] = i;
                kept[j] = false;
                break;
            }
        }
    }
    return duplicate_of;
#else
    (void)max_bits;
    throw std::runtime_error("the Quality Analyzer needs the ArrayFire build");
#endif
}

std::string ValidateImageQualityChecks(const ImageQualityChecks& checks) {
    if (checks.blur && !(checks.blur_min >= 0.0f)) return "the blur threshold must be 0 or more";
    if (checks.brightness &&
        !(checks.brightness_min >= 0.0f && checks.brightness_min < checks.brightness_max && checks.brightness_max <= 255.0f)) {
        return "brightness needs 0 <= darkest < brightest <= 255";
    }
    if (checks.contrast && !(checks.contrast_min >= 0.0f && checks.contrast_min <= 1.0f)) {
        return "the contrast threshold must be between 0 and 1";
    }
    if (checks.duplicates && (checks.duplicate_bits < 0 || checks.duplicate_bits > 32)) {
        return "the near-duplicate distance must be 0 to 32 bits";
    }
    return {};
}

std::vector<uint32_t> JudgeImageQuality(const std::vector<ImageQualityMetrics>& metrics,
                                        const std::vector<int64_t>& duplicate_of,
                                        const ImageQualityChecks& checks) {
    std::vector<uint32_t> reasons(metrics.size(), 0);
    for (size_t i = 0; i < metrics.size(); ++i) {
        const ImageQualityMetrics& m = metrics[i];
        uint32_t r = 0;
        if (checks.blur && m.blur < checks.blur_min) r |= kQualityBlurry;
        if (checks.brightness && m.brightness < checks.brightness_min) r |= kQualityDark;
        if (checks.brightness && m.brightness > checks.brightness_max) r |= kQualityBright;
        if (checks.contrast && m.contrast < checks.contrast_min) r |= kQualityLowContrast;
        if (checks.duplicates && i < duplicate_of.size() && duplicate_of[i] >= 0) r |= kQualityDuplicate;
        reasons[i] = r;
    }
    return reasons;
}

}  // namespace cyxwiz::image
