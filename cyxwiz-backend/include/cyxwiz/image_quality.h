#pragma once

// Image quality measurements for the Quality Analyzer (TOFIX140).
//
// A batch is rows [N, H*W*C], each row one [H, W, C] RGB (or 1-channel)
// image in [0, 1], as the image batcher decodes it at the Resize size. Every
// metric is taken on the luminance L = 255 * (0.299 R + 0.587 G + 0.114 B),
// the OpenCV RGB2GRAY weights, in float:
//
// - blur: variance of the 3x3 Laplacian of L (cv2.Laplacian ksize=1,
//   reflect-101 border); low = blurry.
// - brightness: mean of L, 0 to 255.
// - contrast: standard deviation of L / 255.
// - hash: 64-bit difference hash. L is area-averaged to 8 rows x 9 columns
//   (cv2.resize INTER_AREA); bit r*8+c is set when column c+1 is brighter
//   than column c in row r.
//
// The measurements run on the ArrayFire device; there is no CPU copy, and a
// build without ArrayFire refuses them.

#include "api_export.h"
#include "image_augmentation.h"
#include "tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz::image {

// Bump when a metric's definition changes, so cached analyses go stale.
constexpr int kImageQualityVersion = 1;

struct ImageQualityMetrics {
    float blur = 0.0f;
    float brightness = 0.0f;
    float contrast = 0.0f;
    uint64_t hash = 0;
};

// Empty when the shape can be measured, else why not (the hash needs at least
// 8 rows and 9 columns, and 1 or 3 channels).
CYXWIZ_API std::string ValidateImageQualityShape(const ImageShape& shape);

CYXWIZ_API std::vector<ImageQualityMetrics> MeasureImageQuality(const Tensor& rows, const ImageShape& shape);

CYXWIZ_API int HashDistance(uint64_t a, uint64_t b);

// For each image, the index of an earlier kept image whose hash differs in at
// most max_bits bits, or -1. Images are taken in order and an image is kept
// when no earlier kept image is that close (the first of each group stays).
// The pairwise comparison runs on the device in tiles.
CYXWIZ_API std::vector<int64_t> FindNearDuplicates(const std::vector<uint64_t>& hashes, int max_bits);

// Why an image is left out; a bit set.
enum ImageQualityReason : uint32_t {
    kQualityBlurry = 1u << 0,
    kQualityDark = 1u << 1,
    kQualityBright = 1u << 2,
    kQualityLowContrast = 1u << 3,
    kQualityDuplicate = 1u << 4,
};

struct ImageQualityChecks {
    bool blur = true;
    float blur_min = 400.0f;
    bool brightness = true;
    float brightness_min = 50.0f;
    float brightness_max = 220.0f;
    bool contrast = true;
    float contrast_min = 0.12f;
    bool duplicates = true;
    int duplicate_bits = 4;
};

// Empty when the checks are usable, else why not.
CYXWIZ_API std::string ValidateImageQualityChecks(const ImageQualityChecks& checks);

// The reasons each image fails; duplicate_of is FindNearDuplicates' result
// (ignored when the duplicate check is off).
CYXWIZ_API std::vector<uint32_t> JudgeImageQuality(const std::vector<ImageQualityMetrics>& metrics,
                                                   const std::vector<int64_t>& duplicate_of,
                                                   const ImageQualityChecks& checks);

}  // namespace cyxwiz::image
