#pragma once

// Image transforms for training batches (TOFIX140 image transforms).
//
// A batch is rows [N, H*W*C], each row one [H, W, C] image in [0, 1]. The
// transforms follow torchvision.transforms.functional and run on the whole
// batch on the ArrayFire device; the host only draws each sample's settings
// (flip yes/no, crop position, angle, jitter factors and order). There is no
// second CPU implementation: without ArrayFire the build refuses them.

#include "api_export.h"
#include "tensor.h"

#include <cstddef>
#include <random>
#include <string>
#include <vector>

namespace cyxwiz::image {

enum class ImageOpKind {
    CenterCrop,
    RandomCrop,
    HorizontalFlip,
    VerticalFlip,
    Rotate,
    ColorJitter,
    GaussianBlur,
    Grayscale,
    Morphology,
    Erase,
    RandAugment,
};

// torchvision.transforms.v2.RandAugment's 14 ops, in its order; each image
// takes num_ops of them at random, signed ones with a random sign.
enum class RandAugmentOp {
    Identity, ShearX, ShearY, TranslateX, TranslateY, Rotate, Brightness, Color,
    Contrast, Sharpness, Posterize, Solarize, AutoContrast, Equalize,
};
constexpr int kRandAugmentOps = 14;
constexpr int kRandAugmentBins = 31;

// The magnitude torchvision uses for op at bin `magnitude` (0..30), unsigned,
// on an image of this height and width.
CYXWIZ_API float RandAugmentMagnitude(RandAugmentOp op, int magnitude, size_t height, size_t width);
CYXWIZ_API bool RandAugmentSigned(RandAugmentOp op);

enum class Interpolation { Nearest, Bilinear };

// Flat square structuring element; pixels outside the image are ignored
// (kornia.morphology / torch max_pool2d borders).
enum class MorphologyOp { Erode, Dilate, Open, Close, Gradient, TopHat, BlackHat };

// Advanced Augment's erasing methods: Cutout (DeVries & Taylor 2017: a square
// of cutout_size centred at a random pixel, clipped to the image) and Random
// Erasing (torchvision RandomErasing: random area and aspect ratio).
enum class EraseMethod { Cutout, RandomErasing };

struct ImageShape {
    size_t height = 0;
    size_t width = 0;
    size_t channels = 0;
    size_t Size() const { return height * width * channels; }
    bool operator==(const ImageShape&) const = default;
};

struct ImageOp {
    ImageOpKind kind = ImageOpKind::HorizontalFlip;
    int height = 0;               // crops
    int width = 0;
    int padding = 0;              // random crop: zero border added first (torchvision padding)
    float probability = 0.5f;     // flips, rotate
    float max_angle = 15.0f;      // rotate: angle drawn in [-max_angle, max_angle] degrees
    Interpolation interpolation = Interpolation::Nearest;
    float brightness = 0.0f;      // color jitter: factor in [max(0, 1-b), 1+b]
    float contrast = 0.0f;
    float saturation = 0.0f;
    float hue = 0.0f;             // shift in [-hue, hue], hue <= 0.5
    int kernel_size = 5;          // gaussian blur and morphology, odd
    float sigma = 1.0f;
    MorphologyOp morphology = MorphologyOp::Erode;
    EraseMethod erase_method = EraseMethod::Cutout;  // erase (probability above applies)
    int cutout_size = 16;
    float scale_min = 0.02f;      // random erasing: area fraction range
    float scale_max = 0.33f;
    float ratio_min = 0.3f;       // random erasing: aspect ratio (h / w) range
    float ratio_max = 3.3f;
    float value = 0.0f;           // erased pixel value in [0, 1]
    int num_ops = 2;              // randaugment: ops per image
    int magnitude = 9;            // randaugment: bin 0..30
};

// True for the nodes that draw random settings and run on Train batches only.
CYXWIZ_API bool IsRandomImageOp(ImageOpKind kind);

// Empty when the op is valid for an input of this shape; otherwise the reason.
CYXWIZ_API std::string ValidateImageOp(const ImageOp& op, const ImageShape& input);

// The output shape of a valid op.
CYXWIZ_API ImageShape ImageShapeAfter(const ImageOp& op, const ImageShape& input);

// One batch's per-sample settings for one op. Unused fields stay empty.
struct ImageOpDraws {
    std::vector<int> top;                  // random crop (in the padded image), erase box
    std::vector<int> left;
    std::vector<int> box_height;           // erase: box size, 0 = this sample is unchanged
    std::vector<int> box_width;
    std::vector<int> apply;                // flips: 1 = flip this sample
    std::vector<float> angle;              // rotate, degrees (0 = unchanged)
    std::vector<std::vector<float>> factors;  // jitter: brightness, contrast, saturation, hue;
                                              // randaugment: each pick's signed magnitude
    std::vector<std::vector<int>> order;      // jitter: application order of those four;
                                              // randaugment: each pick's RandAugmentOp
};

// Draws the settings torchvision's random transform would draw, per sample.
// Outside training, random ops draw nothing: Random Crop centres, the others
// pass the batch through.
CYXWIZ_API ImageOpDraws DrawImageOp(const ImageOp& op, const ImageShape& input,
                                    size_t batch, bool training, std::mt19937& rng);

// Applies one op with the given settings to rows [N, input.Size()].
// Returns rows [N, ImageShapeAfter(op, input).Size()], left on the device.
CYXWIZ_API Tensor ApplyImageOp(const ImageOp& op, const ImageOpDraws& draws,
                               const ImageShape& input, const Tensor& rows);

// Batch mixing (torchvision.transforms.v2 MixUp / CutMix): one lambda per
// batch from Beta(alpha, alpha); each image is mixed with the batch rolled by
// one, and the one-hot labels the same way. Training batches only.
enum class BatchMix { None, MixUp, CutMix };

struct BatchMixDraw {
    bool apply = false;
    float lambda = 1.0f;          // weight of each sample's own image and label
    int top = 0, left = 0;        // CutMix box pasted from the rolled batch
    int height = 0, width = 0;
};

CYXWIZ_API BatchMixDraw DrawBatchMix(BatchMix method, float alpha, float probability,
                                     const ImageShape& shape, std::mt19937& rng);

// Mixes rows [N, shape.Size()] and one-hot labels [N, C] on the device.
CYXWIZ_API void ApplyBatchMix(BatchMix method, const BatchMixDraw& draw, const ImageShape& shape,
                              Tensor& rows, Tensor& labels);

// The ordered transforms between Resize and the model, then Normalize.
struct CYXWIZ_API ImageAugmentation {
    std::vector<ImageOp> ops;
    BatchMix mix = BatchMix::None;
    float mix_alpha = 1.0f;
    float mix_probability = 1.0f;
    bool normalize = false;
    float mean = 0.0f;
    float std_dev = 1.0f;

    bool Empty() const { return ops.empty() && mix == BatchMix::None && !normalize; }
    ImageShape ShapeAfter(const ImageShape& input) const;
    // Uploads the rows once, runs every op, the batch mix (training, when
    // labels are given) and Normalize on the device, and returns rows
    // [N, ShapeAfter(input).Size()] resident on the device.
    Tensor Apply(const Tensor& rows, const ImageShape& input, bool training,
                 std::mt19937& rng, Tensor* labels = nullptr) const;
};

}  // namespace cyxwiz::image
