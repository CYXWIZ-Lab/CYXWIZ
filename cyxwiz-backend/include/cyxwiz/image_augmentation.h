// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

// Image transforms for training batches (TOFIX140 image transforms).
//
// A batch is rows [N, H*W*C], each row one [H, W, C] image in [0, 1]. The
// transforms follow torchvision.transforms.functional and run on the whole
// batch on the ArrayFire device; the host only draws each sample's settings
// (flip yes/no, crop position, angle, jitter factors and order). There is no
// second CPU implementation.

#include "api_export.h"
#include "tensor.h"

#include <cstddef>
#include <cstdint>
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
    RandomResizedCrop,
    RandomAffine,
    RandomPerspective,
    GaussianNoise,
    RandomGrayscale,
    AutoAugment,
    TrivialAugment,
    AugMix,
};

// The operations the policy augmentations pick from: RandAugment's 14, in
// torchvision.transforms.v2.RandAugment's order, then AutoAugment's Invert.
enum class AugmentOp {
    Identity, ShearX, ShearY, TranslateX, TranslateY, Rotate, Brightness, Color,
    Contrast, Sharpness, Posterize, Solarize, AutoContrast, Equalize, Invert,
};
constexpr int kAugmentOps = 15;
constexpr int kRandAugmentOps = 14;  // RandAugment draws from Identity..Equalize
constexpr int kRandAugmentBins = 31;

// The magnitude tables of torchvision's policies: Standard (RandAugment,
// AutoAugment), Wide (TrivialAugmentWide) and AugMix's.
enum class AugmentSpace { Standard, Wide, AugMix };

// The magnitude torchvision uses for op at bin `index` of `bins`, unsigned,
// on an image of this height and width (0 for the ops without one).
CYXWIZ_API float AugmentMagnitude(AugmentSpace space, AugmentOp op, int bins, int index, size_t height,
                                  size_t width);
CYXWIZ_API bool AugmentSigned(AugmentOp op);

// torchvision AutoAugmentPolicy: 25 sub-policies of two (op, probability,
// magnitude bin of 10) each.
enum class AutoAugmentPolicy { ImageNet, Cifar10, Svhn };

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
    int height = 0;               // crops; random resized crop: the output size
    int width = 0;
    int padding = 0;              // random crop: zero border added first (torchvision padding)
    float probability = 0.5f;     // flips, rotate, perspective, random grayscale, policies
    float max_angle = 15.0f;      // rotate, random affine: angle drawn in [-max_angle, max_angle] degrees
    Interpolation interpolation = Interpolation::Nearest;
    float brightness = 0.0f;      // color jitter: factor in [max(0, 1-b), 1+b]
    float contrast = 0.0f;
    float saturation = 0.0f;
    float hue = 0.0f;             // shift in [-hue, hue], hue <= 0.5
    int kernel_size = 5;          // gaussian blur and morphology, odd
    float sigma = 1.0f;           // gaussian blur; gaussian noise: the noise's standard deviation
    MorphologyOp morphology = MorphologyOp::Erode;
    EraseMethod erase_method = EraseMethod::Cutout;  // erase (probability above applies)
    int cutout_size = 16;
    float scale_min = 0.02f;      // random erasing, random resized crop: area fraction range
    float scale_max = 0.33f;
    float ratio_min = 0.3f;       // aspect ratio range: random erasing h / w, random resized crop w / h
    float ratio_max = 3.3f;
    float value = 0.0f;           // erased pixel value in [0, 1]
    int num_ops = 2;              // randaugment: ops per image
    int magnitude = 9;            // randaugment: bin 0..30
    float translate_x = 0.0f;     // random affine: shift up to this fraction of the width / height
    float translate_y = 0.0f;
    float zoom_min = 1.0f;        // random affine: scale factor range
    float zoom_max = 1.0f;
    float shear_x_min = 0.0f;     // random affine: shear ranges, degrees
    float shear_x_max = 0.0f;
    float shear_y_min = 0.0f;
    float shear_y_max = 0.0f;
    float fill = 0.0f;            // random affine, perspective: value outside the image, in [0, 1]
    float distortion = 0.5f;      // random perspective: distortion_scale in [0, 1]
    float noise_mean = 0.0f;      // gaussian noise
    bool clip = true;             // gaussian noise: clamp the result to [0, 1]
    AutoAugmentPolicy policy = AutoAugmentPolicy::ImageNet;
    int magnitude_bins = 31;      // trivial augment
    int severity = 3;             // augmix: magnitude bins drawn from (of 10)
    int mixture_width = 3;        // augmix: chains mixed
    int chain_depth = -1;         // augmix: ops per chain, -1 = 1 to 3 at random
    float alpha = 1.0f;           // augmix: Dirichlet / Beta concentration
};

// True for the nodes that draw random settings and run on Train batches only.
CYXWIZ_API bool IsRandomImageOp(ImageOpKind kind);

// Empty when the op is valid for an input of this shape; otherwise the reason.
CYXWIZ_API std::string ValidateImageOp(const ImageOp& op, const ImageShape& input);

// The output shape of a valid op.
CYXWIZ_API ImageShape ImageShapeAfter(const ImageOp& op, const ImageShape& input);

// One batch's per-sample settings for one op. Unused fields stay empty.
struct ImageOpDraws {
    std::vector<int> top;                  // random crop (in the padded image), erase box, resized crop box
    std::vector<int> left;
    std::vector<int> box_height;           // erase: box size, 0 = this sample is unchanged; resized crop box
    std::vector<int> box_width;
    std::vector<int> apply;                // flips, perspective, random grayscale, noise: 1 = this sample
    std::vector<float> angle;              // rotate, degrees (0 = unchanged)
    std::vector<std::vector<float>> factors;  // jitter: brightness, contrast, saturation, hue;
                                              // policies: each pick's signed magnitude;
                                              // affine: angle, tx, ty (pixels), scale, shear x, shear y
    std::vector<std::vector<int>> order;      // jitter: application order of those four;
                                              // policies: each pick's AugmentOp (augmix: chain k's
                                              // steps at 3k..3k+2, Identity past its depth)
    std::vector<std::vector<int>> corners;    // perspective: end points tl, tr, br, bl as x, y
    std::vector<std::vector<float>> weights;  // augmix: the original's weight, then each chain's
    std::vector<float> noise;                 // gaussian noise: every value, [N, H*W*C] row-major;
    uint32_t noise_seed = 0;                  // when empty, drawn on the device from this seed
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
    // Per channel when set (one value for each channel of the images after
    // the ops), used instead of mean / std_dev.
    std::vector<float> channel_mean;
    std::vector<float> channel_std;

    bool Empty() const { return ops.empty() && mix == BatchMix::None && !normalize; }
    ImageShape ShapeAfter(const ImageShape& input) const;
    // Uploads the rows once, runs every op, the batch mix (training, when
    // labels are given) and Normalize on the device, and returns rows
    // [N, ShapeAfter(input).Size()] resident on the device.
    Tensor Apply(const Tensor& rows, const ImageShape& input, bool training,
                 std::mt19937& rng, Tensor* labels = nullptr) const;
};

}  // namespace cyxwiz::image
