#pragma once

// "What the model sees" (TOFIX178 1.3e): one image through a graph's image
// steps with the code training runs. Decoded at the Resize size, then each
// fixed transform (done once with the images), then draws of the random
// transforms (done per batch), on the ArrayFire device. Normalize is left out
// so the pictures stay in 0..1.

#include "prepared_image_set.h"

#include <cyxwiz/image_augmentation.h>

#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz {

struct PreviewPicture {
    std::string title;               // "Center Crop 64 x 64", "Draw 1"
    std::string detail;              // "64 x 64, once", "flipped, rotated 7.5 degrees"
    int width = 0, height = 0;
    std::vector<uint8_t> rgba;       // width * height * 4
};

struct ImagePipelinePreview {
    PreviewPicture decoded;          // at the Resize size
    std::vector<PreviewPicture> fixed;
    std::vector<std::string> random_ops;  // the per-batch transforms, in order
    std::vector<PreviewPicture> draws;    // empty when there are none
    std::string error;               // the image cannot be read, or a step fails
};

// Changes with the Resize or any image step: views that depend on them redo.
std::string ImageStepsKey(const PreparedImageSpec& spec, const image::ImageAugmentation& augmentation);

ImagePipelinePreview PreviewImagePipeline(const std::string& path, const PreparedImageSpec& spec,
                                          const image::ImageAugmentation& augmentation, int draws, uint32_t seed);

}  // namespace cyxwiz
