#pragma once

// Unit weights as images (TOFIX178 6.4c, approved mockup board Dense): a
// Dense layer that reads the model's image flattened has one weight per
// pixel and colour, so each unit's weight row is a picture of what it looks
// for. Grey images draw a unit's weights red where a bright pixel raises the
// unit and blue where it lowers it; colour images draw them as the colours
// that raise it most (each unit on its own scale, as the Filters view).

#include <cyxwiz/tensor.h>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz::unit_weights {

// The model's image as the Dense reads it.
struct ImageShape {
    size_t height = 0;
    size_t width = 0;
    size_t channels = 0;  // 1 or 3
    size_t Pixels() const { return height * width * channels; }
};

// The image this Dense reads, when it reads the model's image flattened:
// every layer before the Dense has no parameters (Flatten, Dropout,
// Reshape), the model's input is a picture (H x W, C 1 or 3) and the Dense
// has H x W x C inputs. Such a model takes the image batch first [N, H, W, C]
// and its Flatten keeps [H, W, C] row-major order, so weight k of a unit
// belongs to row k / (W C), column (k / C) % W, channel k % C.
std::optional<ImageShape> ReadsImage(size_t dense_inputs, size_t image_height, size_t image_width,
                                     size_t image_channels, bool only_parameter_free_layers_before);

// Each unit's weight norm (rows of a [units, inputs] matrix), computed on
// the device; only the norms come back.
std::vector<float> RowNorms(const Tensor& weights);

// The units to draw: the largest norm first (ties by unit) or in unit order;
// `count` of them, 0 for all.
std::vector<size_t> Order(const std::vector<float>& norms, bool by_norm, size_t count);

// The rows `units` of a [units, inputs] matrix, gathered on the device, one
// after another on the host.
std::vector<float> Rows(const Tensor& weights, const std::vector<size_t>& units);

// One unit's weights as a picture: `pixels` row-major [H, W, C]. Grey (C 1):
// each weight over the unit's largest |weight|, in [-1, 1] (0 stays 0, so
// red and blue keep their meaning). Colour (C 3): scaled to the unit's own
// range over all its colours, in [0, 1].
struct Picture {
    std::vector<float> pixels;
    float low = 0.0f;   // the smallest weight
    float high = 0.0f;  // the largest weight
};
Picture Draw(const float* row, const ImageShape& image);

// "Each unit's 784 input weights drawn as the 28 × 28 image they read: ..."
std::string Lead(const ImageShape& image, size_t units, size_t shown, bool by_norm);
// The Inspector's note for Unit weights.
const std::string& Hint();

}  // namespace cyxwiz::unit_weights
