#pragma once

// Data Studio > Prepare for an image graph (TOFIX178 1.6): the graph's image
// steps as training runs them - Resize, the steps done once when the images
// are prepared, the ones drawn per batch on training images, the batch mix,
// Normalize - each with its node, in words, with its output size. Built from
// the live compile, so the groups are training's own (SplitFixedImageTransforms:
// fixed until the first random step). Pure data; gui/data_studio/image_recipe_view
// draws it.

#include "graph_compiler.h"

#include <functional>
#include <string>
#include <vector>

namespace cyxwiz::image_recipe {

enum class Group { Resize, Once, PerBatch, Normalize };

struct Row {
    int node_id = -1;
    std::string name;        // the node's name on the canvas
    std::string what;        // "Center Crop 64 x 64", "Image Contrast (CLAHE) 2 / 8"
    Group group = Group::Once;
    bool opencv = false;     // an OpenCV step: done once, never after a random step
    bool random = false;     // a random device step (per batch from here on)
    image::ImageShape output;  // after this step
    std::string output_text; // "64 x 64 x 3" (a times sign)
    // The compiled ops up to and including this row (image_augmentation.ops),
    // for a preview of the steps up to it.
    size_t ops_through = 0;
};

struct Input {
    const TrainingConfiguration* config = nullptr;  // the canvas graph's compile
    int resize_node_id = -1;                        // the graph's Resize node
    std::function<std::string(int)> name_of;        // a node's name by id
    std::string dataset;                            // "Cats_Dogs Data"
};

struct View {
    bool known = false;       // an image graph that has compiled
    std::string reason;       // why not, when not known
    std::vector<Row> rows;    // in training's order
    std::string subtitle;     // "Cats_Dogs Data to model input [64, 64, 3] · 4 steps once, 3 per batch"
    size_t once = 0, per_batch = 0;
};

View Build(const Input& in);

// Empty when `rows[from]` may move to just after `rows[after]` (after = -1:
// first of the steps); otherwise why not. Resize and Normalize stay where
// they are, and an OpenCV step may not follow a random one (it is done once
// with the images; the compiler refuses that graph).
std::string MoveRefusal(const View& view, size_t from, long long after);

// True when moving `rows[from]` to just after `rows[after]` leaves the order
// as it is (after itself, or after the row already before it).
bool MoveChangesNothing(const View& view, size_t from, long long after);

}  // namespace cyxwiz::image_recipe
