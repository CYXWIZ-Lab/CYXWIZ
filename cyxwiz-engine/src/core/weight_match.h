#pragma once

// Pairs the tensors of a weight file with the layers of a built graph, by
// node name, and turns PyTorch's layouts into the Engine's. Pure: it only
// reads names, shapes and values, so the same report serves the Data Loader
// dialog before training and the run itself.
//
// A layer takes the entries whose name is "<node name>.<parameter>": node
// "layer1.0.conv1" takes "layer1.0.conv1.weight". A layer is loaded only when
// every one of its parameters is in the file with the right shape; otherwise
// it starts fresh and the report says why.

#include "graph_model.h"
#include "weight_file.h"

#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::weight_match {

// One layer of the built model that holds parameters.
struct LayerSlot {
    std::string node_name;
    gui::NodeType type = gui::NodeType::Dense;
    std::string type_label;   // "Conv2D", for the report
    std::string key_prefix;   // "layer7." (the model's parameter keys start with it)
    // The module's parameters, state included: key ("weights", "gamma", "running_mean") and shape.
    std::vector<std::pair<std::string, std::vector<size_t>>> parameters;
};

struct LayerRow {
    std::string node_name;
    std::string type_label;
    std::vector<std::string> file_entries;  // "conv1.weight [64, 3, 7, 7]"; empty when the file has none
    std::string graph_shapes;               // "[7, 7, 3, 64]" or "[64] x 4"
    bool loaded = false;
    std::string reason;                     // why it starts fresh; empty when loaded
};

struct UnusedEntry {
    std::string name;    // "fc.weight [1000, 512]"
    std::string reason;  // "no node is named fc", "a counter, not a weight", ...
};

// A tensor to hand to the model: its parameter key and row-major values in the Engine's layout.
struct LoadedTensor {
    std::string key;  // "layer7.weights"
    std::vector<size_t> shape;
    std::vector<float> values;
};

struct Report {
    std::string format;                // the file's format
    size_t file_entries = 0;
    std::vector<LayerRow> layers;      // every layer with parameters, in model order
    std::vector<UnusedEntry> unused;   // file entries no layer took
    std::vector<LoadedTensor> tensors; // what to set on the model
    size_t loaded = 0;                 // layers that take their weights
    size_t fresh = 0;                  // layers that start fresh

    // "40 of 41 layers take their weights; 1 starts fresh: fc (shape differs ...)".
    std::string Summary() const;
};

Report Match(const weight_file::WeightFile& file, const std::vector<LayerSlot>& layers);

// "[64, 3, 7, 7]"
std::string ShapeText(const std::vector<size_t>& shape);

}  // namespace cyxwiz::weight_match
