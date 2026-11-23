#pragma once

// "Inside this layer" in the Properties panel: the steps a composite layer
// node runs, in order, each with its output for one sample and its learnable
// weights. Pure data in and out: no ImGui, no registry, no backend calls.

#include "graph_model.h"

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace cyxwiz {

struct LayerInsideStep {
    std::string name;     // "Dense 32 to 8", "ReLU"
    std::string output;   // "[8]"; "" when the sizes are not known
    std::string weights;  // "264"; "" for a step without weights
};

struct LayerInside {
    bool applies = false;  // the node type has steps inside
    std::vector<LayerInsideStep> steps;
    std::string total;     // the steps' weights summed; "" when the sizes are not known
    std::string note;
};

// `input_sample` is the node's compiled input for one sample ([H, W, C]);
// empty before the graph has compiled.
LayerInside BuildLayerInside(gui::NodeType type,
                             const std::map<std::string, std::string>& parameters,
                             const std::vector<size_t>& input_sample);

}  // namespace cyxwiz
