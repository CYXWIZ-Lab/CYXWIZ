#pragma once

// Local Debug's picked sample (TOFIX178 6.0b): item N of the graph's data,
// made by the batcher training builds for that data kind and prepared the way
// validation sees it (fixed steps, Normalize, the label form; no random
// augmentation), with the Data Split part it falls in.

#include "graph_compiler.h"

#include <cyxwiz/tensor.h>

#include <cstddef>
#include <functional>
#include <string>

namespace cyxwiz {

struct DebugSample {
    Tensor features;
    Tensor labels;
    // "image 120 of cat_dog, validation"; empty when unavailable.
    std::string description;
    // Why the picked item cannot run (Local Debug then runs a synthetic sample
    // and says so); empty when it can.
    std::string unavailable;

    bool ok() const { return unavailable.empty(); }
};

// `stop`: polled while a data kind prepares its items (true stops).
DebugSample MakeDebugSample(const TrainingConfiguration& config, size_t index, std::function<bool()> stop = {});

}  // namespace cyxwiz
