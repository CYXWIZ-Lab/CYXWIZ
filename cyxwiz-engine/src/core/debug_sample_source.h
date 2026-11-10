#pragma once

// Local Debug's picked sample (TOFIX178 6.0b): item N of the graph's data,
// made by the batcher training builds for that data kind and prepared the way
// validation sees it (fixed steps, Normalize, the label form; no random
// augmentation), with the Data Split part it falls in.

#include "graph_compiler.h"
#include "graph_model.h"
#include "materialization_cache.h"
#include "training_input_preparation.h"

#include <cyxwiz/tensor.h>

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace cyxwiz {

struct DebugSample {
    Tensor features;
    Tensor labels;
    // "row 120 of reviews, validation"; empty when unavailable.
    std::string description;
    // Each item's name in its source ("cat.10599.jpg"; empty when it has
    // none) and the classes in label order (empty when unknown).
    std::vector<std::string> item_names;
    // Each item's number in its source, in batch order.
    std::vector<size_t> item_indices;
    std::vector<std::string> class_names;
    // Why the picked item cannot run (Local Debug then runs a synthetic sample
    // and says so); empty when it can.
    std::string unavailable;

    bool ok() const { return unavailable.empty(); }
};

struct DebugSampleRequest {
    size_t index = 0;
    // Items index .. index + count - 1 as one batch (a layer capture's
    // "batch of 8", TOFIX178 6.1c); the picked item alone when the data ends
    // before them.
    size_t count = 1;
    // Spread the `count` items evenly over the whole data from `index`
    // (every total / count-th item; Top images, TOFIX178 6.3e) instead of
    // consecutive ones.
    bool spread = false;
    const std::vector<gui::MLNode>* nodes = nullptr;
    const std::vector<gui::NodeLink>* links = nullptr;
    // The project folders and prepared-data cache Train uses, for the data
    // steps before the Data Split.
    StepRunRoots roots;
    MaterializationCacheConfig cache_config;
    // Polled while the data is prepared (true stops).
    std::function<bool()> stop;
};

// `config` is reconciled to the data the sample comes from as Train
// reconciles it (label column, target, input width), so the model Local Debug
// builds from it reads the sample.
DebugSample MakeDebugSample(TrainingConfiguration& config, const DebugSampleRequest& request);

}  // namespace cyxwiz
