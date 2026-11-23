#pragma once

// The Model window's Health tab (TOFIX129 D11): one row per weight layer of
// the training's health record (core/training_health) - its weight size and
// mean gradient size by epoch, its weights at the last epoch, and the dead
// units of the ReLU after it - with a headline, what to try, and the words
// for the selected layer. Pure data in and out.

#include "training_health.h"

#include <string>
#include <vector>

namespace cyxwiz {

struct HealthRow {
    size_t layer = 0;
    std::string name;            // the layer's name on the canvas
    std::string weight_norms;    // "3.74 → 3.98 → 4.13"
    std::string gradient_norms;  // "0.152 → 0.133 → 0.135"
    std::vector<long long> histogram;  // the last epoch's
    std::string low, high;       // its range, "-0.463", "0.455"
    size_t count = 0;
    std::string dead;            // "15 of 64 (23 %)", "no ReLU", "not checked"
    bool dead_warning = false;   // a fifth or more dead
    std::string words;           // the selected layer's sentence
};

struct HealthView {
    bool has_health = false;
    std::string empty_text;
    std::vector<HealthRow> rows;
    std::string headline;
    bool warning = false;
    std::string detail;
    std::string advice;
    std::string epochs;          // "3 epochs"
};

// `layer_names`: the compiled layers' names by index (the N of "layerN").
HealthView BuildHealthView(const TrainingHealth& health, const std::vector<std::string>& layer_names);

}  // namespace cyxwiz
