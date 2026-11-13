#pragma once

// What Neural Architecture Search shows (TOFIX178 T4 H2, approved mockup), as
// data: the input and output read from the compiled graph, the quick score's
// parts and summary, the change buttons, what a change did, a version's
// layers, and the saved defaults set on a search. No ImGui.

#include "engine_config.h"
#include "graph_compiler.h"
#include "nas_evaluator.h"
#include "trial_runner.h"

#include <string>
#include <vector>

namespace cyxwiz::nas_presentation {

static_assert(core::kArchitectureSearchMaxTrain == kNASRanked, "Train the best N trains from the ranked versions");

// The model's input and output, read from the compiled graph.
struct GraphShape {
    bool ok = false;
    int outputs = 0;
    std::string task;  // "classification", "regression" or "image_classification" (the score column)
    std::string text;  // "784 features in, 10 classes out"
    std::string inputs_text;   // "784 features"
    std::string outputs_text;  // "10 classes"
    std::string why;           // why not ok
};
GraphShape ReadShape(const TrainingConfiguration& compiled);

// "The head (Dense 3, the 10 outputs) is never removed, resized or swapped."
std::string HeadNote(const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links,
                     const GraphShape& shape);

struct ScorePart {
    std::string label;
    double value = 0.0;  // 0..1
    std::string tip;
};
// Size, Efficiency, Depth, Variety, Skip links.
std::vector<ScorePart> ScoreParts(const ArchitectureScore& score);
// "Parameters 109,386 · model layers: 3x Dense, 2x ReLU · no skip connections"
std::string ScoreSummary(const ArchitectureScore& score, const std::vector<gui::MLNode>& nodes,
                         const std::vector<gui::NodeLink>& links);
std::string ScoreText(double value);  // "0.31"
std::string Grouped(int64_t value);   // "109,386"

struct Mutation {
    MutationType type;
    const char* label;
    const char* tip;
};
// Add layer, Remove layer, Swap layer, Change units, Change activation, Random.
const std::vector<Mutation>& Mutations();

// What a change did, from the graph before and after: "Added BatchNorm before
// Dense 3", "Removed Dense 2 (ReLU feeds Dense 3)", "Dense 1 units 128 to 192",
// "ReLU became GELU". Empty when nothing changed.
std::string DescribeChange(const std::vector<gui::MLNode>& before, const std::vector<gui::NodeLink>& before_links,
                           const std::vector<gui::MLNode>& after, const std::vector<gui::NodeLink>& after_links);

// A version's model layers in order: "Dense 192, BatchNorm, ReLU, Dense 10".
std::string LayersText(const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links);

// A trained version's validation loss and accuracy (or MAE): "0.079", "97.7 %".
std::string LossText(const TrialScore& score);
std::string AccuracyText(const TrialScore& score);


// The saved defaults (Preferences > Search) set on a search, and read back.
void ApplyDefaults(NASSearchConfig& config, int& train_best, const core::ArchitectureSearchDefaults& defaults);
core::ArchitectureSearchDefaults DefaultsOf(const NASSearchConfig& config, int train_best);

}  // namespace cyxwiz::nas_presentation
