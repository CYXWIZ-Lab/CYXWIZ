#pragma once

// The Studio Debugger's "Where it looks" view (TOFIX178 6.3b, approved
// mockup board WhereItLooks): what can be explained, how the input is
// covered, and the result in words. Pure data in and out; the occlusion
// itself is core/occlusion.

#include "layer_view_presentation.h"
#include "occlusion.h"

#include <string>
#include <vector>

namespace cyxwiz::where_it_looks {

// One choice under "Explain": the predicted class, the true class, or
// another output.
struct Target {
    int index = 0;
    std::string label;  // "Predicted: cat", "True: dog", "dog"
    std::string why;    // hover text
};
// The classes (or outputs) of a model: from its class names, else
// "class N" for each of `outputs` (2 for one logit through a sigmoid).
std::vector<std::string> OutputNames(layer_view::OutputRule rule, size_t outputs,
                                     const std::vector<std::string>& class_names);
// Predicted and True first (when known; True left out when it is the
// predicted class), then every other output.
std::vector<Target> Targets(const std::vector<std::string>& names, int predicted, int truth);

// "Patch 12 × 12 px, every 4 px, filled with the image's mean per channel".
std::string CoverText(const occlusion::Plan& plan, const std::string& fill_source);
// "196 forward passes (14 × 14 patch positions), 64 at a time on the device."
std::string PassesText(const occlusion::Plan& plan, size_t pieces, size_t chunk);

// Changes scaled to [-1, 1] by the largest size (0 when nothing moved).
std::vector<float> Scaled(const std::vector<double>& change);

// What a piece is, in words: "rows 16-27, columns 24-35", "\"restless\""
// (token words when known, else "token 4123 at 7"), "plas".
std::string PieceText(const occlusion::Plan& plan, const occlusion::Piece& piece,
                      const std::vector<std::string>& piece_names);

// "Covering rows 16-27, columns 24-35 lowers P(cat) most: 0.940 to 0.531."
std::string Headline(const occlusion::Plan& plan, const std::vector<occlusion::Piece>& pieces,
                     const occlusion::Result& result, const std::string& target_name,
                     const std::vector<std::string>& piece_names);

// "-0.409": a change for the hover and the feature rows.
std::string ChangeText(double change);

// The Inspector's note for the view.
const std::string& InspectorHint();

}  // namespace cyxwiz::where_it_looks
