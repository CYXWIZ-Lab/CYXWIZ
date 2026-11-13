#pragma once

// "Where it looks" by occlusion (TOFIX178 6.3b): cover one piece of the input
// at a time (an image patch, a token, a feature), run the model again and see
// how the picked output's score moves. Forward passes only, on the device,
// in batches; the host gets one score per piece.
//
// The sample is in the model's input layout with one item: an image
// [H, W, C, 1] (batch last), tokens or features [1, T] (batch first).

#include "debug_model_session.h"
#include "layer_view_presentation.h"

#include <cyxwiz/tensor.h>

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz::occlusion {

enum class InputKind { Image, Tokens, Features };

// What the input is, from its shape and whether the first layer is an
// Embedding (token ids); nullopt when it is none of the three.
std::optional<InputKind> KindOf(const std::vector<size_t>& sample_shape, bool first_layer_embeds);

// One covered piece: an image patch (rows, columns) or one position.
struct Piece {
    size_t row = 0;
    size_t column = 0;
    size_t rows = 1;
    size_t columns = 1;
    size_t position = 0;  // token or feature index
};

struct Plan {
    InputKind kind = InputKind::Features;
    std::vector<size_t> sample_shape;
    size_t patch = 12;  // image: patch side in pixels
    size_t step = 4;    // image: between patch positions
    // The value a covered element takes: per channel (image), one value
    // (tokens: the pad id), per feature (features).
    std::vector<float> fill;
};

// The pieces in order: image patches row by row (positions 0, step, ... up to
// the last that fits, and one flush with the far edge), every token that is
// not padding (`token_ids` row-major, `pad_id`), every feature.
std::vector<Piece> PiecesOf(const Plan& plan, const std::vector<float>& token_ids = {}, float pad_id = 0.0f);
// The grid of patch positions (rows x columns) for an image plan.
std::pair<size_t, size_t> PatchGrid(const Plan& plan);

// The picked output's score for one item's output values: a class's
// probability through the rule (softmax, sigmoid, ...), else the output value.
double TargetScore(const std::vector<float>& item_output, layer_view::OutputRule rule, int target);

// The sample with each piece covered, one item per piece, in the model's
// input layout (built on the device).
Tensor CoveredBatch(const Tensor& sample, const Plan& plan, const std::vector<Piece>& pieces);

struct Result {
    double baseline = 0.0;        // the target's score on the unchanged sample
    std::vector<double> change;   // per piece: covered score - baseline
    bool cancelled = false;
};
// Runs the baseline and every piece, `chunk` items a forward. `progress`
// gets (pieces done, pieces); `cancelled` is polled between forwards.
Result Run(DebugModelSession& session, const Tensor& sample, const Plan& plan, const std::vector<Piece>& pieces,
           layer_view::OutputRule rule, int target, size_t chunk,
           const std::function<void(size_t, size_t)>& progress = {}, const std::function<bool()>& cancelled = {});

}  // namespace cyxwiz::occlusion
