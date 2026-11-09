#pragma once

// "Where it looks" by saliency (TOFIX178 6.3c, approved mockup board
// Saliency): the gradient of the picked output's score with respect to the
// input, from one forward and one backward pass on the device. Token ids
// have no gradient, so for tokens it is taken at the Embedding's output
// (DebugModelSession::LayerOutputGradient, TOFIX204), one vector per token.
//
// The sample is in the model's input layout with one item, as occlusion's:
// an image [H, W, C, 1] (batch last), tokens or features [1, T] (batch first).

#include "debug_model_session.h"
#include "layer_view_presentation.h"
#include "occlusion.h"

#include <cyxwiz/tensor.h>

#include <vector>

namespace cyxwiz::saliency {

// The output weighting whose gradient is taken: 1 at the target's logit
// (the score before a softmax or sigmoid, which does not saturate), or at
// its value. A single sigmoid or probability output's class 0 is -1: its
// score falls as the output rises.
std::vector<float> TargetWeights(size_t outputs, layer_view::OutputRule rule, int target);

struct Result {
    // One value per pixel (row by row), token or feature.
    // size: image, the largest |gradient| over the channels; tokens, the
    // gradient's length; features, |gradient|.
    std::vector<double> size;
    // times_input: image, gradient x value summed over the channels; tokens,
    // gradient . embedding; features, gradient x value.
    std::vector<double> times_input;
    size_t rows = 0;     // image
    size_t columns = 0;  // image
};

// The per-element values from a gradient and what it is the gradient of:
// the sample (image [H, W, C, 1], features [1, F]) or, for tokens, the
// Embedding's output [1, T, D]. Shapes must match. On the device; the host
// gets one value per pixel, token or feature.
Result Reduce(occlusion::InputKind kind, const Tensor& gradient, const Tensor& values);

// One forward and one backward pass. `embedding_layer` is the module index of
// the Embedding the tokens enter (tokens only).
Result Run(DebugModelSession& session, const Tensor& sample, occlusion::InputKind kind, layer_view::OutputRule rule,
           int target, size_t embedding_layer = 0);

}  // namespace cyxwiz::saliency
