#pragma once

// A debug run's model, kept so it can be asked again (TOFIX202, for the
// Studio Debugger's TOFIX178 6.3 views: occlusion, saliency, top images).
// Rebuilt from the run's frozen configuration with the weights its layer
// capture held, which are the weights the run's forward used (captured
// before the optimizer step). The same path serves a live Local Debug run,
// a saved run and a training snapshot. Runs in eval mode on the device.

#include "executable_model.h"
#include "graph_compiler.h"
#include "layer_capture.h"

#include <cyxwiz/tensor.h>

#include <memory>
#include <string>

namespace cyxwiz {

class DebugModelSession {
public:
    // nullptr with `error` set when the model cannot be built or the capture
    // lacks a layer's weights (capture "What" without parameters, or tensors
    // past the capture budget).
    static std::unique_ptr<DebugModelSession> FromCapture(const TrainingConfiguration& config,
                                                          const LayerCapture& capture, std::string* error);

    // The model's output for `input`, in the model's input layout (the
    // capture's first layer input; any number of samples), in eval mode.
    Tensor Forward(const Tensor& input);

    // d(sum of output x target_weights) / d input, in eval mode.
    // `target_weights` has the output's shape: a one-hot class, a token's
    // logit, a weighting of several outputs. Throws when the shapes differ.
    Tensor InputGradient(const Tensor& input, const Tensor& target_weights);

private:
    explicit DebugModelSession(std::unique_ptr<IExecutableModel> model);

    std::unique_ptr<IExecutableModel> model_;
};

}  // namespace cyxwiz
