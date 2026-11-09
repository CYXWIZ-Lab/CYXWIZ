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
    // nullptr with `error` set when the model cannot be built, is not the
    // run's (a captured layer is another node in it), or the capture lacks a
    // layer's weights (capture "What" without parameters, or tensors past the
    // capture budget).
    static std::unique_ptr<DebugModelSession> FromCapture(const TrainingConfiguration& config,
                                                          const LayerCapture& capture, std::string* error);

    // The model's output for `input`, in the model's input layout (the
    // capture's first layer input; any number of samples), in eval mode.
    Tensor Forward(const Tensor& input);

    // d(sum of output x target_weights) / d input, in eval mode.
    // `target_weights` has the output's shape: a one-hot class, a token's
    // logit, a weighting of several outputs. Throws when the shapes differ.
    Tensor InputGradient(const Tensor& input, const Tensor& target_weights);

    // One layer's output for `input` and d(sum of output x target_weights)
    // / d(that output), in eval mode (TOFIX204): what an input gradient
    // cannot give, such as a token's gradient at the Embedding output.
    // `module_index` is the layer's index in the model
    // (BuiltModuleProvenance::module_index, CapturedLayer::module_index).
    // Throws when the shapes differ or the layer is not on the path to the
    // output.
    struct LayerGradient {
        Tensor output;
        Tensor gradient;
    };
    LayerGradient LayerOutputGradient(const Tensor& input, const Tensor& target_weights, size_t module_index);

private:
    explicit DebugModelSession(std::unique_ptr<IExecutableModel> model);

    std::unique_ptr<IExecutableModel> model_;
};

}  // namespace cyxwiz
