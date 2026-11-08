#pragma once

// One optimizer update of a model from its finished gradients, shared by
// the supervised trainer and the reinforcement learning trainer (TOFIX195
// S4.0): the L1/L2 penalty's gradient, clipping by global norm, the
// optimizer's step and the new parameters back into the model, in the order
// torch gives (backward of loss + penalty, clip_grad_norm_, step).

#include "executable_model.h"

#include <cyxwiz/optimizers/optimizer_base.h>
#include <cyxwiz/tensor.h>

#include <map>
#include <string>

namespace cyxwiz {

struct OptimizerStepSettings {
    float regularization_l1 = 0.0f;
    float regularization_l2 = 0.0f;
    float grad_clip_norm = 0.0f;  // 0: no clipping
};

// gradients: averaged over the batch, by parameter name (changed in place by
// the penalty and the clip). Returns the global norm before clipping, or a
// negative value when clipping is off.
float ApplyOptimizerStep(IExecutableModel& model, Optimizer& optimizer, std::map<std::string, Tensor>& gradients,
                         const OptimizerStepSettings& settings);

}  // namespace cyxwiz
