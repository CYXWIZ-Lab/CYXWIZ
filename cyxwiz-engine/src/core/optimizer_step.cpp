#include "optimizer_step.h"

#include <cyxwiz/optimizers/gradient_clipping.h>

namespace cyxwiz {

float ApplyOptimizerStep(IExecutableModel& model, Optimizer& optimizer, std::map<std::string, Tensor>& gradients,
                         const OptimizerStepSettings& settings) {
    auto params = model.GetParameters();
    // Regularization node (TOFIX140 A4): the gradient of
    // l1 x sum(|w|) + l2 x sum(w^2), once per step and before clipping, as
    // torch's backward of loss + penalty gives it.
    const float l1 = settings.regularization_l1;
    const float l2 = settings.regularization_l2;
    if (l1 > 0.0f || l2 > 0.0f) {
        for (auto& [name, grad] : gradients) {
            const auto found = params.find(name);
            if (found == params.end()) continue;
            const Tensor weight = found->second.Shape() == grad.Shape()
                ? found->second
                : found->second.Reshape(grad.Shape());
            if (l1 > 0.0f) grad = grad + weight.Sign() * l1;
            if (l2 > 0.0f) grad = grad + weight * (2.0f * l2);
        }
    }

    if (settings.on_gradients) settings.on_gradients(gradients);

    float norm = -1.0f;
    if (settings.grad_clip_norm > 0.0f) norm = ClipGradientsByGlobalNorm(gradients, settings.grad_clip_norm);

    optimizer.Step(params, gradients);
    model.SetParameters(params);
    return norm;
}

}  // namespace cyxwiz
