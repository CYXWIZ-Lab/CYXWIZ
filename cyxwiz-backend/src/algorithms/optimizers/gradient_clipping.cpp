#include "cyxwiz/optimizers/gradient_clipping.h"
#include "../arrayfire_backend_utils.h"
#include "../arrayfire_host_materialization.h"

#include <arrayfire.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace cyxwiz {

namespace {
// The sum of every gradient's squared elements, as a [1] array on the device
af::array SumOfSquares(const std::map<std::string, Tensor>& gradients) {
    af::array total = af::constant(0.0f, 1);
    for (const auto& [name, grad] : gradients) {
        const af::array g = af::flat(grad.GetSemanticArray());
        total = total + af::sum(g * g);
    }
    return total;
}
}  // namespace

float ClipGradientsByGlobalNorm(std::map<std::string, Tensor>& gradients, float max_norm) {
    if (!std::isfinite(max_norm) || max_norm <= 0.0f) {
        throw std::invalid_argument("Gradient clipping max_norm must be finite and positive");
    }
    if (gradients.empty()) return 0.0f;
    double squared = 0.0;
    try {
        const af::array total = SumOfSquares(gradients);
        float host = 0.0f;
        MaterializeArrayFireToHost(total, &host, ArrayFireHostSyncCategory::OutputMaterialization,
                                   "ClipGradientsByGlobalNorm::Norm", "scalar", "gradient_norm_readback");
        squared = host;
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("ClipGradientsByGlobalNorm failed on the ArrayFire device: ") +
                                 e.what());
    }
    const float norm = static_cast<float>(std::sqrt(squared));
    // As torch: coef clamped at 1, a NaN kept (so a NaN norm scales by NaN)
    const float coef = max_norm / (norm + 1e-6f);
    if (!(coef >= 1.0f))
        for (auto& [name, grad] : gradients) grad = grad * coef;
    return norm;
}

Tensor ClipGradientsByGlobalNormOnDevice(std::map<std::string, Tensor>& gradients, float max_norm) {
    if (!std::isfinite(max_norm) || max_norm <= 0.0f) {
        throw std::invalid_argument("Gradient clipping max_norm must be finite and positive");
    }
    try {
        af::array norm = af::sqrt(SumOfSquares(gradients));
        norm.eval();
        // As torch: coef clamped at 1, a NaN kept (af::min would turn NaN into 1)
        const af::array coef = max_norm / (norm + 1e-6f);
        af::array scale = af::select(coef >= 1.0f, 1.0, coef);
        scale.eval();
        // Left unevaluated: the optimizer's step evaluates the scaled
        // gradient in its own kernel
        for (auto& [name, grad] : gradients) {
            const af::array g = grad.GetSemanticArray();
            grad = Tensor::FromSemanticArray(g * af::tile(scale, g.dims()), grad.Shape());
        }
        return Tensor::FromSemanticArray(norm, {1});
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("ClipGradientsByGlobalNormOnDevice failed on the ArrayFire device: ") +
                                 e.what());
    }
}

} // namespace cyxwiz
