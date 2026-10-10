#include "cyxwiz/optimizers/gradient_clipping.h"
#include "../arrayfire_backend_utils.h"
#include "../arrayfire_host_materialization.h"

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

#include <cmath>
#include <stdexcept>
#include <string>

namespace cyxwiz {

float ClipGradientsByGlobalNorm(std::map<std::string, Tensor>& gradients, float max_norm) {
    if (!std::isfinite(max_norm) || max_norm <= 0.0f) {
        throw std::invalid_argument("Gradient clipping max_norm must be finite and positive");
    }
    if (gradients.empty()) return 0.0f;
    double squared = 0.0;
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array total = af::constant(0.0f, 1);
        for (const auto& [name, grad] : gradients) {
            const af::array g = af::flat(grad.GetSemanticArray());
            total = total + af::sum(g * g);
        }
        float host = 0.0f;
        MaterializeArrayFireToHost(total, &host, ArrayFireHostSyncCategory::OutputMaterialization,
                                   "ClipGradientsByGlobalNorm::Norm", "scalar", "gradient_norm_readback");
        squared = host;
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("ClipGradientsByGlobalNorm failed on the ArrayFire device: ") +
                                 e.what());
    }
#else
    throw std::runtime_error("Gradient clipping runs on ArrayFire, and this build has no ArrayFire");
#endif
    const float norm = static_cast<float>(std::sqrt(squared));
    if (!std::isfinite(norm)) {
        throw std::runtime_error("Gradient norm is not finite; the step cannot be clipped safely");
    }
    if (norm > max_norm) {
        const float scale = max_norm / (norm + 1e-6f);
        for (auto& [name, grad] : gradients) grad = grad * scale;
    }
    return norm;
}

} // namespace cyxwiz
