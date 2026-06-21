#include "cyxwiz/optimizer.h"
#include "cyxwiz/tensor.h"
#include "optimizers/optimizer_utils.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstring>
#include <algorithm>
#include <spdlog/spdlog.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

// ============================================================================
// Factory
// ============================================================================

std::unique_ptr<Optimizer> CreateOptimizer(OptimizerType type, double learning_rate) {
    switch (type) {
        case OptimizerType::SGD:
            return std::make_unique<SGDOptimizer>(learning_rate);
        case OptimizerType::Adam:
            return std::make_unique<AdamOptimizer>(learning_rate);
        case OptimizerType::AdamW:
            return std::make_unique<AdamWOptimizer>(learning_rate);
        case OptimizerType::RMSprop:
            return std::make_unique<RMSpropOptimizer>(learning_rate);
        case OptimizerType::AdaGrad:
            return std::make_unique<AdaGradOptimizer>(learning_rate);
        case OptimizerType::NAdam:
            return std::make_unique<NAdamOptimizer>(learning_rate);
        case OptimizerType::Adadelta:
            // Adadelta doesn't use learning_rate, but we accept it for consistency
            return std::make_unique<AdadeltaOptimizer>();
        case OptimizerType::LAMB:
            return std::make_unique<LAMBOptimizer>(learning_rate);
        default:
            return nullptr;
    }
}

} // namespace cyxwiz






