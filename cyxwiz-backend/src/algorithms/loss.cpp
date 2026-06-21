#include "cyxwiz/loss.h"
#include "cyxwiz/tensor.h"
#include "losses/loss_utils.h"
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <vector>
#include <spdlog/spdlog.h>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

// Undefine Windows macros that conflict with ArrayFire functions
// Must be AFTER all includes (Windows headers define these)
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz {

using namespace loss_detail;
// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<Loss> CreateLoss(LossType type, Reduction reduction, float delta) {
    switch (type) {
        case LossType::MSE:
            return std::make_unique<MSELoss>(reduction);
        case LossType::CrossEntropy:
            return std::make_unique<CrossEntropyLoss>(reduction);
        case LossType::BinaryCrossEntropy:
            return std::make_unique<BCELoss>(reduction);
        case LossType::BCEWithLogits:
            return std::make_unique<BCEWithLogitsLoss>(reduction);
        case LossType::NLLLoss:
            return std::make_unique<NLLLoss>(reduction);
        case LossType::L1:
            return std::make_unique<L1Loss>(reduction);
        case LossType::SmoothL1:
        case LossType::Huber:
            return std::make_unique<SmoothL1Loss>(delta, reduction);
        case LossType::KLDivergence:
            return std::make_unique<KLDivLoss>(reduction);
        case LossType::CosineEmbedding:
            return std::make_unique<CosineEmbeddingLoss>(0.0f, reduction);
        case LossType::Focal:
            return std::make_unique<FocalLoss>(0.25f, 2.0f, reduction);
        case LossType::Triplet:
            return std::make_unique<TripletLoss>(1.0f, TripletLoss::DistanceType::Euclidean, reduction);
        case LossType::Contrastive:
            return std::make_unique<ContrastiveLoss>(1.0f, reduction);
        default:
            throw std::runtime_error("Unknown loss type");
    }
}


} // namespace cyxwiz






