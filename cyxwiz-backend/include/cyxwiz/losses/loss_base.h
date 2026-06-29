#pragma once

#include "cyxwiz/api_export.h"
#include "cyxwiz/tensor.h"

#include <memory>
#include <string>

namespace cyxwiz {

enum class LossType {
    MSE,
    CrossEntropy,
    BinaryCrossEntropy,
    BCEWithLogits,
    NLLLoss,
    L1,
    SmoothL1,
    Huber,
    KLDivergence,
    CosineEmbedding,
    Focal,
    SoftDice,
    Tversky,
    Jaccard,
    Triplet,
    Contrastive
};

enum class Reduction {
    None,
    Mean,
    Sum
};

class CYXWIZ_API Loss {
public:
    explicit Loss(Reduction reduction = Reduction::Mean) : reduction_(reduction) {}
    virtual ~Loss() = default;

    virtual Tensor Forward(const Tensor& predictions, const Tensor& targets) = 0;
    virtual Tensor Backward(const Tensor& predictions, const Tensor& targets) = 0;

    virtual std::string GetName() const { return "Loss"; }

    Reduction GetReduction() const { return reduction_; }
    void SetReduction(Reduction reduction) { reduction_ = reduction; }

protected:
    Reduction reduction_;
    Tensor cached_loss_;
};

CYXWIZ_API std::unique_ptr<Loss> CreateLoss(
    LossType type,
    Reduction reduction = Reduction::Mean,
    float delta = 1.0f
);

} // namespace cyxwiz
