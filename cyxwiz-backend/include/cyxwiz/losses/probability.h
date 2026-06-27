#pragma once

#include "cyxwiz/losses/loss_base.h"

namespace cyxwiz {

class CYXWIZ_API BCELoss : public Loss {
public:
    explicit BCELoss(Reduction reduction = Reduction::Mean, float eps = 1e-7f)
        : Loss(reduction), eps_(eps) {}

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "BCE"; }

private:
    float eps_;
};

class CYXWIZ_API BCEWithLogitsLoss : public Loss {
public:
    explicit BCEWithLogitsLoss(Reduction reduction = Reduction::Mean,
                               float pos_weight = 1.0f)
        : Loss(reduction), pos_weight_(pos_weight) {}

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "BCEWithLogits"; }

    float GetPosWeight() const { return pos_weight_; }
    void SetPosWeight(float pos_weight) { pos_weight_ = pos_weight; }

private:
    float pos_weight_ = 1.0f;
};

class CYXWIZ_API KLDivLoss : public Loss {
public:
    explicit KLDivLoss(Reduction reduction = Reduction::Mean, bool log_target = false)
        : Loss(reduction), log_target_(log_target) {}

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "KLDiv"; }

private:
    bool log_target_;
};

} // namespace cyxwiz
