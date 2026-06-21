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
    explicit BCEWithLogitsLoss(Reduction reduction = Reduction::Mean)
        : Loss(reduction) {}

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "BCEWithLogits"; }
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
