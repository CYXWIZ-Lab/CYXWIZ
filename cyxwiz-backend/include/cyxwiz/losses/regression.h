#pragma once

#include "cyxwiz/losses/loss_base.h"

namespace cyxwiz {

class CYXWIZ_API MSELoss : public Loss {
public:
    explicit MSELoss(Reduction reduction = Reduction::Mean) : Loss(reduction) {}

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "MSE"; }
};

class CYXWIZ_API L1Loss : public Loss {
public:
    explicit L1Loss(Reduction reduction = Reduction::Mean) : Loss(reduction) {}

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "L1"; }
};

class CYXWIZ_API SmoothL1Loss : public Loss {
public:
    explicit SmoothL1Loss(float delta = 1.0f, Reduction reduction = Reduction::Mean)
        : Loss(reduction), delta_(delta) {}

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "SmoothL1"; }

    float GetDelta() const { return delta_; }

private:
    float delta_;
};

using HuberLoss = SmoothL1Loss;

} // namespace cyxwiz
