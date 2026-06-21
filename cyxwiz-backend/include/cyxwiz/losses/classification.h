#pragma once

#include "cyxwiz/losses/loss_base.h"

namespace cyxwiz {

class CYXWIZ_API CrossEntropyLoss : public Loss {
public:
    explicit CrossEntropyLoss(Reduction reduction = Reduction::Mean, int ignore_index = -100)
        : Loss(reduction), ignore_index_(ignore_index) {}

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "CrossEntropy"; }

private:
    int ignore_index_;
    Tensor cached_softmax_;
};

class CYXWIZ_API NLLLoss : public Loss {
public:
    explicit NLLLoss(Reduction reduction = Reduction::Mean, int ignore_index = -100)
        : Loss(reduction), ignore_index_(ignore_index) {}

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "NLL"; }

private:
    int ignore_index_;
};

class CYXWIZ_API FocalLoss : public Loss {
public:
    explicit FocalLoss(float alpha = 0.25f, float gamma = 2.0f,
                       Reduction reduction = Reduction::Mean)
        : Loss(reduction), alpha_(alpha), gamma_(gamma) {}

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "Focal"; }

    float GetAlpha() const { return alpha_; }
    float GetGamma() const { return gamma_; }
    void SetAlpha(float alpha) { alpha_ = alpha; }
    void SetGamma(float gamma) { gamma_ = gamma; }

private:
    float alpha_;
    float gamma_;
    Tensor cached_probs_;
};

} // namespace cyxwiz
