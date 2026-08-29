#pragma once

#include "cyxwiz/losses/loss_base.h"

#include <utility>
#include <vector>

namespace cyxwiz {

class CYXWIZ_API CrossEntropyLoss : public Loss {
public:
    explicit CrossEntropyLoss(Reduction reduction = Reduction::Mean,
                              int ignore_index = -100);

    explicit CrossEntropyLoss(Reduction reduction,
                              int ignore_index,
                              std::vector<float> class_weights);

    CrossEntropyLoss(Reduction reduction,
                     int ignore_index,
                     std::vector<float> class_weights,
                     float label_smoothing);

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "CrossEntropy"; }
    const Tensor* GetLastMeanReductionDenominator() const override {
        return has_cached_mean_denominator_
            ? &cached_mean_denominator_
            : nullptr;
    }
    const std::vector<float>& GetClassWeights() const { return class_weights_; }
    float GetLabelSmoothing() const { return label_smoothing_; }

private:
    int ignore_index_;
    std::vector<float> class_weights_;
    float label_smoothing_;
    Tensor cached_softmax_;
    Tensor cached_class_weights_;
    Tensor cached_mean_denominator_;
    bool has_cached_mean_denominator_ = false;
};

class CYXWIZ_API NLLLoss : public Loss {
public:
    explicit NLLLoss(Reduction reduction = Reduction::Mean,
                     int ignore_index = -100);

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "NLL"; }

private:
    int ignore_index_;
};

class CYXWIZ_API FocalLoss : public Loss {
public:
    explicit FocalLoss(float alpha = 0.25f, float gamma = 2.0f,
                       Reduction reduction = Reduction::Mean);

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "Focal"; }

    float GetAlpha() const { return alpha_; }
    float GetGamma() const { return gamma_; }
    void SetAlpha(float alpha);
    void SetGamma(float gamma);

private:
    float alpha_;
    float gamma_;
};

} // namespace cyxwiz
