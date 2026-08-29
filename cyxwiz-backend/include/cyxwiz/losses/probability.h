#pragma once

#include "cyxwiz/losses/loss_base.h"

namespace cyxwiz {

class CYXWIZ_API BCELoss : public Loss {
public:
    explicit BCELoss(Reduction reduction = Reduction::Mean,
                     float denominator_epsilon = 1e-12f);

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "BCE"; }

private:
    float denominator_epsilon_;
};

class CYXWIZ_API BCEWithLogitsLoss : public Loss {
public:
    explicit BCEWithLogitsLoss(Reduction reduction = Reduction::Mean,
                               float pos_weight = 1.0f);

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "BCEWithLogits"; }

    float GetPosWeight() const { return pos_weight_; }
    void SetPosWeight(float pos_weight);

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

class CYXWIZ_API SoftDiceLoss : public Loss {
public:
    explicit SoftDiceLoss(Reduction reduction = Reduction::Mean,
                          float smooth = 1.0f)
        : Loss(reduction), smooth_(smooth) {}

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "SoftDice"; }

    float GetSmooth() const { return smooth_; }

private:
    float smooth_;
};

class CYXWIZ_API TverskyLoss : public Loss {
public:
    explicit TverskyLoss(Reduction reduction = Reduction::Mean,
                         float alpha = 0.5f,
                         float beta = 0.5f,
                         float smooth = 1.0f);

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "Tversky"; }

    float GetAlpha() const { return alpha_; }
    float GetBeta() const { return beta_; }
    float GetSmooth() const { return smooth_; }

private:
    float alpha_;
    float beta_;
    float smooth_;
};

class CYXWIZ_API JaccardLoss : public Loss {
public:
    explicit JaccardLoss(Reduction reduction = Reduction::Mean,
                         float smooth = 1.0f)
        : Loss(reduction), smooth_(smooth) {}

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "Jaccard"; }

    float GetSmooth() const { return smooth_; }

private:
    float smooth_;
};

} // namespace cyxwiz
