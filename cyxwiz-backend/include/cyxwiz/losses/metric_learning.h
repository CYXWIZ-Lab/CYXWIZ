#pragma once

#include "cyxwiz/losses/loss_base.h"

namespace cyxwiz {

class CYXWIZ_API CosineEmbeddingLoss : public Loss {
public:
    explicit CosineEmbeddingLoss(float margin = 0.0f, Reduction reduction = Reduction::Mean);

    Tensor Forward(const Tensor& x1, const Tensor& x2) override;
    Tensor Backward(const Tensor& x1, const Tensor& x2) override;
    std::string GetName() const override { return "CosineEmbedding"; }

    void SetLabels(const Tensor& labels) { labels_ = labels; }

    float GetMargin() const { return margin_; }
    void SetMargin(float margin);

private:
    float margin_;
    Tensor labels_;
};

struct CYXWIZ_API TripletLossGradients {
    Tensor anchor;
    Tensor positive;
    Tensor negative;
};

class CYXWIZ_API TripletLoss : public Loss {
public:
    enum class DistanceType {
        Euclidean,
        Cosine
    };

    explicit TripletLoss(float margin = 1.0f,
                         DistanceType distance_type = DistanceType::Euclidean,
                         Reduction reduction = Reduction::Mean);

    Tensor Forward(const Tensor& anchor, const Tensor& positive) override;
    Tensor Backward(const Tensor& anchor, const Tensor& positive) override;
    TripletLossGradients BackwardAll(const Tensor& anchor, const Tensor& positive);
    std::string GetName() const override { return "Triplet"; }

    void SetNegative(const Tensor& negative) { negative_ = negative; }
    const Tensor& GetNegative() const { return negative_; }

    float GetMargin() const { return margin_; }
    void SetMargin(float margin);
    DistanceType GetDistanceType() const { return distance_type_; }

private:
    float margin_;
    DistanceType distance_type_;
    Tensor negative_;
};

class CYXWIZ_API ContrastiveLoss : public Loss {
public:
    explicit ContrastiveLoss(float margin = 1.0f, Reduction reduction = Reduction::Mean);

    Tensor Forward(const Tensor& x1, const Tensor& x2) override;
    Tensor Backward(const Tensor& x1, const Tensor& x2) override;
    std::string GetName() const override { return "Contrastive"; }

    void SetLabels(const Tensor& labels) { labels_ = labels; }
    const Tensor& GetLabels() const { return labels_; }

    float GetMargin() const { return margin_; }
    void SetMargin(float margin);

private:
    float margin_;
    Tensor labels_;
};

} // namespace cyxwiz
