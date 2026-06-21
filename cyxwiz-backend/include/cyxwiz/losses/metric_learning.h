#pragma once

#include "cyxwiz/losses/loss_base.h"

namespace cyxwiz {

class CYXWIZ_API CosineEmbeddingLoss : public Loss {
public:
    explicit CosineEmbeddingLoss(float margin = 0.0f, Reduction reduction = Reduction::Mean)
        : Loss(reduction), margin_(margin) {}

    Tensor Forward(const Tensor& x1, const Tensor& x2) override;
    Tensor Backward(const Tensor& x1, const Tensor& x2) override;
    std::string GetName() const override { return "CosineEmbedding"; }

    void SetLabels(const Tensor& labels) { labels_ = labels; }

private:
    float margin_;
    Tensor labels_;
};

class CYXWIZ_API TripletLoss : public Loss {
public:
    enum class DistanceType {
        Euclidean,
        Cosine
    };

    explicit TripletLoss(float margin = 1.0f,
                         DistanceType distance_type = DistanceType::Euclidean,
                         Reduction reduction = Reduction::Mean)
        : Loss(reduction), margin_(margin), distance_type_(distance_type) {}

    Tensor Forward(const Tensor& anchor, const Tensor& positive) override;
    Tensor Backward(const Tensor& anchor, const Tensor& positive) override;
    std::string GetName() const override { return "Triplet"; }

    void SetNegative(const Tensor& negative) { negative_ = negative; }
    const Tensor& GetNegative() const { return negative_; }

    float GetMargin() const { return margin_; }
    void SetMargin(float margin) { margin_ = margin; }
    DistanceType GetDistanceType() const { return distance_type_; }

private:
    float margin_;
    DistanceType distance_type_;
    Tensor negative_;
    Tensor cached_dist_ap_;
    Tensor cached_dist_an_;
};

class CYXWIZ_API ContrastiveLoss : public Loss {
public:
    explicit ContrastiveLoss(float margin = 1.0f, Reduction reduction = Reduction::Mean)
        : Loss(reduction), margin_(margin) {}

    Tensor Forward(const Tensor& x1, const Tensor& x2) override;
    Tensor Backward(const Tensor& x1, const Tensor& x2) override;
    std::string GetName() const override { return "Contrastive"; }

    void SetLabels(const Tensor& labels) { labels_ = labels; }
    const Tensor& GetLabels() const { return labels_; }

    float GetMargin() const { return margin_; }
    void SetMargin(float margin) { margin_ = margin; }

private:
    float margin_;
    Tensor labels_;
    Tensor cached_distances_;
};

} // namespace cyxwiz
