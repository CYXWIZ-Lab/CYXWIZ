#pragma once

#include "api_export.h"
#include "tensor.h"
#include <memory>
#include <string>

namespace cyxwiz {

// ============================================================================
// Loss Types
// ============================================================================

enum class LossType {
    MSE,                    // Mean Squared Error (L2 Loss)
    CrossEntropy,           // Cross Entropy Loss (for classification)
    BinaryCrossEntropy,     // Binary Cross Entropy
    BCEWithLogits,          // BCE + Sigmoid (numerically stable)
    NLLLoss,                // Negative Log Likelihood
    L1,                     // Mean Absolute Error
    SmoothL1,               // Huber Loss
    Huber,                  // Alias for SmoothL1
    KLDivergence,           // KL Divergence
    CosineEmbedding         // Cosine Embedding Loss
};

// Reduction mode for loss computation
enum class Reduction {
    None,   // No reduction - return loss per element
    Mean,   // Average of all losses
    Sum     // Sum of all losses
};

// ============================================================================
// Base Loss Class
// ============================================================================

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
    Tensor cached_loss_;  // For logging/debugging
};

// Factory function to create loss by type
CYXWIZ_API std::unique_ptr<Loss> CreateLoss(LossType type, Reduction reduction = Reduction::Mean, float delta = 1.0f);

// ============================================================================
// MSE Loss - Mean Squared Error
// ============================================================================

class CYXWIZ_API MSELoss : public Loss {
public:
    explicit MSELoss(Reduction reduction = Reduction::Mean) : Loss(reduction) {}

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "MSE"; }
};

// ============================================================================
// L1 Loss - Mean Absolute Error
// ============================================================================

class CYXWIZ_API L1Loss : public Loss {
public:
    explicit L1Loss(Reduction reduction = Reduction::Mean) : Loss(reduction) {}

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "L1"; }
};

// ============================================================================
// Smooth L1 Loss (Huber Loss)
// ============================================================================

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

// Alias
using HuberLoss = SmoothL1Loss;

// ============================================================================
// Cross Entropy Loss (for multi-class classification)
// ============================================================================

class CYXWIZ_API CrossEntropyLoss : public Loss {
public:
    explicit CrossEntropyLoss(Reduction reduction = Reduction::Mean, int ignore_index = -100)
        : Loss(reduction), ignore_index_(ignore_index) {}

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "CrossEntropy"; }

private:
    int ignore_index_;
    Tensor cached_softmax_;  // Store softmax for backward
};

// ============================================================================
// Binary Cross Entropy Loss
// ============================================================================

class CYXWIZ_API BCELoss : public Loss {
public:
    explicit BCELoss(Reduction reduction = Reduction::Mean, float eps = 1e-7f)
        : Loss(reduction), eps_(eps) {}

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "BCE"; }

private:
    float eps_;  // For numerical stability
};

// ============================================================================
// BCE With Logits Loss (more numerically stable)
// ============================================================================

class CYXWIZ_API BCEWithLogitsLoss : public Loss {
public:
    explicit BCEWithLogitsLoss(Reduction reduction = Reduction::Mean)
        : Loss(reduction) {}

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "BCEWithLogits"; }
};

// ============================================================================
// Negative Log Likelihood Loss
// ============================================================================

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

// ============================================================================
// KL Divergence Loss
// ============================================================================

class CYXWIZ_API KLDivLoss : public Loss {
public:
    explicit KLDivLoss(Reduction reduction = Reduction::Mean, bool log_target = false)
        : Loss(reduction), log_target_(log_target) {}

    Tensor Forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override;
    std::string GetName() const override { return "KLDiv"; }

private:
    bool log_target_;  // If true, target is already in log space
};

// ============================================================================
// Cosine Embedding Loss
// ============================================================================

class CYXWIZ_API CosineEmbeddingLoss : public Loss {
public:
    explicit CosineEmbeddingLoss(float margin = 0.0f, Reduction reduction = Reduction::Mean)
        : Loss(reduction), margin_(margin) {}

    // Note: For this loss, predictions = x1, targets = x2
    // Need separate labels tensor indicating if pairs should be similar (1) or dissimilar (-1)
    Tensor Forward(const Tensor& x1, const Tensor& x2) override;
    Tensor Backward(const Tensor& x1, const Tensor& x2) override;
    std::string GetName() const override { return "CosineEmbedding"; }

    void SetLabels(const Tensor& labels) { labels_ = labels; }

private:
    float margin_;
    Tensor labels_;  // 1 for similar, -1 for dissimilar
};

} // namespace cyxwiz
