# Loss Functions API Reference

Loss function implementations in cyxwiz-backend for training neural networks with GPU acceleration.

## Base Loss Class

```cpp
namespace cyxwiz {

class CYXWIZ_API Loss {
public:
    Loss(ReductionType reduction = ReductionType::Mean);
    virtual ~Loss() = default;

    // Compute loss
    virtual Tensor Forward(const Tensor& predictions,
                           const Tensor& targets) = 0;

    // Compute loss and gradients
    virtual Tensor Forward(const Tensor& predictions,
                           const Tensor& targets,
                           Tensor& grad_output);

    // Call operator
    Tensor operator()(const Tensor& predictions, const Tensor& targets);

protected:
    ReductionType reduction_;
};

enum class ReductionType {
    None,   // Return per-element loss
    Mean,   // Average over all elements
    Sum     // Sum all elements
};

} // namespace cyxwiz
```

## Classification Losses

### CrossEntropyLoss

For multi-class classification with class indices as targets.

```cpp
class CYXWIZ_API CrossEntropyLoss : public Loss {
public:
    CrossEntropyLoss(ReductionType reduction = ReductionType::Mean,
                     const Tensor& weight = Tensor(),  // Class weights
                     int ignore_index = -100,
                     float label_smoothing = 0.0f);

    Tensor Forward(const Tensor& predictions,
                   const Tensor& targets) override;

    // predictions: (N, C) logits (NOT softmax)
    // targets: (N,) integer class indices
    // output: scalar (with Mean reduction)

private:
    Tensor weight_;
    int ignore_index_;
    float label_smoothing_;
};
```

### Algorithm

```
# With label smoothing
smoothed_targets = (1 - label_smoothing) * one_hot(targets) + label_smoothing / num_classes

# Cross entropy
loss = -sum(smoothed_targets * log_softmax(predictions), dim=-1)

# Apply class weights if provided
if weight:
    loss = loss * weight[targets]

# Reduce
if reduction == Mean:
    return mean(loss)
elif reduction == Sum:
    return sum(loss)
else:
    return loss
```

### Usage

```cpp
#include <cyxwiz/loss.h>

using namespace cyxwiz;

// Basic cross entropy
CrossEntropyLoss ce_loss;

Tensor logits = Randn({32, 10});  // Batch of 32, 10 classes
Tensor targets = ...;             // Integer tensor (32,) with values 0-9

Tensor loss = ce_loss(logits, targets);

// With class weights for imbalanced data
Tensor weights({0.5f, 0.5f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.5f, 1.5f, 1.0f}, {10});
CrossEntropyLoss weighted_ce(ReductionType::Mean, weights);

// With label smoothing
CrossEntropyLoss smooth_ce(ReductionType::Mean, Tensor(), -100, 0.1f);
```

### BCEWithLogitsLoss

Binary cross entropy with sigmoid applied internally (numerically stable).

```cpp
class CYXWIZ_API BCEWithLogitsLoss : public Loss {
public:
    BCEWithLogitsLoss(ReductionType reduction = ReductionType::Mean,
                      const Tensor& weight = Tensor(),
                      const Tensor& pos_weight = Tensor());

    Tensor Forward(const Tensor& predictions,
                   const Tensor& targets) override;

    // predictions: (N, *) logits
    // targets: (N, *) float 0 or 1
    // output: scalar (with Mean reduction)

private:
    Tensor weight_;
    Tensor pos_weight_;
};
```

### Algorithm

```
# Numerically stable sigmoid cross entropy
max_val = clamp(-predictions, min=0)
loss = max_val + log(exp(-max_val) + exp(-predictions - max_val))
       - predictions * targets

# Apply pos_weight for class imbalance
if pos_weight:
    loss = loss * (1 + (pos_weight - 1) * targets)
```

### Usage

```cpp
// Basic binary cross entropy
BCEWithLogitsLoss bce_loss;

Tensor logits = Randn({32, 1});    // Binary classification
Tensor targets = ...;              // Float tensor (32, 1) with 0.0 or 1.0

Tensor loss = bce_loss(logits, targets);

// For multi-label classification
BCEWithLogitsLoss multilabel_loss;
Tensor multi_logits = Randn({32, 20});  // 20 labels
Tensor multi_targets = ...;              // (32, 20) with 0.0/1.0

// With positive class weight for imbalanced binary
Tensor pos_weight = Full({1}, 2.0f);  // Weight positive class 2x
BCEWithLogitsLoss balanced_bce(ReductionType::Mean, Tensor(), pos_weight);
```

### FocalLoss

For handling class imbalance by down-weighting easy examples.

```cpp
class CYXWIZ_API FocalLoss : public Loss {
public:
    FocalLoss(float alpha = 0.25f,
              float gamma = 2.0f,
              ReductionType reduction = ReductionType::Mean);

    Tensor Forward(const Tensor& predictions,
                   const Tensor& targets) override;

private:
    float alpha_;
    float gamma_;
};
```

### Algorithm

```
p = sigmoid(predictions)
ce_loss = BCE(predictions, targets)
p_t = p * targets + (1 - p) * (1 - targets)
focal_weight = alpha * (1 - p_t)^gamma
loss = focal_weight * ce_loss
```

### Usage

```cpp
// Object detection with class imbalance
FocalLoss focal(0.25f, 2.0f);

Tensor predictions = ...;
Tensor targets = ...;
Tensor loss = focal(predictions, targets);
```

## Regression Losses

### MSELoss (Mean Squared Error)

```cpp
class CYXWIZ_API MSELoss : public Loss {
public:
    MSELoss(ReductionType reduction = ReductionType::Mean);

    Tensor Forward(const Tensor& predictions,
                   const Tensor& targets) override;

    // loss = (predictions - targets)^2
};
```

### Usage

```cpp
MSELoss mse_loss;

Tensor predictions = Randn({32, 1});
Tensor targets = Randn({32, 1});

Tensor loss = mse_loss(predictions, targets);
```

### L1Loss (Mean Absolute Error)

```cpp
class CYXWIZ_API L1Loss : public Loss {
public:
    L1Loss(ReductionType reduction = ReductionType::Mean);

    Tensor Forward(const Tensor& predictions,
                   const Tensor& targets) override;

    // loss = |predictions - targets|
};
```

### SmoothL1Loss (Huber Loss)

```cpp
class CYXWIZ_API SmoothL1Loss : public Loss {
public:
    SmoothL1Loss(float beta = 1.0f,
                 ReductionType reduction = ReductionType::Mean);

    Tensor Forward(const Tensor& predictions,
                   const Tensor& targets) override;

    // loss = 0.5 * x^2 / beta    if |x| < beta
    //      = |x| - 0.5 * beta    otherwise
    // where x = predictions - targets

private:
    float beta_;
};
```

### Usage

```cpp
// More robust to outliers than MSE
SmoothL1Loss huber_loss(1.0f);

Tensor predictions = ...;
Tensor targets = ...;
Tensor loss = huber_loss(predictions, targets);
```

## Contrastive Losses

### TripletMarginLoss

```cpp
class CYXWIZ_API TripletMarginLoss : public Loss {
public:
    TripletMarginLoss(float margin = 1.0f,
                      float p = 2.0f,  // Norm degree
                      bool swap = false,
                      ReductionType reduction = ReductionType::Mean);

    Tensor Forward(const Tensor& anchor,
                   const Tensor& positive,
                   const Tensor& negative);

    // loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)

private:
    float margin_;
    float p_;
    bool swap_;
};
```

### Usage

```cpp
TripletMarginLoss triplet_loss(0.5f);

Tensor anchor = ...;    // (N, embedding_dim)
Tensor positive = ...;  // (N, embedding_dim)
Tensor negative = ...;  // (N, embedding_dim)

Tensor loss = triplet_loss.Forward(anchor, positive, negative);
```

### CosineSimilarityLoss

```cpp
class CYXWIZ_API CosineSimilarityLoss : public Loss {
public:
    CosineSimilarityLoss(float margin = 0.0f,
                         ReductionType reduction = ReductionType::Mean);

    Tensor Forward(const Tensor& x1,
                   const Tensor& x2,
                   const Tensor& labels);  // 1 for similar, -1 for dissimilar

    // loss = 1 - cos_sim(x1, x2)       if label == 1
    //      = max(0, cos_sim - margin)  if label == -1
};
```

### ContrastiveLoss

For learning embeddings from pairs of samples (similarity learning).

```cpp
class CYXWIZ_API ContrastiveLoss : public Loss {
public:
    ContrastiveLoss(float margin = 1.0f,
                    ReductionType reduction = ReductionType::Mean);

    Tensor Forward(const Tensor& x1, const Tensor& x2) override;

    void SetLabels(const Tensor& labels);  // 0=similar, 1=dissimilar

private:
    float margin_;
    Tensor labels_;
};
```

### Algorithm

```
# For each pair (x1, x2) with label y:
distance = ||x1 - x2||²

if y == 0:  # Similar pair
    loss = 0.5 * distance²
else:       # Dissimilar pair
    loss = 0.5 * max(margin - distance, 0)²
```

### Usage

```cpp
ContrastiveLoss contrastive_loss(1.0f);

Tensor x1 = ...;      // (N, embedding_dim)
Tensor x2 = ...;      // (N, embedding_dim)
Tensor labels = ...;  // (N,) with 0=similar, 1=dissimilar

contrastive_loss.SetLabels(labels);
Tensor loss = contrastive_loss(x1, x2);
```

### InfoNCELoss (Contrastive Learning)

```cpp
class CYXWIZ_API InfoNCELoss : public Loss {
public:
    InfoNCELoss(float temperature = 0.07f,
                ReductionType reduction = ReductionType::Mean);

    Tensor Forward(const Tensor& query,
                   const Tensor& positive_key,
                   const Tensor& negative_keys = Tensor());

    // SimCLR-style contrastive loss
};
```

## Segmentation Losses

### DiceLoss

```cpp
class CYXWIZ_API DiceLoss : public Loss {
public:
    DiceLoss(float smooth = 1.0f,
             ReductionType reduction = ReductionType::Mean);

    Tensor Forward(const Tensor& predictions,
                   const Tensor& targets) override;

    // dice = 2 * |P intersection T| / (|P| + |T|)
    // loss = 1 - dice

private:
    float smooth_;
};
```

### Usage

```cpp
// Medical image segmentation
DiceLoss dice_loss;

Tensor predictions = ...;  // (N, C, H, W) after softmax
Tensor targets = ...;      // (N, C, H, W) one-hot encoded

Tensor loss = dice_loss(predictions, targets);
```

### CombinedLoss

```cpp
class CYXWIZ_API CombinedLoss : public Loss {
public:
    CombinedLoss(std::vector<std::pair<std::unique_ptr<Loss>, float>> losses);

    Tensor Forward(const Tensor& predictions,
                   const Tensor& targets) override;

    // loss = sum(weight_i * loss_i)
};
```

### Usage

```cpp
// Combine Dice and Cross Entropy for segmentation
std::vector<std::pair<std::unique_ptr<Loss>, float>> losses;
losses.push_back({std::make_unique<DiceLoss>(), 0.5f});
losses.push_back({std::make_unique<CrossEntropyLoss>(), 0.5f});

CombinedLoss combined(std::move(losses));
```

## GAN Losses

### GANLoss

```cpp
class CYXWIZ_API GANLoss : public Loss {
public:
    enum class Mode {
        Vanilla,    // BCE
        LSGAN,      // MSE
        WGAN,       // Wasserstein
        WGAN_GP     // Wasserstein with gradient penalty
    };

    GANLoss(Mode mode = Mode::Vanilla);

    Tensor DiscriminatorLoss(const Tensor& real_pred,
                             const Tensor& fake_pred);

    Tensor GeneratorLoss(const Tensor& fake_pred);

    // For WGAN-GP
    Tensor GradientPenalty(const Tensor& real_data,
                           const Tensor& fake_data,
                           const Tensor& discriminator_output);

private:
    Mode mode_;
};
```

## Loss Factory

```cpp
namespace cyxwiz {

enum class LossType {
    CrossEntropy,
    BCEWithLogits,
    MSE,
    L1,
    SmoothL1,
    Focal,
    Triplet,
    Contrastive,
    Dice,
    InfoNCE
};

struct LossConfig {
    LossType type = LossType::CrossEntropy;
    ReductionType reduction = ReductionType::Mean;
    float label_smoothing = 0.0f;
    float focal_gamma = 2.0f;
    float focal_alpha = 0.25f;
    float margin = 1.0f;
    float temperature = 0.07f;
};

std::unique_ptr<Loss> CreateLoss(const LossConfig& config);

}
```

### Usage

```cpp
LossConfig config;
config.type = LossType::CrossEntropy;
config.label_smoothing = 0.1f;

auto loss = CreateLoss(config);
```

## Python Bindings

```python
import pycyxwiz as cyx

# Classification losses
ce_loss = cyx.nn.CrossEntropyLoss()
bce_loss = cyx.nn.BCEWithLogitsLoss()
focal_loss = cyx.nn.FocalLoss(alpha=0.25, gamma=2.0)

# Regression losses
mse_loss = cyx.nn.MSELoss()
l1_loss = cyx.nn.L1Loss()
huber_loss = cyx.nn.SmoothL1Loss(beta=1.0)

# Segmentation
dice_loss = cyx.nn.DiceLoss()

# Contrastive
triplet_loss = cyx.nn.TripletMarginLoss(margin=0.5)
infonce_loss = cyx.nn.InfoNCELoss(temperature=0.07)

# Usage
loss = ce_loss(predictions, targets)
loss.backward()
```

## Loss Selection Guide

| Task | Recommended Loss | Notes |
|------|------------------|-------|
| **Multi-class Classification** | CrossEntropyLoss | Use label smoothing for regularization |
| **Binary Classification** | BCEWithLogitsLoss | Use pos_weight for imbalance |
| **Multi-label** | BCEWithLogitsLoss | Per-label independent prediction |
| **Object Detection** | FocalLoss | Handles foreground/background imbalance |
| **Regression** | MSELoss or SmoothL1Loss | SmoothL1 more robust to outliers |
| **Semantic Segmentation** | CrossEntropy + DiceLoss | Combined usually works best |
| **Metric Learning** | TripletMarginLoss | Or InfoNCE for contrastive |
| **GANs** | GANLoss | Choose mode based on architecture |

## Best Practices

1. **Label Smoothing**: Use 0.1 for classification to prevent overconfidence
2. **Class Weights**: Inverse frequency for imbalanced datasets
3. **Focal Loss**: gamma=2.0 is good default for detection
4. **Combined Losses**: Weight by relative magnitude
5. **Gradient Clipping**: Consider when using unstable losses

---

**Next**: [Model API](model.md) | [Tensor API](tensor.md)
