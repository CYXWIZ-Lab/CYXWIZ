# Optimizer API Reference

Optimization algorithms for training neural networks in cyxwiz-backend, implementing gradient descent variants with GPU acceleration.

## Base Optimizer Class

```cpp
namespace cyxwiz {

class CYXWIZ_API Optimizer {
public:
    Optimizer(double learning_rate = 0.001);
    virtual ~Optimizer() = default;

    // Main optimization step
    virtual void Step(std::vector<Tensor*>& parameters,
                      std::vector<Tensor*>& gradients) = 0;

    // Zero all tracked gradients
    virtual void ZeroGrad();

    // Learning rate management
    double LearningRate() const;
    void SetLearningRate(double lr);

    // State management (for checkpointing)
    virtual std::map<std::string, Tensor> State() const;
    virtual void LoadState(const std::map<std::string, Tensor>& state);

    // Step counter
    int64_t Steps() const;
    void ResetSteps();

protected:
    double learning_rate_;
    int64_t step_count_ = 0;
};

} // namespace cyxwiz
```

## SGD (Stochastic Gradient Descent)

```cpp
class CYXWIZ_API SGD : public Optimizer {
public:
    SGD(double learning_rate = 0.01,
        double momentum = 0.0,
        double weight_decay = 0.0,
        bool nesterov = false);

    void Step(std::vector<Tensor*>& parameters,
              std::vector<Tensor*>& gradients) override;

private:
    double momentum_;
    double weight_decay_;
    bool nesterov_;
    std::vector<Tensor> velocity_;  // Momentum buffer
};
```

### Algorithm

```
v_t = momentum * v_{t-1} + grad
if nesterov:
    param = param - lr * (grad + momentum * v_t)
else:
    param = param - lr * v_t

if weight_decay > 0:
    param = param - lr * weight_decay * param
```

### Usage

```cpp
#include <cyxwiz/optimizer.h>

using namespace cyxwiz;

// Basic SGD
SGD sgd(0.01);

// SGD with momentum
SGD sgd_momentum(0.01, 0.9);

// SGD with momentum and weight decay
SGD sgd_full(0.01, 0.9, 1e-4, true);  // Nesterov momentum

// Training step
auto params = model.Parameters();
auto grads = model.Gradients();
sgd.Step(params, grads);
```

## Adam

```cpp
class CYXWIZ_API Adam : public Optimizer {
public:
    Adam(double learning_rate = 0.001,
         double beta1 = 0.9,
         double beta2 = 0.999,
         double epsilon = 1e-8,
         double weight_decay = 0.0,
         bool amsgrad = false);

    void Step(std::vector<Tensor*>& parameters,
              std::vector<Tensor*>& gradients) override;

    std::map<std::string, Tensor> State() const override;
    void LoadState(const std::map<std::string, Tensor>& state) override;

private:
    double beta1_;
    double beta2_;
    double epsilon_;
    double weight_decay_;
    bool amsgrad_;
    std::vector<Tensor> m_;      // First moment
    std::vector<Tensor> v_;      // Second moment
    std::vector<Tensor> v_max_;  // For AMSGrad
};
```

### Algorithm

```
m_t = beta1 * m_{t-1} + (1 - beta1) * grad
v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2

# Bias correction
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)

if amsgrad:
    v_max = max(v_max, v_hat)
    param = param - lr * m_hat / (sqrt(v_max) + epsilon)
else:
    param = param - lr * m_hat / (sqrt(v_hat) + epsilon)

if weight_decay > 0:  # Decoupled weight decay (AdamW style)
    param = param - lr * weight_decay * param
```

### Usage

```cpp
// Default Adam
Adam adam(0.001);

// Adam with custom betas
Adam adam_custom(0.001, 0.9, 0.98, 1e-9);

// AdamW (Adam with decoupled weight decay)
Adam adamw(0.001, 0.9, 0.999, 1e-8, 0.01);

// AMSGrad variant
Adam amsgrad(0.001, 0.9, 0.999, 1e-8, 0.0, true);
```

## AdamW

```cpp
class CYXWIZ_API AdamW : public Optimizer {
public:
    AdamW(double learning_rate = 0.001,
          double beta1 = 0.9,
          double beta2 = 0.999,
          double epsilon = 1e-8,
          double weight_decay = 0.01);

    void Step(std::vector<Tensor*>& parameters,
              std::vector<Tensor*>& gradients) override;

private:
    // Same as Adam, but with properly decoupled weight decay
};
```

### Usage

```cpp
// Recommended for transformers and large models
AdamW adamw(0.001, 0.9, 0.999, 1e-8, 0.01);
```

## RMSprop

```cpp
class CYXWIZ_API RMSprop : public Optimizer {
public:
    RMSprop(double learning_rate = 0.01,
            double alpha = 0.99,
            double epsilon = 1e-8,
            double weight_decay = 0.0,
            double momentum = 0.0,
            bool centered = false);

    void Step(std::vector<Tensor*>& parameters,
              std::vector<Tensor*>& gradients) override;

private:
    double alpha_;
    double epsilon_;
    double weight_decay_;
    double momentum_;
    bool centered_;
    std::vector<Tensor> square_avg_;
    std::vector<Tensor> grad_avg_;  // For centered
    std::vector<Tensor> momentum_buffer_;
};
```

### Algorithm

```
v_t = alpha * v_{t-1} + (1 - alpha) * grad^2

if centered:
    g_t = alpha * g_{t-1} + (1 - alpha) * grad
    v_hat = v_t - g_t^2
else:
    v_hat = v_t

if momentum > 0:
    buf = momentum * buf + grad / (sqrt(v_hat) + epsilon)
    param = param - lr * buf
else:
    param = param - lr * grad / (sqrt(v_hat) + epsilon)
```

### Usage

```cpp
// Basic RMSprop
RMSprop rmsprop(0.01);

// RMSprop with momentum
RMSprop rmsprop_mom(0.01, 0.99, 1e-8, 0.0, 0.9);

// Centered RMSprop
RMSprop rmsprop_centered(0.01, 0.99, 1e-8, 0.0, 0.0, true);
```

## Adagrad

```cpp
class CYXWIZ_API Adagrad : public Optimizer {
public:
    Adagrad(double learning_rate = 0.01,
            double epsilon = 1e-10,
            double weight_decay = 0.0);

    void Step(std::vector<Tensor*>& parameters,
              std::vector<Tensor*>& gradients) override;

private:
    double epsilon_;
    double weight_decay_;
    std::vector<Tensor> sum_;  // Sum of squared gradients
};
```

### Usage

```cpp
// Useful for sparse features
Adagrad adagrad(0.01);
```

## NAdam (Nesterov-accelerated Adam)

```cpp
class CYXWIZ_API NAdam : public Optimizer {
public:
    NAdam(double learning_rate = 0.002,
          double beta1 = 0.9,
          double beta2 = 0.999,
          double epsilon = 1e-8);

    void Step(std::vector<Tensor*>& parameters,
              std::vector<Tensor*>& gradients) override;
};
```

### Algorithm

```
m_t = beta1 * m_{t-1} + (1 - beta1) * grad
v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2

# Bias correction
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)

# Nesterov momentum term
m_nesterov = beta1 * m_hat + (1 - beta1) * grad / (1 - beta1^t)

param = param - lr * m_nesterov / (sqrt(v_hat) + epsilon)
```

### Usage

```cpp
// NAdam often converges faster than Adam
NAdam nadam(0.002);
```

## Adadelta

Adaptive learning rate method that doesn't require setting a learning rate.

```cpp
class CYXWIZ_API Adadelta : public Optimizer {
public:
    Adadelta(double rho = 0.9,
             double epsilon = 1e-6);

    void Step(std::vector<Tensor*>& parameters,
              std::vector<Tensor*>& gradients) override;

private:
    double rho_;
    double epsilon_;
    std::vector<Tensor> acc_grad_;   // E[g²]
    std::vector<Tensor> acc_delta_;  // E[Δx²]
};
```

### Algorithm

```
# Accumulate squared gradient
E[g²]_t = rho * E[g²]_{t-1} + (1 - rho) * grad²

# Compute update using ratio of RMS
delta = -sqrt(E[Δx²]_{t-1} + epsilon) / sqrt(E[g²]_t + epsilon) * grad

# Accumulate squared update
E[Δx²]_t = rho * E[Δx²]_{t-1} + (1 - rho) * delta²

param = param + delta
```

### Usage

```cpp
// No learning rate required - adapts automatically
Adadelta adadelta(0.9, 1e-6);
```

## LAMB (Layer-wise Adaptive Moments for Batch training)

Designed for large batch training (e.g., BERT pre-training with batch size 32K).

```cpp
class CYXWIZ_API LAMB : public Optimizer {
public:
    LAMB(double learning_rate = 0.001,
         double beta1 = 0.9,
         double beta2 = 0.999,
         double epsilon = 1e-6,
         double weight_decay = 0.01);

    void Step(std::vector<Tensor*>& parameters,
              std::vector<Tensor*>& gradients) override;
};
```

### Algorithm

```
# Adam moment computation
m_t = beta1 * m_{t-1} + (1 - beta1) * grad
v_t = beta2 * v_{t-1} + (1 - beta2) * grad²

m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)

# Adam update direction
adam_update = m_hat / (sqrt(v_hat) + epsilon) + weight_decay * param

# Layer-wise trust ratio
trust_ratio = ||param|| / ||adam_update||

# Scaled update
param = param - lr * trust_ratio * adam_update
```

### Usage

```cpp
// For large batch training (BERT, GPT, etc.)
LAMB lamb(0.001, 0.9, 0.999, 1e-6, 0.01);

// Can use much larger batch sizes than Adam
// Typical: batch_size = 32768 with lr = 0.00176
```

## Learning Rate Warmup

Wrapper class for gradual learning rate warmup at training start.

```cpp
class CYXWIZ_API LRWarmup {
public:
    enum class WarmupType { None, Linear, Cosine };

    LRWarmup(std::unique_ptr<Optimizer> optimizer,
             int warmup_steps,
             WarmupType warmup_type = WarmupType::Linear,
             double base_lr = -1.0);  // -1 = use optimizer's LR

    void Step(std::vector<Tensor*>& parameters,
              std::vector<Tensor*>& gradients);
    void ZeroGrad();

    double GetCurrentLR() const;
    double GetWarmupProgress() const;  // 0.0 to 1.0
    bool IsWarmupComplete() const;
};
```

### Warmup Types

- **Linear**: LR increases linearly from 0 to base_lr
- **Cosine**: Smooth ramp-up using cosine curve: `lr = base_lr * 0.5 * (1 - cos(π * progress))`

### Usage

```cpp
// Create optimizer with warmup
auto adam = std::make_unique<Adam>(0.001);
LRWarmup warmup(std::move(adam), 1000, LRWarmup::WarmupType::Linear);

// Training loop
for (int step = 0; step < total_steps; step++) {
    warmup.Step(params, grads);

    if (!warmup.IsWarmupComplete()) {
        std::cout << "Warmup progress: "
                  << warmup.GetWarmupProgress() * 100 << "%" << std::endl;
    }
}
```

### Python Usage

```python
import pycyxwiz as cx

# Create optimizer with warmup
warmup = cx.create_lr_warmup(
    cx.OptimizerType.Adam,
    learning_rate=0.001,
    warmup_steps=1000,
    warmup_type=cx.WarmupType.Linear
)

# Training loop
for step in range(total_steps):
    warmup.step(params, grads)
    print(f"LR: {warmup.get_current_lr():.6f}")
```

## Learning Rate Schedulers

### Base Scheduler

```cpp
class CYXWIZ_API LRScheduler {
public:
    LRScheduler(Optimizer* optimizer);
    virtual ~LRScheduler() = default;

    virtual void Step(int epoch = -1) = 0;
    double GetLR() const;
    double GetLastLR() const;

protected:
    Optimizer* optimizer_;
    double base_lr_;
    double last_lr_;
};
```

### StepLR

```cpp
class CYXWIZ_API StepLR : public LRScheduler {
public:
    StepLR(Optimizer* optimizer,
           int step_size,
           double gamma = 0.1);

    void Step(int epoch = -1) override;

private:
    int step_size_;
    double gamma_;
};
```

### Usage

```cpp
Adam adam(0.001);
StepLR scheduler(&adam, 10, 0.1);  // Decay by 0.1 every 10 epochs

for (int epoch = 0; epoch < 100; epoch++) {
    train_epoch();
    scheduler.Step();
    std::cout << "LR: " << scheduler.GetLR() << std::endl;
}
```

### ExponentialLR

```cpp
class CYXWIZ_API ExponentialLR : public LRScheduler {
public:
    ExponentialLR(Optimizer* optimizer, double gamma);

    void Step(int epoch = -1) override;
    // lr = base_lr * gamma^epoch
};
```

### CosineAnnealingLR

```cpp
class CYXWIZ_API CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(Optimizer* optimizer,
                      int T_max,
                      double eta_min = 0.0);

    void Step(int epoch = -1) override;
    // lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * epoch / T_max))
};
```

### WarmupScheduler

```cpp
class CYXWIZ_API WarmupScheduler : public LRScheduler {
public:
    WarmupScheduler(Optimizer* optimizer,
                    int warmup_steps,
                    LRScheduler* after_warmup = nullptr);

    void Step(int epoch = -1) override;
    // Linear warmup from 0 to base_lr over warmup_steps
};
```

### OneCycleLR

```cpp
class CYXWIZ_API OneCycleLR : public LRScheduler {
public:
    OneCycleLR(Optimizer* optimizer,
               double max_lr,
               int total_steps,
               double pct_start = 0.3,
               double div_factor = 25.0,
               double final_div_factor = 1e4);

    void Step(int epoch = -1) override;
};
```

### ReduceLROnPlateau

```cpp
class CYXWIZ_API ReduceLROnPlateau : public LRScheduler {
public:
    ReduceLROnPlateau(Optimizer* optimizer,
                      std::string mode = "min",
                      double factor = 0.1,
                      int patience = 10,
                      double threshold = 1e-4,
                      double min_lr = 0.0);

    void Step(double metric);  // Different signature
    // Reduce LR when metric stops improving
};
```

### Usage

```cpp
Adam adam(0.001);
ReduceLROnPlateau scheduler(&adam, "min", 0.1, 5);

for (int epoch = 0; epoch < 100; epoch++) {
    train_epoch();
    double val_loss = validate();
    scheduler.Step(val_loss);
}
```

## Optimizer Factory

```cpp
namespace cyxwiz {

enum class OptimizerType {
    SGD,
    Adam,
    AdamW,
    RMSprop,
    Adagrad,
    NAdam,
    Adadelta,
    LAMB
};

struct OptimizerConfig {
    OptimizerType type = OptimizerType::Adam;
    double learning_rate = 0.001;
    double momentum = 0.0;
    double weight_decay = 0.0;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    bool nesterov = false;
    bool amsgrad = false;
};

std::unique_ptr<Optimizer> CreateOptimizer(const OptimizerConfig& config);

}
```

### Usage

```cpp
OptimizerConfig config;
config.type = OptimizerType::AdamW;
config.learning_rate = 0.0001;
config.weight_decay = 0.01;

auto optimizer = CreateOptimizer(config);
```

## Checkpointing

### Save Optimizer State

```cpp
Adam adam(0.001);
// ... training ...

// Save state
auto state = adam.State();
SaveState("optimizer.ckpt", state);
```

### Load Optimizer State

```cpp
Adam adam(0.001);

// Load state
auto state = LoadState("optimizer.ckpt");
adam.LoadState(state);
```

## Python Bindings

```python
import pycyxwiz as cyx

# SGD
sgd = cyx.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam
adam = cyx.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# AdamW
adamw = cyx.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Training loop
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()

# Schedulers
scheduler = cyx.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(epochs):
    train()
    scheduler.step()
```

## Best Practices

### Optimizer Selection

| Optimizer | Best For | Key Parameters |
|-----------|----------|----------------|
| **SGD+Momentum** | Computer vision, CNNs | lr=0.1, momentum=0.9 |
| **Adam** | General purpose, NLP | lr=0.001 |
| **AdamW** | Transformers, large models | lr=0.0001, wd=0.01 |
| **NAdam** | Faster convergence than Adam | lr=0.002 |
| **RMSprop** | RNNs, non-stationary | lr=0.001 |
| **Adadelta** | No LR tuning needed | rho=0.9 |
| **LAMB** | Large batch training (BERT, GPT) | lr=0.001, wd=0.01 |

### Learning Rate Guidelines

1. **Start with defaults**: Adam 0.001, SGD 0.01-0.1
2. **Use warmup**: Especially for large batch sizes
3. **Decay during training**: Cosine or step decay
4. **Monitor loss**: Reduce on plateau for fine-tuning

### Weight Decay

- **L2 regularization**: Built into gradient
- **Decoupled (AdamW)**: Applied directly to weights
- **Typical values**: 1e-4 to 1e-2

---

**Next**: [Loss API](loss.md) | [Model API](model.md)
