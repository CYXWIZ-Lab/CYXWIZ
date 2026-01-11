#pragma once

#include "api_export.h"
#include <map>
#include <string>
#include <memory>

namespace cyxwiz {
    class Tensor;
}

namespace cyxwiz {

enum class OptimizerType {
    SGD = 0,
    Adam = 1,
    AdamW = 2,
    RMSprop = 3,
    AdaGrad = 4,
    NAdam = 5,
    Adadelta = 6,
    LAMB = 7
};

class CYXWIZ_API Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void Step(std::map<std::string, Tensor>& parameters,
                     const std::map<std::string, Tensor>& gradients) = 0;

    virtual void ZeroGrad() = 0;

    void SetLearningRate(double lr) { learning_rate_ = lr; }
    double GetLearningRate() const { return learning_rate_; }

protected:
    double learning_rate_;
    int step_count_;
};

// SGD Optimizer
class CYXWIZ_API SGDOptimizer : public Optimizer {
public:
    SGDOptimizer(double learning_rate = 0.01, double momentum = 0.0);

    void Step(std::map<std::string, Tensor>& parameters,
             const std::map<std::string, Tensor>& gradients) override;

    void ZeroGrad() override;

private:
    double momentum_;
    std::map<std::string, Tensor> velocity_;
};

// Adam Optimizer
class CYXWIZ_API AdamOptimizer : public Optimizer {
public:
    AdamOptimizer(double learning_rate = 0.001,
                  double beta1 = 0.9,
                  double beta2 = 0.999,
                  double epsilon = 1e-8);

    void Step(std::map<std::string, Tensor>& parameters,
             const std::map<std::string, Tensor>& gradients) override;

    void ZeroGrad() override;

private:
    double beta1_;
    double beta2_;
    double epsilon_;
    std::map<std::string, Tensor> m_; // First moment
    std::map<std::string, Tensor> v_; // Second moment
};

// AdamW Optimizer (Adam with weight decay)
class CYXWIZ_API AdamWOptimizer : public AdamOptimizer {
public:
    AdamWOptimizer(double learning_rate = 0.001,
                   double beta1 = 0.9,
                   double beta2 = 0.999,
                   double epsilon = 1e-8,
                   double weight_decay = 0.01);

    void Step(std::map<std::string, Tensor>& parameters,
             const std::map<std::string, Tensor>& gradients) override;

private:
    double weight_decay_;
};

// RMSprop Optimizer - Root Mean Square Propagation
class CYXWIZ_API RMSpropOptimizer : public Optimizer {
public:
    RMSpropOptimizer(double learning_rate = 0.001,
                     double alpha = 0.99,
                     double epsilon = 1e-8,
                     double momentum = 0.0);

    void Step(std::map<std::string, Tensor>& parameters,
             const std::map<std::string, Tensor>& gradients) override;

    void ZeroGrad() override;

private:
    double alpha_;    // Decay rate for running average of squared gradients
    double epsilon_;
    double momentum_;
    std::map<std::string, Tensor> v_;      // Running average of squared gradients
    std::map<std::string, Tensor> buffer_; // Momentum buffer (if momentum > 0)
};

// AdaGrad Optimizer - Adaptive Gradient
class CYXWIZ_API AdaGradOptimizer : public Optimizer {
public:
    AdaGradOptimizer(double learning_rate = 0.01,
                     double epsilon = 1e-10);

    void Step(std::map<std::string, Tensor>& parameters,
             const std::map<std::string, Tensor>& gradients) override;

    void ZeroGrad() override;

private:
    double epsilon_;
    std::map<std::string, Tensor> cache_; // Sum of squared gradients
};

// NAdam Optimizer - Nesterov-accelerated Adam
class CYXWIZ_API NAdamOptimizer : public Optimizer {
public:
    NAdamOptimizer(double learning_rate = 0.002,
                   double beta1 = 0.9,
                   double beta2 = 0.999,
                   double epsilon = 1e-8);

    void Step(std::map<std::string, Tensor>& parameters,
             const std::map<std::string, Tensor>& gradients) override;

    void ZeroGrad() override;

private:
    double beta1_;
    double beta2_;
    double epsilon_;
    std::map<std::string, Tensor> m_; // First moment
    std::map<std::string, Tensor> v_; // Second moment
};

// Adadelta Optimizer - Adaptive learning rate method
// Reference: "ADADELTA: An Adaptive Learning Rate Method" (Zeiler, 2012)
class CYXWIZ_API AdadeltaOptimizer : public Optimizer {
public:
    AdadeltaOptimizer(double rho = 0.9,
                      double epsilon = 1e-6);

    void Step(std::map<std::string, Tensor>& parameters,
             const std::map<std::string, Tensor>& gradients) override;

    void ZeroGrad() override;

    double GetRho() const { return rho_; }

private:
    double rho_;      // Decay rate for running averages
    double epsilon_;
    std::map<std::string, Tensor> acc_grad_;   // Accumulated squared gradients E[g²]
    std::map<std::string, Tensor> acc_delta_;  // Accumulated squared updates E[Δx²]
};

// LAMB Optimizer - Layer-wise Adaptive Moments optimizer for Batch training
// Reference: "Large Batch Optimization for Deep Learning" (You et al., 2019)
// Designed for large batch training (e.g., BERT pre-training with batch size 32K)
class CYXWIZ_API LAMBOptimizer : public Optimizer {
public:
    LAMBOptimizer(double learning_rate = 0.001,
                  double beta1 = 0.9,
                  double beta2 = 0.999,
                  double epsilon = 1e-6,
                  double weight_decay = 0.01);

    void Step(std::map<std::string, Tensor>& parameters,
             const std::map<std::string, Tensor>& gradients) override;

    void ZeroGrad() override;

    double GetBeta1() const { return beta1_; }
    double GetBeta2() const { return beta2_; }
    double GetWeightDecay() const { return weight_decay_; }

private:
    double beta1_;
    double beta2_;
    double epsilon_;
    double weight_decay_;
    std::map<std::string, Tensor> m_; // First moment
    std::map<std::string, Tensor> v_; // Second moment
};

// ============================================================================
// Learning Rate Schedulers
// ============================================================================

enum class WarmupType {
    None,
    Linear,     // Linear warmup from 0 to base_lr
    Cosine      // Cosine warmup from 0 to base_lr
};

// Learning Rate Warmup wrapper
// Wraps any optimizer and applies learning rate warmup for the first N steps
class CYXWIZ_API LRWarmup {
public:
    /**
     * Create learning rate warmup wrapper
     * @param optimizer The underlying optimizer (takes ownership)
     * @param warmup_steps Number of warmup steps
     * @param warmup_type Type of warmup schedule (Linear or Cosine)
     * @param base_lr Target learning rate after warmup (default: use optimizer's initial LR)
     */
    LRWarmup(std::unique_ptr<Optimizer> optimizer,
             int warmup_steps,
             WarmupType warmup_type = WarmupType::Linear,
             double base_lr = -1.0);

    // Forward the Step call with warmup-adjusted learning rate
    void Step(std::map<std::string, Tensor>& parameters,
             const std::map<std::string, Tensor>& gradients);

    void ZeroGrad();

    // Get current learning rate (after warmup adjustment)
    double GetCurrentLR() const;

    // Get warmup progress (0.0 to 1.0)
    double GetWarmupProgress() const;

    // Check if warmup is complete
    bool IsWarmupComplete() const;

    // Access underlying optimizer
    Optimizer* GetOptimizer() { return optimizer_.get(); }

private:
    std::unique_ptr<Optimizer> optimizer_;
    int warmup_steps_;
    WarmupType warmup_type_;
    double base_lr_;
    int current_step_;
};

// Factory function
CYXWIZ_API std::unique_ptr<Optimizer> CreateOptimizer(
    OptimizerType type,
    double learning_rate = 0.001
);

} // namespace cyxwiz
