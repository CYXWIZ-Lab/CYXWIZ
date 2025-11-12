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
    AdaGrad = 4
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

// Factory function
CYXWIZ_API std::unique_ptr<Optimizer> CreateOptimizer(
    OptimizerType type,
    double learning_rate = 0.001
);

} // namespace cyxwiz
