#pragma once

#include "cyxwiz/api_export.h"

#include <map>
#include <memory>
#include <string>

namespace cyxwiz {

class Tensor;

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

CYXWIZ_API std::unique_ptr<Optimizer> CreateOptimizer(
    OptimizerType type,
    double learning_rate = 0.001
);

} // namespace cyxwiz
