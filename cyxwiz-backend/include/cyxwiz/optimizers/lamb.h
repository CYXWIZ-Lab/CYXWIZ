#pragma once

#include "cyxwiz/optimizers/optimizer_base.h"
#include "cyxwiz/tensor.h"

#include <map>
#include <string>

namespace cyxwiz {

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
    std::map<std::string, Tensor> m_;
    std::map<std::string, Tensor> v_;
};

} // namespace cyxwiz
