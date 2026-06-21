#pragma once

#include "cyxwiz/optimizers/optimizer_base.h"
#include "cyxwiz/tensor.h"

#include <map>
#include <string>

namespace cyxwiz {

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
    double alpha_;
    double epsilon_;
    double momentum_;
    std::map<std::string, Tensor> v_;
    std::map<std::string, Tensor> buffer_;
};

class CYXWIZ_API AdaGradOptimizer : public Optimizer {
public:
    AdaGradOptimizer(double learning_rate = 0.01,
                     double epsilon = 1e-10);

    void Step(std::map<std::string, Tensor>& parameters,
              const std::map<std::string, Tensor>& gradients) override;

    void ZeroGrad() override;

private:
    double epsilon_;
    std::map<std::string, Tensor> cache_;
};

class CYXWIZ_API AdadeltaOptimizer : public Optimizer {
public:
    AdadeltaOptimizer(double rho = 0.9,
                      double epsilon = 1e-6);

    void Step(std::map<std::string, Tensor>& parameters,
              const std::map<std::string, Tensor>& gradients) override;

    void ZeroGrad() override;

    double GetRho() const { return rho_; }

private:
    double rho_;
    double epsilon_;
    std::map<std::string, Tensor> acc_grad_;
    std::map<std::string, Tensor> acc_delta_;
};

} // namespace cyxwiz
