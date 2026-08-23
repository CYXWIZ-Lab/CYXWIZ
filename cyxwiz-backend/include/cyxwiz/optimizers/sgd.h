#pragma once

#include "cyxwiz/optimizers/optimizer_base.h"
#include "cyxwiz/tensor.h"

#include <map>
#include <string>

namespace cyxwiz {

class CYXWIZ_API SGDOptimizer : public Optimizer {
public:
    SGDOptimizer(double learning_rate = 0.01, double momentum = 0.0);

    void Step(std::map<std::string, Tensor>& parameters,
              const std::map<std::string, Tensor>& gradients) override;

    void ZeroGrad() override;

    bool ExportState(OptimizerState& state, std::string& error) const override;
    bool ImportState(const OptimizerState& state, std::string& error) override;

private:
    double momentum_;
    std::map<std::string, Tensor> velocity_;
};

} // namespace cyxwiz
