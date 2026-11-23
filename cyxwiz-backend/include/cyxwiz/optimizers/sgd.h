// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

#include "cyxwiz/optimizers/optimizer_base.h"
#include "cyxwiz/tensor.h"

#include <map>
#include <set>
#include <string>

namespace cyxwiz {

class CYXWIZ_API SGDOptimizer : public Optimizer {
public:
    // torch.optim.SGD with dampening 0: the gradient gains weight_decay x
    // parameter, the velocity is momentum x velocity + gradient, and the step
    // is the velocity, or gradient + momentum x velocity with Nesterov (which
    // needs momentum above 0).
    SGDOptimizer(double learning_rate = 0.01, double momentum = 0.0, double weight_decay = 0.0,
                 bool nesterov = false);

    void Step(std::map<std::string, Tensor>& parameters,
              const std::map<std::string, Tensor>& gradients) override;

    void ZeroGrad() override;
    size_t StateTensorsPerParameter() const override { return momentum_ > 0.0 ? 1 : 0; }  // velocity

    // Parameters that take no weight decay (biases and normalization scales, by name).
    void SetNoDecayParameters(std::set<std::string> names) { no_decay_parameters_ = std::move(names); }
    const std::set<std::string>& GetNoDecayParameters() const { return no_decay_parameters_; }
    double GetWeightDecay() const { return weight_decay_; }
    bool IsNesterov() const { return nesterov_; }

    bool ExportState(OptimizerState& state, std::string& error) const override;
    bool ImportState(const OptimizerState& state, std::string& error) override;

private:
    // What a saved state must agree with: momentum always; weight decay and
    // Nesterov only when set, so states saved before them still load.
    std::map<std::string, double> Hyperparameters() const;

    double momentum_;
    double weight_decay_ = 0.0;
    bool nesterov_ = false;
    std::set<std::string> no_decay_parameters_;
    std::map<std::string, Tensor> velocity_;
};

} // namespace cyxwiz
