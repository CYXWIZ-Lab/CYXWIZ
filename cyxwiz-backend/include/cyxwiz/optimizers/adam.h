// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

#include "cyxwiz/optimizers/optimizer_base.h"
#include "cyxwiz/tensor.h"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace cyxwiz {

class CYXWIZ_API AdamOptimizer : public Optimizer {
public:
    AdamOptimizer(double learning_rate = 0.001,
                  double beta1 = 0.9,
                  double beta2 = 0.999,
                  double epsilon = 1e-8);

    void Step(std::map<std::string, Tensor>& parameters,
              const std::map<std::string, Tensor>& gradients) override;

    void ZeroGrad() override;
    size_t StateTensorsPerParameter() const override { return 2; }  // m, v

    bool ExportState(OptimizerState& state, std::string& error) const override;
    bool ImportState(const OptimizerState& state, std::string& error) override;

    // Parameters that skip decoupled weight decay (AdamW): typically biases,
    // normalization scales and embedding tables. Adam ignores it (no decay).
    void SetNoDecayParameters(std::set<std::string> names) { no_decay_parameters_ = std::move(names); }
    const std::set<std::string>& GetNoDecayParameters() const { return no_decay_parameters_; }

    // One optimizer's parameters and their gradients, for StepTogether
    struct Group {
        AdamOptimizer* optimizer = nullptr;
        std::map<std::string, Tensor>* parameters = nullptr;
        const std::map<std::string, Tensor>* gradients = nullptr;
    };
    // One Adam step for several optimizers at once, after a global-norm clip
    // across all of their gradients when max_grad_norm > 0: the same result
    // as clipping the gradients together and calling each Step, in two device
    // launches on the active device's neural provider instead of a few per
    // tensor (TOFIX206 3.2 C). Each optimizer keeps its own moments and steps.
    // False with the reason, and nothing changed, when it does not apply: no
    // provider runs it on this device, an AdamW or other subclass, different
    // hyperparameters, parameters at different steps, more than 32 tensors.
    static bool StepTogether(const std::vector<Group>& groups, float max_grad_norm, std::string& why);

protected:
    std::set<std::string> no_decay_parameters_;
    std::map<std::string, double> AdamHyperparameters() const;
    void StepImpl(std::map<std::string, Tensor>& parameters,
                  const std::map<std::string, Tensor>& gradients,
                  const char* operation_name,
                  double weight_decay);

private:
    double beta1_;
    double beta2_;
    double epsilon_;
    std::map<std::string, Tensor> m_;
    std::map<std::string, Tensor> v_;
    std::map<std::string, int> parameter_steps_;
};

class CYXWIZ_API AdamWOptimizer : public AdamOptimizer {
public:
    AdamWOptimizer(double learning_rate = 0.001,
                   double beta1 = 0.9,
                   double beta2 = 0.999,
                   double epsilon = 1e-8,
                   double weight_decay = 0.01);

    void Step(std::map<std::string, Tensor>& parameters,
              const std::map<std::string, Tensor>& gradients) override;

    bool ExportState(OptimizerState& state, std::string& error) const override;
    bool ImportState(const OptimizerState& state, std::string& error) override;

private:
    double weight_decay_;
};

class CYXWIZ_API NAdamOptimizer : public Optimizer {
public:
    NAdamOptimizer(double learning_rate = 0.002,
                   double beta1 = 0.9,
                   double beta2 = 0.999,
                   double epsilon = 1e-8);

    void Step(std::map<std::string, Tensor>& parameters,
              const std::map<std::string, Tensor>& gradients) override;

    void ZeroGrad() override;
    size_t StateTensorsPerParameter() const override { return 2; }  // m, v
    bool ExportState(OptimizerState& state, std::string& error) const override;
    bool ImportState(const OptimizerState& state, std::string& error) override;

private:
    double beta1_;
    double beta2_;
    double epsilon_;
    std::map<std::string, Tensor> m_;
    std::map<std::string, Tensor> v_;
    std::map<std::string, int> parameter_steps_;
    std::map<std::string, float> mu_products_;
};

} // namespace cyxwiz
