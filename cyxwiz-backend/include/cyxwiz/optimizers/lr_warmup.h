#pragma once

#include "cyxwiz/api_export.h"
#include "cyxwiz/optimizers/optimizer_base.h"

#include <map>
#include <memory>
#include <string>

namespace cyxwiz {

class Tensor;

enum class WarmupType {
    None,
    Linear,
    Cosine
};

class CYXWIZ_API LRWarmup {
public:
    LRWarmup(std::unique_ptr<Optimizer> optimizer,
             int warmup_steps,
             WarmupType warmup_type = WarmupType::Linear,
             double base_lr = -1.0);

    // Applies the wrapped optimizer with the current learning rate, then
    // advances the warmup rate for the next optimizer update. This matches
    // PyTorch's optimizer.step() followed by scheduler.step() ordering.
    void Step(std::map<std::string, Tensor>& parameters,
              const std::map<std::string, Tensor>& gradients);

    void ZeroGrad();

    double GetCurrentLR() const;
    double GetWarmupProgress() const;
    bool IsWarmupComplete() const;

    Optimizer* GetOptimizer() { return optimizer_.get(); }

private:
    std::unique_ptr<Optimizer> optimizer_;
    int warmup_steps_;
    WarmupType warmup_type_;
    double base_lr_;
    int current_step_;
};

} // namespace cyxwiz
