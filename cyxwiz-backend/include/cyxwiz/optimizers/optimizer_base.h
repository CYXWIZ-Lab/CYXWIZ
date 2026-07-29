#pragma once

#include "cyxwiz/api_export.h"
#include "cyxwiz/tensor.h"

#include <map>
#include <memory>
#include <string>

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

/**
 * Typed, backend-owned optimizer state envelope.
 *
 * Checkpoint code serializes this envelope without reaching into concrete
 * optimizer internals. Concrete optimizers remain responsible for validating
 * their identity, hyperparameters, and tensor invariants transactionally.
 */
struct OptimizerState {
    int schema_version = 1;
    std::string optimizer_type;
    double learning_rate = 0.0;
    int step_count = 0;
    std::map<std::string, double> hyperparameters;
    std::map<std::string, Tensor> tensors;
};

class CYXWIZ_API Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void Step(std::map<std::string, Tensor>& parameters,
                      const std::map<std::string, Tensor>& gradients) = 0;

    virtual void ZeroGrad() = 0;

    /** Export complete state required for the next optimizer step. */
    virtual bool ExportState(OptimizerState& state, std::string& error) const {
        state = OptimizerState{};
        error = "This optimizer does not implement exact state export.";
        return false;
    }

    /** Import state transactionally; failure must not mutate the optimizer. */
    virtual bool ImportState(const OptimizerState& state, std::string& error) {
        (void)state;
        error = "This optimizer does not implement exact state import.";
        return false;
    }

    void SetLearningRate(double lr) { learning_rate_ = lr; }
    double GetLearningRate() const { return learning_rate_; }
    int GetStepCount() const { return step_count_; }

protected:
    double learning_rate_;
    int step_count_;
};

CYXWIZ_API std::unique_ptr<Optimizer> CreateOptimizer(
    OptimizerType type,
    double learning_rate = 0.001
);

} // namespace cyxwiz
