#pragma once

#include "../api_export.h"
#include "../tensor.h"

#include <map>
#include <string>

namespace cyxwiz {

// ============================================================================
// Base Layer Class
// ============================================================================
//
// Layer classes are low-level neural-network building blocks. The
// model-facing training/runtime API is SequentialModel + Module in
// sequential.h; modules may wrap these layer primitives to participate in
// serialization, freezing, gradient collection, and optimizer updates.
//
// Keep new model-facing features on Module/SequentialModel first. Add direct
// Layer APIs only when a primitive is also useful outside a SequentialModel.

class CYXWIZ_API Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor Forward(const Tensor& input) = 0;
    virtual Tensor Backward(const Tensor& grad_output) = 0;
    virtual std::map<std::string, Tensor> GetParameters() = 0;
    virtual void SetParameters(const std::map<std::string, Tensor>& params) = 0;

    // Training mode (affects BatchNorm, Dropout, etc.)
    virtual void SetTraining(bool training) { training_ = training; }
    bool IsTraining() const { return training_; }

    // Layer name for debugging/serialization
    virtual std::string GetName() const { return "Layer"; }

protected:
    bool training_ = true;
    Tensor cached_input_;  // For backward pass
};

} // namespace cyxwiz
