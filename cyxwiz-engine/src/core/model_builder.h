#pragma once

#include "graph_compiler.h"
#include <cyxwiz/sequential.h>
#include <cyxwiz/loss.h>
#include <cyxwiz/optimizer.h>
#include <memory>

namespace cyxwiz {

// Result of assembling a SequentialModel + Loss + Optimizer from a
// compiled TrainingConfiguration. On failure, `model` is nullptr and the
// builder has already logged the reason.
struct BuiltModel {
    std::unique_ptr<SequentialModel> model;
    std::unique_ptr<Loss>            loss;
    std::unique_ptr<Optimizer>       optimizer;

    bool ok() const { return model != nullptr; }
};

// Build SequentialModel + Loss + Optimizer from a TrainingConfiguration.
// Pure function — no side effects beyond logging. Shared between
// TrainingExecutor (real training) and DebugExecutor (one-step local
// debug sanity check).
BuiltModel BuildSequentialFromConfig(const TrainingConfiguration& config);

} // namespace cyxwiz
