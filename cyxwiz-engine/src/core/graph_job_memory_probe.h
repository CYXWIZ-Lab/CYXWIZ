#pragma once

// How much device memory a training job needs, measured (TOFIX118 P4b): the
// job's graph and data run through the shared runner for one training step
// on the selected route, and ArrayFire's allocated bytes are read before
// the first step and after it (model, optimizer state and one step's
// activations and gradients). The Engine measures before sending a remote
// job (JobConfig.estimated_memory); the node compares it with its device.

#include "graph_training_job.h"

#include <cstdint>
#include <string>

namespace cyxwiz {

struct GraphJobMemoryProbe {
    bool ok = false;
    std::string error;
    TrainingFailureKind failure = TrainingFailureKind::None;
    std::uint64_t training_bytes = 0;  // measured growth over the first step
    std::string backend;               // route that measured it
};

// Runs one step of `request` (its epochs/checkpoints are overridden).
GraphJobMemoryProbe ProbeGraphJobMemory(GraphTrainingJobRequest request);

}  // namespace cyxwiz
