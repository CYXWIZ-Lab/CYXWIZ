#pragma once

// The Studio Debugger's layer capture control (TOFIX178 6.2d, approved
// Capture mockup): what a capture setting means for a run, the toolbar's
// one-word summary, the epochs a training run snapshots and the size of one
// snapshot from the compiled model. Pure: no ImGui, no device.

#include "engine_config.h"
#include "model_memory_estimate.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz::layer_capture {

// Whether a run captures anything: the When choice covers it (Training also
// covers Local Debug runs) and at least one of activations, parameters and
// gradients is ticked.
bool CapturesAnything(const core::StudioDebuggerConfig& settings, bool for_training);

// Samples a capture runs: 1, or a batch of core::kLayerCaptureBatchSamples.
size_t CaptureSamples(const core::StudioDebuggerConfig& settings);

// "off", "Local Debug runs", "every 5 epochs" (for the toolbar button).
std::string Summary(const core::StudioDebuggerConfig& settings);

// The epochs a run of `epochs` epochs snapshots: every `every`, and the last.
std::vector<int> SnapshotEpochs(int every, int epochs);

struct SizeEstimate {
    bool available = false;  // the model was built and counted
    uint64_t activation_values = 0;  // the input and every layer's output, for the captured samples
    uint64_t activation_bytes = 0;
    uint64_t trained_elements = 0;
    uint64_t state_elements = 0;
    uint64_t parameter_bytes = 0;    // weights and kept state
    uint64_t gradient_bytes = 0;
    uint64_t snapshot_bytes = 0;     // what is ticked, together
    bool over_budget = false;        // past the capture's byte budget: tensors past it are skipped
};

// One snapshot's size, float32, from the compiled model's shapes. A layer's
// output is the next one's input, so it counts once.
SizeEstimate EstimateSize(const ModelMemoryEstimate& model, const core::StudioDebuggerConfig& settings,
                          uint64_t byte_budget);

}  // namespace cyxwiz::layer_capture
