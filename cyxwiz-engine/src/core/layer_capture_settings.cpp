#include "layer_capture_settings.h"

#include <algorithm>

namespace cyxwiz::layer_capture {

using core::LayerCaptureWhen;

bool CapturesAnything(const core::StudioDebuggerConfig& settings, bool for_training) {
    if (settings.capture_when == LayerCaptureWhen::Off) return false;
    if (for_training && settings.capture_when != LayerCaptureWhen::Training) return false;
    return settings.capture_activations || settings.capture_parameters || settings.capture_gradients;
}

size_t CaptureSamples(const core::StudioDebuggerConfig& settings) {
    return settings.capture_batch ? static_cast<size_t>(core::kLayerCaptureBatchSamples) : 1;
}

std::string Summary(const core::StudioDebuggerConfig& settings) {
    if (!CapturesAnything(settings, false)) return "off";
    if (settings.capture_when == LayerCaptureWhen::LocalDebug) return "Local Debug runs";
    const int every = std::max(1, settings.capture_every_epochs);
    return every == 1 ? std::string("every epoch") : "every " + std::to_string(every) + " epochs";
}

std::vector<int> SnapshotEpochs(int every, int epochs) {
    std::vector<int> out;
    if (epochs < 1) return out;
    every = std::max(1, every);
    for (int e = every; e <= epochs; e += every) out.push_back(e);
    if (out.empty() || out.back() != epochs) out.push_back(epochs);
    return out;
}

SizeEstimate EstimateSize(const ModelMemoryEstimate& model, const core::StudioDebuggerConfig& settings,
                          uint64_t byte_budget) {
    SizeEstimate out;
    if (!model.available) return out;
    constexpr uint64_t kFloat = sizeof(float);
    out.available = true;
    out.activation_values = model.output_elements_per_sample * CaptureSamples(settings);
    out.activation_bytes = out.activation_values * kFloat;
    out.trained_elements = model.trained_elements;
    out.state_elements = model.state_elements;
    out.parameter_bytes = (model.trained_elements + model.state_elements) * kFloat;
    out.gradient_bytes = model.trained_elements * kFloat;
    out.snapshot_bytes = (settings.capture_activations ? out.activation_bytes : 0) +
                         (settings.capture_parameters ? out.parameter_bytes : 0) +
                         (settings.capture_gradients ? out.gradient_bytes : 0);
    out.over_budget = out.snapshot_bytes > byte_budget;
    return out;
}

}  // namespace cyxwiz::layer_capture
