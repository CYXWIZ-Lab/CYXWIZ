#include "model_memory_estimate.h"

#include "model_builder.h"

#include <algorithm>
#include <set>

namespace cyxwiz {

namespace {

uint64_t ElementCount(const std::vector<size_t>& shape) {
    if (shape.empty()) return 0;
    uint64_t n = 1;
    for (size_t d : shape) n *= d;
    return n;
}

}  // namespace

ModelMemoryEstimate EstimateTrainingMemory(BuiltExecutableModel& built,
                                           const std::vector<size_t>& sample_input_shape,
                                           int batch_size) {
    ModelMemoryEstimate out;
    if (!built.ok()) return out;
    out.batch_size = std::max(1, batch_size);

    const std::set<std::string> state = built.model->GetStateParameterNames();
    for (const auto& [name, tensor] : built.model->GetParameters()) {
        (state.count(name) ? out.state_elements : out.trained_elements) += tensor.NumElements();
    }
    out.optimizer_tensors_per_parameter = built.optimizer ? built.optimizer->StateTensorsPerParameter() : 0;

    // A layer that makes several modules lists each; its output is kept once.
    uint64_t activations = ElementCount(sample_input_shape);
    uint64_t largest = 0, second = 0, working = 0;
    std::set<size_t> layers_seen;
    for (const auto& entry : built.module_provenance) {
        if (layers_seen.insert(entry.compiled_layer_index).second) {
            const uint64_t output = ElementCount(entry.output_shape);
            activations += output;
            if (output > largest) {
                second = largest;
                largest = output;
            } else if (output > second) {
                second = output;
            }
        }
        if (entry.created()) {
            if (const Module* module = built.model->LayerModule(*entry.module_index)) {
                activations += module->CachedElementsPerSample(entry.input_shape);
                working = std::max<uint64_t>(working, module->BackwardWorkingElementsPerSample(entry.input_shape));
            }
        }
    }
    out.activation_elements_per_sample = activations;
    out.backward_elements_per_sample = largest + second + working;

    constexpr uint64_t kFloat = sizeof(float);
    const uint64_t batch = static_cast<uint64_t>(out.batch_size);
    out.parameter_bytes = out.trained_elements * kFloat;
    out.gradient_bytes = out.trained_elements * kFloat;
    out.optimizer_bytes = out.trained_elements * out.optimizer_tensors_per_parameter * kFloat;
    out.state_bytes = out.state_elements * kFloat;
    out.activation_bytes = (activations + out.backward_elements_per_sample) * batch * kFloat;
    out.total_bytes = out.parameter_bytes + out.gradient_bytes + out.optimizer_bytes + out.state_bytes +
                      out.activation_bytes;
    out.available = true;
    return out;
}

}  // namespace cyxwiz
