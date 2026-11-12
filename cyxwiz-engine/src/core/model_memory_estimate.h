#pragma once

// About how much memory one training step of a built model needs, for every
// layer kind: the parameters, their gradients and the optimizer's state are
// counted from the built model and optimizer; the activations from each
// compiled layer's output shape plus what each module says it keeps for the
// backward pass (Module::CachedElementsPerSample: attention weights,
// recurrent gates, normalized inputs, dropout masks) and the largest working
// set one module's backward holds (Module::BackwardWorkingElementsPerSample:
// a convolution's unfolded columns). float32 throughout.
// It leaves out the device runtime itself and the training data.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz {

struct BuiltExecutableModel;

struct ModelMemoryEstimate {
    bool available = false;
    int batch_size = 0;

    uint64_t trained_elements = 0;       // learnable parameters
    uint64_t state_elements = 0;         // kept, not trained (BatchNorm running statistics, frozen layers)
    size_t optimizer_tensors_per_parameter = 0;
    // Per sample: the input, every layer's output and the modules' caches.
    uint64_t activation_elements_per_sample = 0;
    // Per sample: the input and every layer's output only (what a layer
    // capture keeps; a layer's output is the next one's input).
    uint64_t output_elements_per_sample = 0;
    // Per sample, during backward: the two largest gradients alive at once
    // plus the largest module working set.
    uint64_t backward_elements_per_sample = 0;

    uint64_t parameter_bytes = 0;
    uint64_t gradient_bytes = 0;
    uint64_t optimizer_bytes = 0;
    uint64_t state_bytes = 0;
    uint64_t activation_bytes = 0;       // activations and backward gradients for the batch
    uint64_t total_bytes = 0;

    // The device training would run on, when its memory size is known.
    std::string device_name;
    uint64_t device_memory_bytes = 0;
};

// sample_input_shape: the network input per sample (TrainingConfiguration::
// input_shape); batch_size: the Data Loader's.
ModelMemoryEstimate EstimateTrainingMemory(BuiltExecutableModel& built,
                                           const std::vector<size_t>& sample_input_shape,
                                           int batch_size);

}  // namespace cyxwiz
