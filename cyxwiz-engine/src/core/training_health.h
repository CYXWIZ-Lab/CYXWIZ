#pragma once

// Training health (TOFIX129 D11): recorded by TrainingExecutor once an epoch,
// shown in the Model window's Health tab. For every weight tensor (a
// parameter of two or more dimensions): its size ||W|| at the epoch's end,
// the mean over the epoch's optimizer steps of its gradient's size (no extra
// pass: the gradients training computes, before clipping), and a 24-bin
// histogram of its values; for every ReLU of a layer stack, how many of its
// units (channels of an image layer) gave 0 for every validation sample.
// Statistics accumulate on the device; the host reads them once an epoch.

#include <cyxwiz/tensor.h>

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

class Module;

constexpr int kHealthHistogramBins = 24;

struct WeightHealth {
    std::string parameter;   // "layer3.weights"
    size_t layer = 0;        // the N of "layerN"
    double norm = 0.0;       // ||W|| (Frobenius) at the epoch's end
    double mean_gradient_norm = 0.0;
    std::vector<long long> histogram;  // kHealthHistogramBins counts over [low, high]
    double low = 0.0, high = 0.0;
    size_t count = 0;
};

struct ReluHealth {
    size_t layer = 0;         // the ReLU's module index
    long long dead = 0;       // units that gave 0 for every validation sample
    long long units = 0;
};

struct EpochHealth {
    int epoch = 0;
    std::vector<WeightHealth> weights;
    std::vector<ReluHealth> relus;  // empty when no validation pass ran (or a graph model)
};

struct TrainingHealth {
    std::vector<EpochHealth> epochs;
};

class TrainingHealthRecorder {
public:
    TrainingHealthRecorder();
    ~TrainingHealthRecorder();

    // After the gradients of an optimizer step (before clipping).
    void OnStep(const std::map<std::string, Tensor>& gradients);
    // A module's output during the validation pass (SequentialModel's
    // forward observer); only ReLUs count.
    void OnValidationOutput(size_t module_index, const Module& module, const Tensor& output);
    // Reads the epoch's statistics and starts the next epoch.
    EpochHealth EndEpoch(int epoch, const std::map<std::string, Tensor>& parameters);

private:
    struct DeviceState;
    std::unique_ptr<DeviceState> state_;
};

// The N of a parameter name "layerN.<rest>"; npos when it has none.
size_t HealthLayerIndex(const std::string& parameter);

}  // namespace cyxwiz
