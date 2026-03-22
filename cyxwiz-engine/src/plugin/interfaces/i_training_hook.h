#pragma once

#include <string>
#include <map>

namespace cyxwiz::plugin {

struct TrainingContext {
    int current_epoch = 0;
    int total_epochs = 0;
    int current_batch = 0;
    int total_batches = 0;
    float train_loss = 0.0f;
    float train_accuracy = 0.0f;
    float val_loss = 0.0f;
    float val_accuracy = 0.0f;
    float learning_rate = 0.0f;
    std::map<std::string, float> custom_metrics;
};

class ITrainingHook {
public:
    virtual ~ITrainingHook() = default;

    virtual void OnTrainingStart(TrainingContext& /*ctx*/) {}
    virtual void OnTrainingEnd(TrainingContext& /*ctx*/) {}
    virtual void OnEpochStart(TrainingContext& /*ctx*/) {}
    virtual void OnEpochEnd(TrainingContext& /*ctx*/) {}
    virtual void OnBatchStart(TrainingContext& /*ctx*/) {}
    virtual void OnBatchEnd(TrainingContext& /*ctx*/) {}

    // Return true to request early stopping
    virtual bool ShouldStopEarly(const TrainingContext& /*ctx*/) { return false; }
};

} // namespace cyxwiz::plugin
