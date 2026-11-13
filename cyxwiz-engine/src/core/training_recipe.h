#pragma once

// How to build one training run over prepared data (TOFIX178 trial runner).
// Train starts a recipe through TrainingManager::StartTrainingRecipe; a
// hyperparameter or architecture search builds a fresh TrainingExecutor from
// the same recipe for every trial, so a trial reads the data the way Train
// does, for every kind of data. `make` finishes the configuration where the
// data decides it (input size, class weights, audio feature shape) and builds
// the executor; it returns null with `error` set when it cannot.

#include "data_registry.h"
#include "training_executor.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

class ArrowDataset;
class ParquetBackedDataset;
class ISequenceBatcher;

struct TrainingRecipe {
    std::string task_name = "Training Model";
    std::string start_message = "Training from Node Graph";
    std::function<std::unique_ptr<TrainingExecutor>(TrainingConfiguration config, int batch_size, std::string& error)>
        make;
    std::string error;  // why there is no recipe (make is empty)

    explicit operator bool() const { return static_cast<bool>(make); }
};

TrainingRecipe FailedTrainingRecipe(std::string error);

// In-memory and disk-backed tables, read by the executor's own batchers.
TrainingRecipe ArrowTrainingRecipe(std::shared_ptr<ArrowDataset> dataset, std::string label_column);
TrainingRecipe ParquetTrainingRecipe(std::shared_ptr<ParquetBackedDataset> dataset, std::string label_column);

// Role batchers the training thread builds once the batch size is known.
TrainingRecipe BatcherFactoryTrainingRecipe(ExternalBatcherFactory factory, std::string task_name,
                                            std::string start_message);

// Role batchers built when the run is made; the builder may finish `config`.
using ResolvedBatchersBuilder = std::function<bool(TrainingConfiguration& config, int batch_size,
                                                   ResolvedExternalBatchers& batchers, std::string& error)>;
TrainingRecipe ResolvedBatchersTrainingRecipe(ResolvedBatchersBuilder build, std::string task_name,
                                              std::string start_message);

// Sequence batches (token / tag windows), built for the run's batch size.
using SequenceBatcherBuilder =
    std::function<bool(TrainingConfiguration& config, int batch_size, std::unique_ptr<ISequenceBatcher>& batcher,
                       std::vector<std::string>& id_to_label, std::string& error)>;
TrainingRecipe SequenceTrainingRecipe(SequenceBatcherBuilder build);

#ifndef CYXWIZ_TRAINING_MANAGER_ARROW_HARNESS
// An image folder: decoded on the training thread at the graph's Resize size.
TrainingRecipe ImageTrainingRecipe(DataRegistry::ImageDatasetEntry entry);
// Audio clips: features from the graph's Spectrogram / Mel / MFCC nodes.
TrainingRecipe AudioTrainingRecipe(DataRegistry::AudioDatasetEntry entry);
#endif

}  // namespace cyxwiz
