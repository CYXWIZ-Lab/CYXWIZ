#pragma once

#include "executable_model.h"
#include "graph_compiler.h"
#include "dataset_batcher.h"  // Includes ArrowDatasetBatcher + IBatcher
#include "parquet_arrow_batcher.h"
#include "parquet_backed_dataset.h"
#include "data_registry.h"
#include "arrow_dataset.h"
#include <cyxwiz/tensor.h>
#include <cyxwiz/optimizer.h>
#include <cyxwiz/sequential.h>
#include <cyxwiz/loss.h>
#include <functional>
#include <atomic>
#include <mutex>
#include <memory>
#include <thread>
#include <chrono>

namespace cyxwiz {

/**
 * Training metrics updated during training
 */
struct TrainingMetrics {
    // Current progress
    int current_epoch = 0;
    int total_epochs = 0;
    int current_batch = 0;
    int total_batches = 0;

    // Training metrics
    float train_loss = 0.0f;
    float train_accuracy = 0.0f;

    // Validation metrics
    float val_loss = 0.0f;
    float val_accuracy = 0.0f;

    // Timing
    float epoch_time_seconds = 0.0f;
    float samples_per_second = 0.0f;

    // State
    bool is_training = false;
    bool is_paused = false;
    bool is_complete = false;
    std::string status_message;

    // History (for plotting)
    std::vector<float> loss_history;
    std::vector<float> accuracy_history;
    std::vector<float> val_loss_history;
    std::vector<float> val_accuracy_history;
};

/**
 * Callback types for training progress
 */
using BatchCallback = std::function<void(int epoch, int batch, int total_batches,
                                           float loss, float accuracy)>;
using EpochCallback = std::function<void(int epoch, float train_loss, float train_acc,
                                          float val_loss, float val_acc, float epoch_time)>;
using TrainingCompleteCallback = std::function<void(const TrainingMetrics& final_metrics)>;

/**
 * TrainingExecutor - Executes ML training based on compiled graph configuration
 *
 * This class handles the actual training loop:
 * - Builds the model DYNAMICALLY from TrainingConfiguration
 * - Creates optimizer based on config
 * - Iterates over batches
 * - Performs forward/backward passes
 * - Updates weights
 * - Reports progress via callbacks
 */
class TrainingExecutor {
public:
    /**
     * Create a training executor
     * @param config Compiled training configuration from GraphCompiler
     * @param dataset Dataset handle from DataRegistry
     */
    TrainingExecutor(TrainingConfiguration config, DatasetHandle dataset);

    /**
     * Create a training executor with Arrow dataset (modern API)
     * @param config Compiled training configuration from GraphCompiler
     * @param arrow_dataset Arrow dataset with features and labels
     * @param label_column Name of the label column
     */
    TrainingExecutor(TrainingConfiguration config,
                     std::shared_ptr<ArrowDataset> arrow_dataset,
                     const std::string& label_column);

    /**
     * Create a training executor for a disk-backed Parquet dataset.
     * Used when DataRegistry::LoadTabularCSV picked the disk-backed path
     * because the CSV was too big to fit in RAM.
     */
    TrainingExecutor(TrainingConfiguration config,
                     std::shared_ptr<ParquetBackedDataset> parquet_dataset,
                     const std::string& label_column);

    TrainingExecutor(TrainingConfiguration config,
                     std::unique_ptr<IBatcher> external_batcher);

    ~TrainingExecutor();

    /**
     * Start training (blocking - should be called from a background thread)
     * @param epochs Number of epochs to train
     * @param batch_size Batch size
     * @param batch_cb Callback for each batch (optional)
     * @param epoch_cb Callback for each epoch (optional)
     * @param complete_cb Callback when training completes (optional)
     */
    void Train(
        int epochs,
        int batch_size,
        BatchCallback batch_cb = nullptr,
        EpochCallback epoch_cb = nullptr,
        TrainingCompleteCallback complete_cb = nullptr
    );

    /**
     * Stop training (thread-safe, cooperative cancellation)
     */
    void Stop();

    /**
     * Pause training (thread-safe)
     */
    void Pause();

    /**
     * Resume training after pause (thread-safe)
     */
    void Resume();

    /**
     * Check if training is currently running
     */
    bool IsTraining() const { return is_training_.load(); }

    /**
     * Check if training is paused
     */
    bool IsPaused() const { return is_paused_.load(); }

    /**
     * Get current training metrics (thread-safe)
     */
    TrainingMetrics GetMetrics() const;

    /**
     * Get the training configuration
     */
    const TrainingConfiguration& GetConfig() const { return config_; }

    /**
     * Get the trained model (for export)
     */
    SequentialModel* GetModel() {
        return model_ ? model_->AsSequentialModel() : nullptr;
    }

    /**
     * Get the optimizer (for export)
     */
    Optimizer* GetOptimizer() { return optimizer_.get(); }

    /**
     * Release ownership of the model (transfers ownership to caller)
     */
    std::unique_ptr<SequentialModel> ReleaseModel() {
        return model_ ? model_->ReleaseSequentialModel() : nullptr;
    }

    /**
     * Release ownership of the optimizer (transfers ownership to caller)
     */
    std::unique_ptr<Optimizer> ReleaseOptimizer() { return std::move(optimizer_); }

private:
    // Three possible dataset backings. Exactly one of dataset_ / arrow_dataset_ /
    // parquet_dataset_ is populated at construction time, based on which
    // constructor was called. mode_ is the tag the Train() function uses
    // to pick the right batcher implementation.
    enum class DatasetMode {
        Legacy,   // DatasetHandle + legacy DatasetBatcher
        Arrow,    // ArrowDataset + ArrowDatasetBatcher
        External, // Image/Audio/Text IBatcher constructed by TrainingManager
        Parquet   // ParquetBackedDataset + ParquetArrowBatcher (disk-backed)
    };

    TrainingConfiguration config_;
    DatasetHandle dataset_;
    std::shared_ptr<ArrowDataset> arrow_dataset_;
    std::shared_ptr<ParquetBackedDataset> parquet_dataset_;
    std::string label_column_;
    DatasetMode mode_ = DatasetMode::Legacy;
    std::unique_ptr<IBatcher> external_batcher_;

    // Thread safety
    std::atomic<bool> is_training_{false};
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> is_paused_{false};

    mutable std::mutex metrics_mutex_;
    TrainingMetrics metrics_;

    // Training components - DYNAMIC MODEL
    std::unique_ptr<IExecutableModel> model_;
    std::unique_ptr<Optimizer> optimizer_;
    std::unique_ptr<Loss> loss_;  // Unified loss function (MSE, CrossEntropy, BCE, etc.)

    // Internal training methods

    /**
     * Initialize the training components by building model from config
     */
    bool Initialize(int batch_size);

    /**
     * Run a single training epoch
     */
    void RunTrainingEpoch(
        DatasetBatcher& batcher,
        int epoch,
        BatchCallback batch_cb
    );

    /**
     * Run validation
     */
    void RunValidation(DatasetBatcher& batcher);

    /**
     * Run a single training epoch through any IBatcher implementation.
     * Used for both ArrowDatasetBatcher (in-memory) and ParquetArrowBatcher
     * (disk-backed) — polymorphism via IBatcher lets the same loop drive
     * either one. Name kept as *Arrow for backwards compat with callers
     * that haven't been renamed yet; the implementation is batcher-agnostic.
     */
    void RunTrainingEpochArrow(
        IBatcher& batcher,
        int epoch,
        BatchCallback batch_cb
    );

    /**
     * Run validation through any IBatcher implementation.
     */
    void RunValidationArrow(IBatcher& batcher);

    /**
     * Forward pass through the model
     * @param input Input tensor [batch_size, input_features]
     * @return Output tensor [batch_size, num_classes]
     */
    Tensor Forward(const Tensor& input);

    /**
     * Compute loss between predictions and targets
     */
    float ComputeLoss(const Tensor& predictions, const Tensor& targets);

    /**
     * Compute accuracy (for classification)
     */
    float ComputeAccuracy(const Tensor& predictions, const Tensor& targets);

    /**
     * Backward pass through the model
     */
    void Backward(const Tensor& predictions, const Tensor& targets);

    /**
     * Apply preprocessing to batch data
     */
    void PreprocessBatch(Batch& batch);

    /**
     * Update metrics (thread-safe)
     */
    void UpdateMetrics(const std::function<void(TrainingMetrics&)>& updater);

    /**
     * Check if we should stop (for cooperative cancellation)
     */
    bool ShouldStop() const { return stop_requested_.load(); }

    /**
     * Wait while paused
     */
    void WaitWhilePaused();

    // Cached tensors for backward pass
    Tensor last_predictions_;
    Tensor loss_gradient_;
};

} // namespace cyxwiz
