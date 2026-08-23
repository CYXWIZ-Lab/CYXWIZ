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
#include <utility>

namespace cyxwiz {

inline bool UsesRegressionMetrics(const TrainingConfiguration& config) {
    return UsesContinuousTargetMetrics(config);
}

/**
 * Training metrics updated during training
 */
struct TrainingMetrics {
    // Current progress
    int current_epoch = 0;
    // Run-history truth. This is never rewritten when an earlier checkpoint
    // is restored into the active model after training.
    int last_executed_epoch = 0;
    int total_epochs = 0;
    int current_batch = 0;
    int total_batches = 0;
    int optimizer_step_count = 0;
    size_t train_sample_count = 0;
    size_t val_sample_count = 0;
    size_t test_sample_count = 0;

    // Training metrics
    float train_loss = 0.0f;
    float train_accuracy = 0.0f;
    float train_mae = 0.0f;
    float train_rmse = 0.0f;

    // Validation metrics
    float val_loss = 0.0f;
    float val_accuracy = 0.0f;
    float val_mae = 0.0f;
    float val_rmse = 0.0f;
    bool has_validation_metrics = false;

    // Held-out test metrics, populated after training when a test split exists.
    float test_loss = 0.0f;
    float test_accuracy = 0.0f;
    float test_mae = 0.0f;
    float test_rmse = 0.0f;
    bool has_test_metrics = false;

    // Active-model provenance after training. Run-history fields above remain
    // about executed work; restored checkpoint state is reported separately.
    std::string checkpoint_used;
    int restored_checkpoint_epoch = 0;
    int restored_checkpoint_step = 0;
    std::string active_model_provenance;

    // Token-level sequence tagging metrics. For sequence training,
    // train_accuracy/val_accuracy also mirror token accuracy so existing
    // dashboards and callbacks keep working.
    float train_token_accuracy = 0.0f;
    float val_token_accuracy = 0.0f;
    float train_entity_f1 = 0.0f;
    float val_entity_f1 = 0.0f;
    float test_token_accuracy = 0.0f;
    float test_entity_f1 = 0.0f;
    size_t train_token_count = 0;
    size_t val_token_count = 0;
    size_t test_token_count = 0;

    // Timing
    float epoch_time_seconds = 0.0f;
    float samples_per_second = 0.0f;

    // State
    bool is_training = false;
    bool is_paused = false;
    bool is_complete = false;
    std::string status_message;
    std::string terminal_status;
    std::string terminal_reason;

    // History (for plotting)
    std::vector<float> loss_history;
    std::vector<float> accuracy_history;
    std::vector<float> mae_history;
    std::vector<float> rmse_history;
    std::vector<float> val_loss_history;
    std::vector<float> val_accuracy_history;
    std::vector<float> val_mae_history;
    std::vector<float> val_rmse_history;
};

struct ObjectiveEvaluationMetrics {
    float loss = 0.0f;
    float accuracy = 0.0f;
    float mae = 0.0f;
    float rmse = 0.0f;
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
 * Batchers resolved to semantic Train/Dev/Test roles before execution.
 * A role carries its iteration phase because a supplied Dev/Test dataset is
 * consumed in full, while one source can expose derived partitions by
 * switching the same batcher between phases.
 */
struct ResolvedExternalBatchers {
    std::shared_ptr<IBatcher> train;
    std::shared_ptr<IBatcher> dev;
    std::shared_ptr<IBatcher> test;
    BatcherPhase train_phase = BatcherPhase::Train;
    BatcherPhase dev_phase = BatcherPhase::Val;
    BatcherPhase test_phase = BatcherPhase::Test;
};

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

    TrainingExecutor(TrainingConfiguration config,
                     ResolvedExternalBatchers external_batchers);

    TrainingExecutor(TrainingConfiguration config,
                     std::unique_ptr<ISequenceBatcher> sequence_batcher,
                     std::vector<std::string> id_to_label);

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
    struct SequenceEvaluationMetrics {
        float loss = 0.0f;
        float accuracy = 0.0f;
        float entity_f1 = 0.0f;
        size_t token_count = 0;
    };

    // Three possible dataset backings. Exactly one of dataset_ / arrow_dataset_ /
    // parquet_dataset_ is populated at construction time, based on which
    // constructor was called. mode_ is the tag the Train() function uses
    // to pick the right batcher implementation.
    enum class DatasetMode {
        Legacy,   // DatasetHandle + legacy DatasetBatcher
        Arrow,    // ArrowDataset + ArrowDatasetBatcher
        External, // Image/Audio/Text IBatcher constructed by TrainingManager
        SequenceExternal, // Token-tagging ISequenceBatcher constructed upstream
        Parquet   // ParquetBackedDataset + ParquetArrowBatcher (disk-backed)
    };

    TrainingConfiguration config_;
    DatasetHandle dataset_;
    std::shared_ptr<ArrowDataset> arrow_dataset_;
    std::shared_ptr<ParquetBackedDataset> parquet_dataset_;
    std::string label_column_;
    DatasetMode mode_ = DatasetMode::Legacy;
    ResolvedExternalBatchers external_batchers_;
    std::unique_ptr<ISequenceBatcher> sequence_batcher_;
    std::vector<std::string> sequence_id_to_label_;

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
    ObjectiveEvaluationMetrics EvaluateArrowBatcher(IBatcher& batcher);

    /**
     * Run a single token-tagging epoch through an ISequenceBatcher.
     */
    void RunTrainingEpochSequence(
        ISequenceBatcher& batcher,
        int epoch,
        BatchCallback batch_cb
    );

    /**
     * Run validation for token-tagging batches.
     */
    void RunValidationSequence(ISequenceBatcher& batcher);
    SequenceEvaluationMetrics EvaluateSequenceBatcher(
        ISequenceBatcher& batcher);

    /**
     * Forward pass through the model
     * @param input Input tensor [batch_size, input_features]
     * @return Output tensor [batch_size, num_classes]
     */
    Tensor Forward(const Tensor& input);

    /**
     * Compute loss between predictions and targets
     */
    Tensor ComputeLossTensor(const Tensor& predictions, const Tensor& targets);
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
     * Accumulate current model gradients and step optimizer at the configured
     * grad_accum_steps boundary, or when force_step is true for a final partial
     * accumulation window.
     */
    bool AccumulateGradientsAndMaybeStep(
        int epoch,
        int batch_num,
        int total_batches,
        float batch_loss,
        float current_acc,
        float gradient_weight,
        const Tensor* device_gradient_weight,
        bool force_step
    );

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
    std::map<std::string, Tensor> gradient_accumulator_;
    int gradient_accumulated_batches_ = 0;
    float gradient_accumulation_weight_ = 0.0f;
    Tensor gradient_accumulation_device_weight_;
    bool gradient_accumulation_device_weight_initialized_ = false;
};

} // namespace cyxwiz
