#pragma once

#include "../api_export.h"
#include "process_group.h"
#include "ddp.h"
#include "distributed_sampler.h"
#include "../sequential.h"
#include "../loss.h"
#include "../optimizer.h"
#include "../tensor.h"
#include <functional>
#include <string>
#include <vector>
#include <chrono>

namespace cyxwiz {

/**
 * Configuration for distributed training
 */
struct CYXWIZ_API DistributedTrainingConfig {
    /// Number of training epochs
    int epochs = 10;

    /// Per-GPU batch size (effective batch size = batch_size * world_size)
    size_t batch_size = 64;

    /// Shuffle training data each epoch
    bool shuffle = true;

    /// Random seed for shuffling (deterministic across ranks)
    unsigned int seed = 42;

    /// Only rank 0 saves checkpoints (avoids duplicate I/O)
    bool save_on_master_only = true;

    /// Save checkpoint every N epochs (0 = disabled)
    int checkpoint_every_n_epochs = 0;

    /// Directory for checkpoints
    std::string checkpoint_dir = "./checkpoints";

    /// Print training progress
    bool verbose = true;

    /// Print progress every N batches (0 = only at epoch end)
    int log_every_n_batches = 0;

    /// Validation data ratio (0 = no validation)
    float validation_split = 0.0f;
};

/**
 * Training history with metrics from distributed training
 */
struct CYXWIZ_API DistributedTrainingHistory {
    /// Training loss per epoch (averaged across all ranks)
    std::vector<float> train_losses;

    /// Training accuracy per epoch (if applicable)
    std::vector<float> train_accuracies;

    /// Validation loss per epoch
    std::vector<float> val_losses;

    /// Validation accuracy per epoch
    std::vector<float> val_accuracies;

    /// Total training time in seconds
    double total_time_seconds = 0.0;

    /// Throughput: samples processed per second
    double samples_per_second = 0.0;

    /// Effective batch size (batch_size * world_size)
    size_t effective_batch_size = 0;

    /// Number of ranks used
    int world_size = 1;
};

/**
 * Callback types for training events
 */
using EpochCallback = std::function<void(int epoch, float loss, float accuracy)>;
using BatchCallback = std::function<void(int batch, int total_batches, float loss)>;

/**
 * DistributedTrainer - High-level trainer for distributed data parallel training
 *
 * Combines DDP, DistributedSampler, and training loop into a simple interface.
 * Handles:
 *   - Data distribution across ranks
 *   - Gradient synchronization
 *   - Metric aggregation across ranks
 *   - Checkpointing (master only by default)
 *
 * Usage:
 *   init_distributed();
 *
 *   SequentialModel model;
 *   model.Add<LinearModule>(784, 256);
 *   model.Add<ReLUModule>();
 *   model.Add<LinearModule>(256, 10);
 *
 *   auto loss = std::make_unique<CrossEntropyLoss>();
 *   auto optimizer = std::make_unique<AdamOptimizer>(0.001);
 *
 *   DistributedTrainer trainer(&model, loss.get(), optimizer.get());
 *
 *   DistributedTrainingConfig config;
 *   config.epochs = 10;
 *   config.batch_size = 64;
 *
 *   auto history = trainer.Fit(X_train, y_train, config);
 *
 *   if (trainer.IsMaster()) {
 *       // Save model, print final results, etc.
 *   }
 *
 *   finalize_distributed();
 */
class CYXWIZ_API DistributedTrainer {
public:
    /**
     * Create a distributed trainer
     *
     * @param model Model to train (wrapped in DDP internally)
     * @param loss Loss function
     * @param optimizer Optimizer
     * @param pg Process group (uses default if nullptr)
     */
    DistributedTrainer(SequentialModel* model, Loss* loss, Optimizer* optimizer,
                       ProcessGroup* pg = nullptr);

    ~DistributedTrainer();

    // Non-copyable
    DistributedTrainer(const DistributedTrainer&) = delete;
    DistributedTrainer& operator=(const DistributedTrainer&) = delete;

    /**
     * Train the model
     *
     * @param X_train Training features [num_samples, ...]
     * @param y_train Training labels [num_samples, ...]
     * @param config Training configuration
     * @return Training history with metrics
     */
    DistributedTrainingHistory Fit(const Tensor& X_train, const Tensor& y_train,
                                   const DistributedTrainingConfig& config);

    /**
     * Train with separate validation data
     */
    DistributedTrainingHistory Fit(const Tensor& X_train, const Tensor& y_train,
                                   const Tensor& X_val, const Tensor& y_val,
                                   const DistributedTrainingConfig& config);

    /**
     * Evaluate model on test data
     *
     * @param X_test Test features
     * @param y_test Test labels
     * @return Pair of (loss, accuracy) averaged across all ranks
     */
    std::pair<float, float> Evaluate(const Tensor& X_test, const Tensor& y_test);

    /**
     * Check if this is the master rank (rank 0)
     */
    bool IsMaster() const;

    /**
     * Get current rank
     */
    int GetRank() const;

    /**
     * Get world size
     */
    int GetWorldSize() const;

    /**
     * Save model checkpoint (only on master by default)
     */
    void SaveCheckpoint(const std::string& path);

    /**
     * Load model checkpoint (all ranks load)
     */
    void LoadCheckpoint(const std::string& path);

    /**
     * Set callback for epoch end
     */
    void SetEpochCallback(EpochCallback callback) { epoch_callback_ = callback; }

    /**
     * Set callback for batch end
     */
    void SetBatchCallback(BatchCallback callback) { batch_callback_ = callback; }

    /**
     * Get the DDP wrapper
     */
    DistributedDataParallel* GetDDP() { return ddp_.get(); }

    /**
     * Get the underlying model
     */
    SequentialModel* GetModel() { return model_; }

private:
    std::unique_ptr<DistributedDataParallel> ddp_;
    SequentialModel* model_;
    Loss* loss_;
    Optimizer* optimizer_;
    ProcessGroup* process_group_;

    // Callbacks
    EpochCallback epoch_callback_;
    BatchCallback batch_callback_;

    /**
     * Aggregate a metric value across all ranks
     * Uses AllReduce with AVERAGE
     */
    float AggregateMetric(float local_value);

    /**
     * Aggregate sample count across all ranks
     * Uses AllReduce with SUM
     */
    size_t AggregateSampleCount(size_t local_count);

    /**
     * Extract a batch from data tensors
     */
    std::pair<Tensor, Tensor> ExtractBatch(const Tensor& X, const Tensor& y,
                                           const std::vector<size_t>& indices);

    /**
     * Run one training epoch
     */
    std::pair<float, float> TrainEpoch(const Tensor& X_train, const Tensor& y_train,
                                       DistributedSampler& sampler,
                                       const DistributedTrainingConfig& config,
                                       int epoch);

    /**
     * Run evaluation on validation data
     */
    std::pair<float, float> ValidateEpoch(const Tensor& X_val, const Tensor& y_val,
                                          const DistributedTrainingConfig& config);

    /**
     * Compute accuracy from predictions and targets
     */
    float ComputeAccuracy(const Tensor& predictions, const Tensor& targets);
};

} // namespace cyxwiz
