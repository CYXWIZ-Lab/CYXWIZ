#pragma once

#include "graph_compiler.h"
#include "dataset_batcher.h"
#include "data_registry.h"
#include <cyxwiz/tensor.h>
#include <cyxwiz/optimizer.h>
#include <cyxwiz/layers/linear.h>
#include <cyxwiz/activations/relu.h>
#include <cyxwiz/losses/cross_entropy.h>
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
using BatchCallback = std::function<void(int epoch, int batch, float loss, float accuracy)>;
using EpochCallback = std::function<void(int epoch, float train_loss, float train_acc,
                                          float val_loss, float val_acc, float epoch_time)>;
using TrainingCompleteCallback = std::function<void(const TrainingMetrics& final_metrics)>;

/**
 * TrainingExecutor - Executes ML training based on compiled graph configuration
 *
 * This class handles the actual training loop:
 * - Builds the model from TrainingConfiguration
 * - Creates optimizer
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

private:
    TrainingConfiguration config_;
    DatasetHandle dataset_;

    // Thread safety
    std::atomic<bool> is_training_{false};
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> is_paused_{false};

    mutable std::mutex metrics_mutex_;
    TrainingMetrics metrics_;

    // Training components
    std::unique_ptr<Optimizer> optimizer_;

    // Internal training methods

    /**
     * Initialize the training components
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
     * Backward pass and parameter update
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

    // MLP Model using cyxwiz-backend layers
    struct BackendModel {
        // Layers
        std::unique_ptr<LinearLayer> fc1;  // Input -> Hidden
        std::unique_ptr<ReLU> relu1;       // Activation
        std::unique_ptr<LinearLayer> fc2;  // Hidden -> Output

        // Loss function
        std::unique_ptr<CrossEntropyLoss> loss_fn;

        // Cached tensors for backward pass
        Tensor input_cache;
        Tensor fc1_output;
        Tensor relu_output;
        Tensor fc2_output;

        size_t input_size = 0;
        size_t hidden_size = 128;
        size_t output_size = 0;

        void Initialize(size_t input, size_t hidden, size_t output);
        Tensor Forward(const Tensor& input);
        float ComputeLoss(const Tensor& predictions, const Tensor& targets);
        void Backward(const Tensor& predictions, const Tensor& targets);
        void UpdateWeights(float learning_rate);
    };

    std::unique_ptr<BackendModel> model_;
};

} // namespace cyxwiz
