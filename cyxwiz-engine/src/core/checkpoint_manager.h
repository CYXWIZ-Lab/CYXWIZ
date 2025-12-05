#pragma once

#include <cyxwiz/tensor.h>
#include <cyxwiz/optimizer.h>
#include <cyxwiz/sequential.h>
#include "training_executor.h"
#include <string>
#include <map>
#include <optional>
#include <filesystem>

namespace cyxwiz {

/**
 * Checkpoint metadata stored alongside model weights
 */
struct CheckpointMetadata {
    // Training state
    int epoch = 0;
    int global_step = 0;

    // Metrics at checkpoint time
    float train_loss = 0.0f;
    float train_accuracy = 0.0f;
    float val_loss = 0.0f;
    float val_accuracy = 0.0f;

    // Model architecture info
    std::string model_name;
    std::string optimizer_type;
    float learning_rate = 0.0f;

    // History
    std::vector<float> loss_history;
    std::vector<float> accuracy_history;
    std::vector<float> val_loss_history;
    std::vector<float> val_accuracy_history;

    // Timestamp
    std::string timestamp;
    std::string version = "1.0";
};

/**
 * CheckpointManager - Handles model checkpoint save/load operations
 *
 * Checkpoint format:
 * - Uses a directory structure with JSON metadata and binary tensor files
 * - checkpoint_dir/
 *     - metadata.json  (CheckpointMetadata as JSON)
 *     - model/         (model parameters)
 *         - layer0.weight.bin
 *         - layer0.bias.bin
 *         - ...
 *     - optimizer/     (optimizer state, optional)
 *         - state.bin
 */
class CheckpointManager {
public:
    /**
     * Create a checkpoint manager
     * @param checkpoint_dir Base directory for checkpoints
     */
    explicit CheckpointManager(const std::string& checkpoint_dir);

    ~CheckpointManager() = default;

    /**
     * Save a full checkpoint (model + optimizer + metrics)
     * @param model The model to save
     * @param optimizer The optimizer state to save (optional)
     * @param metrics Training metrics at checkpoint time
     * @param checkpoint_name Name for this checkpoint (e.g., "epoch_10", "best")
     * @return Path to saved checkpoint directory, or empty if failed
     */
    std::string SaveCheckpoint(
        const SequentialModel& model,
        const Optimizer* optimizer,
        const TrainingMetrics& metrics,
        const std::string& checkpoint_name = ""
    );

    /**
     * Load a checkpoint into existing model and optimizer
     * @param model Model to load parameters into
     * @param optimizer Optimizer to load state into (optional)
     * @param checkpoint_name Name of checkpoint to load, or empty for latest
     * @return Metadata from checkpoint, or nullopt if failed
     */
    std::optional<CheckpointMetadata> LoadCheckpoint(
        SequentialModel& model,
        Optimizer* optimizer = nullptr,
        const std::string& checkpoint_name = ""
    );

    /**
     * Save as "best" checkpoint if validation loss improved
     * @param model The model to save
     * @param optimizer The optimizer state
     * @param metrics Current training metrics
     * @param val_loss Current validation loss
     * @return true if this was the best model and was saved
     */
    bool SaveBestModel(
        const SequentialModel& model,
        const Optimizer* optimizer,
        const TrainingMetrics& metrics,
        float val_loss
    );

    /**
     * Get the best validation loss seen so far
     */
    float GetBestValLoss() const { return best_val_loss_; }

    /**
     * List all available checkpoints
     * @return Vector of checkpoint names sorted by time (newest first)
     */
    std::vector<std::string> ListCheckpoints() const;

    /**
     * Delete a checkpoint
     * @param checkpoint_name Name of checkpoint to delete
     * @return true if deleted successfully
     */
    bool DeleteCheckpoint(const std::string& checkpoint_name);

    /**
     * Get path to latest checkpoint
     * @return Path to latest checkpoint directory, or empty if none
     */
    std::string GetLatestCheckpoint() const;

    /**
     * Get path to best checkpoint
     * @return Path to best checkpoint directory, or empty if none
     */
    std::string GetBestCheckpoint() const;

    /**
     * Set maximum number of checkpoints to keep (older ones are deleted)
     * @param max_to_keep Number of checkpoints to keep (0 = unlimited)
     */
    void SetMaxToKeep(size_t max_to_keep) { max_to_keep_ = max_to_keep; }

    /**
     * Enable/disable auto-save of best model
     */
    void SetAutoSaveBest(bool enable) { auto_save_best_ = enable; }

private:
    std::filesystem::path checkpoint_dir_;
    float best_val_loss_ = std::numeric_limits<float>::infinity();
    size_t max_to_keep_ = 5;  // Keep last 5 checkpoints by default
    bool auto_save_best_ = true;

    // Internal helpers

    /**
     * Generate automatic checkpoint name based on epoch/step
     */
    std::string GenerateCheckpointName(const TrainingMetrics& metrics) const;

    /**
     * Save metadata to JSON file
     */
    bool SaveMetadata(const std::filesystem::path& dir, const CheckpointMetadata& metadata);

    /**
     * Load metadata from JSON file
     */
    std::optional<CheckpointMetadata> LoadMetadata(const std::filesystem::path& dir);

    /**
     * Save model parameters to directory
     */
    bool SaveModelParameters(const std::filesystem::path& dir, const SequentialModel& model);

    /**
     * Load model parameters from directory
     */
    bool LoadModelParameters(const std::filesystem::path& dir, SequentialModel& model);

    /**
     * Save a single tensor to binary file
     */
    bool SaveTensor(const std::filesystem::path& path, const Tensor& tensor);

    /**
     * Load a single tensor from binary file
     */
    std::optional<Tensor> LoadTensor(const std::filesystem::path& path);

    /**
     * Clean up old checkpoints to maintain max_to_keep limit
     */
    void CleanupOldCheckpoints();

    /**
     * Get checkpoint directory path for a given name
     */
    std::filesystem::path GetCheckpointPath(const std::string& name) const;
};

} // namespace cyxwiz
