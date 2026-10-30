#pragma once

#include <cyxwiz/tensor.h>
#include <cyxwiz/optimizer.h>
#include <cyxwiz/sequential.h>
#include "training_executor.h"
#include "checkpoint_tokenizer.h"
#include <string>
#include <map>
#include <optional>
#include <filesystem>
#include <limits>

namespace cyxwiz {

class IExecutableModel;

/**
 * Checkpoint metadata stored alongside model weights
 */
// The Data Split a run trained with (absent in older checkpoints). Run Test
// reads the test part from it, so a model is never tested on rows or images
// it trained on when the graph's Data Split changed since.
struct CheckpointDataSplit {
    bool known = false;
    float train = 0.8f, val = 0.1f, test = 0.1f;
    bool stratified = false;
    int seed = 42;
    std::vector<std::string> excluded_files;  // images a Quality node left out
};

struct CheckpointMetadata {
    TrainingRandomness randomness; // Seed provenance, not random-stream continuation.
    // Training state
    int epoch = 0;
    int global_step = 0;

    // Metrics at checkpoint time
    float train_loss = 0.0f;
    float train_accuracy = 0.0f;
    float train_mae = 0.0f;
    float train_rmse = 0.0f;
    float val_loss = 0.0f;
    float val_accuracy = 0.0f;
    float val_mae = 0.0f;
    float val_rmse = 0.0f;

    // Model architecture info
    std::string model_name;
    std::string optimizer_type;
    float learning_rate = 0.0f;

    // History
    std::vector<float> loss_history;
    std::vector<float> accuracy_history;
    std::vector<float> mae_history;
    std::vector<float> rmse_history;
    std::vector<float> val_loss_history;
    std::vector<float> val_accuracy_history;
    std::vector<float> val_mae_history;
    std::vector<float> val_rmse_history;

    // Timestamp
    std::string timestamp;
    std::string version = "1.0";

    // Provenance (optional; empty/0 in checkpoints written before TOFIX143)
    std::string tokenizer_sha256;   // sha256 of the training token-window vocabulary
    int context = 0;                // training sequence length
    std::string initialized_from;   // warm-start checkpoint this run started from
    CheckpointDataSplit split;      // the Data Split training used
};

/**
 * Read-only capability classification for one checkpoint.
 *
 * Loading weights for testing, warm-starting, and exact resume are distinct
 * operations. Callers must not infer exact-resume support from the mere
 * presence of model parameters or a learning rate.
 */
struct CheckpointInspection {
    bool valid = false;
    std::string format_version;
    bool can_load_for_testing = false;
    bool can_warm_start = false;
    bool can_exact_resume = false;
    std::string exact_resume_reason;
    std::string error;
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
    // Model: SequentialModel or IExecutableModel (a graph with branches,
    // TOFIX157); only its parameter map is read or written.
    template <typename Model>
    std::string SaveCheckpoint(
        const Model& model,
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
    template <typename Model>
    std::optional<CheckpointMetadata> LoadCheckpoint(
        Model& model,
        Optimizer* optimizer = nullptr,
        const std::string& checkpoint_name = ""
    );

    /**
     * Inspect checkpoint format and supported operations without mutating a
     * model or optimizer. Version 1 checkpoints are intentionally classified
     * as test-load/warm-start only.
     */
    CheckpointInspection InspectCheckpoint(
        const std::string& checkpoint_name = ""
    );

    /** Detailed failure from the most recent save/load operation. */
    const std::string& GetLastError() const { return last_error_; }

    /**
     * Save as "best" checkpoint if validation loss improved
     * @param model The model to save
     * @param optimizer The optimizer state
     * @param metrics Current training metrics
     * @param val_loss Current validation loss
     * @return true if this was the best model and was saved
     */
    template <typename Model>
    bool SaveBestModel(
        const Model& model,
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

    /** The run's tokenizer, kept in every checkpoint saved after this call
     *  (<checkpoint>/tokenizer/, TOFIX159); none when it is empty. */
    void SetTokenizer(CheckpointTokenizer tokenizer) { tokenizer_ = std::move(tokenizer); }

    /** Provenance written into every checkpoint saved after this call. */
    /** The Data Split written into every checkpoint saved after this call. */
    void SetDataSplit(CheckpointDataSplit split) { provenance_.split = std::move(split); }

    void SetProvenance(std::string tokenizer_sha256, int context, std::string initialized_from) {
        provenance_.tokenizer_sha256 = std::move(tokenizer_sha256);
        provenance_.context = context;
        provenance_.initialized_from = std::move(initialized_from);
    }

private:
    std::filesystem::path checkpoint_dir_;
    float best_val_loss_ = std::numeric_limits<float>::infinity();
    size_t max_to_keep_ = 5;  // Keep last 5 checkpoints by default
    bool auto_save_best_ = true;
    std::string last_error_;
    CheckpointMetadata provenance_;  // only the provenance fields are used
    CheckpointTokenizer tokenizer_;

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
    template <typename Model>
    bool SaveModelParameters(const std::filesystem::path& dir, const Model& model);

    /**
     * Load model parameters from directory
     */
    template <typename Model>
    bool LoadModelParameters(const std::filesystem::path& dir, Model& model);

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

/**
 * Warm start (TOFIX143): load only the model parameters of one checkpoint
 * folder into a new run's model. Optimizer, schedule, RNG, data cursor and
 * epochs stay fresh. Names, shapes and dtypes must match exactly. When
 * expected_tokenizer_sha256 is set (language-model runs) the checkpoint's
 * tokenizer must match it: from metadata.json, or for checkpoints written
 * before TOFIX143 from the same run's generation_previews/*.json.
 */
template <typename Model>
std::optional<CheckpointMetadata> LoadWarmStartCheckpoint(
    const std::filesystem::path& checkpoint,
    Model& model,
    const std::string& expected_tokenizer_sha256,
    std::string& error);

} // namespace cyxwiz
