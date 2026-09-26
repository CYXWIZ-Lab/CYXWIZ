#pragma once

// Resume checkpoints (TOFIX118 P4e): what a training run needs to continue
// exactly where an epoch ended - model parameters, optimizer state, the
// run's counters, metric histories, early-stopping and scheduler state - in
// the v2 checkpoint layout (manifest + verified payloads). Randomness and
// data order are not stored as generator state: each epoch reseeds them from
// (run seed, epoch), so storing the seeds is enough.

#include "checkpoint_manifest.h"
#include "training_scheduler_controller.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

class IExecutableModel;
class Optimizer;

struct TrainingResumeRuntime {
    int completed_epoch = 0;
    int total_epochs = 0;
    int optimizer_step_count = 0;
    int scheduler_step_count = 0;
    float best_val_loss = 0.0f;       // +inf when no validation improved yet
    bool best_val_loss_set = false;
    int epochs_without_improvement = 0;
    std::int64_t model_seed = -1;      // < 0: unseeded run (resume not exact)
    std::int64_t dataloader_seed = -1;
    bool data_order_exact = false;     // every batcher replays its epoch order
    std::vector<float> loss_history, accuracy_history, mae_history, rmse_history;
    std::vector<float> val_loss_history, val_accuracy_history, val_mae_history, val_rmse_history;
    std::vector<double> learning_rate_history;
    std::optional<TrainingSchedulerResumeState> scheduler;
};

// Who the checkpoint belongs to; resume refuses a different graph or data.
struct TrainingResumeIdentity {
    std::string run_id;
    std::string graph_json;             // snapshot stored with the checkpoint
    std::string dataset_manifest_json;  // the data files it trained on
    std::string graph_fingerprint;      // SHA-256 of graph_json
    std::string dataset_fingerprint;
    std::string partition_fingerprint;
    std::string model_type = "executable_model";
    std::string optimizer_type;
    std::string loss_type;
    std::string scheduler_type;         // empty: no scheduler
};

// Writes `directory` (must not exist yet) and publishes its manifest last,
// so a checkpoint is either complete or absent.
bool SaveTrainingResumeCheckpoint(const std::filesystem::path& directory, const TrainingResumeIdentity& identity,
                                  IExecutableModel& model, const Optimizer& optimizer,
                                  const TrainingResumeRuntime& runtime, std::string& error);

// Verifies every payload's SHA-256, then restores the model and optimizer
// (both already built for this graph) and returns the runtime state.
bool LoadTrainingResumeCheckpoint(const std::filesystem::path& directory, IExecutableModel& model,
                                  Optimizer& optimizer, TrainingResumeRuntime& runtime,
                                  TrainingResumeIdentity& identity, std::string& error);

// The newest complete resume checkpoint under `root` (epoch-NNNN folders).
std::optional<std::filesystem::path> FindLatestTrainingResumeCheckpoint(const std::filesystem::path& root);

// Keeps the newest `keep` epoch checkpoints under `root`.
void PruneTrainingResumeCheckpoints(const std::filesystem::path& root, int keep);

// Epoch seed derived from a run seed (SplitMix64 of seed and epoch).
std::uint64_t TrainingEpochSeed(std::uint64_t run_seed, int epoch);

}  // namespace cyxwiz
