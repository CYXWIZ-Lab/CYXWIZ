#pragma once

// One group-training worker's model, kept across rounds.
//
// A graph training job builds its model, loss and optimizer once (Open). Each
// round then starts from the group's state, trains on the rows the round
// leases until its budget is reached, and returns the update
// (state at the start minus state at the end) over the exchanged values of
// the model's layout. The optimizer's own state (Adam's moments) stays in the
// session between rounds. Validation, checkpoints and epochs belong to the
// group's coordinator, not to a round.
//
// Every call must come from the thread that opened the session, and the host
// must keep the same compute device selected for all of them.

#include "graph_training_job.h"
#include "model_state_layout.h"
#include "training_executor.h"
#include "training_failure.h"
#include "training_scheduler_controller.h"

#include <cyxwiz/optimizers/sgd.h>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace cyxwiz {

class ArrowDataset;

// Part of a data file: `rows` rows from `row_offset` (0 rows: to the end).
struct RowRange {
    std::filesystem::path file;
    uint64_t row_offset = 0;
    uint64_t rows = 0;
};

struct RoundData {
    // Trained in this order, without shuffling, so `rows_done` says exactly
    // which rows a round that ran out of time has used.
    std::vector<RowRange> leases;
};

// What a worker needs to know about its group to open a session.
struct RoundSessionOptions {
    // This worker's number in its group: mixed into the graph's model seed,
    // so the members of a group do not draw the same dropout masks, while
    // one worker's rounds stay reproducible.
    uint32_t worker = 0;
    // The length of the job in optimizer updates of one node training alone
    // (rows per epoch / batch size / accumulation x epochs). Needed when the
    // optimizer node has a schedule; 0 when the job did not say.
    uint64_t schedule_total_steps = 0;
};

// Where the group is when a round starts. The learning rate follows it, not
// this worker's own step count: every member is at the same point of the
// schedule whatever its speed, and a replacement starts where the group is.
struct RoundPosition {
    uint64_t round = 1;         // 1, 2, ...: seeds follow it
    uint64_t global_epoch = 0;  // epochs the group has completed
    uint64_t global_step = 0;   // samples the group has trained / batch size / accumulation
};

struct RoundBudget {
    double train_seconds = 0.0;  // 0: no time limit
    uint64_t max_steps = 0;      // optimizer steps; 0: no limit
};

struct RoundCallbacks {
    // Polled between batches; true ends the round as cancelled.
    std::function<bool()> should_stop;
    std::function<void(uint64_t steps, uint64_t samples, double loss)> on_progress;
};

struct RoundOutcome {
    std::vector<float> delta;         // layout.exchanged_count values
    uint64_t steps = 0;
    uint64_t samples = 0;
    double last_loss = 0.0;
    double train_seconds = 0.0;
    double learning_rate = 0.0;       // what the round trained with
    std::vector<uint64_t> rows_done;  // per lease: rows trained, from its start
};

struct RoundFailure {
    TrainingFailureKind kind = TrainingFailureKind::None;
    std::string text;
};

class RoundTrainingSession {
public:
    // `request.dataset_files` maps the graph's training Data Input to one
    // file of the job's data (any part of it): the graph is compiled against
    // its columns. The model is built here. Null with `failure` set when the
    // graph cannot train in rounds.
    static std::unique_ptr<RoundTrainingSession> Open(const GraphTrainingJobRequest& request,
                                                      const RoundSessionOptions& options, RoundFailure& failure);
    ~RoundTrainingSession();

    const ModelStateLayout& Layout() const { return layout_; }
    int BatchSize() const { return batch_size_; }

    // The model as built: the full state, frozen values included (what the
    // group's first worker sends as the starting point).
    const std::vector<float>& InitialState() const { return initial_state_; }

    // Another worker's starting point: the full state, applied when the next
    // round starts. Refused unless it has the layout's full length.
    bool LoadInitialState(std::span<const float> full_state, std::string& error);

    // One round from `theta` (layout.exchanged_count values). The learning
    // rate is the schedule's at `position` and stays there for the round.
    bool TrainRound(const RoundPosition& position, std::span<const float> theta, const RoundData& data, const RoundBudget& budget,
                    const RoundCallbacks& callbacks, RoundOutcome& outcome, RoundFailure& failure);

private:
    RoundTrainingSession();
    std::shared_ptr<ArrowDataset> LoadLeases(const RoundData& data, std::vector<uint64_t>& lease_rows,
                                             RoundFailure& failure);

    // One round's batchers over `dataset` handed to the executor.
    bool RunRound(const TrainingExecutor::RoundRun& run, const std::shared_ptr<ArrowDataset>& dataset,
                  TrainingExecutor::RoundRunResult& result, std::string& error);

    // The schedule's value at the group's position; false when it cannot be evaluated.
    bool LearningRateAt(const RoundPosition& position, double& learning_rate, std::string& error);

    // The graph's schedule on a clock of its own: an optimizer that trains
    // nothing, stepped to the group's position.
    std::optional<TrainingSchedulerSpec> schedule_;
    std::unique_ptr<SGDOptimizer> schedule_optimizer_;
    std::unique_ptr<TrainingSchedulerController> schedule_clock_;
    double base_learning_rate_ = 0.0;

    std::unique_ptr<TrainingExecutor> executor_;
    bool sequence_ = false;  // a language model on token windows
    TrainingConfiguration round_config_;  // the job's, with every row a training row, in order
    std::string label_column_;
    std::string dataset_name_;
    int batch_size_ = 1;
    ModelStateLayout layout_;
    std::vector<float> initial_state_;
    std::vector<float> pending_full_state_;
    // The files of the last round, kept loaded: a worker stays on its chunk.
    std::vector<std::pair<std::string, std::shared_ptr<ArrowDataset>>> loaded_files_;
};

}  // namespace cyxwiz
