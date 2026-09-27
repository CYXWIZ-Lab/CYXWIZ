#pragma once

// Headless training of a saved graph with the shared core (TOFIX118 P2): the
// path a Server Node runs for a training job, equivalent to the Engine's
// Train for graphs whose Data Inputs feed the loader directly.
//
// Supported: Data Input files in Parquet or Arrow IPC; sequence (token
// window / causal-LM) graphs with supplied train/validation/test roles, and
// tabular graphs. Refused with a reason (fail closed, never skipped): graphs
// with preprocessing or data-transform nodes on the data path, other input
// kinds (image/audio/text folders, CSV), or a graph that does not compile.

#include "graph_document.h"
#include "graph_training_job_timing.h"
#include "training_failure.h"
#include "training_executor.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

struct GraphTrainingJobRequest {
    std::string graph_json;
    // Optional: Data Input dataset_name -> local file, replacing the node's
    // file_path (a node maps the job's datasets onto its own storage).
    std::map<std::string, std::string> dataset_files;
    int epochs_override = 0;       // 0 = the graph's Data Loader
    int batch_size_override = 0;   // 0 = the graph's Data Loader
    std::string checkpoint_dir_override;
    // Resume checkpoints (TOFIX118 P4e): written at every epoch end under
    // this folder when set; resume_from continues a run from a checkpoint
    // folder, or from the newest one under resume_checkpoint_root when it is
    // "latest".
    std::string resume_checkpoint_root;
    std::string resume_from;
    // > 0: language-model epochs are also checkpointed every that many
    // optimizer steps (P4e-2); 0: epoch ends only.
    int resume_checkpoint_every_steps = 0;
};

struct GraphTrainingJobCallbacks {
    // Once the graph is compiled and prepared, before the first batch.
    std::function<void(int epochs, int batch_size)> on_start;
    BatchCallback on_batch;
    EpochCallback on_epoch;
    // Polled between batches; true stops the run (reported as cancelled).
    std::function<bool()> should_cancel;
    // Polled likewise; true holds training between batches until it turns
    // false again (TOFIX118 P4d).
    std::function<bool()> should_pause;
};

struct GraphTrainingJobResult {
    bool ok = false;
    bool cancelled = false;
    std::string error;            // why the job did not train (or failed)
    TrainingFailureKind failure = TrainingFailureKind::None;  // category of `error`
    TrainingMetrics metrics;      // final metrics when it ran
    GraphTrainingJobTiming timing;
    // Trained model (for export); on cancel, the partially trained model.
    std::unique_ptr<SequentialModel> model;
};

// Loads the graph's inputs, compiles, trains to completion on the calling
// thread and returns the outcome.
GraphTrainingJobResult RunGraphTrainingJob(const GraphTrainingJobRequest& request,
                                           const GraphTrainingJobCallbacks& callbacks = {});

// Remote jobs ship whole dataset files to the node (owner decision
// 2026-09-26): the Data Input datasets a graph trains from, and the supplied
// Test inputs (Data Split's Test role) that never leave the host.
struct GraphJobDatasetPlan {
    std::vector<std::string> ship;          // dataset_name of each training/validation input
    std::vector<std::string> kept_private;  // dataset_name of each supplied Test input
};

// False with a reason for a graph RunGraphTrainingJob would refuse
// (preparation steps, subgraphs, unreadable) - checked before anything is sent.
bool PlanGraphJobDatasets(const std::string& graph_json, GraphJobDatasetPlan& plan, std::string& error);

// The job's graph with each shipped Data Input reading its local file and the
// `drop` inputs (with their links) removed. Every remaining Data Input needs
// a file. The rest of the document is unchanged.
bool BindGraphJobDatasets(const std::string& graph_json, const std::map<std::string, std::string>& files,
                          const std::vector<std::string>& drop, std::string& bound_json, std::string& error);

}  // namespace cyxwiz
