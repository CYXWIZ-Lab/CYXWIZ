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
#include "training_executor.h"

#include <functional>
#include <map>
#include <memory>
#include <string>

namespace cyxwiz {

struct GraphTrainingJobRequest {
    std::string graph_json;
    // Optional: Data Input dataset_name -> local file, replacing the node's
    // file_path (a node maps the job's datasets onto its own storage).
    std::map<std::string, std::string> dataset_files;
    int epochs_override = 0;       // 0 = the graph's Data Loader
    int batch_size_override = 0;   // 0 = the graph's Data Loader
    std::string checkpoint_dir_override;
};

struct GraphTrainingJobCallbacks {
    // Once the graph is compiled and prepared, before the first batch.
    std::function<void(int epochs, int batch_size)> on_start;
    BatchCallback on_batch;
    EpochCallback on_epoch;
    // Polled between batches; true stops the run (reported as cancelled).
    std::function<bool()> should_cancel;
};

struct GraphTrainingJobResult {
    bool ok = false;
    bool cancelled = false;
    std::string error;            // why the job did not train (or failed)
    TrainingMetrics metrics;      // final metrics when it ran
    std::unique_ptr<SequentialModel> model;  // trained model (for export)
};

// Loads the graph's inputs, compiles, trains to completion on the calling
// thread and returns the outcome.
GraphTrainingJobResult RunGraphTrainingJob(const GraphTrainingJobRequest& request,
                                           const GraphTrainingJobCallbacks& callbacks = {});

}  // namespace cyxwiz
