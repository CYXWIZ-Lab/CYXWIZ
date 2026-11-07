#pragma once

// One trial of a hyperparameter or architecture search (TOFIX178): the graph
// trained once with some node properties changed, scored on its validation
// part. The run is built from Train's recipe for the prepared data
// (training_recipe.h) on a fresh TrainingExecutor, so it reads the data the
// way Train does for every kind of data, and never touches TrainingManager
// (the model in use, the Training Dashboard's runs, the project's
// checkpoints: a trial's go to a scratch folder removed after).
//
// A trial changes properties of the nodes that shape the model and its
// training: a compiled layer's (units, filters, dropout rate, ...), the
// optimizer's (learning rate, weight decay, ...), the scheduler's, and the
// DataLoader's batch size and epochs. The edited graph is compiled again and
// its model, optimizer and training-loop settings replace the launched
// configuration's; everything Train's launcher worked out about the data
// (datasets, label column, input width, splits) stays as launched.

#include "graph_compiler.h"
#include "training_recipe.h"

#include <functional>
#include <string>
#include <vector>

namespace cyxwiz {

struct TrialChange {
    int node_id = -1;
    std::string property;
    std::string value;
};

struct TrialSetup {
    std::vector<gui::MLNode> nodes;
    std::vector<gui::NodeLink> links;
    TrainingConfiguration launched;  // as Train's launcher handed it to the dispatch
    TrainingRecipe recipe;           // how that prepared data trains
};

struct TrialScore {
    bool regression = false;
    bool has_validation = false;
    float val_loss = 0.0f, val_accuracy = 0.0f;  // classes: accuracy 0..1
    float val_mae = 0.0f, val_rmse = 0.0f;       // numbers
    float train_loss = 0.0f, train_accuracy = 0.0f;
};

struct TrialProgress {
    int epoch = 0, epochs = 0;  // 1-based epoch just finished
    float train_loss = 0.0f, val_loss = 0.0f, val_accuracy = 0.0f;
};

struct TrialCallbacks {
    std::function<bool()> should_cancel;  // checked between batches
    std::function<void(const TrialProgress&)> on_epoch;
};

struct TrialOutcome {
    bool ok = false;
    bool cancelled = false;
    std::string error;
    TrialScore score;
    int epochs = 0;  // epochs run
    double seconds = 0.0;
};

// Whether `property` of `node` can be a trial's change; `why` says why not.
bool IsTrialTarget(const TrainingConfiguration& launched, const gui::MLNode& node, const std::string& property,
                   std::string* why = nullptr);

// The launched configuration with the changed graph's model, optimizer and
// training-loop settings. False with `error` when a change names no target
// or the changed graph does not compile.
bool TrialConfiguration(const TrialSetup& setup, const std::vector<TrialChange>& changes, TrainingConfiguration& out,
                        std::string& error);

// Blocking: run it on a task.
TrialOutcome RunTrial(const TrialSetup& setup, const std::vector<TrialChange>& changes,
                      const TrialCallbacks& callbacks = {});

}  // namespace cyxwiz
