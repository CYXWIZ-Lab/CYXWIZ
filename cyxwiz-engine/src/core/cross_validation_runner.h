#pragma once

// Cross-validation and learning curves that train the graph (TOFIX129 D4).
// The graph is compiled and its data prepared by Train's own launcher; this
// runner then trains it k times, each time with fresh weights on k-1 folds,
// and scores the fold it did not see. It uses the Engine's training, not a
// copy: each fold's rows go to the training batchers as the supplied
// training and validation tables (BuildResolvedTabularTrainingBatchers), a
// new TrainingExecutor trains them, and TestExecutor scores the fold with the
// test code (accuracy, macro F1, loss; MAE, RMSE, R-squared).
//
// The rows are the prepared table's, less the graph's own test split (read
// from the Engine's test batcher, so they stay unseen). Text labels are
// numbered once over the whole table, so every fold names the classes alike.
// A learning curve trains each fold on shares of its training rows (a seeded
// random subset) and scores both those rows and the fold.
//
// It never touches TrainingManager: the trained model in use, the Training
// Dashboard's runs and the checkpoints stay as they are.

#include "graph_compiler.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

class ArrowDataset;

struct CrossValidationSettings {
    int folds = 5;                     // 2 .. 20
    bool stratified = true;            // classes: keep each fold's class shares
    uint32_t seed = 42;                // folds and learning-curve subsets
    std::vector<double> train_shares;  // empty: cross-validation; else a learning curve over these (0, 1]
    // Given folds (1 .. folds per row of the rows used, in table order) instead
    // of drawing them: a column of fold numbers, or a test's fixed folds.
    std::vector<int> fold_of_row;
};

struct CrossValidationScores {
    double accuracy = 0.0, macro_f1 = 0.0, loss = 0.0;  // classes
    double mae = 0.0, rmse = 0.0, r2 = 0.0;             // numbers
};

// One training: a fold, at a share of its training rows (1 for cross-validation).
struct CrossValidationRun {
    int fold = 0;  // 1-based
    double share = 1.0;
    int64_t train_rows = 0, val_rows = 0;
    CrossValidationScores val;
    CrossValidationScores train;  // learning curve only: the rows it trained on
    int epochs = 0;
    double seconds = 0.0;
};

struct CrossValidationProgress {
    int run = 0, runs = 0;  // 1-based run of all
    int fold = 0, folds = 0;
    double share = 1.0;
    int epoch = 0, epochs = 0;
    bool regression = false;
};

struct CrossValidationCallbacks {
    std::function<bool()> should_cancel;
    std::function<void(const CrossValidationProgress&)> on_progress;
    std::function<void(const CrossValidationRun&)> on_run;  // each training as it finishes
};

struct CrossValidationOutcome {
    bool ok = false;
    bool cancelled = false;
    std::string error;
    bool regression = false;
    std::string target;                    // the label column
    std::vector<std::string> class_names;  // classes, when text
    int64_t rows = 0;                      // rows used (the table less the graph's test rows)
    int64_t test_rows_left_out = 0;
    int folds = 0;
    std::vector<CrossValidationRun> runs;  // fold order; a learning curve share by share
    double seconds = 0.0;
};

// `config` is the compiled graph as Train's launcher hands it on (dataset roles
// and label set), `dataset` the prepared table it trains from. Blocking: run it
// on a task.
CrossValidationOutcome RunCrossValidation(const TrainingConfiguration& config, const std::shared_ptr<ArrowDataset>& dataset,
                                         const std::string& label_column, int epochs, int batch_size,
                                         const CrossValidationSettings& settings,
                                         const CrossValidationCallbacks& callbacks = {});

// Mean and (sample) standard deviation of one score over the runs at `share`.
struct ScoreSummary {
    double mean = 0.0, sd = 0.0, low = 0.0, high = 0.0;
    int count = 0;
};
ScoreSummary SummariseRuns(const std::vector<CrossValidationRun>& runs, double share,
                           double (*pick)(const CrossValidationRun&));

}  // namespace cyxwiz
