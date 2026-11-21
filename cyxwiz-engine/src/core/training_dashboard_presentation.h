#pragma once

// What the Training Dashboard says, without ImGui (TOFIX129 B4, approved
// mockup 2026-11-18): the metric cards, the "How the run is going" rows and
// the time words. The panel (gui/panels/training_plot_panel.cpp) draws them;
// test_training_dashboard_presentation checks them on a real run's numbers.

#include "missing_columns.h"
#include "training_run_comparison_record.h"

#include <optional>
#include <string>
#include <vector>

namespace cyxwiz::training_dashboard {

// A curve: x is the epoch (a fraction inside an epoch for batch points).
struct Series {
    std::vector<double> x;
    std::vector<double> y;
    bool Empty() const { return x.empty() || y.empty(); }
    size_t Size() const { return x.size() < y.size() ? x.size() : y.size(); }
};

// A chart point: x is the epoch position, y the value.
struct Point {
    double x = 0.0;
    double y = 0.0;
};

// What the panel knows about the run.
struct RunFacts {
    bool training = false;
    // Training: the latest batch point (x = epoch - 1 + batch / batches; its
    // value is the mean of the batches since the point before, not the
    // epoch's running mean) and one value per finished epoch (x = the epoch).
    std::optional<Point> latest_loss_point;
    std::optional<Point> latest_accuracy_point;
    Series train_loss_epochs;
    Series train_accuracy_epochs;
    // Validation: one value per finished epoch.
    Series val_loss;
    Series val_accuracy;
    // Progress and time.
    int current_epoch = 0;
    int total_epochs = 0;
    int last_executed_epoch = 0;
    int current_batch = 0;
    int total_batches = 0;
    std::optional<double> remaining_seconds;  // the ETA, once it has a rate
    double batches_per_second = 0.0;
    double samples_per_second = 0.0;
    double avg_epoch_seconds = 0.0;
    std::optional<double> epoch_end_seconds;  // measured time of an epoch's end (validation, previews, checkpoint)
    double total_seconds = 0.0;
};

enum class Tone { Good, Bad, Error, Plain, Faint };

// One metric card: its name, the value, the change since the epoch before,
// then up to two more lines.
struct Card {
    std::string id;      // a stable ImGui id
    std::string label;   // "Loss · epoch 10"
    int series = -1;     // which curve's colour: 0 train loss, 1 val loss, 2 train acc, 3 val acc, -1 time
    std::string value;   // "0.2223"
    std::string tooltip;
    std::string change;  // "0.0376 since epoch 9" (the caret is drawn by the panel)
    int direction = 0;   // -1 down, +1 up, 0 none
    Tone change_tone = Tone::Faint;
    std::string line2;
    Tone line2_tone = Tone::Faint;
    std::string line3;   // "latest batches 0.5287"
    Tone line3_tone = Tone::Plain;
};
std::vector<Card> BuildCards(const RunFacts& facts);

// "How the run is going": rows in plain words and the suggestion.
struct InsightRow {
    std::string label;
    std::string value;
    Tone tone = Tone::Plain;
};
struct Insights {
    std::string suggestion;  // "Suggested: keep going"
    Tone suggestion_tone = Tone::Good;
    std::vector<InsightRow> rows;
};
Insights BuildInsights(const RunFacts& facts);

// The header card (B4-4): the state, where the run ran and on what data,
// how it ended and which model is active, in plain words; the codes and
// paths go under "Run details".
enum class RunState {
    Idle,
    WaitingForGpu,
    Training,
    Paused,
    PreparationFailed,
    Preparing,
    EarlyStopped,
    Stopped,
    Failed,
    Completed,
    ModelLoaded,
};

// Where the run ran, from its execution trace.
struct Route {
    std::string requested_backend;  // "arrayfire_cuda"
    std::string effective_backend;  // "arrayfire_cpu"
    int effective_device_id = 0;
    std::string device_name;
    std::string policy;  // "compatibility", "strict"
    bool fallback = false;  // the chosen device failed its check
};

// An engine task beside the run (reading data, building the cache).
struct EngineTask {
    std::string name;
    float progress = 0.0f;
    std::string status;
};

// What the header knows beyond RunFacts.
struct HeaderFacts {
    bool paused = false;
    std::string waiting_for_gpu;  // the run holding the GPU; empty when not waiting
    bool preparing = false;
    bool preparation_failed = false;
    std::string preparation_message;
    std::string preparation_error;
    double preparation_progress = 0.0;
    std::string terminal_status;  // "completed", "early_stopped", "cancelled", "stopped", "failed"
    std::string terminal_reason;  // "completed_all_epochs", "validation_loss_plateau_patience_3", ...
    // A run refused for missing values: its columns (the Dashboard offers the fix).
    std::vector<MissingColumn> missing_columns;
    std::optional<Route> route;
    // The data.
    std::string dataset;
    size_t train_rows = 0, validation_rows = 0, test_rows = 0;
    int samples_per_batch = 0;
    int batches_per_update = 1;
    int metric_interval = 0;  // a point every N batches; 0: first and last batch
    float last_batch_loss = 0.0f;
    // The active model.
    std::string active_model_provenance;  // "restored_best_checkpoint", "run_final_state", "loaded_checkpoint_for_testing"
    bool checkpoint_loaded = false;       // loaded for testing, no run
    std::string checkpoint_path;
    int checkpoint_epoch = 0;
    int checkpoint_step = 0;
    bool has_checkpoint_validation = false;
    double checkpoint_val_loss = 0.0;
    double checkpoint_val_accuracy = 0.0;  // a fraction
    bool classification = true;           // accuracy means something (not a regression)
    // The held-out test after the run.
    bool has_test = false;
    double test_loss = 0.0;
    double test_accuracy = 0.0;  // a fraction
    double test_mae = 0.0;
    std::vector<EngineTask> tasks;
};

struct DetailRow {
    std::string key;
    std::string value;
};

struct Header {
    RunState state = RunState::Idle;
    std::string pill;          // "Training", "Completed"
    std::string status;        // "Epoch 6 of 10 · batch 10 of 12", "All 10 epochs in 2.2s"
    std::string status_tooltip;
    bool show_progress = false;
    double progress = 0.0;     // 0..1
    std::string progress_tooltip;
    std::string progress_text;  // under the bar while preparing
    std::string where;          // "Runs on the CPU (ArrayFire CPU) · compatibility policy"
    std::string where_tooltip;  // the route's codes
    std::string data;           // "digits_8x8 · 1,433 training, 176 validation, 188 test rows · batches of 128"
    std::string now;            // while training: "Latest batches: loss 0.5287 · accuracy 90.70%"
    std::string now_detail;     // "(batch 10 of 12, 204 batches/s)"
    std::string outcome;        // "Finished: every epoch ran."
    Tone outcome_tone = Tone::Plain;
    std::string active_model;   // "Active model: the best validation checkpoint, epoch 10"
    std::string active_model_detail;  // "(validation loss 0.2554, accuracy 92.61%)"
    std::string test;           // "Held-out test: accuracy 94.15%, loss 0.2220"
    std::string test_detail;    // "(188 rows the model never saw)"
    bool missing_values = false;  // failed for missing values: the fix card goes under the outcome
    // Run details > The run: the stop code, checkpoint, batches, engine tasks.
    std::vector<DetailRow> run_rows;
};
Header BuildHeader(const RunFacts& run, const HeaderFacts& facts);
// The engine tasks as rows: up to three ("Engine task": "name · 55% · status"),
// then "Engine tasks": "and N more".
std::vector<DetailRow> TaskRows(const std::vector<EngineTask>& tasks);

// The Runs card (B4-5): this session's runs, best test accuracy first. The
// key columns come first; "All columns" shows the rest.
struct RunCell {
    std::string text;
    std::string tooltip;
    Tone tone = Tone::Plain;
};
struct RunsTable {
    static constexpr size_t kKeyColumns = 8;
    std::vector<std::string> columns;         // the key columns, then the rest
    std::vector<std::vector<RunCell>> rows;   // one cell per column
    std::string key_note;     // under the key columns: what All columns adds
    std::string all_note;     // under all columns: the partition match
    std::string empty;        // no runs yet
    std::string loaded_note;  // a loaded checkpoint is not a run
};
// `records` in the panel's order (best first: SortTrainingRunComparisonsByBestMetric).
RunsTable BuildRunsTable(const std::vector<TrainingRunComparisonRecord>& records, bool checkpoint_loaded);

// The metrics CSV: "Epoch", then one column per series; a row for every
// epoch position where any series has a point (the batch points inside an
// epoch and the epoch ends), a series without a point there left empty.
struct CsvColumn {
    std::string name;
    Series values;
};
std::string MetricsCsv(const std::vector<CsvColumn>& columns);

// "23,000"
std::string Thousands(double value);
// "0.22s" for a time per epoch.
std::string PerEpoch(double seconds);

}  // namespace cyxwiz::training_dashboard
