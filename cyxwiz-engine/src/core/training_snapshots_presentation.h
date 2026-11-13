#pragma once

// The Studio Debugger's training snapshots (TOFIX178 6.3d, approved mockup
// board Snapshots): saved runs in Overview > Runs (Local Debug runs and
// training runs that kept layer snapshots), a training run's snapshot table
// in Overview > Compare, and the snapshot stepper in Model > Layers. Pure
// data in and out.

#include "debug_run_store.h"
#include "layer_capture_store.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::training_snapshots {

// "2026-11-05 14:12" (local time) from a run id ending in milliseconds since
// 1970 ("train-1793417384228", "local-debug-<hash>-1794247562482"); "" when
// it has none.
std::string WhenFromRunId(const std::string& run_id);

// One saved run in Overview > Runs.
struct RunRow {
    std::string run_id;
    bool training = false;
    std::string kind;    // "Training", "Local Debug"
    std::string title;   // run id, and the graph for a training run
    std::string when;
    std::string detail;  // one plain line
    std::vector<std::pair<std::string, std::string>> details;  // under Details; empty values skipped
};
RunRow TrainingRunRow(const std::string& run_id, const std::optional<TrainingRunRecord>& record,
                      const std::vector<LayerCaptureInfo>& captures);
RunRow DebugRunRow(const DebugRunStoreSummary& run);

// A training run's snapshots in Overview > Compare.
struct SnapshotRow {
    int epoch = 0;
    std::string label;  // the capture label to open
    std::string train_loss;
    std::string val_loss;
    std::string val_metric;
    std::string size;
};
struct SnapshotTable {
    std::string heading;        // "Snapshots of train-1793417384228 · cats_dogs (20 epochs)"
    std::string metric_header;  // "Validation accuracy", "Validation MAE"
    std::vector<SnapshotRow> rows;
    std::string note;
};
SnapshotTable SnapshotTableFor(const std::string& run_id, const std::optional<TrainingRunRecord>& record,
                               const std::vector<LayerCaptureInfo>& captures);

// The stepper's note in Model > Layers: "4 of 4 snapshots · validation 92.97 %".
std::string StepperText(size_t index, size_t count, int epoch, const std::optional<TrainingRunRecord>& record);
// "epoch 20" for a snapshot label, the label itself otherwise ("local_debug").
std::string SnapshotName(const std::string& label);
// A side of a comparison: "Epoch 20 · validation 92.97 %" ("Epoch 20" when
// the epoch has no validation or no record).
std::string SnapshotTitle(const std::string& label, const std::optional<TrainingRunRecord>& record);

// "18.4 MB".
std::string SizeText(std::uintmax_t bytes);

}  // namespace cyxwiz::training_snapshots
