#pragma once

// Layer captures saved with their run (TOFIX178 6.1c): in the debug run
// store, <store>/<run id>/layers/<label>/, beside the run's session.json.
// capture.json lists the layers and their tensors; each tensor is one
// tensor file (tensor_file.h). capture.json is written last, so a folder
// without it is an unfinished save and is not listed.

#include "layer_capture.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

inline constexpr const char* kLocalDebugCaptureLabel = "local_debug";

// "epoch_0005": training snapshots sort by epoch.
std::string EpochCaptureLabel(int epoch);

std::filesystem::path LayerCaptureDir(const std::string& run_id, const std::string& label);

// Copies every tensor to the host once. Replaces a capture saved before
// under the same run and label.
bool SaveLayerCapture(const LayerCapture& capture, const std::string& run_id, const std::string& label,
                      std::string* error = nullptr);

std::shared_ptr<LayerCapture> LoadLayerCapture(const std::string& run_id, const std::string& label,
                                               std::string* error = nullptr);

// The run's saved captures, by label.
std::vector<std::string> ListLayerCaptures(const std::string& run_id);

// One saved capture: its label, the epoch of a training snapshot (0 for
// Local Debug) and the bytes its files take on disk.
struct LayerCaptureInfo {
    std::string label;
    int epoch = 0;
    std::uintmax_t bytes = 0;
};
std::vector<LayerCaptureInfo> ListLayerCaptureInfo(const std::string& run_id);

// The runs that have saved captures, newest first.
std::vector<std::string> ListRunsWithLayerCaptures();

// A training run's epochs, saved beside its snapshots (TOFIX178 6.3d):
// <store>/<run id>/layers/training.json, rewritten at each snapshot, so each
// snapshot's epoch and the run's best epoch so far are known.
struct TrainingEpochRecord {
    int epoch = 0;
    float train_loss = 0.0f;
    float train_metric = 0.0f;  // accuracy, or the MAE of a regression
    std::optional<float> val_loss;
    std::optional<float> val_metric;
};
struct TrainingRunRecord {
    std::string run_id;
    std::string graph_name;
    int total_epochs = 0;
    bool regression = false;  // the metric is the MAE, not accuracy
    std::vector<TrainingEpochRecord> epochs;
    // The snapshots' items, for the layer views' status line.
    std::vector<std::string> item_names;   // empty when items have no name
    std::vector<std::string> class_names;  // empty when unknown
    std::vector<float> label_values;       // row-major, as the run's batches carry them
    std::vector<size_t> label_shape;
    std::string output_rule;               // layer_view::OutputRuleKey; empty when unknown
};
bool SaveTrainingRunRecord(const TrainingRunRecord& record, std::string* error = nullptr);
// nullopt when the run has none (a Local Debug run, or a snapshot saved before 6.3d).
std::optional<TrainingRunRecord> LoadTrainingRunRecord(const std::string& run_id);

// Deletes the captures of all but the `keep_runs` runs saved last (a run's
// session.json and traces stay). Returns the number of runs cleared.
size_t PruneLayerCaptures(size_t keep_runs);

}  // namespace cyxwiz
