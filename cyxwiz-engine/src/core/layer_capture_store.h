#pragma once

// Layer captures saved with their run (TOFIX178 6.1c): in the debug run
// store, <store>/<run id>/layers/<label>/, beside the run's session.json.
// capture.json lists the layers and their tensors; each tensor is one
// tensor file (tensor_file.h). capture.json is written last, so a folder
// without it is an unfinished save and is not listed.

#include "layer_capture.h"

#include <cstddef>
#include <filesystem>
#include <memory>
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

// Deletes the captures of all but the `keep_runs` runs saved last (a run's
// session.json and traces stay). Returns the number of runs cleared.
size_t PruneLayerCaptures(size_t keep_runs);

}  // namespace cyxwiz
