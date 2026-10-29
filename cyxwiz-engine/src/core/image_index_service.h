#pragma once

// Builds an image dataset's index table on a background task after the
// dataset loads, and registers it as DataRegistry::ImageIndexName(name)
// (TOFIX178 1.3e). Training never waits for it.

#include "data_registry.h"
#include "image_quality_analysis.h"

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz {

struct ImageIndexStatus {
    enum class State { None, Building, Ready, Failed };
    State state = State::None;
    size_t done = 0, total = 0;
    std::string error;
};

// Starts (or restarts) the index of `dataset`; a build still running for it
// is cancelled. label_names: a display name per class index ("" keeps the
// class name).
void BuildImageIndexInBackground(const std::string& dataset, const DataRegistry::ImageDatasetEntry& entry,
                                 const std::vector<std::string>& label_names = {});

ImageIndexStatus GetImageIndexStatus(const std::string& dataset);

// A measured quality analysis for `dataset`: its index gets the quality
// columns now and after every rebuild (renamed classes).
void AttachImageQuality(const std::string& dataset, std::shared_ptr<const ImageQualityAnalysis> analysis,
                        const ImageQualityVerdict& verdict);

// New class names for a loaded image dataset: the registry entry takes them
// and the index is rebuilt with them (label_name). False when not loaded.
bool RenameImageClasses(const std::string& dataset, const std::vector<std::string>& display_names);

}  // namespace cyxwiz
