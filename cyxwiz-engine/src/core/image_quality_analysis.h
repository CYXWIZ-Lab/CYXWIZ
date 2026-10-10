#pragma once

// Quality Analyzer (TOFIX140): measure an image dataset once at the Resize
// size, keep the measurements, and judge them against a node's checks.
//
// The measurements depend only on the images and the Resize size, so they
// are cached under a key built from both (every file's path, size and
// modification time). Changing a check re-judges the cached measurements;
// changing the images or the Resize size needs a new analysis. Training
// leaves the rejected files out; they stay on disk.

#include "data_registry.h"
#include "dataset_base.h"

#include <cyxwiz/image_quality.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

struct ImageQualityAnalysis {
    std::string key;
    std::string folder_path;
    int width = 0;
    int height = 0;
    std::vector<std::string> class_names;
    std::vector<std::string> files;  // per image, in dataset order
    std::vector<int> labels;
    std::vector<image::ImageQualityMetrics> metrics;
    std::string device;  // where it was measured, e.g. "CUDA - GeForce GTX 1050 Ti"
    double seconds = 0.0;
};

struct ImageQualityVerdict {
    std::vector<uint32_t> reasons;  // image::ImageQualityReason bits per image
    std::vector<int64_t> duplicate_of;
    size_t rejected = 0;
    size_t blurry = 0, dark = 0, bright = 0, low_contrast = 0, duplicates = 0;
    size_t multiple = 0;  // images failing more than one check
    std::vector<size_t> class_total, class_rejected;
};

// The dataset the image batcher would build for this entry at this size.
std::shared_ptr<Dataset> OpenImageQualityDataset(const DataRegistry::ImageDatasetEntry& entry, int width, int height);

// The cache key of the measurements: the files (path, size, modification
// time), the size and the metric version. Remembered per folder while the
// folder's directories are unchanged, so compiling on every edit stays cheap.
std::string ImageQualityKey(const DataRegistry::ImageDatasetEntry& entry, int width, int height);

// Where measurements run on this thread, e.g. "CUDA - GeForce GTX 1050 Ti".
std::string ImageQualityDeviceLabel();

// <project>/cache/quality when a project is open, else <temp>/cyxwiz/quality.
// The Engine sets the root as projects open and close.
void SetImageQualityProjectRoot(const std::string& root);
std::filesystem::path ImageQualityCacheDir();

// Decodes every image on CPU workers and measures it on the ArrayFire device
// in batches. progress(done, total) after each batch; returns nullopt when
// cancel became true. Throws when the images cannot be measured.
std::optional<ImageQualityAnalysis> AnalyzeImageQuality(const DataRegistry::ImageDatasetEntry& entry,
                                                        int width, int height,
                                                        const std::function<void(size_t, size_t)>& progress,
                                                        const std::atomic<bool>* cancel);

bool SaveImageQualityAnalysis(const ImageQualityAnalysis& analysis, std::string* error);
// The cached analysis for key, or null. Shared and remembered, so repeated
// compiles do not re-read the file.
std::shared_ptr<const ImageQualityAnalysis> LoadImageQualityAnalysis(const std::string& key);

// Node parameters <-> checks. Unknown or unparsable values keep the default.
image::ImageQualityChecks ImageQualityChecksFromParameters(const std::map<std::string, std::string>& parameters);
void WriteImageQualityChecks(const image::ImageQualityChecks& checks, std::map<std::string, std::string>& parameters);

ImageQualityVerdict JudgeImageQualityAnalysis(const ImageQualityAnalysis& analysis,
                                              const image::ImageQualityChecks& checks);

// The compiler's view (GraphDatasetCatalog::image_quality, installed by this
// file): the files to leave out, or why the analysis is missing or stale.
struct GraphImageQualityResult;
GraphImageQualityResult ResolveImageQuality(const std::string& dataset_name, int width, int height,
                                            const std::map<std::string, std::string>& parameters);

}  // namespace cyxwiz
