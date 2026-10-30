#pragma once

// The image batchers of a compiled graph (TOFIX178 1.5): training's train /
// validation / test parts and Run Test's test part are made here, the same
// way - the images prepared once with the graph's fixed steps, then the
// per-batch steps, Normalize and the label form on the device.

#include "data_registry.h"
#include "image_dataset_batcher.h"
#include "sample_split.h"
#include "test_dataset_selection.h"
#include "test_executor.h"

#include <functional>
#include <memory>

namespace cyxwiz {

struct TrainingConfiguration;

// `stop`: polled while the images are decoded (true stops). Throws when the
// images cannot be prepared.
std::unique_ptr<ImageDatasetBatcher> MakeImageBatcher(const DataRegistry::ImageDatasetEntry& entry,
                                                      const TrainingConfiguration& run, int batch_size,
                                                      const SampleSplitSettings& split,
                                                      std::function<bool()> stop = {});

// The graph's Data Split settings for the training images (a wired role's
// share is already 0: the compiler gave it to train).
SampleSplitSettings ImageSplitOf(const TrainingConfiguration& run);

// Run Test on images: the Data Split's test part of the training images, or
// all of a dataset given whole (a wired Test dataset, EntireProvidedDataset),
// labelled with the dataset's class names.
// `dataset_name`: the image Data Input's dataset (its index table names the rows).
ExternalTestSource MakeImageTestSource(const std::string& dataset_name, const DataRegistry::ImageDatasetEntry& entry,
                                       TestDatasetScope scope);

}  // namespace cyxwiz
