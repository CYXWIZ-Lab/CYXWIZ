#include "debug_sample_source.h"

#include "data_registry.h"
#include "image_batchers.h"

#include <exception>
#include <string>

namespace cyxwiz {

namespace {

const char* PartName(const std::optional<BatcherPhase>& part) {
    if (!part) return "left out of the split";
    switch (*part) {
        case BatcherPhase::Train: return "train";
        case BatcherPhase::Val: return "validation";
        case BatcherPhase::Test: return "test";
    }
    return "left out of the split";
}

// Item `index` from a batcher of the graph's data, as validation sees it.
DebugSample FromBatcher(IBatcher& batcher, size_t index, const std::string& item, const std::string& dataset) {
    DebugSample sample;
    const auto batch = batcher.GetSourceSamples({index});
    if (!batch) {
        sample.unavailable = "this data's batcher cannot reach an item by number yet";
        return sample;
    }
    sample.features = batch->data;
    sample.labels = batch->labels;
    sample.description = item + " " + std::to_string(index) + " of " + dataset + ", " +
                         PartName(batcher.SourceSamplePart(index));
    return sample;
}

}  // namespace

DebugSample MakeDebugSample(const TrainingConfiguration& config, size_t index, std::function<bool()> stop) {
    DebugSample sample;
    try {
        if (const auto* found = DataRegistry::Instance().GetImageDatasetEntry(config.dataset_name)) {
            const DataRegistry::ImageDatasetEntry entry = *found;  // a copy: the registry may reload
            // The batcher training and Run Test make (MakeImageBatcher), one image per batch.
            auto batcher = MakeImageBatcher(entry, config, 1, ImageSplitOf(config), std::move(stop));
            return FromBatcher(*batcher, index, "image", config.dataset_name);
        }
        sample.unavailable = config.dataset_name.empty()
                                 ? "the graph's data is not loaded"
                                 : "picking a sample is built for image data so far; '" + config.dataset_name +
                                       "' is another kind";
    } catch (const std::exception& e) {
        sample.unavailable = e.what();
    }
    return sample;
}

}  // namespace cyxwiz
