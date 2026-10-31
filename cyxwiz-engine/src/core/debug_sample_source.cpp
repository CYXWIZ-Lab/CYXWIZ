#include "debug_sample_source.h"

#include "data_registry.h"
#include "graph_training_prep.h"
#include "image_batchers.h"
#include "training_batcher_setup.h"

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

// Item `index` from `source` (as validation sees it), named with the part of
// the first of `parts` that holds it.
DebugSample Take(IBatcher& source, std::initializer_list<const IBatcher*> parts, size_t index,
                 const std::string& item, const std::string& dataset) {
    DebugSample sample;
    const auto batch = source.GetSourceSamples({index});
    if (!batch) {
        sample.unavailable = "this data's batcher cannot reach an item by number yet";
        return sample;
    }
    std::optional<BatcherPhase> part;
    for (const IBatcher* batcher : parts)
        if (batcher && !part) part = batcher->SourceSamplePart(index);
    sample.features = batch->data;
    sample.labels = batch->labels;
    sample.description = item + " " + std::to_string(index) + " of " + dataset + ", " + PartName(part);
    return sample;
}

// A table, as Train reads it: the data steps before the Data Split (run, or
// from the prepared-data cache), the runtime label and width, then the
// batchers training builds (split, label encoding, Normalize, label form).
DebugSample TableSample(TrainingConfiguration& config, const DebugSampleRequest& request) {
    DebugSample sample;
    auto& registry = DataRegistry::Instance();
    std::string dataset = config.dataset_name;
    if (config.training_data_needs_pipeline_run && request.nodes && request.links) {
        const int feeder = TrainingInputFeeder(*request.nodes, *request.links);
        if (feeder >= 0) {
            const auto prepared = PrepareStepResult(*request.nodes, *request.links, feeder, registry, request.roots,
                                                    request.cache_config, request.stop);
            if (prepared.cancelled) {
                sample.unavailable = "stopped while the data steps ran";
                return sample;
            }
            if (!prepared.ok) {
                sample.unavailable = "the data steps before the Data Split failed: " + prepared.error;
                return sample;
            }
            dataset = prepared.dataset;
        }
    }
    const auto table = registry.GetArrowDataset(dataset);
    if (!table) {
        sample.unavailable = "'" + dataset + "' is not a table in memory (picking a row of a table kept on disk or of "
                             "sparse features comes later)";
        return sample;
    }
    std::string label = config.dataset_roles.train.label_column;
    if (label.empty() && request.nodes)
        label = FindGraphLabelColumn(*request.nodes, config.dataset_name, config.data_source_node_id);
    label = ResolveRuntimeArrowLabelColumn(dataset, label);
    ReconcileRuntimeDatasetTarget(config, label, dataset);
    ReconcileRuntimeTabularFeatureWidth(dataset, label, config);
    auto set = BuildArrowTrainingBatchers(config, table, label, 1);
    if (!set.arrow_val) {
        sample.unavailable = "the table's batchers could not be built";
        return sample;
    }
    return Take(*set.arrow_val, {set.arrow_train.get(), set.arrow_val.get(), set.arrow_test.get()}, request.index,
                "row", dataset);
}

}  // namespace

DebugSample MakeDebugSample(TrainingConfiguration& config, const DebugSampleRequest& request) {
    DebugSample sample;
    try {
        auto& registry = DataRegistry::Instance();
        if (const auto* found = registry.GetImageDatasetEntry(config.dataset_name)) {
            const DataRegistry::ImageDatasetEntry entry = *found;  // a copy: the registry may reload
            // The batcher training and Run Test make (MakeImageBatcher), one image per batch.
            auto batcher = MakeImageBatcher(entry, config, 1, ImageSplitOf(config), request.stop);
            return Take(*batcher, {batcher.get()}, request.index, "image", config.dataset_name);
        }
        if (config.sequence_batch.enabled) {
            sample.unavailable = "picking a sequence (token tagging) sample comes later";
            return sample;
        }
        if (registry.GetArrowDataset(config.dataset_name) || config.training_data_needs_pipeline_run)
            return TableSample(config, request);
        sample.unavailable = config.dataset_name.empty()
                                 ? "the graph's data is not loaded"
                                 : "picking a sample of '" + config.dataset_name + "' (this data kind) comes later";
    } catch (const std::exception& e) {
        sample.unavailable = e.what();
    }
    return sample;
}

}  // namespace cyxwiz
