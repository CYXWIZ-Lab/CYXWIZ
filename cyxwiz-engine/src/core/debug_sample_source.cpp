#include "debug_sample_source.h"

#include "audio_dataset_batcher.h"
#include "data_registry.h"
#include "graph_training_prep.h"
#include "image_batchers.h"
#include "sparse_feature_dataset.h"
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

// Items `index` .. `index + count - 1` from `source` (as validation sees
// them), named with the part of the first of `parts` that holds the first.
DebugSample Take(IBatcher& source, std::initializer_list<const IBatcher*> parts, const DebugSampleRequest& request,
                 const std::string& item, const std::string& dataset) {
    DebugSample sample;
    const size_t index = request.index;
    size_t count = request.count;
    size_t stride = 1;
    if (request.spread) {
        if (const size_t total = source.SourceSampleCount(); total > index) {
            const auto spread = SpreadItems(total, index, count);
            count = spread.count;
            stride = spread.stride;
        }
    }
    size_t taken = 0;
    const auto batch = GetSourceSampleRange(source, index, count, &taken, stride);
    if (!batch) {
        sample.unavailable = "this data's batcher cannot reach an item by number yet";
        return sample;
    }
    if (batch->data.NumElements() == 0) {
        sample.unavailable = "this data's batches carry sparse (CSR) features, which Local Debug does not feed yet";
        return sample;
    }
    std::optional<BatcherPhase> part;
    for (const IBatcher* batcher : parts)
        if (batcher && !part) part = batcher->SourceSamplePart(index);
    sample.features = batch->data;
    sample.labels = batch->labels;
    for (size_t i = 0; i < taken; ++i) {
        sample.item_indices.push_back(index + i * stride);
        sample.item_names.push_back(source.SourceSampleName(index + i * stride));
    }
    sample.class_names = source.ClassNames();
    sample.description = taken == 1 ? item + " " + std::to_string(index) + " of " + dataset + ", " + PartName(part)
                         : stride > 1 ? std::to_string(taken) + " " + item + "s of " + dataset + ", every " +
                                            std::to_string(stride) + "th from " + item + " " + std::to_string(index)
                                      : item + "s " + std::to_string(index) + "-" + std::to_string(index + taken - 1) +
                                            " of " + dataset + ", the first in " + PartName(part);
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
    // In memory, on disk (Parquet) or sparse features, as Train picks them.
    const auto table = registry.GetArrowDataset(dataset);
    const auto disk = table ? nullptr : registry.GetParquetBackedDataset(dataset);
    const auto sparse = table || disk ? nullptr : registry.GetSparseFeatureDataset(dataset);
    if (!table && !disk && !sparse) {
        sample.unavailable = "'" + dataset + "' is not a loaded table";
        return sample;
    }
    std::string label = config.dataset_roles.train.label_column;
    if (label.empty() && request.nodes)
        label = FindGraphLabelColumn(*request.nodes, config.dataset_name, config.data_source_node_id);
    label = ResolveRuntimeArrowLabelColumn(dataset, label);
    ReconcileRuntimeDatasetTarget(config, label, dataset);
    ReconcileRuntimeTabularFeatureWidth(dataset, label, config);
    TrainingBatcherSet set;
    if (table) {
        set = BuildArrowTrainingBatchers(config, table, label, 1);
    } else if (disk) {
        set = BuildParquetTrainingBatchers(config, disk, label, 1);
    } else {
        // Sparse features train through the resolved partitions (TabularLoader).
        if (!config.dataset_roles.train.IsSupplied()) {
            config.dataset_roles.train.dataset_name = dataset;
            config.dataset_roles.train.label_column = sparse->GetLabelName();
        }
        ResolvedTabularDatasets datasets;
        datasets.train_sparse = sparse;
        auto built = BuildResolvedTabularTrainingBatchers(config, datasets, 1);
        if (!built.ok()) {
            sample.unavailable = "the sparse batchers could not be built: " + built.error_message;
            return sample;
        }
        set = std::move(built.batchers);
    }
    if (set.arrow_val)
        return Take(*set.arrow_val, {set.arrow_train.get(), set.arrow_val.get(), set.arrow_test.get()},
                    request, "row", dataset);
    if (set.parquet_val)
        return Take(*set.parquet_val, {set.parquet_train.get(), set.parquet_val.get(), set.parquet_test.get()},
                    request, "row", dataset);
    if (set.sparse_val)
        return Take(*set.sparse_val, {set.sparse_train.get(), set.sparse_val.get(), set.sparse_test.get()},
                    request, "row", dataset);
    sample.unavailable = "the table's batchers could not be built";
    return sample;
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
            DebugSample taken = Take(*batcher, {batcher.get()}, request, "image",
                                     config.dataset_name);
            taken.class_names = entry.ShownClassNames();  // with the names typed on the Data Input
            return taken;
        }
        if (const auto* found = registry.GetAudioDatasetEntry(config.dataset_name)) {
            const DataRegistry::AudioDatasetEntry entry = *found;  // a copy: the registry may reload
            if (!config.audio_preprocessing.has_feature_node) {
                sample.unavailable = kAudioFeaturesMissingMessage;
                return sample;
            }
            // The batcher training makes, one clip per batch; the model's
            // input follows its features, as it does for Train.
            auto batcher = MakeAudioTrainingBatcher(entry, config, 1, AudioTrainingSplit(config));
            if (batcher->GetNumSamples() == 0) {
                sample.unavailable = "the audio dataset has no clips";
                return sample;
            }
            ApplyAudioInputShape(*batcher, config);
            return Take(*batcher, {batcher.get()}, request, "clip", config.dataset_name);
        }
        if (config.sequence_batch.enabled) {
            sample.unavailable = "picking a sequence (token tagging) sample comes later";
            return sample;
        }
        if (registry.GetArrowDataset(config.dataset_name) || registry.GetParquetBackedDataset(config.dataset_name) ||
            registry.GetSparseFeatureDataset(config.dataset_name) || config.training_data_needs_pipeline_run)
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
