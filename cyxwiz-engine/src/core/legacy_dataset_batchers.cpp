#include "legacy_dataset_batchers.h"

#include "classification_decision.h"
#include "../preprocessing/preprocessing_config.h"
#include "../preprocessing/statistics_calculator.h"

#include <spdlog/spdlog.h>

#include <algorithm>

namespace cyxwiz {

DatasetBatcherAdapter::DatasetBatcherAdapter(std::unique_ptr<DatasetBatcher> batcher)
    : batcher_(std::move(batcher)) {}

namespace {

// Label and input settings shared by every legacy split.
void ApplyLegacyLabelAndInputSettings(const TrainingConfiguration& config, DatasetBatcher& batcher) {
    if (config.preprocessing.has_normalization) {
        batcher.SetLegacyNormalization(config.preprocessing.norm_mean, config.preprocessing.norm_std);
    }
    if (UsesScalarBinaryTargets(config.loss_type)) {
        batcher.SetLegacyScalarLabelMode(true);
    } else if (config.preprocessing.has_onehot) {
        batcher.SetLegacyOneHotEncoding(config.preprocessing.num_classes);
    }
    batcher.SetFlatten(true);
}

}  // namespace

ResolvedExternalBatchers BuildLegacyTrainingBatchers(const TrainingConfiguration& config,
                                                     const DatasetHandle& dataset, int batch_size) {
    spdlog::info("Legacy dataset batchers: batch_size={}, shuffle={}, drop_last={}, num_workers={}", batch_size,
                 config.shuffle, config.drop_last, config.num_workers);
    const auto seed = static_cast<uint32_t>(config.dataloader_seed);
    auto train = std::make_unique<DatasetBatcher>(dataset, batch_size, DatasetSplit::Train, config.shuffle,
                                                  config.drop_last, config.num_workers, seed);
    auto val = std::make_unique<DatasetBatcher>(dataset, batch_size, DatasetSplit::Validation, false, false,
                                                config.num_workers, seed);

    // Registry preprocessing pipeline, fitted on the dataset's statistics.
    const std::string dataset_name = !config.dataset_name.empty() ? config.dataset_name : dataset.GetName();
    DataRegistry& registry = DataRegistry::Instance();
    if (registry.HasPreprocessingConfig(dataset_name)) {
        spdlog::info("Legacy dataset batchers: preprocessing config found for '{}'", dataset_name);
        const PreprocessingConfig preprocessing = registry.GetPreprocessingConfig(dataset_name);
        if (preprocessing.enabled) {
            train->SetPreprocessingConfig(preprocessing);
            val->SetPreprocessingConfig(preprocessing);
            const DatasetStatistics stats = StatisticsCalculator::Compute(
                dataset_name, &registry,
                [](float progress) { spdlog::debug("Statistics computation: {:.1f}%", progress * 100.0f); });
            if (stats.is_valid) {
                train->InitializePreprocessing(stats);
                val->InitializePreprocessing(stats);
            }
        }
    }
    if (registry.HasAugmentationPipeline(dataset_name)) {
        if (auto pipeline = registry.GetAugmentationPipeline(dataset_name)) {
            train->SetAugmentationPipeline(pipeline);
            train->SetApplyAugmentationOnTrain(true);
        }
    }
    ApplyLegacyLabelAndInputSettings(config, *train);
    ApplyLegacyLabelAndInputSettings(config, *val);

    ResolvedExternalBatchers batchers;
    batchers.train = std::make_shared<DatasetBatcherAdapter>(std::move(train));
    batchers.dev = std::make_shared<DatasetBatcherAdapter>(std::move(val));
    return batchers;
}

ExternalBatcherFactory LegacyTrainingBatcherFactory(DatasetHandle dataset) {
    return [dataset = std::move(dataset)](const TrainingConfiguration& config, int batch_size) {
        return BuildLegacyTrainingBatchers(config, dataset, batch_size);
    };
}

std::unique_ptr<IBatcher> BuildLegacyTestBatcher(const TrainingConfiguration& config, const DatasetHandle& dataset,
                                                 int batch_size) {
    auto test = std::make_unique<DatasetBatcher>(dataset, batch_size, DatasetSplit::Test, false, false,
                                                 config.num_workers, static_cast<uint32_t>(config.dataloader_seed));
    ApplyLegacyLabelAndInputSettings(config, *test);
    return std::make_unique<DatasetBatcherAdapter>(std::move(test));
}

ExternalTestSource LegacyTestSource(DatasetHandle dataset) {
    ExternalTestSource source;
    source.make_batcher = [dataset = std::move(dataset)](const TrainingConfiguration& config, int batch_size) {
        return BuildLegacyTestBatcher(config, dataset, batch_size);
    };
    return source;
}

}  // namespace cyxwiz
