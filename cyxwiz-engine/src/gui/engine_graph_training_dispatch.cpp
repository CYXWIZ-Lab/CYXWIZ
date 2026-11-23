#include "engine_graph_training_dispatch.h"

#include "loaders/data_loader.h"
#include "../core/data_registry.h"
#include "../core/arrow_dataset.h"
#ifndef CYXWIZ_TRAINING_MANAGER_ARROW_HARNESS
#include "../core/legacy_dataset_batchers.h"
#endif
#include "../core/sequence_arrow_batcher.h"
#include "../core/training_manager.h"
#include "../core/training_recipe.h"

#include <spdlog/spdlog.h>

namespace gui {

cyxwiz::TrainingRecipe GraphTrainingRecipe(cyxwiz::DataRegistry& registry, const cyxwiz::TrainingConfiguration& config,
                                           const std::string& dataset_name, const std::string& label_column) {
    if (config.sequence_batch.enabled) {
        auto arrow_dataset = registry.GetArrowDataset(dataset_name);
        // A role's windows: a table in memory, or a file kept on disk.
        auto source_of = [&registry](const std::string& name) {
            cyxwiz::SequenceWindowSource source;
            if (auto memory = registry.GetArrowDataset(name)) source.table = memory->GetArrowTable();
            else source.file = registry.GetParquetBackedDataset(name);
            return source;
        };
        const auto& roles = config.dataset_roles;
        const auto dev_source = roles.dev.IsSupplied() ? source_of(roles.dev.dataset_name) : cyxwiz::SequenceWindowSource{};
        const auto test_source = roles.test.IsSupplied() ? source_of(roles.test.dataset_name) : cyxwiz::SequenceWindowSource{};
        if ((roles.dev.IsSupplied() && dev_source.Empty()) || (roles.test.IsSupplied() && test_source.Empty())) {
            return cyxwiz::FailedTrainingRecipe(
                "supplied sequence roles require registered datasets; Apply each source before Train");
        }
        // A training set kept on disk streams a row group at a time (TOFIX154).
        const bool on_disk = !arrow_dataset && registry.GetParquetBackedDataset(dataset_name);
        auto dev = roles.dev.IsSupplied() ? registry.GetArrowDataset(roles.dev.dataset_name) : nullptr;
        auto test = roles.test.IsSupplied() ? registry.GetArrowDataset(roles.test.dataset_name) : nullptr;
        if (!on_disk && ((roles.dev.IsSupplied() && !dev) || (roles.test.IsSupplied() && !test))) {
            return cyxwiz::FailedTrainingRecipe(
                "supplied sequence roles kept on disk need the training set on disk too; raise the data budget or "
                "keep all of them in memory");
        }
        const auto train_source = on_disk ? source_of(dataset_name) : cyxwiz::SequenceWindowSource{};
        // The batches are built for each run's batch size.
        return cyxwiz::SequenceTrainingRecipe(
            [on_disk, train_source, arrow_dataset, dev, test, dev_source, test_source, dataset_name](
                cyxwiz::TrainingConfiguration& run, int batch_size, std::unique_ptr<cyxwiz::ISequenceBatcher>& batcher,
                std::vector<std::string>& id_to_label, std::string& error) {
                auto sequence = on_disk
                    ? cyxwiz::BuildSequenceBatcherFromWindowSources(train_source, run, batch_size, dev_source,
                                                                    test_source)
                    : cyxwiz::BuildSequenceBatcherFromArrowDataset(arrow_dataset, run, batch_size, dev, test);
                if (!sequence.success()) {
                    error = "sequence batcher materialization failed: " + sequence.error_message;
                    return false;
                }
                cyxwiz::ApplySequenceBatcherBuildResultToTrainingConfig(sequence, run);
                spdlog::info("StartTrainingFromGraph: sequence training from '{}' ({} samples, {} labels, "
                             "seq_len={}, token_vocab={})", dataset_name, sequence.sample_count,
                             sequence.id_to_label.size(), sequence.sequence_length, sequence.token_vocabulary_size);
                batcher = std::move(sequence.batcher);
                id_to_label = std::move(sequence.id_to_label);
                return true;
            });
    }

    // The loader that owns the runtime dataset says how it trains.
    if (auto* loader = cyxwiz::loaders::GetByRegisteredDataset(dataset_name)) {
        return loader->TrainingRecipeFor(config, dataset_name, label_column);
    }

#ifndef CYXWIZ_TRAINING_MANAGER_ARROW_HARNESS
    // Fall back to the legacy DatasetHandle path kept for back-compat.
    auto dataset = registry.GetDataset(dataset_name);
    if (!dataset) return cyxwiz::FailedTrainingRecipe("dataset '" + dataset_name + "' not found; load data first");
    spdlog::info("Starting legacy training: dataset={}", dataset_name);
    return cyxwiz::BatcherFactoryTrainingRecipe(cyxwiz::LegacyTrainingBatcherFactory(dataset), "Training Model",
                                                "Training from Node Graph");
#else
    // Arrow-harness builds (tests) carry no legacy DatasetHandle training.
    return cyxwiz::FailedTrainingRecipe("dataset '" + dataset_name +
                                        "' needs the legacy training path, not built into this harness");
#endif
}

GraphTrainingDispatch MakeEngineGraphTrainingDispatch(cyxwiz::DataRegistry& registry, cyxwiz::TrainingManager& tm) {
    return [&registry, &tm](
        cyxwiz::TrainingConfiguration dispatch_config,
        const std::string& dataset_name,
        const std::string& label_column,
        int epochs,
        int batch_size,
        std::weak_ptr<cyxwiz::TrainingPlotPanel> plot_panel,
        std::function<void(bool)> callback) {
        const auto recipe = GraphTrainingRecipe(registry, dispatch_config, dataset_name, label_column);
        return tm.StartTrainingRecipe(recipe, std::move(dispatch_config), epochs, batch_size, plot_panel,
                                      std::move(callback));
    };
}

}  // namespace gui
