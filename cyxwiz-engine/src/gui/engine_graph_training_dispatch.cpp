#include "engine_graph_training_dispatch.h"

#include "loaders/data_loader.h"
#include "../core/data_registry.h"
#ifndef CYXWIZ_TRAINING_MANAGER_ARROW_HARNESS
#include "../core/legacy_dataset_batchers.h"
#endif
#include "../core/sequence_arrow_batcher.h"
#include "../core/training_manager.h"

#include <spdlog/spdlog.h>

namespace gui {

GraphTrainingDispatch MakeEngineGraphTrainingDispatch(cyxwiz::DataRegistry& registry, cyxwiz::TrainingManager& tm) {
    return [&registry, &tm](
        cyxwiz::TrainingConfiguration dispatch_config,
        const std::string& dataset_name,
        const std::string& label_column,
        int epochs,
        int batch_size,
        std::weak_ptr<cyxwiz::TrainingPlotPanel> plot_panel,
        std::function<void(bool)> callback) {

        if (dispatch_config.sequence_batch.enabled) {
            auto arrow_dataset = registry.GetArrowDataset(dataset_name);
            auto dev = dispatch_config.dataset_roles.dev.IsSupplied()
                ? registry.GetArrowDataset(dispatch_config.dataset_roles.dev.dataset_name) : nullptr;
            auto test = dispatch_config.dataset_roles.test.IsSupplied()
                ? registry.GetArrowDataset(dispatch_config.dataset_roles.test.dataset_name) : nullptr;
            if ((dispatch_config.dataset_roles.dev.IsSupplied() && !dev) ||
                (dispatch_config.dataset_roles.test.IsSupplied() && !test)) {
                spdlog::error("Supplied sequence roles require registered in-memory Arrow datasets; Apply each source before Train");
                return false;
            }
            auto sequence = cyxwiz::BuildSequenceBatcherFromArrowDataset(
                arrow_dataset, dispatch_config, batch_size, dev, test);
            if (!sequence.success()) {
                spdlog::error("StartTrainingFromGraph: sequence batcher "
                              "materialization failed: {}",
                              sequence.error_message);
                return false;
            }

            cyxwiz::ApplySequenceBatcherBuildResultToTrainingConfig(
                sequence, dispatch_config);

            spdlog::info("StartTrainingFromGraph: starting sequence "
                         "training from '{}' ({} samples, {} labels, "
                         "seq_len={}, token_vocab={})",
                         dataset_name, sequence.sample_count,
                         sequence.id_to_label.size(),
                         sequence.sequence_length,
                         sequence.token_vocabulary_size);
            return tm.StartTrainingSequence(
                std::move(dispatch_config),
                std::move(sequence.batcher),
                std::move(sequence.id_to_label),
                epochs, batch_size, plot_panel, std::move(callback));
        }

        // Dispatch via loader polymorphism. GetByRegisteredDataset walks
        // registered loaders and picks the one that owns the runtime dataset.
        if (auto* loader = cyxwiz::loaders::GetByRegisteredDataset(dataset_name)) {
            return loader->LaunchTraining(
                std::move(dispatch_config), dataset_name, label_column,
                epochs, batch_size, plot_panel, std::move(callback));
        }

#ifndef CYXWIZ_TRAINING_MANAGER_ARROW_HARNESS
        // Fall back to the legacy DatasetHandle path kept for back-compat.
        auto dataset = registry.GetDataset(dataset_name);
        if (!dataset) {
            spdlog::error("Dataset '{}' not found in registry. Load data first.",
                          dataset_name);
            return false;
        }
        spdlog::info("Starting legacy training: dataset={}, epochs={}, batch_size={}",
                     dataset_name, epochs, batch_size);
        return tm.StartTraining(
            std::move(dispatch_config),
            cyxwiz::LegacyTrainingBatcherFactory(dataset), epochs, batch_size,
            plot_panel, std::move(callback));
#else
        // Arrow-harness builds (tests) carry no legacy DatasetHandle training.
        spdlog::error("Dataset '{}' needs the legacy training path, not built into this harness", dataset_name);
        return false;
#endif
    };
}

}  // namespace gui
