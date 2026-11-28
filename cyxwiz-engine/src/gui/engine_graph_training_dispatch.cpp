#include "engine_graph_training_dispatch.h"

#include "../core/data_registry.h"
#ifndef CYXWIZ_TRAINING_MANAGER_ARROW_HARNESS
#include "../core/legacy_dataset_batchers.h"
#include "../core/media_training_recipe.h"
#endif
#include "../core/run_resolver.h"
#include "../core/training_manager.h"

#include <spdlog/spdlog.h>

#include <optional>

namespace gui {

cyxwiz::TrainingRecipe GraphTrainingRecipe(cyxwiz::DataRegistry& registry,
                                           const cyxwiz::TrainingConfiguration& config) {
    cyxwiz::RunDatasets data;
    data.arrow = [&registry](const std::string& name) { return registry.GetArrowDataset(name); };
    data.parquet = [&registry](const std::string& name) { return registry.GetParquetBackedDataset(name); };
    data.sparse = [&registry](const std::string& name) { return registry.GetSparseFeatureDataset(name); };
    // Images before audio before text, as the dataset catalog ranks a name
    // that has more than one kind of entry. The harness builds (tests) carry
    // no image or audio batchers, so there the configuration goes unread.
    data.media = [&registry]([[maybe_unused]] const cyxwiz::TrainingConfiguration& run,
                             const std::string& name) -> std::optional<cyxwiz::TrainingRecipe> {
#ifndef CYXWIZ_TRAINING_MANAGER_ARROW_HARNESS
        if (const auto* image = registry.GetImageDatasetEntry(name)) {
            spdlog::info("GraphTrainingRecipe: image training: dataset={}, {} images, {} classes", name,
                         image->num_images, image->num_classes);
            return cyxwiz::ImageTrainingRecipe(*image);
        }
        if (const auto* audio = registry.GetAudioDatasetEntry(name)) {
            spdlog::info("GraphTrainingRecipe: audio training: dataset={}, {} samples, {} classes, feature_type={}",
                         name, audio->num_samples, audio->num_classes,
                         static_cast<int>(run.audio_preprocessing.feature_type));
            return cyxwiz::AudioTrainingRecipe(*audio);
        }
#endif
        // Raw text also keeps its table under the same name; it trains only
        // through a tokenized (or vectorized) table.
        if (registry.IsTextDataset(name)) {
            return cyxwiz::FailedTrainingRecipe("text dataset '" + name +
                                                "' cannot train as raw text. Add a Text Tokenizer (or TF-IDF / Count "
                                                "Vectorizer) node; text trains through its tokenized table.");
        }
        return std::nullopt;
    };
    auto recipe = cyxwiz::ResolveRun(config, data);

#ifndef CYXWIZ_TRAINING_MANAGER_ARROW_HARNESS
    // A name in none of those stores may still be a legacy DatasetHandle. A
    // sequence graph's refusal and raw text's are final. This fallback is
    // removed with the DatasetHandle training path.
    const std::string& dataset_name = config.dataset_name;
    if (!recipe && !config.sequence_batch.enabled && !registry.IsTextDataset(dataset_name)) {
        if (auto dataset = registry.GetDataset(dataset_name)) {
            spdlog::info("Starting legacy training: dataset={}", dataset_name);
            return cyxwiz::BatcherFactoryTrainingRecipe(cyxwiz::LegacyTrainingBatcherFactory(dataset),
                                                        "Training Model", "Training from Node Graph");
        }
    }
#endif
    return recipe;
}

GraphTrainingDispatch MakeEngineGraphTrainingDispatch(cyxwiz::DataRegistry& registry, cyxwiz::TrainingManager& tm) {
    return [&registry, &tm](
        cyxwiz::TrainingConfiguration dispatch_config,
        const std::string& /*dataset_name*/,
        const std::string& /*label_column*/,
        int epochs,
        int batch_size,
        std::weak_ptr<cyxwiz::TrainingPlotPanel> plot_panel,
        std::function<void(bool)> callback) {
        // The launcher wrote the dataset name and label into the configuration.
        const auto recipe = GraphTrainingRecipe(registry, dispatch_config);
        return tm.StartTrainingRecipe(recipe, std::move(dispatch_config), epochs, batch_size, plot_panel,
                                      std::move(callback));
    };
}

}  // namespace gui
