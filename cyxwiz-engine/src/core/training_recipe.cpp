#include "training_recipe.h"

#include "arrow_dataset.h"
#include "parquet_backed_dataset.h"
#include "training_batcher_setup.h"
#include "worker_defaults.h"

#include <spdlog/spdlog.h>

#include <utility>

namespace cyxwiz {

namespace {

void NormalizeTrainingNumWorkers(TrainingConfiguration& config) {
    const int requested = config.num_workers;
    const int normalized = ClampNumWorkersToPlatform(requested);
    if (normalized != requested) {
        spdlog::warn("TrainingRecipe: clamping num_workers from {} to {} based on platform", requested, normalized);
        config.num_workers = normalized;
    } else if (config.num_workers > 0) {
        spdlog::info("TrainingRecipe: using num_workers={}", config.num_workers);
    }
}

// The model's input width from the table: every column but the label (and
// the compiler's own width for time-series windows).
void ResolveTableInputSize(TrainingConfiguration& config, const std::shared_ptr<arrow::Schema>& schema,
                           const std::string& label_column, size_t rows, size_t columns, const char* kind) {
    const auto input_resolution = ResolveTabularTrainingInputSize(config, schema, label_column);
    config.input_size = input_resolution.input_size;
    if (input_resolution.used_compiled_override) {
        spdlog::info("TrainingRecipe: {} dataset has {} rows, {} cols, input_size={} (from GraphCompiler's "
                     "TimeSeriesWindow override)", kind, rows, columns, config.input_size);
    } else if (input_resolution.has_separate_label_column) {
        spdlog::info("TrainingRecipe: {} dataset has {} rows, {} cols, input_size={} (label '{}')", kind, rows,
                     columns, config.input_size, label_column);
    } else {
        spdlog::warn("TrainingRecipe: {} dataset has no label column '{}' ({} cols, input_size={})", kind,
                     label_column, columns, config.input_size);
    }
}

}  // namespace

TrainingRecipe FailedTrainingRecipe(std::string error) {
    TrainingRecipe recipe;
    recipe.error = std::move(error);
    return recipe;
}

TrainingRecipe MakeTrainingRecipe(
    std::string task_name, std::string start_message,
    std::function<std::unique_ptr<TrainingExecutor>(TrainingConfiguration, int, std::string&)> make) {
    TrainingRecipe recipe;
    recipe.task_name = std::move(task_name);
    recipe.start_message = std::move(start_message);
    recipe.make = [make = std::move(make)](TrainingConfiguration config, int batch_size, std::string& error) {
        NormalizeTrainingNumWorkers(config);
        return make(std::move(config), batch_size, error);
    };
    return recipe;
}

TrainingRecipe ArrowTrainingRecipe(std::shared_ptr<ArrowDataset> dataset, std::string label_column) {
    if (!dataset) return FailedTrainingRecipe("the Arrow dataset is missing");
    return MakeTrainingRecipe(
        "Training Model (Arrow)", "Training from Arrow Dataset",
        [dataset, label_column](TrainingConfiguration config, int, std::string&) {
            ResolveTableInputSize(config, dataset->GetSchema(), label_column, dataset->GetNumRows(),
                                  dataset->GetNumColumns(), "Arrow");
            return std::make_unique<TrainingExecutor>(std::move(config), dataset, label_column);
        });
}

TrainingRecipe ParquetTrainingRecipe(std::shared_ptr<ParquetBackedDataset> dataset, std::string label_column) {
    if (!dataset) return FailedTrainingRecipe("the Parquet dataset is missing");
    return MakeTrainingRecipe(
        "Training Model (Parquet)", "Training from Parquet-backed Dataset",
        [dataset, label_column](TrainingConfiguration config, int, std::string&) {
            // Column counts come from the cached schema: metadata, not a disk read.
            ResolveTableInputSize(config, dataset->GetSchema(), label_column, dataset->GetNumRows(),
                                  dataset->GetNumColumns(), "Parquet");
            return std::make_unique<TrainingExecutor>(std::move(config), dataset, label_column);
        });
}

TrainingRecipe BatcherFactoryTrainingRecipe(ExternalBatcherFactory factory, std::string task_name,
                                            std::string start_message) {
    if (!factory) return FailedTrainingRecipe("no batcher factory");
    return MakeTrainingRecipe(std::move(task_name), std::move(start_message),
                              [factory](TrainingConfiguration config, int, std::string&) {
                                  return std::make_unique<TrainingExecutor>(std::move(config), factory);
                              });
}

TrainingRecipe ResolvedBatchersTrainingRecipe(ResolvedBatchersBuilder build, std::string task_name,
                                              std::string start_message) {
    if (!build) return FailedTrainingRecipe("no batcher builder");
    return MakeTrainingRecipe(
        std::move(task_name), std::move(start_message),
        [build](TrainingConfiguration config, int batch_size, std::string& error) -> std::unique_ptr<TrainingExecutor> {
            ResolvedExternalBatchers batchers;
            if (!build(config, batch_size, batchers, error)) return nullptr;
            if (!batchers.train) {
                error = "there is no training data";
                return nullptr;
            }
            return std::make_unique<TrainingExecutor>(std::move(config), std::move(batchers));
        });
}

TrainingRecipe SequenceTrainingRecipe(SequenceBatcherBuilder build) {
    if (!build) return FailedTrainingRecipe("no sequence batcher builder");
    return MakeTrainingRecipe(
        "Training Model (Sequence)", "Training from sequence dataset",
        [build](TrainingConfiguration config, int batch_size, std::string& error) -> std::unique_ptr<TrainingExecutor> {
            std::unique_ptr<ISequenceBatcher> batcher;
            std::vector<std::string> id_to_label;
            if (!build(config, batch_size, batcher, id_to_label, error)) return nullptr;
            if (!batcher) {
                error = "the sequence batcher is missing";
                return nullptr;
            }
            return std::make_unique<TrainingExecutor>(std::move(config), std::move(batcher), std::move(id_to_label));
        });
}

}  // namespace cyxwiz
