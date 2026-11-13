#include "training_recipe.h"

#include "arrow_dataset.h"
#include "parquet_backed_dataset.h"
#include "training_batcher_setup.h"
#include "worker_defaults.h"
#ifndef CYXWIZ_TRAINING_MANAGER_ARROW_HARNESS
#include "audio_dataset_batcher.h"
#include "image_batchers.h"
#include "image_dataset_batcher.h"
#include <cyxwiz/error_codes.h>
#endif

#include <spdlog/spdlog.h>

#include <optional>
#include <stdexcept>
#include <utility>

namespace cyxwiz {

namespace {

void NormalizeTrainingNumWorkers(TrainingConfiguration& config) {
    const int requested = config.num_workers;
    const int normalized = ClampNumWorkersToPlatform(requested);
    if (normalized != requested) {
        spdlog::warn("TrainingManager: clamping num_workers from {} to {} based on platform", requested, normalized);
        config.num_workers = normalized;
    } else if (config.num_workers > 0) {
        spdlog::info("TrainingManager: using num_workers={}", config.num_workers);
    }
}

TrainingRecipe Recipe(std::string task_name, std::string start_message,
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

// The model's input width from the table: every column but the label (and
// the compiler's own width for time-series windows).
void ResolveTableInputSize(TrainingConfiguration& config, const std::shared_ptr<arrow::Schema>& schema,
                           const std::string& label_column, size_t rows, size_t columns, const char* kind) {
    const auto input_resolution = ResolveTabularTrainingInputSize(config, schema, label_column);
    config.input_size = input_resolution.input_size;
    if (input_resolution.used_compiled_override) {
        spdlog::info("TrainingManager: {} dataset has {} rows, {} cols, input_size={} (from GraphCompiler's "
                     "TimeSeriesWindow override)", kind, rows, columns, config.input_size);
    } else if (input_resolution.has_separate_label_column) {
        spdlog::info("TrainingManager: {} dataset has {} rows, {} cols, input_size={} (label '{}')", kind, rows,
                     columns, config.input_size, label_column);
    } else {
        spdlog::warn("TrainingManager: {} dataset has no label column '{}' ({} cols, input_size={})", kind,
                     label_column, columns, config.input_size);
    }
}

}  // namespace

TrainingRecipe FailedTrainingRecipe(std::string error) {
    TrainingRecipe recipe;
    recipe.error = std::move(error);
    return recipe;
}

TrainingRecipe ArrowTrainingRecipe(std::shared_ptr<ArrowDataset> dataset, std::string label_column) {
    if (!dataset) return FailedTrainingRecipe("the Arrow dataset is missing");
    return Recipe("Training Model (Arrow)", "Training from Arrow Dataset",
                  [dataset, label_column](TrainingConfiguration config, int, std::string&) {
                      ResolveTableInputSize(config, dataset->GetSchema(), label_column, dataset->GetNumRows(),
                                            dataset->GetNumColumns(), "Arrow");
                      TryApplyBalancedClassWeightsFromArrowTable(config, dataset->GetArrowTable(), label_column,
                                                                 /*partition_column=*/"", "TrainingManager Arrow");
                      return std::make_unique<TrainingExecutor>(std::move(config), dataset, label_column);
                  });
}

TrainingRecipe ParquetTrainingRecipe(std::shared_ptr<ParquetBackedDataset> dataset, std::string label_column) {
    if (!dataset) return FailedTrainingRecipe("the Parquet dataset is missing");
    return Recipe("Training Model (Parquet)", "Training from Parquet-backed Dataset",
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
    return Recipe(std::move(task_name), std::move(start_message),
                  [factory](TrainingConfiguration config, int, std::string&) {
                      return std::make_unique<TrainingExecutor>(std::move(config), factory);
                  });
}

TrainingRecipe ResolvedBatchersTrainingRecipe(ResolvedBatchersBuilder build, std::string task_name,
                                              std::string start_message) {
    if (!build) return FailedTrainingRecipe("no batcher builder");
    return Recipe(std::move(task_name), std::move(start_message),
                  [build](TrainingConfiguration config, int batch_size,
                          std::string& error) -> std::unique_ptr<TrainingExecutor> {
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
    return Recipe("Training Model (Sequence)", "Training from sequence dataset",
                  [build](TrainingConfiguration config, int batch_size,
                          std::string& error) -> std::unique_ptr<TrainingExecutor> {
                      std::unique_ptr<ISequenceBatcher> batcher;
                      std::vector<std::string> id_to_label;
                      if (!build(config, batch_size, batcher, id_to_label, error)) return nullptr;
                      if (!batcher) {
                          error = "the sequence batcher is missing";
                          return nullptr;
                      }
                      return std::make_unique<TrainingExecutor>(std::move(config), std::move(batcher),
                                                                std::move(id_to_label));
                  });
}

#ifndef CYXWIZ_TRAINING_MANAGER_ARROW_HARNESS
TrainingRecipe ImageTrainingRecipe(DataRegistry::ImageDatasetEntry image_entry) {
    return Recipe(
        "Training Model (Image)", "Training from Image Dataset",
        [image_entry](TrainingConfiguration config, int, std::string& error) -> std::unique_ptr<TrainingExecutor> {
            // A Validation or Test image dataset wired to the Data Split (TOFIX178
            // 1.5): read in full as that part, through the same image steps.
            std::optional<DataRegistry::ImageDatasetEntry> dev_entry, test_entry;
            for (auto [role, entry] : {std::pair{&config.dataset_roles.dev, &dev_entry},
                                       std::pair{&config.dataset_roles.test, &test_entry}}) {
                if (!role->IsSupplied() || role->joined_to_training) continue;
                const auto* found = DataRegistry::Instance().GetImageDatasetEntry(role->dataset_name);
                if (!found) {
                    error = "'" + role->dataset_name + "' is wired as " +
                            (role == &config.dataset_roles.dev ? "validation" : "test") +
                            " but is not a loaded image dataset";
                    return nullptr;
                }
                *entry = *found;
            }

            // On the training thread: the images decoded once at the graph's
            // Resize size and mode (from the project's cache/images when they
            // were before; TOFIX178 1.3b), then the batcher gathers from them and
            // runs the image transforms and Normalize on the device. A Stop stops
            // the decoding.
            auto stop_requested = std::make_shared<std::function<bool()>>();
            ExternalBatcherFactory factory = [image_entry, dev_entry, test_entry, stop_requested](
                                                 const TrainingConfiguration& run, int run_batch_size) {
                const auto stop = [stop_requested] { return *stop_requested && (*stop_requested)(); };
                const auto make_batcher = [&](const DataRegistry::ImageDatasetEntry& entry,
                                              const SampleSplitSettings& split) {
                    return std::shared_ptr<ImageDatasetBatcher>(
                        MakeImageBatcher(entry, run, run_batch_size, split, stop));
                };
                // The Data Split's shares (a wired role's share is 0: the compiler
                // gave it to train) and its seed, by class when stratified.
                const SampleSplitSettings split = ImageSplitOf(run);
                auto batcher = make_batcher(image_entry, split);
                if (batcher->SamplesIn(BatcherPhase::Train) == 0)
                    throw std::runtime_error("the image dataset has no images to train on");

                // The compiler set input_shape [H, W, C] from the same Resize size
                // and transforms (ApplyImageInputShape), so the model and the
                // batches agree.
                spdlog::info("TrainingManager: Image dataset {} train / {} validation / {} test images, "
                             "input_size={}, {} image steps, split seed={}",
                             batcher->SamplesIn(BatcherPhase::Train), batcher->SamplesIn(BatcherPhase::Val),
                             batcher->SamplesIn(BatcherPhase::Test), run.input_size,
                             run.image_augmentation.ops.size() + run.image_cv_steps.size(), run.split_seed);

                // One batcher, three parts by phase; a wired role reads its own set.
                ResolvedExternalBatchers batchers;
                batchers.train = batcher;
                if (dev_entry) {
                    batchers.dev = make_batcher(*dev_entry, SampleSplitSettings{0.0f, 1.0f, 0.0f, false, split.seed});
                } else if (batcher->SamplesIn(BatcherPhase::Val) > 0) {
                    batchers.dev = batcher;
                }
                if (test_entry) {
                    batchers.test = make_batcher(*test_entry, SampleSplitSettings{0.0f, 0.0f, 1.0f, false, split.seed});
                } else if (batcher->SamplesIn(BatcherPhase::Test) > 0) {
                    batchers.test = batcher;
                }
                return batchers;
            };

            auto executor = std::make_unique<TrainingExecutor>(std::move(config), std::move(factory));
            *stop_requested = [run = executor.get()] { return run->StopRequested(); };
            return executor;
        });
}

TrainingRecipe AudioTrainingRecipe(DataRegistry::AudioDatasetEntry audio_entry) {
    return Recipe(
        "Training Model (Audio)", "Training from Audio Dataset",
        [audio_entry](TrainingConfiguration config, int batch_size,
                      std::string& error) -> std::unique_ptr<TrainingExecutor> {
            if (!config.audio_preprocessing.has_feature_node) {
                error = std::string("[") + errors::Compiler::AudioFeaturesMissing + "] " + kAudioFeaturesMissingMessage;
                return nullptr;
            }

            // AudioDatasetBatcher loads with the entry's settings, extracts the
            // features config.audio_preprocessing describes (Spectrogram /
            // MelSpectrogram / MFCC / AudioAugmentation nodes) and probes a
            // sample for the actual feature shape.
            auto batcher = MakeAudioTrainingBatcher(audio_entry, config, batch_size, AudioTrainingSplit(config));
            if (batcher->GetNumSamples() == 0) {
                error = "the audio dataset has 0 samples";
                return nullptr;
            }

            // input_size is feature_rows x feature_cols (the flattened feature
            // map): the model's first Dense layer sees this many features.
            ApplyAudioInputShape(*batcher, config);

            spdlog::info("TrainingManager: Audio dataset {} train / {} validation / {} test clips, input_size={} "
                         "({}x{}), num_workers={}, split seed={}", batcher->SamplesIn(BatcherPhase::Train),
                         batcher->SamplesIn(BatcherPhase::Val), batcher->SamplesIn(BatcherPhase::Test),
                         config.input_size, batcher->GetFeatureRows(), batcher->GetFeatureCols(), config.num_workers,
                         config.split_seed);

            // A Validation or Test audio dataset wired to the Data Split (the
            // compiler gave its share to train): read whole as that part, with
            // the same features. Its classes must be the training set's, in the
            // same order, and its features the same shape, or the model's inputs
            // and outputs would not line up.
            std::shared_ptr<AudioDatasetBatcher> wired_dev, wired_test;
            for (auto [role, wired] : {std::pair{&config.dataset_roles.dev, &wired_dev},
                                       std::pair{&config.dataset_roles.test, &wired_test}}) {
                if (!role->IsSupplied() || role->joined_to_training) continue;
                const bool dev = role == &config.dataset_roles.dev;
                const std::string part = dev ? "validation" : "test";
                const auto* entry = DataRegistry::Instance().GetAudioDatasetEntry(role->dataset_name);
                if (!entry) {
                    error = "'" + role->dataset_name + "' is wired as " + part + " but is not a loaded audio dataset";
                    return nullptr;
                }
                if (!entry->class_names.empty() && !audio_entry.class_names.empty() &&
                    entry->class_names != audio_entry.class_names) {
                    error = "'" + role->dataset_name + "' (wired as " + part +
                            ") has other classes than the training dataset; they must be the same, in the same order";
                    return nullptr;
                }
                auto set = MakeAudioTrainingBatcher(*entry, config, batch_size,
                                                    SampleSplitSettings{0.0f, dev ? 1.0f : 0.0f, dev ? 0.0f : 1.0f,
                                                                        false,
                                                                        static_cast<uint32_t>(config.split_seed)});
                if (set->SamplesIn(dev ? BatcherPhase::Val : BatcherPhase::Test) == 0) {
                    error = "'" + role->dataset_name + "' (wired as " + part + ") has no clips";
                    return nullptr;
                }
                if (set->GetFeatureRows() != batcher->GetFeatureRows() ||
                    set->GetFeatureCols() != batcher->GetFeatureCols()) {
                    error = "'" + role->dataset_name + "' (wired as " + part + ") gives " +
                            std::to_string(set->GetFeatureRows()) + "x" + std::to_string(set->GetFeatureCols()) +
                            " features, the training dataset " + std::to_string(batcher->GetFeatureRows()) + "x" +
                            std::to_string(batcher->GetFeatureCols()) +
                            "; load it with the same sample rate and duration";
                    return nullptr;
                }
                spdlog::info("TrainingManager: '{}' is the {} part: {} clips", role->dataset_name, part,
                             set->SamplesIn(dev ? BatcherPhase::Val : BatcherPhase::Test));
                *wired = std::move(set);
            }

            // One batcher, three parts by phase (the held-out test runs at the
            // end); a wired role reads its own set.
            ResolvedExternalBatchers batchers;
            batchers.train = batcher;
            if (wired_dev) batchers.dev = wired_dev;
            else if (batcher->SamplesIn(BatcherPhase::Val) > 0) batchers.dev = batcher;
            if (wired_test) batchers.test = wired_test;
            else if (batcher->SamplesIn(BatcherPhase::Test) > 0) batchers.test = batcher;
            return std::make_unique<TrainingExecutor>(std::move(config), std::move(batchers));
        });
}
#endif

}  // namespace cyxwiz
