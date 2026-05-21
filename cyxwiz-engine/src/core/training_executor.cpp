#include "training_executor.h"
#include "checkpoint_manager.h"
#include "crash_run_recorder.h"
#include <cyxwiz/debug_hooks.h>
#include "training_trace_collector.h"
#include "model_builder.h"
#include "data_registry.h"
#include "../preprocessing/preprocessing_config.h"
#include "../preprocessing/statistics_calculator.h"
#include "../plugin/registries/plugin_training_hook_manager.h"
#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <limits>

namespace cyxwiz {

// ============================================================================
// TrainingExecutor Implementation
// ============================================================================

TrainingExecutor::TrainingExecutor(TrainingConfiguration config, DatasetHandle dataset)
    : config_(std::move(config))
    , dataset_(dataset)
    , mode_(DatasetMode::Legacy)
{
    spdlog::info("TrainingExecutor: Created with {} layers, input_size={}, output_size={}",
                 config_.layers.size(), config_.input_size, config_.output_size);
}

TrainingExecutor::TrainingExecutor(TrainingConfiguration config,
                                   std::shared_ptr<ArrowDataset> arrow_dataset,
                                   const std::string& label_column)
    : config_(std::move(config))
    , arrow_dataset_(arrow_dataset)
    , label_column_(label_column)
    , mode_(DatasetMode::Arrow)
{
    spdlog::info("TrainingExecutor: Created with Arrow dataset ({} rows, {} cols), label='{}'",
                 arrow_dataset_ ? arrow_dataset_->GetNumRows() : 0,
                 arrow_dataset_ ? arrow_dataset_->GetNumColumns() : 0,
                 label_column_);
    spdlog::info("TrainingExecutor: Model has {} layers, input_size={}, output_size={}",
                 config_.layers.size(), config_.input_size, config_.output_size);
}

TrainingExecutor::TrainingExecutor(TrainingConfiguration config,
                                   std::shared_ptr<ParquetBackedDataset> parquet_dataset,
                                   const std::string& label_column)
    : config_(std::move(config))
    , parquet_dataset_(parquet_dataset)
    , label_column_(label_column)
    , mode_(DatasetMode::Parquet)
{
    spdlog::info("TrainingExecutor: Created with Parquet-backed dataset "
                 "({} rows, {} cols, {} row groups, {:.1f} MB on disk), label='{}'",
                 parquet_dataset_ ? parquet_dataset_->GetNumRows() : 0,
                 parquet_dataset_ ? parquet_dataset_->GetNumColumns() : 0,
                 parquet_dataset_ ? parquet_dataset_->GetNumRowGroups() : 0,
                 parquet_dataset_ ? parquet_dataset_->GetFileSizeBytes() / (1024.0 * 1024.0) : 0.0,
                 label_column_);
    spdlog::info("TrainingExecutor: Model has {} layers, input_size={}, output_size={}",
                 config_.layers.size(), config_.input_size, config_.output_size);
}

TrainingExecutor::TrainingExecutor(TrainingConfiguration config,
                                   std::unique_ptr<IBatcher> external_batcher)
    : config_(std::move(config))
    , external_batcher_(std::move(external_batcher))
    , mode_(DatasetMode::Image)
{
    spdlog::info("TrainingExecutor: Created with external IBatcher, "
                 "{} layers, input_size={}, output_size={}",
                 config_.layers.size(), config_.input_size, config_.output_size);
}

TrainingExecutor::~TrainingExecutor() {
    Stop();
}

bool TrainingExecutor::Initialize(int /*batch_size*/) {
    auto built = BuildSequentialFromConfig(config_);
    if (!built.ok()) {
        spdlog::error("TrainingExecutor: Failed to build model from config");
        return false;
    }
    model_ = std::move(built.model);
    loss_ = std::move(built.loss);
    optimizer_ = std::move(built.optimizer);
    return true;
}

void TrainingExecutor::Train(
    int epochs,
    int batch_size,
    BatchCallback batch_cb,
    EpochCallback epoch_cb,
    TrainingCompleteCallback complete_cb)
{
    if (is_training_.load()) {
        spdlog::warn("TrainingExecutor: Already training");
        return;
    }

    is_training_.store(true);
    stop_requested_.store(false);
    is_paused_.store(false);

    try {
        // Initialize
        if (!Initialize(batch_size)) {
            spdlog::error("TrainingExecutor: Failed to initialize");
            is_training_.store(false);
            return;
        }

    // Setup metrics
    UpdateMetrics([epochs](TrainingMetrics& m) {
        m.total_epochs = epochs;
        m.current_epoch = 0;
        m.is_training = true;
        m.is_complete = false;
        m.status_message = "Starting training...";
        m.loss_history.clear();
        m.accuracy_history.clear();
        m.val_loss_history.clear();
        m.val_accuracy_history.clear();
    });

    // Create batchers - Arrow in-memory, Parquet disk-backed, or legacy.
    // All three end up driving the training loop through IBatcher pointers.
    std::unique_ptr<ArrowDatasetBatcher> arrow_train_batcher;
    std::unique_ptr<ArrowDatasetBatcher> arrow_val_batcher;
    std::unique_ptr<ParquetArrowBatcher> parquet_train_batcher;
    std::unique_ptr<ParquetArrowBatcher> parquet_val_batcher;
    std::unique_ptr<DatasetBatcher> legacy_train_batcher;
    std::unique_ptr<DatasetBatcher> legacy_val_batcher;

    // Non-owning IBatcher pointers — point at whichever concrete batcher
    // the mode selected. The Arrow and Parquet paths both flow through the
    // same IBatcher-aware training loops; the legacy path stays on the
    // legacy-specific functions.
    IBatcher* active_train_ibatcher = nullptr;
    IBatcher* active_val_ibatcher = nullptr;

    size_t num_train_samples = 0;

    if (mode_ == DatasetMode::Arrow) {
        // Arrow-based batching (Data Studio)
        spdlog::info("TrainingExecutor: Using Arrow dataset for training "
                     "(batch_size={}, shuffle={}, train_ratio={:.2f}, time_series={})",
                     batch_size, config_.shuffle, config_.train_ratio,
                     config_.is_time_series);

        // Phase 4 Time-Series dispatch. When GraphCompiler detected a
        // TimeSeriesWindow node, the Arrow table handed to us has
        // already been materialized by PipelineMaterializer — windowed
        // rows and an `__partition__` int8 column. Build the train and
        // val batchers by filtering rows on partition value instead of
        // the legacy first-N% train_split slicing, and flip regression
        // mode on so labels round-trip as float.
        const std::string partition_col = config_.is_time_series
            ? "__partition__" : "";
        // For time-series the effective label column is always the
        // `y` emitted by TimeSeriesWindowOperator. Override whatever
        // the DataInput node recorded (which was a column from the
        // pre-materialization source table, not the windowed one).
        const std::string effective_label =
            config_.is_time_series ? "y" : label_column_;

        // Honor DataLoader/DataSplit node config (or defaults if no such nodes)
        // Validation batcher never shuffles regardless of config.
        arrow_train_batcher = std::make_unique<ArrowDatasetBatcher>(
            arrow_dataset_, effective_label, batch_size,
            config_.shuffle, config_.train_ratio, true,
            partition_col, /*partition_value=*/0, config_.num_workers);
        arrow_val_batcher = std::make_unique<ArrowDatasetBatcher>(
            arrow_dataset_, effective_label, batch_size,
            false, config_.train_ratio, false,
            partition_col, /*partition_value=*/1, config_.num_workers);

        if (config_.drop_last) {
            spdlog::warn("TrainingExecutor: drop_last=true requested but ArrowDatasetBatcher "
                         "does not yet support it - last partial batch will be kept");
        }
        if (!config_.is_time_series &&
            config_.has_data_split && config_.test_ratio > 0.01f) {
            spdlog::warn("TrainingExecutor: test_ratio={:.2f} configured on DataSplit but "
                         "ArrowDatasetBatcher has no held-out test split - the test portion "
                         "will be merged into validation. train={:.2f}, val+test={:.2f}",
                         config_.test_ratio, config_.train_ratio, 1.0f - config_.train_ratio);
        }

        // Apply preprocessing from config
        if (config_.preprocessing.has_normalization) {
            arrow_train_batcher->SetNormalization(config_.preprocessing.norm_mean,
                                                   config_.preprocessing.norm_std);
            arrow_val_batcher->SetNormalization(config_.preprocessing.norm_mean,
                                                 config_.preprocessing.norm_std);
        }

        if (config_.is_time_series) {
            // Time-series regression: float labels [batch, 1], no one-hot.
            arrow_train_batcher->SetRegressionMode(true);
            arrow_val_batcher->SetRegressionMode(true);
        } else if (config_.preprocessing.has_onehot) {
            // One-hot encoding for classification
            arrow_train_batcher->SetOneHotEncoding(config_.preprocessing.num_classes);
            arrow_val_batcher->SetOneHotEncoding(config_.preprocessing.num_classes);
        } else {
            // Default to output_size for classification
            arrow_train_batcher->SetOneHotEncoding(config_.output_size);
            arrow_val_batcher->SetOneHotEncoding(config_.output_size);
        }

        num_train_samples = arrow_train_batcher->GetNumSamples();
        active_train_ibatcher = arrow_train_batcher.get();
        active_val_ibatcher = arrow_val_batcher.get();
    } else if (mode_ == DatasetMode::Parquet) {
        // Disk-backed Parquet batching — rows are fetched lazily from the
        // memory-mapped Parquet cache one row group at a time. Same output
        // shape as the Arrow path, so training loops don't need to know.
        spdlog::info("TrainingExecutor: Using Parquet-backed dataset for training "
                     "(batch_size={}, shuffle={}, train_ratio={:.2f})",
                     batch_size, config_.shuffle, config_.train_ratio);

        parquet_train_batcher = std::make_unique<ParquetArrowBatcher>(
            parquet_dataset_, label_column_, batch_size,
            config_.shuffle, config_.train_ratio, true, config_.num_workers);
        parquet_val_batcher = std::make_unique<ParquetArrowBatcher>(
            parquet_dataset_, label_column_, batch_size,
            false, config_.train_ratio, false, config_.num_workers);

        if (config_.drop_last) {
            spdlog::warn("TrainingExecutor: drop_last=true requested but ParquetArrowBatcher "
                         "does not yet support it - last partial batch will be kept");
        }
        if (config_.has_data_split && config_.test_ratio > 0.01f) {
            spdlog::warn("TrainingExecutor: test_ratio={:.2f} configured on DataSplit but "
                         "ParquetArrowBatcher has no held-out test split - the test portion "
                         "will be merged into validation. train={:.2f}, val+test={:.2f}",
                         config_.test_ratio, config_.train_ratio, 1.0f - config_.train_ratio);
        }

        // Apply preprocessing from config (same as Arrow path).
        if (config_.preprocessing.has_normalization) {
            parquet_train_batcher->SetNormalization(config_.preprocessing.norm_mean,
                                                     config_.preprocessing.norm_std);
            parquet_val_batcher->SetNormalization(config_.preprocessing.norm_mean,
                                                   config_.preprocessing.norm_std);
        }

        if (config_.preprocessing.has_onehot) {
            parquet_train_batcher->SetOneHotEncoding(config_.preprocessing.num_classes);
            parquet_val_batcher->SetOneHotEncoding(config_.preprocessing.num_classes);
        } else {
            parquet_train_batcher->SetOneHotEncoding(config_.output_size);
            parquet_val_batcher->SetOneHotEncoding(config_.output_size);
        }

        num_train_samples = parquet_train_batcher->GetNumSamples();
        active_train_ibatcher = parquet_train_batcher.get();
        active_val_ibatcher = parquet_val_batcher.get();
    } else if (mode_ == DatasetMode::Image) {
        // Image mode: the batcher was constructed externally by
        // StartTrainingImage with the correct target size and
        // augmentation pipeline. We just point the training loop at it.
        // Validation uses the same batcher reset for now — Phase 1.4
        // can add a separate validation batcher without augmentation.
        spdlog::info("TrainingExecutor: Using external batcher for training "
                     "(batch_size={}, num_workers={}, {} samples)",
                     batch_size, config_.num_workers,
                     external_batcher_ ? external_batcher_->GetNumSamples() : 0);

        if (!external_batcher_) {
            spdlog::error("TrainingExecutor: external batcher mode but no external batcher");
            return;
        }

        num_train_samples = external_batcher_->GetNumSamples();
        active_train_ibatcher = external_batcher_.get();
        active_val_ibatcher = external_batcher_.get();
    } else {
        // Legacy DatasetHandle batching
        spdlog::info("TrainingExecutor: Using legacy dataset for training "
                     "(batch_size={}, shuffle={}, drop_last={}, num_workers={})",
                     batch_size, config_.shuffle, config_.drop_last, config_.num_workers);

        // Honor DataLoader node config (or defaults if no such node).
        // Validation batcher never shuffles and never drops the last batch.
        legacy_train_batcher = std::make_unique<DatasetBatcher>(
            dataset_, batch_size, DatasetSplit::Train,
            config_.shuffle, config_.drop_last, config_.num_workers);
        legacy_val_batcher = std::make_unique<DatasetBatcher>(
            dataset_, batch_size, DatasetSplit::Validation, false, false, config_.num_workers);

        // Apply NEW preprocessing pipeline (if configured)
        std::string dataset_name = dataset_.GetName();
        DataRegistry& registry = DataRegistry::Instance();

        if (registry.HasPreprocessingConfig(dataset_name)) {
            spdlog::info("TrainingExecutor: Found preprocessing config for dataset '{}'", dataset_name);

            PreprocessingConfig preprocessing_config = registry.GetPreprocessingConfig(dataset_name);

            if (preprocessing_config.enabled) {
                legacy_train_batcher->SetPreprocessingConfig(preprocessing_config);
                legacy_val_batcher->SetPreprocessingConfig(preprocessing_config);

                spdlog::info("TrainingExecutor: Computing dataset statistics...");
                DatasetStatistics stats = StatisticsCalculator::Compute(
                    dataset_name, &registry,
                    [](float progress) {
                        spdlog::debug("Statistics computation: {:.1f}%", progress * 100.0f);
                    }
                );

                if (stats.is_valid) {
                    legacy_train_batcher->InitializePreprocessing(stats);
                    legacy_val_batcher->InitializePreprocessing(stats);
                    spdlog::info("TrainingExecutor: Preprocessing pipeline initialized");
                }
            }
        }

        // Load augmentation pipeline
        std::string dataset_name_aug = dataset_.GetName();
        if (registry.HasAugmentationPipeline(dataset_name_aug)) {
            auto aug_pipeline = registry.GetAugmentationPipeline(dataset_name_aug);
            if (aug_pipeline) {
                legacy_train_batcher->SetAugmentationPipeline(aug_pipeline);
                legacy_train_batcher->SetApplyAugmentationOnTrain(true);
            }
        }

        // Apply OLD preprocessing settings
        if (config_.preprocessing.has_normalization) {
            legacy_train_batcher->SetLegacyNormalization(config_.preprocessing.norm_mean,
                                                         config_.preprocessing.norm_std);
            legacy_val_batcher->SetLegacyNormalization(config_.preprocessing.norm_mean,
                                                       config_.preprocessing.norm_std);
        }

        if (config_.preprocessing.has_onehot) {
            legacy_train_batcher->SetLegacyOneHotEncoding(config_.preprocessing.num_classes);
            legacy_val_batcher->SetLegacyOneHotEncoding(config_.preprocessing.num_classes);
        }

        legacy_train_batcher->SetFlatten(true);
        legacy_val_batcher->SetFlatten(true);

        num_train_samples = legacy_train_batcher->GetNumSamples();
    }

    spdlog::info("TrainingExecutor: Starting training for {} epochs, batch_size={}, samples={}",
                 epochs, batch_size, num_train_samples);

    spdlog::debug("TrainingExecutor: Step 1 - Notifying plugin hooks");
    // Notify plugin hooks: training start
    {
        cyxwiz::plugin::TrainingContext ctx;
        ctx.total_epochs = epochs;
        ctx.learning_rate = config_.learning_rate;
        cyxwiz::plugin::PluginTrainingHookManager::Instance().NotifyTrainingStart(ctx);
    }

    spdlog::debug("TrainingExecutor: Step 2 - Setting model to training mode");
    // Set model to training mode
    model_->SetTraining(true);

    spdlog::debug("TrainingExecutor: Step 3 - Entering training loop");
    CrashRunRecorder::Instance().StartTrainingRun(config_, epochs, batch_size, num_train_samples);
    BackendDebugHooks::SetDebugEventCallback([](const std::string& source,
                                                const std::string& message) {
        if (source.rfind("Model", 0) == 0) {
            TrainingTraceCollector::Instance().RecordRuntimeEvent(source, message);
        } else {
            CrashRunRecorder::Instance().MarkBackendEvent(source, message);
            TrainingTraceCollector::Instance().RecordRuntimeWarning(source, message);
        }
    });
    const auto last_run = CrashRunRecorder::LoadLastRun();
    TrainingTraceCollector::Instance().StartRun(
        last_run ? last_run->run_id : "training-run");

    std::unique_ptr<CheckpointManager> checkpoint_manager;
    float best_val_loss = std::numeric_limits<float>::infinity();
    int epochs_without_improvement = 0;
    const int early_stopping_patience = std::max(0, config_.early_stopping_patience);
    const bool save_best_checkpoint = config_.save_best_checkpoint;
    bool validation_ran = false;

    std::filesystem::path checkpoint_root = config_.checkpoint_dir.empty()
        ? (std::filesystem::current_path() / ".cyxwiz" / "checkpoints")
        : std::filesystem::path(config_.checkpoint_dir);
    checkpoint_root /= last_run ? last_run->run_id : "training-run";
    checkpoint_manager = std::make_unique<CheckpointManager>(checkpoint_root.string());

    // Training loop
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        spdlog::debug("TrainingExecutor: Epoch {} starting", epoch);
        if (ShouldStop()) break;
        // Check plugin early stopping
        {
            cyxwiz::plugin::TrainingContext stop_ctx;
            stop_ctx.current_epoch = epoch;
            stop_ctx.total_epochs = epochs;
            stop_ctx.learning_rate = config_.learning_rate;
            if (cyxwiz::plugin::PluginTrainingHookManager::Instance().ShouldStopEarly(stop_ctx)) {
                spdlog::info("TrainingExecutor: Plugin requested early stop");
                break;
            }
        }
        WaitWhilePaused();

        auto epoch_start = std::chrono::steady_clock::now();

        UpdateMetrics([epoch](TrainingMetrics& m) {
            m.current_epoch = epoch;
            m.status_message = "Training epoch " + std::to_string(epoch) + "...";
        });

        // Notify plugin hooks: epoch start
        {
            cyxwiz::plugin::TrainingContext ctx;
            ctx.current_epoch = epoch;
            ctx.total_epochs = epochs;
            ctx.learning_rate = config_.learning_rate;
            cyxwiz::plugin::PluginTrainingHookManager::Instance().NotifyEpochStart(ctx);
        }

        spdlog::debug("TrainingExecutor: About to call RunTrainingEpoch");
        // Run training epoch - dispatch by dataset mode. Arrow and Parquet
        // batchers both flow through RunTrainingEpochArrow via their shared
        // IBatcher base; legacy DatasetBatcher stays on its own path.
        if (mode_ == DatasetMode::Legacy) {
            RunTrainingEpoch(*legacy_train_batcher, epoch, batch_cb);
        } else if (active_train_ibatcher) {
            RunTrainingEpochArrow(*active_train_ibatcher, epoch, batch_cb);
        }

        if (ShouldStop()) break;

        // Run validation (eval mode).
        //
        // For image/audio batchers, active_train_ibatcher and
        // active_val_ibatcher point to the *same* instance — the batcher
        // holds both train_indices_ and val_indices_ internally and switches
        // between them via SetPhase. Without SetPhase(Val) the val pass
        // would iterate the training indices, producing bogus "perfect val"
        // metrics (this was the source of the suspicious 100% val acc).
        // For Arrow/Parquet/legacy paths these are separate instances so
        // SetPhase is a no-op on the default IBatcher impl.
        model_->SetTraining(false);
        if (mode_ == DatasetMode::Legacy) {
            RunValidation(*legacy_val_batcher);
            validation_ran = true;
        } else if (active_val_ibatcher) {
            active_val_ibatcher->SetPhase(BatcherPhase::Val);
            RunValidationArrow(*active_val_ibatcher);
            active_val_ibatcher->SetPhase(BatcherPhase::Train);
            validation_ran = true;
        }
        model_->SetTraining(true);

        auto epoch_end = std::chrono::steady_clock::now();
        float epoch_time = std::chrono::duration<float>(epoch_end - epoch_start).count();

        // Get current metrics for callback
        TrainingMetrics current = GetMetrics();

        // Compute samples per second
        float samples_per_sec = static_cast<float>(num_train_samples) / epoch_time;

        // Update history
        UpdateMetrics([&](TrainingMetrics& m) {
            m.epoch_time_seconds = epoch_time;
            m.samples_per_second = samples_per_sec;
            m.loss_history.push_back(m.train_loss);
            m.accuracy_history.push_back(m.train_accuracy);
            m.val_loss_history.push_back(m.val_loss);
            m.val_accuracy_history.push_back(m.val_accuracy);
        });

        // Epoch callback
        if (epoch_cb) {
            epoch_cb(epoch, current.train_loss, current.train_accuracy,
                     current.val_loss, current.val_accuracy, epoch_time);
        }

        spdlog::info("Epoch {}/{}: loss={:.4f}, acc={:.2f}%, val_loss={:.4f}, val_acc={:.2f}% ({:.1f}s, {:.0f} samples/sec)",
                     epoch, epochs, current.train_loss, current.train_accuracy * 100,
                     current.val_loss, current.val_accuracy * 100, epoch_time, samples_per_sec);

        if (validation_ran && checkpoint_manager && save_best_checkpoint && std::isfinite(current.val_loss)) {
            if (current.val_loss < best_val_loss) {
                best_val_loss = current.val_loss;
                epochs_without_improvement = 0;
                if (checkpoint_manager->SaveBestModel(*model_, optimizer_.get(), current, current.val_loss)) {
                    spdlog::info("TrainingExecutor: Best validation checkpoint saved at epoch {} (val_loss={:.4f})",
                                 epoch, current.val_loss);
                }
            } else {
                ++epochs_without_improvement;
                spdlog::info("TrainingExecutor: No validation improvement for {} epoch(s) (best val_loss={:.4f})",
                             epochs_without_improvement, best_val_loss);
                if (early_stopping_patience > 0 &&
                    epochs_without_improvement >= early_stopping_patience) {
                    spdlog::info("TrainingExecutor: Early stopping after {} non-improving epoch(s) (patience={})",
                                 epochs_without_improvement, early_stopping_patience);
                    UpdateMetrics([](TrainingMetrics& m) {
                        m.status_message = "Early stopping: validation loss plateaued";
                    });
                    break;
                }
            }
        }

        // Notify plugin hooks: epoch end
        {
            cyxwiz::plugin::TrainingContext ctx;
            ctx.current_epoch = epoch;
            ctx.total_epochs = epochs;
            ctx.train_loss = current.train_loss;
            ctx.train_accuracy = current.train_accuracy;
            ctx.val_loss = current.val_loss;
            ctx.val_accuracy = current.val_accuracy;
            ctx.learning_rate = config_.learning_rate;
            cyxwiz::plugin::PluginTrainingHookManager::Instance().NotifyEpochEnd(ctx);
        }

        // Reset batchers for next epoch
        if (mode_ == DatasetMode::Legacy) {
            legacy_train_batcher->Reset();
            legacy_val_batcher->Reset();
        } else {
            active_train_ibatcher->Reset();
            active_val_ibatcher->Reset();
        }
    }

    TrainingMetrics final_metrics = GetMetrics();

    // Mark complete
    UpdateMetrics([](TrainingMetrics& m) {
        m.is_training = false;
        m.is_complete = true;
        m.status_message = "Training complete";
    });

    if (!stop_requested_.load() && checkpoint_manager && save_best_checkpoint) {
        const std::string best_checkpoint = checkpoint_manager->GetBestCheckpoint();
        if (!best_checkpoint.empty()) {
            auto restored = checkpoint_manager->LoadCheckpoint(*model_, optimizer_.get(), "best");
            if (restored) {
                final_metrics.current_epoch = restored->epoch;
                final_metrics.current_batch = restored->global_step;
                final_metrics.train_loss = restored->train_loss;
                final_metrics.train_accuracy = restored->train_accuracy;
                final_metrics.val_loss = restored->val_loss;
                final_metrics.val_accuracy = restored->val_accuracy;
                final_metrics.loss_history = restored->loss_history;
                final_metrics.accuracy_history = restored->accuracy_history;
                final_metrics.val_loss_history = restored->val_loss_history;
                final_metrics.val_accuracy_history = restored->val_accuracy_history;
                UpdateMetrics([&](TrainingMetrics& m) {
                    m.current_epoch = restored->epoch;
                    m.current_batch = restored->global_step;
                    m.train_loss = restored->train_loss;
                    m.train_accuracy = restored->train_accuracy;
                    m.val_loss = restored->val_loss;
                    m.val_accuracy = restored->val_accuracy;
                    m.loss_history = restored->loss_history;
                    m.accuracy_history = restored->accuracy_history;
                    m.val_loss_history = restored->val_loss_history;
                    m.val_accuracy_history = restored->val_accuracy_history;
                    m.status_message = "Restored best validation checkpoint";
                });
                spdlog::info("TrainingExecutor: Restored best checkpoint from epoch {} (val_loss={:.4f})",
                             restored->epoch, restored->val_loss);
            }
        }
    }

    // Notify plugin hooks: training end
    {
        cyxwiz::plugin::TrainingContext ctx;
        ctx.current_epoch = final_metrics.current_epoch;
        ctx.total_epochs = final_metrics.total_epochs;
        ctx.train_loss = final_metrics.train_loss;
        ctx.train_accuracy = final_metrics.train_accuracy;
        ctx.val_loss = final_metrics.val_loss;
        ctx.val_accuracy = final_metrics.val_accuracy;
        ctx.learning_rate = config_.learning_rate;
        cyxwiz::plugin::PluginTrainingHookManager::Instance().NotifyTrainingEnd(ctx);
    }

    if (stop_requested_.load()) {
        CrashRunRecorder::Instance().MarkCancelled();
        TrainingTraceCollector::Instance().FinishRun("cancelled");
    } else {
        CrashRunRecorder::Instance().MarkCompleted();
        TrainingTraceCollector::Instance().FinishRun("completed");
    }
    BackendDebugHooks::SetDebugEventCallback({});

    is_training_.store(false);

    // Complete callback
    if (complete_cb) {
        complete_cb(GetMetrics());
    }

    spdlog::info("TrainingExecutor: Training complete");
    } catch (const std::exception& e) {
        CrashRunRecorder::Instance().MarkFailed(e.what());
        TrainingTraceCollector::Instance().FinishRun("failed");
        BackendDebugHooks::SetDebugEventCallback({});
        is_training_.store(false);
        spdlog::error("TrainingExecutor: Training failed: {}", e.what());
        throw;
    } catch (...) {
        CrashRunRecorder::Instance().MarkFailed("unknown native exception");
        TrainingTraceCollector::Instance().FinishRun("failed");
        BackendDebugHooks::SetDebugEventCallback({});
        is_training_.store(false);
        spdlog::error("TrainingExecutor: Training failed with unknown exception");
        throw;
    }
}

void TrainingExecutor::RunTrainingEpoch(
    DatasetBatcher& batcher,
    int epoch,
    BatchCallback batch_cb)
{
    float epoch_loss = 0.0f;
    int correct = 0;
    int total = 0;
    int batch_num = 0;

    size_t total_batches = batcher.GetNumBatches();

    UpdateMetrics([total_batches](TrainingMetrics& m) {
        m.total_batches = static_cast<int>(total_batches);
        m.current_batch = 0;
    });

    // Epoch wall-clock start for the periodic progress log below —
    // mirrors RunTrainingEpochArrow so both training paths have the
    // same "training is alive" feedback loop.
    const auto epoch_start_time = std::chrono::steady_clock::now();

    while (!batcher.IsEpochComplete()) {
        if (ShouldStop()) break;
        WaitWhilePaused();

        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::GetNextBatch, epoch, batch_num + 1,
            static_cast<int>(total_batches));
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::GetNextBatch, epoch, batch_num + 1,
            static_cast<int>(total_batches));
        Batch batch = batcher.GetNextBatch();
        if (!batch.IsValid()) break;

        batch_num++;

        // Forward pass through model
        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::Forward, epoch, batch_num,
            static_cast<int>(total_batches));
        const auto forward_start = std::chrono::steady_clock::now();
        Tensor predictions = Forward(batch.data);
        const auto forward_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - forward_start).count();
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::Forward, epoch, batch_num,
            static_cast<int>(total_batches), 0.0f, 0.0f, forward_ms);

        // DEBUG: Log sample values for first batch of first epoch
        if (epoch == 1 && batch_num == 1) {
            const float* input_data = batch.data.Data<float>();
            const float* pred_data_debug = predictions.Data<float>();
            const float* target_data_debug = batch.labels.Data<float>();

            // Log input data range
            float min_input = input_data[0], max_input = input_data[0];
            const auto& input_shape = batch.data.Shape();
            if (input_shape.size() < 2) {
                spdlog::error("TrainingExecutor: Expected 2D input, got {}D", input_shape.size());
                break;
            }
            size_t input_size = input_shape[0] * input_shape[1];
            for (size_t i = 1; i < std::min(input_size, size_t(1000)); ++i) {
                min_input = std::min(min_input, input_data[i]);
                max_input = std::max(max_input, input_data[i]);
            }
            spdlog::info("DEBUG: Input data range: [{:.4f}, {:.4f}]", min_input, max_input);

            // Log first sample prediction
            spdlog::info("DEBUG: First sample predictions:");
            std::string pred_str = "  [";
            for (size_t c = 0; c < config_.output_size; ++c) {
                pred_str += fmt::format("{:.4f}", pred_data_debug[c]);
                if (c < config_.output_size - 1) pred_str += ", ";
            }
            pred_str += "]";
            spdlog::info("{}", pred_str);

            // Log first sample target
            spdlog::info("DEBUG: First sample target:");
            std::string target_str = "  [";
            for (size_t c = 0; c < config_.output_size; ++c) {
                target_str += fmt::format("{:.1f}", target_data_debug[c]);
                if (c < config_.output_size - 1) target_str += ", ";
            }
            target_str += "]";
            spdlog::info("{}", target_str);
        }

        // Compute loss
        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::ComputeLoss, epoch, batch_num,
            static_cast<int>(total_batches));
        float batch_loss = ComputeLoss(predictions, batch.labels);
        const std::string loss_status = std::isfinite(batch_loss) ? "ok" : "failed";
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::ComputeLoss, epoch, batch_num,
            static_cast<int>(total_batches), batch_loss, 0.0f, 0.0f,
            loss_status,
            std::isfinite(batch_loss) ? "" : "Training loss became NaN or Inf.");
        epoch_loss += batch_loss;

        // Compute accuracy
        const float* pred_data = predictions.Data<float>();
        const float* target_data = batch.labels.Data<float>();

        for (size_t b = 0; b < batch.size; ++b) {
            int pred_class = 0, true_class = 0;
            float max_pred = pred_data[b * config_.output_size];
            float max_target = target_data[b * config_.output_size];

            // Start from c=0 to properly compare all classes including class 0
            for (size_t c = 0; c < config_.output_size; ++c) {
                if (pred_data[b * config_.output_size + c] > max_pred) {
                    max_pred = pred_data[b * config_.output_size + c];
                    pred_class = static_cast<int>(c);
                }
                if (target_data[b * config_.output_size + c] > max_target) {
                    max_target = target_data[b * config_.output_size + c];
                    true_class = static_cast<int>(c);
                }
            }
            if (pred_class == true_class) correct++;
            total++;
        }

        // Backward pass
        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::Backward, epoch, batch_num,
            static_cast<int>(total_batches), batch_loss);
        const auto backward_start = std::chrono::steady_clock::now();
        Backward(predictions, batch.labels);
        const auto backward_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - backward_start).count();
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::Backward, epoch, batch_num,
            static_cast<int>(total_batches), batch_loss, 0.0f, backward_ms);

        // Update weights using optimizer
        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::UpdateParameters, epoch, batch_num,
            static_cast<int>(total_batches), batch_loss);
        model_->UpdateParameters(optimizer_.get());
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::UpdateParameters, epoch, batch_num,
            static_cast<int>(total_batches), batch_loss);

        // Update metrics
        float current_loss = epoch_loss / batch_num;
        float current_acc = static_cast<float>(correct) / total;

        UpdateMetrics([batch_num, current_loss, current_acc](TrainingMetrics& m) {
            m.current_batch = batch_num;
            m.train_loss = current_loss;
            m.train_accuracy = current_acc;
        });

        // Periodic progress log — mirror of RunTrainingEpochArrow's
        // version so non-Arrow training paths (legacy DatasetBatcher)
        // also get per-50-batch liveness signals.
        if (batch_num == 1 || batch_num % 50 == 0) {
            const auto now = std::chrono::steady_clock::now();
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - epoch_start_time).count();
            const float elapsed_s = elapsed_ms / 1000.0f;
            const float rate = elapsed_ms > 0
                ? (batch_num * 1000.0f / static_cast<float>(elapsed_ms))
                : 0.0f;
            spdlog::info("Epoch {} [{}/{}] loss={:.4f} acc={:.2f}% "
                         "({:.1f}s, {:.1f} batches/s)",
                         epoch, batch_num, total_batches,
                         current_loss, current_acc * 100.0f,
                         elapsed_s, rate);
        }

        // Batch callback
        if (batch_cb) {
            CrashRunRecorder::Instance().MarkStage(
                TrainingTraceStage::BatchCallback, epoch, batch_num,
                static_cast<int>(total_batches), batch_loss, current_acc);
            TrainingTraceCollector::Instance().RecordStage(
                TrainingTraceStage::BatchCallback, epoch, batch_num,
                static_cast<int>(total_batches), batch_loss, current_acc);
            batch_cb(epoch, batch_num, static_cast<int>(total_batches), batch_loss, current_acc);
        }
    }

    // Final epoch metrics
    float final_loss = batch_num > 0 ? epoch_loss / batch_num : 0.0f;
    float final_acc = total > 0 ? static_cast<float>(correct) / total : 0.0f;

    UpdateMetrics([final_loss, final_acc](TrainingMetrics& m) {
        m.train_loss = final_loss;
        m.train_accuracy = final_acc;
    });
    CrashRunRecorder::Instance().MarkStage(
        TrainingTraceStage::EpochComplete, epoch, batch_num,
        static_cast<int>(total_batches), final_loss, final_acc);
    TrainingTraceCollector::Instance().RecordStage(
        TrainingTraceStage::EpochComplete, epoch, batch_num,
        static_cast<int>(total_batches), final_loss, final_acc);
}

void TrainingExecutor::RunValidation(DatasetBatcher& batcher) {
    float val_loss = 0.0f;
    int correct = 0;
    int total = 0;
    int batch_num = 0;

    batcher.Reset();

    while (!batcher.IsEpochComplete()) {
        if (ShouldStop()) break;

        Batch batch = batcher.GetNextBatch();
        if (!batch.IsValid()) break;

        batch_num++;

        // Forward pass only (no backprop)
        Tensor predictions = Forward(batch.data);

        // Compute loss
        float batch_loss = ComputeLoss(predictions, batch.labels);
        val_loss += batch_loss;

        // Compute accuracy
        const float* pred_data = predictions.Data<float>();
        const float* target_data = batch.labels.Data<float>();

        for (size_t b = 0; b < batch.size; ++b) {
            int pred_class = 0, true_class = 0;
            float max_pred = pred_data[b * config_.output_size];
            float max_target = target_data[b * config_.output_size];

            // Start from c=0 to properly compare all classes including class 0
            for (size_t c = 0; c < config_.output_size; ++c) {
                if (pred_data[b * config_.output_size + c] > max_pred) {
                    max_pred = pred_data[b * config_.output_size + c];
                    pred_class = static_cast<int>(c);
                }
                if (target_data[b * config_.output_size + c] > max_target) {
                    max_target = target_data[b * config_.output_size + c];
                    true_class = static_cast<int>(c);
                }
            }
            if (pred_class == true_class) correct++;
            total++;
        }
    }

    float final_loss = batch_num > 0 ? val_loss / batch_num : 0.0f;
    float final_acc = total > 0 ? static_cast<float>(correct) / total : 0.0f;

    UpdateMetrics([final_loss, final_acc](TrainingMetrics& m) {
        m.val_loss = final_loss;
        m.val_accuracy = final_acc;
    });
}

Tensor TrainingExecutor::Forward(const Tensor& input) {
    if (!model_) {
        spdlog::error("TrainingExecutor::Forward: Model not initialized");
        return Tensor();
    }

    last_predictions_ = model_->Forward(input);
    return last_predictions_;
}

float TrainingExecutor::ComputeLoss(const Tensor& predictions, const Tensor& targets) {
    if (!loss_) {
        spdlog::error("TrainingExecutor::ComputeLoss: No loss function");
        return 0.0f;
    }

    Tensor loss_tensor = loss_->Forward(predictions, targets);
    const float* loss_data = loss_tensor.Data<float>();
    return loss_data[0];
}

float TrainingExecutor::ComputeAccuracy(const Tensor& predictions, const Tensor& targets) {
    const auto& shape = predictions.Shape();
    if (shape.size() != 2) return 0.0f;

    size_t batch_size = shape[0];
    size_t num_classes = shape[1];

    const float* pred_data = predictions.Data<float>();
    const float* target_data = targets.Data<float>();

    int correct = 0;
    for (size_t b = 0; b < batch_size; ++b) {
        int pred_class = 0, true_class = 0;
        float max_pred = pred_data[b * num_classes];
        float max_target = target_data[b * num_classes];

        // Start from c=0 to properly compare all classes including class 0
        for (size_t c = 0; c < num_classes; ++c) {
            if (pred_data[b * num_classes + c] > max_pred) {
                max_pred = pred_data[b * num_classes + c];
                pred_class = static_cast<int>(c);
            }
            if (target_data[b * num_classes + c] > max_target) {
                max_target = target_data[b * num_classes + c];
                true_class = static_cast<int>(c);
            }
        }
        if (pred_class == true_class) correct++;
    }

    return static_cast<float>(correct) / batch_size;
}

void TrainingExecutor::Backward(const Tensor& predictions, const Tensor& targets) {
    if (!model_) {
        spdlog::error("TrainingExecutor::Backward: Model not initialized");
        return;
    }

    if (!loss_) {
        spdlog::error("TrainingExecutor::Backward: No loss function");
        return;
    }

    // Compute loss gradient
    Tensor grad = loss_->Backward(predictions, targets);

    // AfToTensor can flatten trailing-1 dimensions (e.g. [16,1] → [16])
    // when converting from column-major ArrayFire arrays back to row-major
    // CyxWiz tensors. The model's backward pass expects the gradient to
    // match the forward output shape exactly. Re-wrap the raw buffer with
    // the correct shape if dimensions dropped.
    if (grad.Shape() != predictions.Shape() &&
        grad.NumElements() == predictions.NumElements()) {
        grad = Tensor(predictions.Shape(), grad.Data<float>());
    }

    // Backward through model
    model_->Backward(grad);
}

void TrainingExecutor::Stop() {
    stop_requested_.store(true);
    is_paused_.store(false);  // Unpause so thread can exit
}

void TrainingExecutor::Pause() {
    is_paused_.store(true);
    UpdateMetrics([](TrainingMetrics& m) {
        m.is_paused = true;
        m.status_message = "Training paused";
    });
}

void TrainingExecutor::Resume() {
    is_paused_.store(false);
    UpdateMetrics([](TrainingMetrics& m) {
        m.is_paused = false;
        m.status_message = "Training resumed";
    });
}

TrainingMetrics TrainingExecutor::GetMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void TrainingExecutor::UpdateMetrics(const std::function<void(TrainingMetrics&)>& updater) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    updater(metrics_);
}

void TrainingExecutor::WaitWhilePaused() {
    while (is_paused_.load() && !stop_requested_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void TrainingExecutor::PreprocessBatch(Batch& /*batch*/) {
    // Preprocessing is handled by DatasetBatcher
}

// =============================================================================
// Arrow-specific training methods
// =============================================================================

void TrainingExecutor::RunTrainingEpochArrow(
    IBatcher& batcher,
    int epoch,
    BatchCallback batch_cb)
{
    spdlog::debug("RunTrainingEpochArrow: Entered, epoch={}", epoch);

    float epoch_loss = 0.0f;
    int correct = 0;
    int total = 0;
    int batch_num = 0;

    spdlog::debug("RunTrainingEpochArrow: Getting num batches");
    size_t total_batches = batcher.GetNumBatches();
    spdlog::debug("RunTrainingEpochArrow: total_batches={}", total_batches);

    UpdateMetrics([total_batches](TrainingMetrics& m) {
        m.total_batches = static_cast<int>(total_batches);
        m.current_batch = 0;
    });

    // Epoch wall-clock start for the periodic progress log below.
    // Without per-batch logging the training run goes completely
    // silent between the first-batch debug dump and the epoch-end
    // summary, which can be 100+ seconds on a real dataset — making
    // it impossible to tell "alive but slow" from "hung". This is
    // the fix for the tofix.md entry "Training logs silent mid-epoch".
    const auto epoch_start_time = std::chrono::steady_clock::now();

    spdlog::debug("RunTrainingEpochArrow: Entering batch loop");
    while (!batcher.IsEpochComplete()) {
        if (ShouldStop()) break;
        WaitWhilePaused();

        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::GetNextBatch, epoch, batch_num + 1,
            static_cast<int>(total_batches));
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::GetNextBatch, epoch, batch_num + 1,
            static_cast<int>(total_batches));
        Batch batch = batcher.GetNextBatch();
        if (!batch.IsValid()) break;

        batch_num++;

        // Forward pass through model
        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::Forward, epoch, batch_num,
            static_cast<int>(total_batches));
        const auto forward_start = std::chrono::steady_clock::now();
        Tensor predictions = Forward(batch.data);
        const auto forward_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - forward_start).count();
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::Forward, epoch, batch_num,
            static_cast<int>(total_batches), 0.0f, 0.0f, forward_ms);

        // DEBUG: Log sample values for first batch of first epoch
        if (epoch == 1 && batch_num == 1) {
            const float* input_data = batch.data.Data<float>();
            float min_input = input_data[0], max_input = input_data[0];
            const auto& input_shape = batch.data.Shape();
            if (input_shape.size() >= 2) {
                size_t input_size = input_shape[0] * input_shape[1];
                for (size_t i = 1; i < std::min(input_size, size_t(1000)); ++i) {
                    min_input = std::min(min_input, input_data[i]);
                    max_input = std::max(max_input, input_data[i]);
                }
                spdlog::info("DEBUG Arrow: Input data range: [{:.4f}, {:.4f}]", min_input, max_input);
            }

            // Debug labels
            const auto& label_shape = batch.labels.Shape();
            std::string shape_str;
            for (auto d : label_shape) shape_str += std::to_string(d) + " ";
            spdlog::info("DEBUG Arrow: Label shape: [{}], output_size={}", shape_str, config_.output_size);

            const float* label_data = batch.labels.Data<float>();
            size_t label_count = 0;
            if (!label_shape.empty()) {
                label_count = 1;
                for (size_t dim : label_shape) {
                    label_count *= dim;
                }
            }
            if (label_data && label_count > 0) {
                const size_t sample_count = std::min<size_t>(label_count, 3);
                std::string label_str = "  [";
                for (size_t i = 0; i < sample_count; ++i) {
                    label_str += fmt::format("{:.1f}", label_data[i]);
                    if (i + 1 < sample_count) label_str += ", ";
                }
                label_str += "]";
                spdlog::info("DEBUG Arrow: First label values: {}", label_str);
            } else {
                spdlog::error("DEBUG Arrow: Labels tensor is empty or invalid!");
            }

            // Debug predictions
            const auto& pred_shape = predictions.Shape();
            std::string pred_shape_str;
            for (auto d : pred_shape) pred_shape_str += std::to_string(d) + " ";
            spdlog::info("DEBUG Arrow: Predictions shape: [{}]", pred_shape_str);
        }

        // Compute loss
        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::ComputeLoss, epoch, batch_num,
            static_cast<int>(total_batches));
        float batch_loss = ComputeLoss(predictions, batch.labels);
        const std::string loss_status = std::isfinite(batch_loss) ? "ok" : "failed";
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::ComputeLoss, epoch, batch_num,
            static_cast<int>(total_batches), batch_loss, 0.0f, 0.0f,
            loss_status,
            std::isfinite(batch_loss) ? "" : "Training loss became NaN or Inf.");
        epoch_loss += batch_loss;

        // Compute accuracy
        const float* pred_data = predictions.Data<float>();
        const float* target_data = batch.labels.Data<float>();

        for (size_t b = 0; b < batch.size; ++b) {
            int pred_class = 0, true_class = 0;
            float max_pred = pred_data[b * config_.output_size];
            float max_target = target_data[b * config_.output_size];

            for (size_t c = 0; c < config_.output_size; ++c) {
                if (pred_data[b * config_.output_size + c] > max_pred) {
                    max_pred = pred_data[b * config_.output_size + c];
                    pred_class = static_cast<int>(c);
                }
                if (target_data[b * config_.output_size + c] > max_target) {
                    max_target = target_data[b * config_.output_size + c];
                    true_class = static_cast<int>(c);
                }
            }
            if (pred_class == true_class) correct++;
            total++;
        }

        // Backward pass
        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::Backward, epoch, batch_num,
            static_cast<int>(total_batches), batch_loss);
        const auto backward_start = std::chrono::steady_clock::now();
        Backward(predictions, batch.labels);
        const auto backward_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - backward_start).count();
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::Backward, epoch, batch_num,
            static_cast<int>(total_batches), batch_loss, 0.0f, backward_ms);

        // Update weights using optimizer
        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::UpdateParameters, epoch, batch_num,
            static_cast<int>(total_batches), batch_loss);
        model_->UpdateParameters(optimizer_.get());
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::UpdateParameters, epoch, batch_num,
            static_cast<int>(total_batches), batch_loss);

        // Update metrics
        float current_loss = epoch_loss / batch_num;
        float current_acc = static_cast<float>(correct) / total;

        UpdateMetrics([batch_num, current_loss, current_acc](TrainingMetrics& m) {
            m.current_batch = batch_num;
            m.train_loss = current_loss;
            m.train_accuracy = current_acc;
        });

        // Periodic progress log so the user knows training is alive.
        // Fires on batch 1 (so they see immediate feedback that the
        // loop entered) and every 50 batches after. Throughput is
        // computed against epoch_start_time so the "batches/s" reading
        // naturally warms up as the batcher + GPU pools stabilize.
        if (batch_num == 1 || batch_num % 50 == 0) {
            const auto now = std::chrono::steady_clock::now();
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - epoch_start_time).count();
            const float elapsed_s = elapsed_ms / 1000.0f;
            const float rate = elapsed_ms > 0
                ? (batch_num * 1000.0f / static_cast<float>(elapsed_ms))
                : 0.0f;
            spdlog::info("Epoch {} [{}/{}] loss={:.4f} acc={:.2f}% "
                         "({:.1f}s, {:.1f} batches/s)",
                         epoch, batch_num, total_batches,
                         current_loss, current_acc * 100.0f,
                         elapsed_s, rate);
        }

        // Batch callback
        if (batch_cb) {
            CrashRunRecorder::Instance().MarkStage(
                TrainingTraceStage::BatchCallback, epoch, batch_num,
                static_cast<int>(total_batches), batch_loss, current_acc);
            TrainingTraceCollector::Instance().RecordStage(
                TrainingTraceStage::BatchCallback, epoch, batch_num,
                static_cast<int>(total_batches), batch_loss, current_acc);
            batch_cb(epoch, batch_num, static_cast<int>(total_batches), batch_loss, current_acc);
        }
    }

    // Final epoch metrics
    float final_loss = batch_num > 0 ? epoch_loss / batch_num : 0.0f;
    float final_acc = total > 0 ? static_cast<float>(correct) / total : 0.0f;

    UpdateMetrics([final_loss, final_acc](TrainingMetrics& m) {
        m.train_loss = final_loss;
        m.train_accuracy = final_acc;
    });
    CrashRunRecorder::Instance().MarkStage(
        TrainingTraceStage::EpochComplete, epoch, batch_num,
        static_cast<int>(total_batches), final_loss, final_acc);
    TrainingTraceCollector::Instance().RecordStage(
        TrainingTraceStage::EpochComplete, epoch, batch_num,
        static_cast<int>(total_batches), final_loss, final_acc);
}

void TrainingExecutor::RunValidationArrow(IBatcher& batcher) {
    float val_loss = 0.0f;
    int correct = 0;
    int total = 0;
    int batch_num = 0;

    batcher.Reset();

    while (!batcher.IsEpochComplete()) {
        if (ShouldStop()) break;

        Batch batch = batcher.GetNextBatch();
        if (!batch.IsValid()) break;

        batch_num++;

        // Forward pass only (no backprop)
        Tensor predictions = Forward(batch.data);

        // Compute loss
        float batch_loss = ComputeLoss(predictions, batch.labels);
        val_loss += batch_loss;

        // Compute accuracy
        const float* pred_data = predictions.Data<float>();
        const float* target_data = batch.labels.Data<float>();

        for (size_t b = 0; b < batch.size; ++b) {
            int pred_class = 0, true_class = 0;
            float max_pred = pred_data[b * config_.output_size];
            float max_target = target_data[b * config_.output_size];

            for (size_t c = 0; c < config_.output_size; ++c) {
                if (pred_data[b * config_.output_size + c] > max_pred) {
                    max_pred = pred_data[b * config_.output_size + c];
                    pred_class = static_cast<int>(c);
                }
                if (target_data[b * config_.output_size + c] > max_target) {
                    max_target = target_data[b * config_.output_size + c];
                    true_class = static_cast<int>(c);
                }
            }
            if (pred_class == true_class) correct++;
            total++;
        }
    }

    float final_loss = batch_num > 0 ? val_loss / batch_num : 0.0f;
    float final_acc = total > 0 ? static_cast<float>(correct) / total : 0.0f;

    UpdateMetrics([final_loss, final_acc](TrainingMetrics& m) {
        m.val_loss = final_loss;
        m.val_accuracy = final_acc;
    });
}

} // namespace cyxwiz
