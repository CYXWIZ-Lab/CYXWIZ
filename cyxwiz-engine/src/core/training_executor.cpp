#include "training_executor.h"
#include "classification_decision.h"
#include "checkpoint_manager.h"
#include "crash_run_recorder.h"
#include "error_codes.h"
#include <cyxwiz/debug_hooks.h>
#include "training_trace_collector.h"
#include "model_builder.h"
#include "sequence_model_input.h"
#include "sequence_training_step.h"
#include "training_batcher_setup.h"
#ifndef CYXWIZ_TRAINING_EXECUTOR_MODERN_ONLY
#include "data_registry.h"
#include "../preprocessing/preprocessing_config.h"
#include "../preprocessing/statistics_calculator.h"
#endif
#include "../plugin/registries/plugin_training_hook_manager.h"
#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <filesystem>
#include <limits>
#include <optional>
#include <stdexcept>

namespace cyxwiz {

namespace {

void LogTrainingBackendPlacementPlan(const TrainingConfiguration& config) {
    if (config.backend_placements.empty()) {
        return;
    }

    const auto summary = config.SummarizeBackendPlacements();
    spdlog::info(
        "TrainingExecutor: Backend placement plan: total={}, gpu={}, cpu={}, mixed={}, risk={}, unsupported={}, unknown={}",
        summary.total,
        summary.gpu,
        summary.cpu,
        summary.mixed,
        summary.risk,
        summary.unsupported,
        summary.unknown);
    for (const auto& placement : config.backend_placements) {
        const std::string layer =
            placement.node_type +
            (placement.node_name.empty()
                 ? std::string()
                 : " '" + placement.node_name + "'");
        const std::string detail =
            layer + " -> expected=" + placement.expected_backend +
            (placement.fallback_backend.empty()
                 ? std::string()
                 : ", fallback=" + placement.fallback_backend) +
            (placement.reason_code.empty()
                 ? std::string()
                 : ", reason=" + placement.reason_code);

        if (placement.NeedsUserAttention()) {
            spdlog::warn("TrainingExecutor: {}", detail);
            if (!placement.explanation.empty()) {
                spdlog::warn("TrainingExecutor: {}", placement.explanation);
            }
        } else {
            spdlog::info("TrainingExecutor: {}", detail);
        }
    }
}

std::string DescribePinMemoryTransferStatus(
    const PinMemoryTransferStatus& status) {
    return fmt::format(
        "pin_memory requested={}, effective_mode={}, reason={}, backend={}, "
        "batch_size={}, node_id={}, node_name='{}': {}",
        status.requested ? "true" : "false",
        status.effective_mode,
        status.reason_code,
        status.backend,
        status.batch_size,
        status.node_id,
        status.node_name,
        status.message);
}

void ReportPinMemoryTransferStatus(const TrainingConfiguration& config) {
    const auto& status = config.pin_memory_transfer;
    if (!status.requested) {
        return;
    }

    const std::string detail = DescribePinMemoryTransferStatus(status);
    const std::string severity = status.NeedsUserWarning() ? "warning" : "ok";
    if (status.NeedsUserWarning()) {
        spdlog::warn("TrainingExecutor: {}", detail);
    } else {
        spdlog::info("TrainingExecutor: {}", detail);
    }
    TrainingTraceCollector::Instance().RecordPinMemoryTransferStatus(
        status,
        detail,
        severity);
}

bool ShouldLogTrainingBatch(const TrainingConfiguration& config, int batch_num) {
    if (batch_num <= 1) {
        return true;
    }
    return config.log_interval > 0 && batch_num % config.log_interval == 0;
}

bool ShouldRunValidationEpoch(const TrainingConfiguration& config,
                              int epoch,
                              int total_epochs) {
    const int validation_freq = std::max(1, config.validation_freq);
    return epoch == total_epochs || validation_freq <= 1 ||
           epoch % validation_freq == 0;
}

} // namespace

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
    , mode_(DatasetMode::External)
{
    external_batchers_.train = std::shared_ptr<IBatcher>(std::move(external_batcher));
    external_batchers_.dev = external_batchers_.train;
    spdlog::info("TrainingExecutor: Created with external IBatcher, "
                 "{} layers, input_size={}, output_size={}",
                 config_.layers.size(), config_.input_size, config_.output_size);
}

TrainingExecutor::TrainingExecutor(TrainingConfiguration config,
                                   ResolvedExternalBatchers external_batchers)
    : config_(std::move(config))
    , external_batchers_(std::move(external_batchers))
    , mode_(DatasetMode::External)
{
    spdlog::info("TrainingExecutor: Created with resolved external role batchers, "
                 "{} layers, input_size={}, output_size={}",
                 config_.layers.size(), config_.input_size, config_.output_size);
}

TrainingExecutor::TrainingExecutor(
    TrainingConfiguration config,
    std::unique_ptr<ISequenceBatcher> sequence_batcher,
    std::vector<std::string> id_to_label)
    : config_(std::move(config))
    , mode_(DatasetMode::SequenceExternal)
    , sequence_batcher_(std::move(sequence_batcher))
    , sequence_id_to_label_(std::move(id_to_label))
{
    spdlog::info("TrainingExecutor: Created with external ISequenceBatcher, "
                 "{} layers, input_size={}, output_size={}, labels={}",
                 config_.layers.size(), config_.input_size,
                 config_.output_size, sequence_id_to_label_.size());
}

TrainingExecutor::~TrainingExecutor() {
    Stop();
}

bool TrainingExecutor::Initialize(int /*batch_size*/) {
    std::string target_transform_error;
    if (!ResolveRegressionTargetTransform(
            config_.regression_target_transform,
            target_transform_error) ||
        !config_.regression_target_transform.IsResolvedForWidth(
            config_.output_size)) {
        if (target_transform_error.empty()) {
            target_transform_error =
                "resolved target transform width does not match model output";
        }
        spdlog::error(
            "TrainingExecutor: regression target transform is invalid: {}",
            target_transform_error);
        return false;
    }
    if (config_.regression_target_transform.enabled) {
        spdlog::info(
            "TrainingExecutor: resolved StandardScaler regression target "
            "state '{}' for {} outputs; loss is transformed-space and "
            "MAE/RMSE are original-unit metrics",
            config_.regression_target_transform.state_path,
            config_.regression_target_transform.scales.size());
    }

    if (config_.sequence_batch.enabled &&
        mode_ != DatasetMode::SequenceExternal) {
        spdlog::error("TrainingExecutor: {}",
                      errors::FormatError(
                          errors::Runtime::UnsupportedNode,
                          SequenceBatchRuntimeUnsupportedMessage()));
        return false;
    }
    if (mode_ == DatasetMode::SequenceExternal) {
        if (!config_.sequence_batch.enabled) {
            spdlog::error("TrainingExecutor: {}",
                          errors::FormatError(
                              errors::Training::InvalidTrainingSetup,
                              "sequence batch config is not enabled"));
            return false;
        }
        if (!sequence_batcher_) {
            spdlog::error("TrainingExecutor: {}",
                          errors::FormatError(
                              errors::Training::InvalidTrainingSetup,
                              "sequence batcher is null"));
            return false;
        }
        if (sequence_id_to_label_.empty()) {
            spdlog::error("TrainingExecutor: {}",
                          errors::FormatError(
                              errors::Data::RequiredLabelColumnMissing,
                              "sequence label vocabulary is empty"));
            return false;
        }
    }

    auto built = BuildExecutableFromConfig(config_);
    if (!built.ok()) {
        spdlog::error("TrainingExecutor: {}",
                      built.error_message.empty()
                          ? errors::FormatError(
                                errors::Training::ModelBuildFailed,
                                "Failed to build model from config")
                          : built.error_message);
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
            spdlog::error("TrainingExecutor: {}",
                          errors::FormatError(
                              errors::Training::InvalidTrainingSetup,
                              "Failed to initialize"));
            is_training_.store(false);
            return;
        }
        LogTrainingBackendPlacementPlan(config_);

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
        m.has_validation_metrics = false;
        m.optimizer_step_count = 0;
        m.has_test_metrics = false;
        m.train_token_accuracy = 0.0f;
        m.val_token_accuracy = 0.0f;
        m.train_entity_f1 = 0.0f;
        m.val_entity_f1 = 0.0f;
        m.train_token_count = 0;
        m.val_token_count = 0;
    });

    // Create batchers - Arrow in-memory, Parquet disk-backed, external, or
    // legacy. Modern paths flow through IBatcher pointers; legacy keeps the
    // existing DatasetBatcher loop for now.
    TrainingBatcherSet modern_batchers;
#ifndef CYXWIZ_TRAINING_EXECUTOR_MODERN_ONLY
    std::unique_ptr<DatasetBatcher> legacy_train_batcher;
    std::unique_ptr<DatasetBatcher> legacy_val_batcher;
#endif

    // Non-owning IBatcher pointers point at whichever concrete batcher the
    // selected mode owns. Arrow, Parquet, image, audio, and text all share the
    // IBatcher-aware loop; legacy DatasetHandle keeps its older loop.
    IBatcher* active_train_ibatcher = nullptr;
    IBatcher* active_val_ibatcher = nullptr;
    IBatcher* active_test_ibatcher = nullptr;
    BatcherPhase train_batcher_phase = BatcherPhase::Train;
    BatcherPhase val_batcher_phase = BatcherPhase::Val;
    BatcherPhase test_batcher_phase = BatcherPhase::Test;
    ISequenceBatcher* active_sequence_batcher = nullptr;

    size_t num_train_samples = 0;
    size_t num_val_samples = 0;
    size_t num_test_samples = 0;

    if (mode_ == DatasetMode::Arrow) {
        modern_batchers = BuildArrowTrainingBatchers(
            config_, arrow_dataset_, label_column_, batch_size);
        num_train_samples = modern_batchers.num_train_samples;
        active_train_ibatcher = modern_batchers.train;
        active_val_ibatcher = modern_batchers.val;
        active_test_ibatcher = modern_batchers.test;
    } else if (mode_ == DatasetMode::Parquet) {
        modern_batchers = BuildParquetTrainingBatchers(
            config_, parquet_dataset_, label_column_, batch_size);
        num_train_samples = modern_batchers.num_train_samples;
        active_train_ibatcher = modern_batchers.train;
        active_val_ibatcher = modern_batchers.val;
        active_test_ibatcher = modern_batchers.test;
    } else if (mode_ == DatasetMode::External) {
        // External batchers are constructed by TrainingManager for
        // image/audio/text datasets with the compiled graph config already
        // applied. The executor only owns the common training loop.
        spdlog::info("TrainingExecutor: Using external batcher for training "
                     "(batch_size={}, num_workers={}, {} samples)",
                     batch_size, config_.num_workers,
                     external_batchers_.train ? external_batchers_.train->GetNumSamples() : 0);

        if (!external_batchers_.train) {
            spdlog::error("TrainingExecutor: external batcher mode but no Train batcher");
            return;
        }

        active_train_ibatcher = external_batchers_.train.get();
        active_val_ibatcher = external_batchers_.dev
            ? external_batchers_.dev.get() : active_train_ibatcher;
        active_test_ibatcher = external_batchers_.test.get();
        train_batcher_phase = external_batchers_.train_phase;
        val_batcher_phase = external_batchers_.dev
            ? external_batchers_.dev_phase : BatcherPhase::Val;
        test_batcher_phase = external_batchers_.test_phase;
    } else if (mode_ == DatasetMode::SequenceExternal) {
        spdlog::info("TrainingExecutor: Using external sequence batcher for "
                     "token tagging ({} samples, {} batches)",
                     sequence_batcher_ ? sequence_batcher_->GetNumSamples() : 0,
                     sequence_batcher_ ? sequence_batcher_->GetNumBatches() : 0);

        if (!sequence_batcher_) {
            spdlog::error("TrainingExecutor: sequence mode but no sequence batcher");
            return;
        }

        num_train_samples = sequence_batcher_->GetNumSamples();
        active_sequence_batcher = sequence_batcher_.get();
    } else {
#ifndef CYXWIZ_TRAINING_EXECUTOR_MODERN_ONLY
        // Legacy DatasetHandle batching
        spdlog::info("TrainingExecutor: Using legacy dataset for training "
                     "(batch_size={}, shuffle={}, drop_last={}, num_workers={})",
                     batch_size, config_.shuffle, config_.drop_last, config_.num_workers);

        // Honor DataLoader node config (or defaults if no such node).
        // Validation batcher never shuffles and never drops the last batch.
        legacy_train_batcher = std::make_unique<DatasetBatcher>(
            dataset_, batch_size, DatasetSplit::Train,
            config_.shuffle, config_.drop_last, config_.num_workers,
            static_cast<uint32_t>(config_.dataloader_seed));
        legacy_val_batcher = std::make_unique<DatasetBatcher>(
            dataset_, batch_size, DatasetSplit::Validation, false, false, config_.num_workers,
            static_cast<uint32_t>(config_.dataloader_seed));

        // Apply NEW preprocessing pipeline (if configured)
        const std::string dataset_name = !config_.dataset_name.empty()
            ? config_.dataset_name
            : dataset_.GetName();
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
        if (registry.HasAugmentationPipeline(dataset_name)) {
            auto aug_pipeline = registry.GetAugmentationPipeline(dataset_name);
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

        if (UsesScalarBinaryTargets(config_.loss_type)) {
            legacy_train_batcher->SetLegacyScalarLabelMode(true);
            legacy_val_batcher->SetLegacyScalarLabelMode(true);
        } else if (config_.preprocessing.has_onehot) {
            legacy_train_batcher->SetLegacyOneHotEncoding(config_.preprocessing.num_classes);
            legacy_val_batcher->SetLegacyOneHotEncoding(config_.preprocessing.num_classes);
        }

        legacy_train_batcher->SetFlatten(true);
        legacy_val_batcher->SetFlatten(true);

        num_train_samples = legacy_train_batcher->GetNumSamples();
        num_val_samples = legacy_val_batcher->GetNumSamples();
#else
        spdlog::error("TrainingExecutor: legacy DatasetHandle mode is disabled "
                      "for this modern-only test build");
        is_training_.store(false);
        return;
#endif
    }

    if (mode_ == DatasetMode::SequenceExternal && active_sequence_batcher) {
        active_sequence_batcher->SetPhase(BatcherPhase::Train);
        num_train_samples = active_sequence_batcher->GetNumSamples();
        active_sequence_batcher->SetPhase(BatcherPhase::Val);
        num_val_samples = active_sequence_batcher->GetNumSamples();
        active_sequence_batcher->SetPhase(BatcherPhase::Train);
    } else if (active_train_ibatcher) {
        active_train_ibatcher->SetPhase(train_batcher_phase);
        num_train_samples = active_train_ibatcher->GetNumSamples();

        if (active_val_ibatcher) {
            active_val_ibatcher->SetPhase(val_batcher_phase);
            num_val_samples = active_val_ibatcher->GetNumSamples();
            active_val_ibatcher->SetPhase(train_batcher_phase);
        }

        if (active_test_ibatcher) {
            active_test_ibatcher->SetPhase(test_batcher_phase);
            num_test_samples = active_test_ibatcher->GetNumSamples();
            active_test_ibatcher->SetPhase(train_batcher_phase);
        }

        active_train_ibatcher->SetPhase(train_batcher_phase);
    }

    UpdateMetrics([num_train_samples, num_val_samples, num_test_samples](
                      TrainingMetrics& m) {
        m.train_sample_count = num_train_samples;
        m.val_sample_count = num_val_samples;
        m.test_sample_count = num_test_samples;
    });
    FinalizePartitionManifest(
        config_.dataset_roles,
        static_cast<int64_t>(num_train_samples),
        static_cast<int64_t>(num_val_samples),
        static_cast<int64_t>(num_test_samples));

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
    const auto trace_snapshot = TrainingTraceCollector::Instance().Snapshot();
    if (!trace_snapshot.available || trace_snapshot.status != "running") {
        TrainingTraceCollector::Instance().StartRun(
            last_run ? last_run->run_id : "training-run");
    } else {
        TrainingTraceCollector::Instance().RecordRuntimeEvent(
            "TrainingLoop",
            "Training loop attached to existing setup trace");
    }
    ReportPinMemoryTransferStatus(config_);
    gradient_accumulator_.clear();
    gradient_accumulated_batches_ = 0;

    std::unique_ptr<CheckpointManager> checkpoint_manager;
    float best_val_loss = std::numeric_limits<float>::infinity();
    int epochs_without_improvement = 0;
    const int early_stopping_patience = std::max(0, config_.early_stopping_patience);
    const bool save_best_checkpoint = config_.save_best_checkpoint;
    const int validation_freq = std::max(1, config_.validation_freq);
    const int log_interval = std::max(0, config_.log_interval);
    spdlog::info("TrainingExecutor: DataLoader runtime policy validation_freq={} epoch(s), log_interval={} batch(es), seed={}, grad_accum_steps={}",
                 validation_freq, log_interval, config_.dataloader_seed,
                 std::max(1, config_.grad_accum_steps));
    std::string terminal_status;
    std::string terminal_reason;

    std::filesystem::path checkpoint_root = config_.checkpoint_dir.empty()
        ? (std::filesystem::current_path() / ".cyxwiz" / "checkpoints")
        : std::filesystem::path(config_.checkpoint_dir);
    checkpoint_root /= last_run ? last_run->run_id : "training-run";
    checkpoint_manager = std::make_unique<CheckpointManager>(checkpoint_root.string());

    // Training loop
    const bool regression_metrics = UsesRegressionMetrics(config_);
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
                terminal_status = "early_stopped";
                terminal_reason = "plugin_requested_early_stop";
                UpdateMetrics([&](TrainingMetrics& m) {
                    m.terminal_status = terminal_status;
                    m.terminal_reason = terminal_reason;
                    m.status_message = "Early stopping: plugin requested stop";
                });
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
#ifndef CYXWIZ_TRAINING_EXECUTOR_MODERN_ONLY
            RunTrainingEpoch(*legacy_train_batcher, epoch, batch_cb);
#else
            break;
#endif
        } else if (mode_ == DatasetMode::SequenceExternal &&
                   active_sequence_batcher) {
            RunTrainingEpochSequence(*active_sequence_batcher, epoch, batch_cb);
        } else if (active_train_ibatcher) {
            RunTrainingEpochArrow(*active_train_ibatcher, epoch, batch_cb);
        }

        if (ShouldStop()) break;

        // Run validation (eval mode).
        //
        // For image/audio batchers, active_train_ibatcher and
        // active_val_ibatcher point to the *same* instance - the batcher
        // holds both train_indices_ and val_indices_ internally and switches
        // between them via SetPhase. Without SetPhase(Val) the val pass
        // would iterate the training indices, producing bogus "perfect val"
        // metrics (this was the source of the suspicious 100% val acc).
        // For Arrow/Parquet/legacy paths these are separate instances so
        // SetPhase is a no-op on the default IBatcher impl.
        const bool should_validate_this_epoch =
            ShouldRunValidationEpoch(config_, epoch, epochs);
        bool validation_ran_this_epoch = false;
        if (should_validate_this_epoch) {
            model_->SetTraining(false);
        }
        if (should_validate_this_epoch && mode_ == DatasetMode::Legacy) {
#ifndef CYXWIZ_TRAINING_EXECUTOR_MODERN_ONLY
            RunValidation(*legacy_val_batcher);
            validation_ran_this_epoch = true;
#endif
        } else if (should_validate_this_epoch &&
                   mode_ == DatasetMode::SequenceExternal &&
                   active_sequence_batcher) {
            active_sequence_batcher->SetPhase(BatcherPhase::Val);
            RunValidationSequence(*active_sequence_batcher);
            active_sequence_batcher->SetPhase(BatcherPhase::Train);
            validation_ran_this_epoch = true;
        } else if (should_validate_this_epoch && active_val_ibatcher) {
            active_val_ibatcher->SetPhase(val_batcher_phase);
            RunValidationArrow(*active_val_ibatcher);
            active_val_ibatcher->SetPhase(train_batcher_phase);
            validation_ran_this_epoch = true;
        } else if (!should_validate_this_epoch) {
            spdlog::debug("TrainingExecutor: Skipping validation at epoch {} (validation_freq={})",
                          epoch, validation_freq);
        }
        if (should_validate_this_epoch) {
            model_->SetTraining(true);
        }

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
            if (regression_metrics) {
                m.mae_history.push_back(m.train_mae);
                m.rmse_history.push_back(m.train_rmse);
            } else {
                m.accuracy_history.push_back(m.train_accuracy);
            }
            if (validation_ran_this_epoch) {
                m.val_loss_history.push_back(m.val_loss);
                if (regression_metrics) {
                    m.val_mae_history.push_back(m.val_mae);
                    m.val_rmse_history.push_back(m.val_rmse);
                } else {
                    m.val_accuracy_history.push_back(m.val_accuracy);
                }
            }
        });

        // Epoch callback
        if (epoch_cb) {
            epoch_cb(epoch, current.train_loss, current.train_accuracy,
                     validation_ran_this_epoch ? current.val_loss : -1.0f,
                     validation_ran_this_epoch ? current.val_accuracy : -1.0f,
                     epoch_time);
        }

        if (validation_ran_this_epoch && regression_metrics) {
            spdlog::info(
                "Epoch {}/{}: loss={:.4f}, mae={:.4f}, rmse={:.4f}, "
                "val_loss={:.4f}, val_mae={:.4f}, val_rmse={:.4f} "
                "({:.1f}s, {:.0f} samples/sec)",
                epoch, epochs, current.train_loss, current.train_mae,
                current.train_rmse, current.val_loss, current.val_mae,
                current.val_rmse, epoch_time, samples_per_sec);
            TrainingTraceCollector::Instance().RecordValidationMetrics(
                epoch, current.train_loss, 0.0f, current.val_loss, 0.0f,
                epoch_time * 1000.0f);
        } else if (validation_ran_this_epoch) {
            spdlog::info("Epoch {}/{}: loss={:.4f}, acc={:.2f}%, val_loss={:.4f}, val_acc={:.2f}% ({:.1f}s, {:.0f} samples/sec)",
                         epoch, epochs, current.train_loss, current.train_accuracy * 100,
                         current.val_loss, current.val_accuracy * 100, epoch_time, samples_per_sec);
            TrainingTraceCollector::Instance().RecordValidationMetrics(
                epoch,
                current.train_loss,
                current.train_accuracy,
                current.val_loss,
                current.val_accuracy,
                epoch_time * 1000.0f);
        } else if (regression_metrics) {
            spdlog::info(
                "Epoch {}/{}: loss={:.4f}, mae={:.4f}, rmse={:.4f}, "
                "validation skipped (validation_freq={}) "
                "({:.1f}s, {:.0f} samples/sec)",
                epoch, epochs, current.train_loss, current.train_mae,
                current.train_rmse, validation_freq, epoch_time,
                samples_per_sec);
        } else {
            spdlog::info("Epoch {}/{}: loss={:.4f}, acc={:.2f}%, validation skipped (validation_freq={}) ({:.1f}s, {:.0f} samples/sec)",
                         epoch, epochs, current.train_loss, current.train_accuracy * 100,
                         validation_freq, epoch_time, samples_per_sec);
        }

        if (validation_ran_this_epoch && checkpoint_manager && save_best_checkpoint && std::isfinite(current.val_loss)) {
            if (current.val_loss < best_val_loss) {
                best_val_loss = current.val_loss;
                epochs_without_improvement = 0;
                if (SequentialModel* sequential_model = model_->AsSequentialModel()) {
                    if (checkpoint_manager->SaveBestModel(*sequential_model,
                                                          optimizer_.get(),
                                                          current,
                                                          current.val_loss)) {
                        TrainingTraceCollector::Instance().RecordCheckpointSaved(
                            epoch,
                            checkpoint_manager->GetBestCheckpoint(),
                            current.val_loss,
                            current.val_accuracy,
                            true);
                        spdlog::info("TrainingExecutor: Best validation checkpoint saved at epoch {} (val_loss={:.4f})",
                                     epoch, current.val_loss);
                    }
                } else {
                    spdlog::warn("TrainingExecutor: save_best_checkpoint is not available for graph executable models yet");
                    checkpoint_manager.reset();
                }
            } else {
                ++epochs_without_improvement;
                spdlog::info("TrainingExecutor: No validation improvement for {} epoch(s) (best val_loss={:.4f})",
                             epochs_without_improvement, best_val_loss);
                if (early_stopping_patience > 0 &&
                    epochs_without_improvement >= early_stopping_patience) {
                    spdlog::info("TrainingExecutor: Early stopping after {} non-improving epoch(s) (patience={})",
                                 epochs_without_improvement, early_stopping_patience);
                    terminal_status = "early_stopped";
                    terminal_reason =
                        "validation_loss_plateau_patience_" +
                        std::to_string(early_stopping_patience);
                    UpdateMetrics([&](TrainingMetrics& m) {
                        m.terminal_status = terminal_status;
                        m.terminal_reason = terminal_reason;
                        m.status_message = "Early stopping: validation loss plateaued";
                    });
                    TrainingTraceCollector::Instance().RecordTerminalEvent(
                        terminal_status,
                        terminal_reason,
                        epoch,
                        current.train_loss,
                        current.train_accuracy);
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
#ifndef CYXWIZ_TRAINING_EXECUTOR_MODERN_ONLY
            legacy_train_batcher->Reset();
            legacy_val_batcher->Reset();
#endif
        } else {
            if (mode_ == DatasetMode::SequenceExternal &&
                active_sequence_batcher) {
                active_sequence_batcher->Reset();
            } else {
                active_train_ibatcher->Reset();
                active_val_ibatcher->Reset();
                if (active_test_ibatcher) {
                    active_test_ibatcher->Reset();
                }
            }
        }
    }

    TrainingMetrics final_metrics = GetMetrics();

    // Mark complete
    if (terminal_status.empty()) {
        terminal_status = stop_requested_.load() ? "cancelled" : "completed";
        terminal_reason = stop_requested_.load()
            ? "user_cancelled"
            : "completed_all_epochs";
    }
    UpdateMetrics([&](TrainingMetrics& m) {
        m.is_training = false;
        m.is_complete = true;
        m.terminal_status = terminal_status;
        m.terminal_reason = terminal_reason;
        if (terminal_status == "early_stopped") {
            m.status_message = "Training early-stopped: " + terminal_reason;
        } else if (terminal_status == "cancelled") {
            m.status_message = "Training cancelled";
        } else {
            m.status_message = "Training complete";
        }
    });

    if (!stop_requested_.load() && checkpoint_manager && save_best_checkpoint) {
        const std::string best_checkpoint = checkpoint_manager->GetBestCheckpoint();
        if (!best_checkpoint.empty()) {
            std::optional<CheckpointMetadata> restored;
            if (SequentialModel* sequential_model = model_->AsSequentialModel()) {
                restored = checkpoint_manager->LoadCheckpoint(*sequential_model,
                                                              optimizer_.get(),
                                                              "best");
            }
            if (restored) {
                final_metrics.current_epoch = restored->epoch;
                final_metrics.current_batch = restored->global_step;
                final_metrics.train_loss = restored->train_loss;
                final_metrics.train_accuracy = restored->train_accuracy;
                final_metrics.train_mae = restored->train_mae;
                final_metrics.train_rmse = restored->train_rmse;
                final_metrics.val_loss = restored->val_loss;
                final_metrics.val_accuracy = restored->val_accuracy;
                final_metrics.val_mae = restored->val_mae;
                final_metrics.val_rmse = restored->val_rmse;
                final_metrics.has_validation_metrics = true;
                final_metrics.loss_history = restored->loss_history;
                final_metrics.accuracy_history = restored->accuracy_history;
                final_metrics.mae_history = restored->mae_history;
                final_metrics.rmse_history = restored->rmse_history;
                final_metrics.val_loss_history = restored->val_loss_history;
                final_metrics.val_accuracy_history = restored->val_accuracy_history;
                final_metrics.val_mae_history = restored->val_mae_history;
                final_metrics.val_rmse_history = restored->val_rmse_history;
                final_metrics.checkpoint_used = best_checkpoint;
                UpdateMetrics([&](TrainingMetrics& m) {
                    m.current_epoch = restored->epoch;
                    m.current_batch = restored->global_step;
                    m.train_loss = restored->train_loss;
                    m.train_accuracy = restored->train_accuracy;
                    m.train_mae = restored->train_mae;
                    m.train_rmse = restored->train_rmse;
                    m.val_loss = restored->val_loss;
                    m.val_accuracy = restored->val_accuracy;
                    m.val_mae = restored->val_mae;
                    m.val_rmse = restored->val_rmse;
                    m.has_validation_metrics = true;
                    m.loss_history = restored->loss_history;
                    m.accuracy_history = restored->accuracy_history;
                    m.mae_history = restored->mae_history;
                    m.rmse_history = restored->rmse_history;
                    m.val_loss_history = restored->val_loss_history;
                    m.val_accuracy_history = restored->val_accuracy_history;
                    m.val_mae_history = restored->val_mae_history;
                    m.val_rmse_history = restored->val_rmse_history;
                    m.checkpoint_used = best_checkpoint;
                    m.terminal_status = terminal_status;
                    m.terminal_reason = terminal_reason;
                    m.status_message = std::string(
                        "Restored best validation checkpoint after ") +
                        terminal_status + ": " + terminal_reason;
                });
                TrainingTraceCollector::Instance().RecordCheckpointSaved(
                    restored->epoch,
                    best_checkpoint,
                    restored->val_loss,
                    restored->val_accuracy,
                    true);
                spdlog::info("TrainingExecutor: Restored best checkpoint from epoch {} (val_loss={:.4f})",
                             restored->epoch, restored->val_loss);
            }
        }
    }

    if (!stop_requested_.load() && active_test_ibatcher &&
        active_test_ibatcher->GetNumSamples() > 0) {
        model_->SetTraining(false);
        active_test_ibatcher->SetPhase(test_batcher_phase);
        const auto test_evaluation = EvaluateArrowBatcher(*active_test_ibatcher);
        active_test_ibatcher->SetPhase(train_batcher_phase);
        UpdateMetrics([test_evaluation](TrainingMetrics& m) {
            m.test_loss = test_evaluation.loss;
            m.test_accuracy = test_evaluation.accuracy;
            m.test_mae = test_evaluation.mae;
            m.test_rmse = test_evaluation.rmse;
            m.has_test_metrics = true;
        });
        final_metrics.test_loss = test_evaluation.loss;
        final_metrics.test_accuracy = test_evaluation.accuracy;
        final_metrics.test_mae = test_evaluation.mae;
        final_metrics.test_rmse = test_evaluation.rmse;
        final_metrics.has_test_metrics = true;
        if (regression_metrics) {
            spdlog::info(
                "TrainingExecutor: Held-out test metrics test_loss={:.4f}, "
                "test_mae={:.4f}, test_rmse={:.4f} ({} samples)",
                test_evaluation.loss, test_evaluation.mae,
                test_evaluation.rmse,
                active_test_ibatcher->GetNumSamples());
        } else {
            spdlog::info("TrainingExecutor: Held-out test metrics test_loss={:.4f}, test_acc={:.2f}% ({} samples)",
                         test_evaluation.loss, test_evaluation.accuracy * 100.0f,
                         active_test_ibatcher->GetNumSamples());
        }
    } else if (!stop_requested_.load() && active_test_ibatcher) {
        spdlog::warn("TrainingExecutor: configured test split produced 0 held-out samples; "
                     "test metrics were skipped");
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
    } else if (terminal_status == "early_stopped") {
        CrashRunRecorder::Instance().MarkEarlyStopped(terminal_reason);
        TrainingTraceCollector::Instance().FinishRun("early_stopped");
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
        const std::string coded_error = errors::FormatError(
            errors::Training::TrainingExecutionFailed,
            "Training failed",
            e.what());
        CrashRunRecorder::Instance().MarkFailed(coded_error);
        TrainingTraceCollector::Instance().FinishRun("failed");
        BackendDebugHooks::SetDebugEventCallback({});
        is_training_.store(false);
        spdlog::error("TrainingExecutor: {}", coded_error);
        throw;
    } catch (...) {
        const std::string coded_error = errors::FormatError(
            errors::Training::TrainingExecutionFailed,
            "Training failed with unknown native exception");
        CrashRunRecorder::Instance().MarkFailed(coded_error);
        TrainingTraceCollector::Instance().FinishRun("failed");
        BackendDebugHooks::SetDebugEventCallback({});
        is_training_.store(false);
        spdlog::error("TrainingExecutor: {}", coded_error);
        throw;
    }
}

#ifndef CYXWIZ_TRAINING_EXECUTOR_MODERN_ONLY
void TrainingExecutor::RunTrainingEpoch(
    DatasetBatcher& batcher,
    int epoch,
    BatchCallback batch_cb)
{
    float epoch_loss = 0.0f;
    const bool regression_metrics = UsesRegressionMetrics(config_);
    RegressionMetricAccumulator regression(
        &config_.regression_target_transform);
    int correct = 0;
    int total = 0;
    int batch_num = 0;

    size_t total_batches = batcher.GetNumBatches();

    UpdateMetrics([total_batches](TrainingMetrics& m) {
        m.total_batches = static_cast<int>(total_batches);
        m.current_batch = 0;
    });

    // Epoch wall-clock start for the periodic progress log below -
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

        // Compute objective-appropriate metrics.
        const float* pred_data = predictions.Data<float>();
        const float* target_data = batch.labels.Data<float>();
        if (regression_metrics) {
            regression.Add(pred_data, target_data,
                           batch.size * config_.output_size,
                           config_.output_size);
        } else {
            const auto accuracy_count = CountClassificationDecisions(
                pred_data, target_data, batch.size, config_.output_size,
                ClassificationDecisionModeForLoss(config_.loss_type));
            correct += static_cast<int>(accuracy_count.correct);
            total += static_cast<int>(accuracy_count.total);
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

        // Update metrics
        float current_loss = epoch_loss / batch_num;
        float current_acc = total > 0
            ? static_cast<float>(correct) / total : 0.0f;
        const float current_mae = regression.Mae();
        const float current_rmse = regression.Rmse();

        AccumulateGradientsAndMaybeStep(
            epoch, batch_num, static_cast<int>(total_batches),
            batch_loss, current_acc, batcher.IsEpochComplete());

        UpdateMetrics([batch_num, current_loss, current_acc,
                       current_mae, current_rmse,
                       regression_metrics](TrainingMetrics& m) {
            m.current_batch = batch_num;
            m.train_loss = current_loss;
            m.train_accuracy = current_acc;
            if (regression_metrics) {
                m.train_mae = current_mae;
                m.train_rmse = current_rmse;
            }
        });

        // Periodic progress log - mirror of RunTrainingEpochArrow's
        // version so non-Arrow training paths (legacy DatasetBatcher)
        // also get per-50-batch liveness signals.
        if (ShouldLogTrainingBatch(config_, batch_num)) {
            const auto now = std::chrono::steady_clock::now();
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - epoch_start_time).count();
            const float elapsed_s = elapsed_ms / 1000.0f;
            const float rate = elapsed_ms > 0
                ? (batch_num * 1000.0f / static_cast<float>(elapsed_ms))
                : 0.0f;
            if (regression_metrics) {
                spdlog::info(
                    "Epoch {} [{}/{}] loss={:.4f} mae={:.4f} rmse={:.4f} "
                    "({:.1f}s, {:.1f} batches/s)",
                    epoch, batch_num, total_batches, current_loss,
                    current_mae, current_rmse, elapsed_s, rate);
            } else {
                spdlog::info("Epoch {} [{}/{}] loss={:.4f} acc={:.2f}% "
                             "({:.1f}s, {:.1f} batches/s)",
                             epoch, batch_num, total_batches,
                             current_loss, current_acc * 100.0f,
                             elapsed_s, rate);
            }
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
    const float final_mae = regression.Mae();
    const float final_rmse = regression.Rmse();

    UpdateMetrics([final_loss, final_acc, final_mae, final_rmse,
                   regression_metrics](TrainingMetrics& m) {
        m.train_loss = final_loss;
        m.train_accuracy = final_acc;
        if (regression_metrics) {
            m.train_mae = final_mae;
            m.train_rmse = final_rmse;
        }
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
    const bool regression_metrics = UsesRegressionMetrics(config_);
    RegressionMetricAccumulator regression(
        &config_.regression_target_transform);
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

        // Compute objective-appropriate metrics.
        const float* pred_data = predictions.Data<float>();
        const float* target_data = batch.labels.Data<float>();
        if (regression_metrics) {
            regression.Add(pred_data, target_data,
                           batch.size * config_.output_size,
                           config_.output_size);
        } else {
            const auto accuracy_count = CountClassificationDecisions(
                pred_data, target_data, batch.size, config_.output_size,
                ClassificationDecisionModeForLoss(config_.loss_type));
            correct += static_cast<int>(accuracy_count.correct);
            total += static_cast<int>(accuracy_count.total);
        }
    }

    float final_loss = batch_num > 0 ? val_loss / batch_num : 0.0f;
    float final_acc = total > 0 ? static_cast<float>(correct) / total : 0.0f;
    const float final_mae = regression.Mae();
    const float final_rmse = regression.Rmse();

    UpdateMetrics([final_loss, final_acc, final_mae, final_rmse,
                   regression_metrics](TrainingMetrics& m) {
        m.val_loss = final_loss;
        m.val_accuracy = final_acc;
        if (regression_metrics) {
            m.val_mae = final_mae;
            m.val_rmse = final_rmse;
        }
        m.has_validation_metrics = true;
    });
}
#endif

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

    return ClassificationAccuracy(
        pred_data, target_data, batch_size, num_classes,
        ClassificationDecisionModeForLoss(config_.loss_type));
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

    // AfToTensor can flatten trailing-1 dimensions (e.g. [16,1] -> [16])
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

bool TrainingExecutor::AccumulateGradientsAndMaybeStep(
    int epoch,
    int batch_num,
    int total_batches,
    float batch_loss,
    float current_acc,
    bool force_step) {
    if (!model_ || !optimizer_) {
        return false;
    }

    const auto grads = model_->GetGradients();
    if (grads.empty()) {
        return false;
    }

    for (const auto& [name, grad] : grads) {
        if (grad.GetDataType() != DataType::Float32) {
            throw std::runtime_error(
                "TrainingExecutor: gradient accumulation requires Float32 gradients for '" +
                name + "'");
        }

        auto found = gradient_accumulator_.find(name);
        if (found == gradient_accumulator_.end()) {
            gradient_accumulator_[name] = grad.Clone();
            continue;
        }

        Tensor& accumulated = found->second;
        if (accumulated.Shape() != grad.Shape()) {
            throw std::runtime_error(
                "TrainingExecutor: gradient accumulation shape mismatch for '" +
                name + "'");
        }

        float* dst = accumulated.Data<float>();
        const float* src = grad.Data<float>();
        for (size_t i = 0; i < accumulated.NumElements(); ++i) {
            dst[i] += src[i];
        }
    }

    ++gradient_accumulated_batches_;
    const int grad_accum_steps = std::max(1, config_.grad_accum_steps);
    if (!force_step && gradient_accumulated_batches_ < grad_accum_steps) {
        return false;
    }

    std::map<std::string, Tensor> averaged_grads;
    const float scale = 1.0f / static_cast<float>(gradient_accumulated_batches_);
    for (const auto& [name, accumulated] : gradient_accumulator_) {
        Tensor averaged = accumulated.Clone();
        float* values = averaged.Data<float>();
        for (size_t i = 0; i < averaged.NumElements(); ++i) {
            values[i] *= scale;
        }
        averaged_grads[name] = std::move(averaged);
    }

    CrashRunRecorder::Instance().MarkStage(
        TrainingTraceStage::UpdateParameters, epoch, batch_num,
        total_batches, batch_loss, current_acc);

    auto params = model_->GetParameters();
    optimizer_->Step(params, averaged_grads);
    model_->SetParameters(params);

    TrainingTraceCollector::Instance().RecordStage(
        TrainingTraceStage::UpdateParameters, epoch, batch_num,
        total_batches, batch_loss, current_acc);

    UpdateMetrics([](TrainingMetrics& m) {
        ++m.optimizer_step_count;
    });

    gradient_accumulator_.clear();
    gradient_accumulated_batches_ = 0;
    return true;
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

namespace {

int64_t SequenceTargetIdAt(const Tensor& targets, size_t offset) {
    if (targets.GetDataType() == DataType::Int64) {
        return targets.Data<int64_t>()[offset];
    }
    if (targets.GetDataType() == DataType::Int32) {
        return static_cast<int64_t>(targets.Data<int32_t>()[offset]);
    }
    throw std::runtime_error("language-model target_ids must be Int64 or Int32");
}

struct NextTokenAccuracyCount {
    size_t correct = 0;
    size_t valid = 0;
};

NextTokenAccuracyCount CountNextTokenAccuracyFromLogits(
    const Tensor& logits,
    const Tensor& target_ids,
    int64_t ignore_index) {

    const auto& logit_shape = logits.Shape();
    const auto& target_shape = target_ids.Shape();
    if (logits.GetDataType() != DataType::Float32 ||
        logit_shape.size() != 3 ||
        target_shape.size() != 2 ||
        target_shape[0] != logit_shape[0] ||
        target_shape[1] != logit_shape[1]) {
        throw std::runtime_error(
            "language-model accuracy expects logits [batch, seq, vocab] "
            "and target_ids [batch, seq]");
    }

    const size_t batch_size = logit_shape[0];
    const size_t sequence_length = logit_shape[1];
    const size_t vocab_size = logit_shape[2];
    const float* data = logits.Data<float>();

    NextTokenAccuracyCount result;
    for (size_t row = 0; row < batch_size; ++row) {
        for (size_t col = 0; col < sequence_length; ++col) {
            const size_t target_offset = row * sequence_length + col;
            const int64_t target = SequenceTargetIdAt(target_ids, target_offset);
            if (target == ignore_index) {
                continue;
            }
            if (target < 0 || static_cast<size_t>(target) >= vocab_size) {
                throw std::runtime_error(
                    "language-model target id is outside the vocabulary range");
            }

            const size_t logit_offset = target_offset * vocab_size;
            size_t predicted = 0;
            float best = data[logit_offset];
            for (size_t vocab = 1; vocab < vocab_size; ++vocab) {
                const float value = data[logit_offset + vocab];
                if (value > best) {
                    best = value;
                    predicted = vocab;
                }
            }
            if (predicted == static_cast<size_t>(target)) {
                ++result.correct;
            }
            ++result.valid;
        }
    }

    return result;
}

} // namespace

void TrainingExecutor::RunTrainingEpochSequence(
    ISequenceBatcher& batcher,
    int epoch,
    BatchCallback batch_cb)
{
    float epoch_loss = 0.0f;
    int batch_num = 0;
    size_t sample_count = 0;
    SequenceTagMetrics aggregate_metrics;

    const size_t total_batches = batcher.GetNumBatches();
    UpdateMetrics([total_batches](TrainingMetrics& m) {
        m.total_batches = static_cast<int>(total_batches);
        m.current_batch = 0;
    });

    const auto epoch_start_time = std::chrono::steady_clock::now();
    batcher.Reset();

    while (!batcher.IsEpochComplete()) {
        if (ShouldStop()) break;
        WaitWhilePaused();

        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::GetNextBatch, epoch, batch_num + 1,
            static_cast<int>(total_batches));
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::GetNextBatch, epoch, batch_num + 1,
            static_cast<int>(total_batches));

        SequenceBatch batch = batcher.GetNextSequenceBatch();
        if (!batch.IsValid()) break;
        if (!batch.IsSupervised()) {
            throw std::runtime_error(
                "TrainingExecutor: sequence batch is missing tag_ids or target_ids");
        }
        const bool is_language_modeling = batch.HasTargetIds();
        const Tensor& targets =
            is_language_modeling ? batch.target_ids : batch.tag_ids;
        const int64_t ignore_index = is_language_modeling
            ? config_.sequence_batch.target_ignore_index
            : config_.sequence_batch.ignore_index;

        ++batch_num;
        sample_count += batch.size;

        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::Forward, epoch, batch_num,
            static_cast<int>(total_batches));
        const auto forward_start = std::chrono::steady_clock::now();
        Tensor model_input = BuildSequenceModelInput(batch, config_);
        Tensor predictions = Forward(model_input);
        const auto forward_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - forward_start).count();
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::Forward, epoch, batch_num,
            static_cast<int>(total_batches), 0.0f, 0.0f, forward_ms);

        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::ComputeLoss, epoch, batch_num,
            static_cast<int>(total_batches));
        const float batch_loss = ComputeLoss(predictions, targets);
        const std::string loss_status =
            std::isfinite(batch_loss) ? "ok" : "failed";
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::ComputeLoss, epoch, batch_num,
            static_cast<int>(total_batches), batch_loss, 0.0f, 0.0f,
            loss_status,
            std::isfinite(batch_loss) ? "" :
                "Sequence training loss became NaN or Inf.");
        if (!std::isfinite(batch_loss)) {
            throw std::runtime_error(
                "TrainingExecutor: sequence training loss is not finite");
        }
        epoch_loss += batch_loss;

        if (is_language_modeling) {
            const auto accuracy_count = CountNextTokenAccuracyFromLogits(
                predictions, targets, ignore_index);
            aggregate_metrics.total_tokens += accuracy_count.valid;
            aggregate_metrics.correct_tokens += accuracy_count.correct;
            aggregate_metrics.token_accuracy =
                aggregate_metrics.total_tokens == 0 ? 0.0 :
                    static_cast<double>(aggregate_metrics.correct_tokens) /
                    static_cast<double>(aggregate_metrics.total_tokens);
        } else {
            const auto batch_metrics = ComputeSequenceTagMetricsFromLogits(
                predictions, targets, sequence_id_to_label_, ignore_index);
            AccumulateSequenceTagMetrics(aggregate_metrics, batch_metrics);
            FinalizeSequenceTagMetricRates(aggregate_metrics);
        }

        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::Backward, epoch, batch_num,
            static_cast<int>(total_batches), batch_loss);
        const auto backward_start = std::chrono::steady_clock::now();
        Backward(predictions, targets);
        const auto backward_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - backward_start).count();
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::Backward, epoch, batch_num,
            static_cast<int>(total_batches), batch_loss, 0.0f, backward_ms);

        const float current_loss = epoch_loss / batch_num;
        const float current_acc =
            static_cast<float>(aggregate_metrics.token_accuracy);
        const float current_f1 =
            static_cast<float>(aggregate_metrics.entity_f1);

        AccumulateGradientsAndMaybeStep(
            epoch, batch_num, static_cast<int>(total_batches),
            batch_loss, current_acc, batcher.IsEpochComplete());

        UpdateMetrics([batch_num,
                       current_loss,
                       current_acc,
                       current_f1,
                       token_count = aggregate_metrics.total_tokens]
                      (TrainingMetrics& m) {
            m.current_batch = batch_num;
            m.train_loss = current_loss;
            m.train_accuracy = current_acc;
            m.train_token_accuracy = current_acc;
            m.train_entity_f1 = current_f1;
            m.train_token_count = token_count;
        });

        if (ShouldLogTrainingBatch(config_, batch_num)) {
            const auto now = std::chrono::steady_clock::now();
            const auto elapsed_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - epoch_start_time).count();
            const float elapsed_s = elapsed_ms / 1000.0f;
            const float rate = elapsed_ms > 0
                ? (batch_num * 1000.0f / static_cast<float>(elapsed_ms))
                : 0.0f;
            spdlog::info("Epoch {} [{}/{}] seq_loss={:.4f} "
                         "token_acc={:.2f}% entity_f1={:.2f}% "
                         "({:.1f}s, {:.1f} batches/s)",
                         epoch, batch_num, total_batches, current_loss,
                         current_acc * 100.0f, current_f1 * 100.0f,
                         elapsed_s, rate);
        }

        if (batch_cb) {
            CrashRunRecorder::Instance().MarkStage(
                TrainingTraceStage::BatchCallback, epoch, batch_num,
                static_cast<int>(total_batches), batch_loss, current_acc);
            TrainingTraceCollector::Instance().RecordStage(
                TrainingTraceStage::BatchCallback, epoch, batch_num,
                static_cast<int>(total_batches), batch_loss, current_acc);
            batch_cb(epoch, batch_num, static_cast<int>(total_batches),
                     batch_loss, current_acc);
        }
    }

    FinalizeSequenceTagMetricRates(aggregate_metrics);
    const float final_loss = batch_num > 0 ? epoch_loss / batch_num : 0.0f;
    const float final_acc =
        static_cast<float>(aggregate_metrics.token_accuracy);
    const float final_f1 =
        static_cast<float>(aggregate_metrics.entity_f1);

    UpdateMetrics([final_loss,
                   final_acc,
                   final_f1,
                   token_count = aggregate_metrics.total_tokens]
                  (TrainingMetrics& m) {
        m.train_loss = final_loss;
        m.train_accuracy = final_acc;
        m.train_token_accuracy = final_acc;
        m.train_entity_f1 = final_f1;
        m.train_token_count = token_count;
    });
    CrashRunRecorder::Instance().MarkStage(
        TrainingTraceStage::EpochComplete, epoch, batch_num,
        static_cast<int>(total_batches), final_loss, final_acc);
    TrainingTraceCollector::Instance().RecordStage(
        TrainingTraceStage::EpochComplete, epoch, batch_num,
        static_cast<int>(total_batches), final_loss, final_acc);

    spdlog::debug("TrainingExecutor: sequence epoch {} consumed {} samples",
                  epoch, sample_count);
}

void TrainingExecutor::RunValidationSequence(ISequenceBatcher& batcher) {
    float val_loss = 0.0f;
    int batch_num = 0;
    SequenceTagMetrics aggregate_metrics;

    batcher.Reset();
    while (!batcher.IsEpochComplete()) {
        if (ShouldStop()) break;

        SequenceBatch batch = batcher.GetNextSequenceBatch();
        if (!batch.IsValid()) break;
        if (!batch.IsSupervised()) {
            throw std::runtime_error(
                "TrainingExecutor: sequence validation batch is missing tag_ids or target_ids");
        }
        const bool is_language_modeling = batch.HasTargetIds();
        const Tensor& targets =
            is_language_modeling ? batch.target_ids : batch.tag_ids;
        const int64_t ignore_index = is_language_modeling
            ? config_.sequence_batch.target_ignore_index
            : config_.sequence_batch.ignore_index;

        ++batch_num;
        Tensor model_input = BuildSequenceModelInput(batch, config_);
        Tensor predictions = Forward(model_input);
        const float batch_loss = ComputeLoss(predictions, targets);
        if (!std::isfinite(batch_loss)) {
            throw std::runtime_error(
                "TrainingExecutor: sequence validation loss is not finite");
        }
        val_loss += batch_loss;

        if (is_language_modeling) {
            const auto accuracy_count = CountNextTokenAccuracyFromLogits(
                predictions, targets, ignore_index);
            aggregate_metrics.total_tokens += accuracy_count.valid;
            aggregate_metrics.correct_tokens += accuracy_count.correct;
            aggregate_metrics.token_accuracy =
                aggregate_metrics.total_tokens == 0 ? 0.0 :
                    static_cast<double>(aggregate_metrics.correct_tokens) /
                    static_cast<double>(aggregate_metrics.total_tokens);
        } else {
            const auto batch_metrics = ComputeSequenceTagMetricsFromLogits(
                predictions, targets, sequence_id_to_label_, ignore_index);
            AccumulateSequenceTagMetrics(aggregate_metrics, batch_metrics);
        }
    }

    FinalizeSequenceTagMetricRates(aggregate_metrics);
    const float final_loss = batch_num > 0 ? val_loss / batch_num : 0.0f;
    const float final_acc =
        static_cast<float>(aggregate_metrics.token_accuracy);
    const float final_f1 =
        static_cast<float>(aggregate_metrics.entity_f1);

    UpdateMetrics([final_loss,
                   final_acc,
                   final_f1,
                   token_count = aggregate_metrics.total_tokens]
                  (TrainingMetrics& m) {
        m.val_loss = final_loss;
        m.val_accuracy = final_acc;
        m.has_validation_metrics = true;
        m.val_token_accuracy = final_acc;
        m.val_entity_f1 = final_f1;
        m.val_token_count = token_count;
    });
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
    const bool regression_metrics = UsesRegressionMetrics(config_);
    RegressionMetricAccumulator regression(
        &config_.regression_target_transform);
    int correct = 0;
    int total = 0;
    int batch_num = 0;
    double fetch_total_ms = 0.0;
    double fetch_max_ms = 0.0;

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
    // summary, which can be 100+ seconds on a real dataset - making
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
        const auto fetch_start = std::chrono::steady_clock::now();
        Batch batch = batcher.GetNextBatch();
        const double fetch_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - fetch_start).count();
        if (!batch.IsValid()) break;

        batch_num++;
        fetch_total_ms += fetch_ms;
        fetch_max_ms = std::max(fetch_max_ms, fetch_ms);

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

        if (regression_metrics) {
            regression.Add(
                pred_data, target_data, batch.size * config_.output_size,
                config_.output_size);
        } else {
            const auto accuracy_count = CountClassificationDecisions(
                pred_data, target_data, batch.size, config_.output_size,
                ClassificationDecisionModeForLoss(config_.loss_type));
            correct += static_cast<int>(accuracy_count.correct);
            total += static_cast<int>(accuracy_count.total);
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

        // Update metrics
        float current_loss = epoch_loss / batch_num;
        float current_acc = total > 0
            ? static_cast<float>(correct) / total
            : 0.0f;
        const float current_mae = regression.Mae();
        const float current_rmse = regression.Rmse();

        AccumulateGradientsAndMaybeStep(
            epoch, batch_num, static_cast<int>(total_batches),
            batch_loss, current_acc, batcher.IsEpochComplete());

        UpdateMetrics([batch_num, current_loss, current_acc,
                       current_mae, current_rmse](TrainingMetrics& m) {
            m.current_batch = batch_num;
            m.train_loss = current_loss;
            m.train_accuracy = current_acc;
            m.train_mae = current_mae;
            m.train_rmse = current_rmse;
        });

        // Periodic progress log so the user knows training is alive.
        // Fires on batch 1 (so they see immediate feedback that the
        // loop entered) and every 50 batches after. Throughput is
        // computed against epoch_start_time so the "batches/s" reading
        // naturally warms up as the batcher + GPU pools stabilize.
        if (ShouldLogTrainingBatch(config_, batch_num)) {
            const auto now = std::chrono::steady_clock::now();
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - epoch_start_time).count();
            const float elapsed_s = elapsed_ms / 1000.0f;
            const float rate = elapsed_ms > 0
                ? (batch_num * 1000.0f / static_cast<float>(elapsed_ms))
                : 0.0f;
            if (regression_metrics) {
                spdlog::info(
                    "Epoch {} [{}/{}] loss={:.4f} mae={:.4f} rmse={:.4f} "
                    "({:.1f}s, {:.1f} batches/s)",
                    epoch, batch_num, total_batches,
                    current_loss, current_mae, current_rmse,
                    elapsed_s, rate);
            } else {
                spdlog::info("Epoch {} [{}/{}] loss={:.4f} acc={:.2f}% "
                             "({:.1f}s, {:.1f} batches/s)",
                             epoch, batch_num, total_batches,
                             current_loss, current_acc * 100.0f,
                             elapsed_s, rate);
            }
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
    const float final_mae = regression.Mae();
    const float final_rmse = regression.Rmse();

    UpdateMetrics([final_loss, final_acc, final_mae, final_rmse](TrainingMetrics& m) {
        m.train_loss = final_loss;
        m.train_accuracy = final_acc;
        m.train_mae = final_mae;
        m.train_rmse = final_rmse;
    });
    if (batch_num > 0) {
        spdlog::info(
            "Arrow loader timing epoch {}: batches={}, avg_fetch={:.2f} ms, max_fetch={:.2f} ms",
            epoch,
            batch_num,
            fetch_total_ms / static_cast<double>(batch_num),
            fetch_max_ms);
    }
    CrashRunRecorder::Instance().MarkStage(
        TrainingTraceStage::EpochComplete, epoch, batch_num,
        static_cast<int>(total_batches), final_loss, final_acc);
    TrainingTraceCollector::Instance().RecordStage(
        TrainingTraceStage::EpochComplete, epoch, batch_num,
        static_cast<int>(total_batches), final_loss, final_acc);
}

ObjectiveEvaluationMetrics TrainingExecutor::EvaluateArrowBatcher(
    IBatcher& batcher) {
    float val_loss = 0.0f;
    const bool regression_metrics = UsesRegressionMetrics(config_);
    RegressionMetricAccumulator regression(
        &config_.regression_target_transform);
    int correct = 0;
    int total = 0;
    int batch_num = 0;
    double fetch_total_ms = 0.0;
    double fetch_max_ms = 0.0;

    batcher.Reset();

    while (!batcher.IsEpochComplete()) {
        if (ShouldStop()) break;

        const auto fetch_start = std::chrono::steady_clock::now();
        Batch batch = batcher.GetNextBatch();
        const double fetch_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - fetch_start).count();
        if (!batch.IsValid()) break;

        batch_num++;
        fetch_total_ms += fetch_ms;
        fetch_max_ms = std::max(fetch_max_ms, fetch_ms);

        // Forward pass only (no backprop)
        Tensor predictions = Forward(batch.data);

        // Compute loss
        float batch_loss = ComputeLoss(predictions, batch.labels);
        val_loss += batch_loss;

        // Compute accuracy
        const float* pred_data = predictions.Data<float>();
        const float* target_data = batch.labels.Data<float>();

        if (regression_metrics) {
            regression.Add(
                pred_data, target_data, batch.size * config_.output_size,
                config_.output_size);
        } else {
            const auto accuracy_count = CountClassificationDecisions(
                pred_data, target_data, batch.size, config_.output_size,
                ClassificationDecisionModeForLoss(config_.loss_type));
            correct += static_cast<int>(accuracy_count.correct);
            total += static_cast<int>(accuracy_count.total);
        }
    }

    float final_loss = batch_num > 0 ? val_loss / batch_num : 0.0f;
    float final_acc = total > 0 ? static_cast<float>(correct) / total : 0.0f;

    if (batch_num > 0) {
        spdlog::info(
            "Arrow loader timing validation: batches={}, avg_fetch={:.2f} ms, max_fetch={:.2f} ms",
            batch_num,
            fetch_total_ms / static_cast<double>(batch_num),
            fetch_max_ms);
    }

    return {final_loss, final_acc, regression.Mae(), regression.Rmse()};
}

void TrainingExecutor::RunValidationArrow(IBatcher& batcher) {
    const auto evaluation = EvaluateArrowBatcher(batcher);

    UpdateMetrics([evaluation](TrainingMetrics& m) {
        m.val_loss = evaluation.loss;
        m.val_accuracy = evaluation.accuracy;
        m.val_mae = evaluation.mae;
        m.val_rmse = evaluation.rmse;
        m.has_validation_metrics = true;
    });
}

} // namespace cyxwiz
