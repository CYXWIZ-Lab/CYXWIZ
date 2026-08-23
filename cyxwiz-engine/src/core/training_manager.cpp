#include "training_manager.h"
#include "classification_decision.h"
#include "model_builder.h"

#include <cstdint>
#include "crash_run_recorder.h"
#include "image_dataset_batcher.h"
#include "audio_dataset_batcher.h"
#include "text_dataset_batcher.h"
#include "training_batcher_setup.h"
#include "training_trace_collector.h"
#include "training_run_comparison.h"
#include "worker_defaults.h"
#include "../gui/panels/training_plot_panel.h"
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <filesystem>

namespace cyxwiz {

namespace {

void NormalizeTrainingNumWorkers(TrainingConfiguration& config,
                                 const char* context) {
    const int requested = config.num_workers;
    const int normalized = ClampNumWorkersToPlatform(requested);
    if (normalized != requested) {
        spdlog::warn("{}: clamping num_workers from {} to {} based on platform",
                     context, requested, normalized);
        config.num_workers = normalized;
    } else if (config.num_workers > 0) {
        spdlog::info("{}: using num_workers={}", context, config.num_workers);
    }
}

std::string BuildTrainingTaskTerminalMessage(
    const TrainingMetrics& metrics) {
    std::string message;
    if (metrics.terminal_status == "early_stopped") {
        message = "Early stopped";
    } else if (metrics.terminal_status == "cancelled") {
        message = "Cancelled";
    } else if (metrics.terminal_status == "failed") {
        message = "Failed";
    } else {
        message = "Completed";
    }

    message += " after " + std::to_string(metrics.last_executed_epoch) +
        "/" + std::to_string(metrics.total_epochs) + " executed epochs";
    if (!metrics.terminal_reason.empty()) {
        message += ": " + metrics.terminal_reason;
    }
    if (metrics.restored_checkpoint_epoch > 0) {
        message += "; active model restored from checkpoint epoch " +
            std::to_string(metrics.restored_checkpoint_epoch);
        if (metrics.restored_checkpoint_step > 0) {
            message += " step " +
                std::to_string(metrics.restored_checkpoint_step);
        }
    } else if (!metrics.active_model_provenance.empty()) {
        message += "; active model provenance=" +
            metrics.active_model_provenance;
    }
    return message;
}

} // namespace

TrainingManager& TrainingManager::Instance() {
    static TrainingManager instance;
    return instance;
}

TrainingManager::~TrainingManager() {
    StopTraining();
    if (training_thread_ && training_thread_->joinable()) {
        training_thread_->join();
    }
}

TrainingMetrics TrainingManager::GetCurrentMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return cached_metrics_;
}

bool TrainingManager::StartTrainingCommon(
    std::unique_ptr<TrainingExecutor> executor,
    int epochs,
    int batch_size,
    std::weak_ptr<TrainingPlotPanel> plot_panel,
    std::function<void(bool)> node_editor_callback,
    const std::string& task_name,
    const std::string& start_msg)
{
    // Caller has already verified !is_training_.load() and acquired
    // mutex_. This helper carries the common plumbing that used to
    // be duplicated across StartTraining{Legacy, Arrow, Parquet,
    // Image, Audio, Text} verbatim.
    is_training_.store(true);
    stop_requested_.store(false);
    const bool sequence_mode =
        executor && executor->GetConfig().sequence_batch.enabled;
    const bool regression_mode =
        executor && UsesRegressionMetrics(executor->GetConfig());

    if (node_editor_callback) {
        node_editor_callback(true);
    }

    if (auto panel = plot_panel.lock()) {
        panel->Clear();
        panel->ShowAccuracyPlot(!regression_mode);
        panel->ShowCustomMetrics(sequence_mode || regression_mode);
        panel->SetTrainingState(true, 0, epochs, 0.0f, 0.0f);
        panel->SetVisible(true);
    }

    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto run_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    const auto trace_snapshot = TrainingTraceCollector::Instance().Snapshot();
    if (!trace_snapshot.available || trace_snapshot.status != "running") {
        TrainingTraceCollector::Instance().StartRun(
            "train-" + std::to_string(run_ms));
    } else {
        TrainingTraceCollector::Instance().RecordRuntimeEvent(
            "TrainingSetup",
            "Training manager attached to existing preparation trace");
    }
    TrainingTraceCollector::Instance().RecordTaskProgress(
        0,
        task_name,
        "TrainingSetup",
        0.0f,
        start_msg,
        "running");

    // Tasks-panel progress tracker. Poll loop reads TrainingManager
    // state — the actual training work lives in TrainingThreadFunc.
    spdlog::info("TrainingManager: submitting task '{}'", task_name);
    current_task_id_.store(AsyncTaskManager::Instance().RunAsync(
        task_name,
        [](LambdaTask& task) {
            auto& mgr = TrainingManager::Instance();
            bool stop_forwarded = false;
            while (mgr.IsTrainingActive()) {
                if (task.ShouldStop() && !stop_forwarded) {
                    // Task-panel cancellation is a request to stop the
                    // training run. Keep observing until the training thread
                    // publishes its authoritative terminal metrics.
                    mgr.StopTraining();
                    stop_forwarded = true;
                }
                if (!mgr.IsTrainingActive()) {
                    break;
                }
                auto metrics = mgr.GetCurrentMetrics();
                float progress = static_cast<float>(metrics.current_epoch) /
                    std::max(1, metrics.total_epochs);
                task.ReportProgress(progress,
                    "Epoch " + std::to_string(metrics.current_epoch) + "/" +
                    std::to_string(metrics.total_epochs) +
                    " - Loss: " + std::to_string(metrics.train_loss).substr(0, 6));
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            const auto final_metrics =
                TrainingManager::Instance().GetCurrentMetrics();
            const std::string message =
                BuildTrainingTaskTerminalMessage(final_metrics);
            if (final_metrics.terminal_status == "early_stopped") {
                task.MarkCompleted(message, "early_stopped");
            } else if (final_metrics.terminal_status == "cancelled") {
                task.MarkCancelled(message);
            } else if (final_metrics.terminal_status == "failed") {
                task.MarkFailed(message);
            } else {
                task.MarkCompleted(message, "completed");
            }
        },
        nullptr,
        nullptr
    ));
    spdlog::info("TrainingManager: task '{}' submitted with id {}",
                 task_name, current_task_id_.load());

    if (on_training_start_) {
        on_training_start_(start_msg);
    }

    // Join any still-running prior training thread before replacing
    // it. Done here (inside the mutex) so two concurrent starts can't
    // race over training_thread_.
    if (training_thread_ && training_thread_->joinable()) {
        spdlog::info("TrainingManager: joining previous training thread before starting '{}'",
                     task_name);
        training_thread_->join();
    }

    spdlog::info("TrainingManager: creating training thread for '{}'", task_name);
    training_thread_ = std::make_unique<std::thread>(
        &TrainingManager::TrainingThreadFunc, this,
        std::move(executor), epochs, batch_size, plot_panel, node_editor_callback
    );

    spdlog::info("TrainingManager: Started {} ({} epochs, batch_size={})",
                 task_name, epochs, batch_size);
    return true;
}

#ifndef CYXWIZ_TRAINING_MANAGER_ARROW_HARNESS
bool TrainingManager::StartTraining(
    TrainingConfiguration config,
    DatasetHandle dataset,
    int epochs,
    int batch_size,
    std::weak_ptr<TrainingPlotPanel> plot_panel,
    std::function<void(bool)> node_editor_callback)
{
    // Check if already training
    if (is_training_.load()) {
        spdlog::warn("TrainingManager: Cannot start training - already training");
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Double-check after acquiring lock
    if (is_training_.load()) {
        return false;
    }

    NormalizeTrainingNumWorkers(config, "TrainingManager");

    auto executor = std::make_unique<TrainingExecutor>(std::move(config), dataset);
    return StartTrainingCommon(
        std::move(executor), epochs, batch_size, plot_panel,
        std::move(node_editor_callback),
        "Training Model", "Training from Node Graph");
}
#endif

bool TrainingManager::StartTrainingExternal(
    TrainingConfiguration config,
    ResolvedExternalBatchers batchers,
    int epochs,
    int batch_size,
    std::weak_ptr<TrainingPlotPanel> plot_panel,
    std::function<void(bool)> node_editor_callback)
{
    if (is_training_.load() || !batchers.train) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (is_training_.load()) return false;

    NormalizeTrainingNumWorkers(config, "TrainingManager");
    auto executor = std::make_unique<TrainingExecutor>(
        std::move(config), std::move(batchers));
    return StartTrainingCommon(
        std::move(executor), epochs, batch_size, plot_panel,
        std::move(node_editor_callback),
        "Training Model (Resolved Roles)", "Training from resolved dataset roles");
}

std::shared_ptr<SequentialModel> TrainingManager::GetActiveModel() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_trained_model_;
}

ActiveModelInfo TrainingManager::GetActiveModelInfo() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_model_info_;
}

bool TrainingManager::HasTrainedModel() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_trained_model_ != nullptr;
}

CheckpointEvaluationLoadResult TrainingManager::LoadCheckpointForEvaluation(
    const TrainingConfiguration& config,
    const std::string& checkpoint_path,
    const std::string& graph_fingerprint,
    std::function<bool()> cancel_requested) {
    CheckpointEvaluationLoadResult result;
    namespace fs = std::filesystem;

    const auto was_cancelled = [&cancel_requested]() {
        return cancel_requested && cancel_requested();
    };

    if (was_cancelled()) {
        result.error_message = "Checkpoint loading was cancelled.";
        return result;
    }

    if (IsTrainingActive()) {
        result.error_message =
            "Cannot load a checkpoint while training is active.";
        return result;
    }
    if (!config.is_valid) {
        result.error_message =
            "The active graph must compile before loading a checkpoint.";
        return result;
    }
    if (checkpoint_path.empty()) {
        result.error_message = "Checkpoint path is empty.";
        return result;
    }

    std::error_code ec;
    fs::path resolved = fs::absolute(fs::path(checkpoint_path), ec);
    if (ec || !fs::is_directory(resolved, ec)) {
        result.error_message = "Checkpoint directory was not found: " +
                               checkpoint_path;
        return result;
    }
    if (!fs::exists(resolved / "metadata.json", ec) &&
        fs::exists(resolved / "best" / "metadata.json", ec)) {
        resolved /= "best";
    }
    if (!fs::exists(resolved / "metadata.json", ec)) {
        result.error_message =
            "Select a checkpoint directory containing metadata.json, or a "
            "training-run directory containing best/metadata.json.";
        return result;
    }

    auto built = BuildSequentialFromConfig(config);
    if (!built.ok() || !built.model) {
        result.error_message = built.error_message.empty()
            ? "The active graph could not build a sequential model for this checkpoint."
            : built.error_message;
        return result;
    }

    if (was_cancelled()) {
        result.error_message = "Checkpoint loading was cancelled.";
        return result;
    }

    CheckpointManager manager(resolved.parent_path().string());
    auto metadata = manager.LoadCheckpoint(
        *built.model, nullptr, resolved.filename().string());
    if (!metadata) {
        result.error_message = manager.GetLastError().empty()
            ? "Checkpoint loading failed."
            : manager.GetLastError();
        return result;
    }

    if (was_cancelled()) {
        result.error_message =
            "Checkpoint loading was cancelled; the active model was not replaced.";
        return result;
    }

    auto loaded_model = std::shared_ptr<SequentialModel>(std::move(built.model));
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (was_cancelled()) {
            result.error_message =
                "Checkpoint loading was cancelled; the active model was not replaced.";
            return result;
        }
        if (is_training_.load()) {
            result.error_message =
                "Training started while the checkpoint was loading; the "
                "active model was not replaced.";
            return result;
        }

        last_trained_model_ = std::move(loaded_model);
        last_optimizer_.reset();
        last_metrics_ = TrainingMetrics{};
        last_metrics_.current_epoch = metadata->epoch;
        last_metrics_.current_batch = metadata->global_step;
        last_metrics_.train_loss = metadata->train_loss;
        last_metrics_.train_accuracy = metadata->train_accuracy;
        last_metrics_.train_mae = metadata->train_mae;
        last_metrics_.train_rmse = metadata->train_rmse;
        last_metrics_.val_loss = metadata->val_loss;
        last_metrics_.val_accuracy = metadata->val_accuracy;
        last_metrics_.val_mae = metadata->val_mae;
        last_metrics_.val_rmse = metadata->val_rmse;
        last_metrics_.loss_history = metadata->loss_history;
        last_metrics_.accuracy_history = metadata->accuracy_history;
        last_metrics_.mae_history = metadata->mae_history;
        last_metrics_.rmse_history = metadata->rmse_history;
        last_metrics_.val_loss_history = metadata->val_loss_history;
        last_metrics_.val_accuracy_history = metadata->val_accuracy_history;
        last_metrics_.val_mae_history = metadata->val_mae_history;
        last_metrics_.val_rmse_history = metadata->val_rmse_history;
        last_metrics_.has_validation_metrics =
            !metadata->val_loss_history.empty() ||
            !metadata->val_accuracy_history.empty() ||
            !metadata->val_mae_history.empty() ||
            !metadata->val_rmse_history.empty() ||
            metadata->val_loss != 0.0f || metadata->val_accuracy != 0.0f ||
            metadata->val_mae != 0.0f || metadata->val_rmse != 0.0f;
        last_metrics_.checkpoint_used = resolved.string();

        active_model_info_ = ActiveModelInfo{};
        active_model_info_.origin = ActiveModelOrigin::LoadedCheckpoint;
        active_model_info_.checkpoint_path = resolved.string();
        active_model_info_.graph_fingerprint = graph_fingerprint;
        active_model_info_.effective_dataset_name = config.dataset_name;
        active_model_info_.effective_label_column =
            config.dataset_roles.train.label_column.empty()
                ? config.target.primary_column
                : config.dataset_roles.train.label_column;
        active_model_info_.checkpoint_metadata = *metadata;
    }

    result.success = true;
    result.resolved_checkpoint_path = resolved.string();
    result.metadata = *metadata;
    spdlog::info(
        "TrainingManager: Loaded checkpoint '{}' for evaluation (epoch={}, val_loss={:.4f})",
        result.resolved_checkpoint_path,
        result.metadata.epoch,
        result.metadata.val_loss);
    return result;
}
bool TrainingManager::StartTrainingArrow(
    TrainingConfiguration config,
    std::shared_ptr<ArrowDataset> arrow_dataset,
    const std::string& label_column,
    int epochs,
    int batch_size,
    std::weak_ptr<TrainingPlotPanel> plot_panel,
    std::function<void(bool)> node_editor_callback)
{
    // Check if already training
    if (is_training_.load()) {
        spdlog::warn("TrainingManager: Cannot start training - already training");
        return false;
    }

    if (!arrow_dataset) {
        spdlog::error("TrainingManager: Arrow dataset is null");
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Double-check after acquiring lock
    if (is_training_.load()) {
        return false;
    }

    // Update input size from actual data
    size_t num_cols = arrow_dataset->GetNumColumns();
    size_t num_rows = arrow_dataset->GetNumRows();

    // Input size = all columns except label
    // ArrowDatasetBatcher auto-detects label columns, so always reserve one column for label
    // when dataset has multiple columns (typical ML dataset structure).
    //
    // Phase 4 Time-Series: trust the GraphCompiler's input_size override.
    // For TS graphs GraphCompiler sets input_size from the TimeSeriesWindow
    // operator's input_width param. The materialized Arrow table ALSO has
    // an extra `__partition__` metadata column, so `num_cols - 1` would
    // double-count it as a feature and build a Linear layer of the wrong
    // size (crash at first forward pass due to dimension mismatch with the
    // batch tensor, which correctly excludes both label AND __partition__).
    const auto input_resolution =
        ResolveTabularTrainingInputSize(config, num_cols);
    config.input_size = input_resolution.input_size;

    if (input_resolution.used_compiled_override) {
        spdlog::info("TrainingManager: Arrow dataset has {} rows, {} cols, "
                     "input_size={} (from GraphCompiler's TimeSeriesWindow override)",
                     num_rows, num_cols, config.input_size);
    } else if (input_resolution.has_separate_label_column) {
        spdlog::info("TrainingManager: Arrow dataset has {} rows, {} cols, input_size={} (assuming 1 label column)",
                     num_rows, num_cols, config.input_size);
    } else {
        spdlog::warn("TrainingManager: Arrow dataset has only {} column - no separate label column", num_cols);
    }

    NormalizeTrainingNumWorkers(config, "TrainingManager");
    spdlog::info("TrainingManager: resolving Arrow class-weight/balancing hints");
    TryApplyBalancedClassWeightsFromArrowTable(
        config,
        arrow_dataset ? arrow_dataset->GetArrowTable() : nullptr,
        label_column,
        /*partition_column=*/"",
        "TrainingManager Arrow");
    spdlog::info("TrainingManager: Arrow class-weight/balancing hint resolution complete");

    spdlog::info("TrainingManager: creating Arrow TrainingExecutor");
    auto executor = std::make_unique<TrainingExecutor>(std::move(config), arrow_dataset, label_column);
    spdlog::info("TrainingManager: Arrow TrainingExecutor created");
    return StartTrainingCommon(
        std::move(executor), epochs, batch_size, plot_panel,
        std::move(node_editor_callback),
        "Training Model (Arrow)", "Training from Arrow Dataset");
}

bool TrainingManager::StartTrainingParquet(
    TrainingConfiguration config,
    std::shared_ptr<ParquetBackedDataset> parquet_dataset,
    const std::string& label_column,
    int epochs,
    int batch_size,
    std::weak_ptr<TrainingPlotPanel> plot_panel,
    std::function<void(bool)> node_editor_callback)
{
    // Mirrors StartTrainingArrow; the only difference is the dataset type
    // carried through to the TrainingExecutor, which picks its Parquet
    // batcher based on that type.
    if (is_training_.load()) {
        spdlog::warn("TrainingManager: Cannot start training - already training");
        return false;
    }

    if (!parquet_dataset) {
        spdlog::error("TrainingManager: Parquet dataset is null");
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (is_training_.load()) {
        return false;
    }

    // Update input size from actual data (same logic as StartTrainingArrow).
    // Parquet column counts come from the dataset's cached schema, so this
    // is a metadata lookup, not a disk read.
    size_t num_cols = parquet_dataset->GetNumColumns();
    size_t num_rows = parquet_dataset->GetNumRows();
    const auto input_resolution =
        ResolveTabularTrainingInputSize(config, num_cols);
    config.input_size = input_resolution.input_size;

    if (input_resolution.used_compiled_override) {
        spdlog::info("TrainingManager: Parquet dataset has {} rows, {} cols, "
                     "input_size={} (from GraphCompiler's TimeSeriesWindow override)",
                     num_rows, num_cols, config.input_size);
    } else if (input_resolution.has_separate_label_column) {
        spdlog::info("TrainingManager: Parquet dataset has {} rows, {} cols, input_size={} (assuming 1 label column)",
                     num_rows, num_cols, config.input_size);
    } else {
        spdlog::warn("TrainingManager: Parquet dataset has only {} column - no separate label column", num_cols);
    }

    NormalizeTrainingNumWorkers(config, "TrainingManager");

    auto executor = std::make_unique<TrainingExecutor>(std::move(config), parquet_dataset, label_column);
    return StartTrainingCommon(
        std::move(executor), epochs, batch_size, plot_panel,
        std::move(node_editor_callback),
        "Training Model (Parquet)", "Training from Parquet-backed Dataset");
}

bool TrainingManager::StartTrainingSequence(
    TrainingConfiguration config,
    std::unique_ptr<ISequenceBatcher> sequence_batcher,
    std::vector<std::string> id_to_label,
    int epochs,
    int batch_size,
    std::weak_ptr<TrainingPlotPanel> plot_panel,
    std::function<void(bool)> node_editor_callback)
{
    if (is_training_.load()) {
        spdlog::warn("TrainingManager: Cannot start sequence training - already training");
        return false;
    }

    if (!sequence_batcher) {
        spdlog::error("TrainingManager: sequence batcher is null");
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (is_training_.load()) {
        return false;
    }

    NormalizeTrainingNumWorkers(config, "TrainingManager");

    auto executor = std::make_unique<TrainingExecutor>(
        std::move(config), std::move(sequence_batcher), std::move(id_to_label));
    return StartTrainingCommon(
        std::move(executor), epochs, batch_size, plot_panel,
        std::move(node_editor_callback),
        "Training Model (Sequence)", "Training from sequence dataset");
}

#ifndef CYXWIZ_TRAINING_MANAGER_ARROW_HARNESS
bool TrainingManager::StartTrainingImage(
    TrainingConfiguration config,
    const DataRegistry::ImageDatasetEntry& image_entry,
    int epochs,
    int batch_size,
    std::weak_ptr<TrainingPlotPanel> plot_panel,
    std::function<void(bool)> node_editor_callback)
{
    if (is_training_.load()) {
        spdlog::warn("TrainingManager: Cannot start training - already training");
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (is_training_.load()) return false;

    NormalizeTrainingNumWorkers(config, "TrainingManager");

    // Build the ImageDatasetBatcher with the image preprocessing config
    // extracted from the graph's Resize / Normalize / Augmentation nodes.
    auto batcher = std::make_unique<ImageDatasetBatcher>(
        image_entry, config.image_preprocessing,
        batch_size, config.train_ratio, config.shuffle, config.num_workers,
        static_cast<uint32_t>(config.dataloader_seed));
    batcher->SetDropLast(config.drop_last);

    if (batcher->GetNumSamples() == 0) {
        spdlog::error("TrainingManager: Image dataset has 0 samples");
        return false;
    }

    // Update input_size from the target dimensions. The batcher flattens
    // images to [H*W*C] per sample by default.
    int tw = config.image_preprocessing.target_width > 0
        ? config.image_preprocessing.target_width : 224;
    int th = config.image_preprocessing.target_height > 0
        ? config.image_preprocessing.target_height : 224;
    int ch = config.image_preprocessing.convert_to_grayscale ? 1 : 3;
    config.input_size = static_cast<size_t>(tw * th * ch);

    spdlog::info("TrainingManager: Image dataset {} samples, input_size={} ({}x{}x{}), num_workers={}, seed={}",
                 batcher->GetNumSamples(), config.input_size, tw, th, ch,
                 config.num_workers, config.dataloader_seed);

    // Set up normalization / one-hot from the compiled graph config
    if (config.preprocessing.has_normalization) {
        batcher->SetNormalization(config.preprocessing.norm_mean,
                                  config.preprocessing.norm_std);
    }
    if (UsesScalarBinaryTargets(config.loss_type)) {
        batcher->SetScalarLabelMode(true);
    } else if (config.preprocessing.has_onehot &&
               config.preprocessing.num_classes > 0) {
        batcher->SetOneHotEncoding(config.preprocessing.num_classes);
    }

    auto executor = std::make_unique<TrainingExecutor>(std::move(config), std::move(batcher));
    return StartTrainingCommon(
        std::move(executor), epochs, batch_size, plot_panel,
        std::move(node_editor_callback),
        "Training Model (Image)", "Training from Image Dataset");
}

bool TrainingManager::StartTrainingAudio(
    TrainingConfiguration config,
    const DataRegistry::AudioDatasetEntry& audio_entry,
    int epochs,
    int batch_size,
    std::weak_ptr<TrainingPlotPanel> plot_panel,
    std::function<void(bool)> node_editor_callback)
{
    if (is_training_.load()) {
        spdlog::warn("TrainingManager: Cannot start training - already training");
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (is_training_.load()) return false;

    NormalizeTrainingNumWorkers(config, "TrainingManager");

    // Build the audio batcher. AudioDatasetBatcher constructs the
    // underlying AudioDataset from the entry, applies any graph-driven
    // preprocessing overrides from config.audio_preprocessing
    // (Spectrogram/MelSpectrogram/MFCC/AudioAugmentation nodes), and
    // probes a sample to discover the actual feature shape.
    auto batcher = std::make_unique<AudioDatasetBatcher>(
        audio_entry,
        config.audio_preprocessing,
        batch_size, config.train_ratio, config.shuffle, config.num_workers,
        static_cast<uint32_t>(config.dataloader_seed));
    batcher->SetDropLast(config.drop_last);

    if (batcher->GetNumSamples() == 0) {
        spdlog::error("TrainingManager: Audio dataset has 0 samples");
        return false;
    }

    // input_size is feature_rows × feature_cols (the flattened feature map
    // — Spectrogram or MelSpec or MFCC). The model's first Dense layer
    // sees this many features.
    config.input_size = static_cast<size_t>(batcher->GetFeatureRows()) *
                        static_cast<size_t>(batcher->GetFeatureCols());
    config.input_shape = {
        static_cast<size_t>(batcher->GetFeatureRows()),
        static_cast<size_t>(batcher->GetFeatureCols())
    };

    spdlog::info("TrainingManager: Audio dataset {} samples, input_size={} ({}x{}), num_workers={}, seed={}",
                 batcher->GetNumSamples(), config.input_size,
                 batcher->GetFeatureRows(), batcher->GetFeatureCols(),
                 config.num_workers, config.dataloader_seed);

    // Mirror image path: hand normalization / one-hot through to the batcher.
    if (config.preprocessing.has_normalization) {
        batcher->SetNormalization(config.preprocessing.norm_mean,
                                  config.preprocessing.norm_std);
    }
    if (UsesScalarBinaryTargets(config.loss_type)) {
        batcher->SetScalarLabelMode(true);
    } else if (config.preprocessing.has_onehot &&
               config.preprocessing.num_classes > 0) {
        batcher->SetOneHotEncoding(config.preprocessing.num_classes);
    }

    auto executor = std::make_unique<TrainingExecutor>(std::move(config), std::move(batcher));
    return StartTrainingCommon(
        std::move(executor), epochs, batch_size, plot_panel,
        std::move(node_editor_callback),
        "Training Model (Audio)", "Training from Audio Dataset");
}

bool TrainingManager::StartTrainingText(
    TrainingConfiguration config,
    const DataRegistry::TextDatasetEntry& text_entry,
    int epochs,
    int batch_size,
    std::weak_ptr<TrainingPlotPanel> plot_panel,
    std::function<void(bool)> node_editor_callback)
{
    if (is_training_.load()) {
        spdlog::warn("TrainingManager: Cannot start training - already training");
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (is_training_.load()) return false;

    // Normalize before constructing TextDatasetBatcher so its delegated
    // ArrowDatasetBatchers receive the actual worker budget.
    NormalizeTrainingNumWorkers(config, "TrainingManager");

    // Build the text batcher. TextDatasetBatcher owns a TextDataset
    // internally which performs tokenization + vocab building on
    // construction. Graph preprocessing nodes (Tokenizer / Vocabulary /
    // Padding) override the dialog defaults stored on text_entry.
    auto batcher = std::make_unique<TextDatasetBatcher>(
        text_entry,
        config.text_preprocessing,
        batch_size,
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
        config.shuffle,
        config.num_workers,
        static_cast<uint32_t>(config.dataloader_seed),
        config.stratified,
        static_cast<uint32_t>(std::max(0, config.split_seed)),
        config.balance_classes,
        config.balance_mode,
        config.balance_target,
        static_cast<uint32_t>(std::max(0, config.balance_seed)));
    batcher->SetDropLast(config.drop_last);

    if (batcher->GetNumSamples() == 0) {
        spdlog::error("TrainingManager: Text dataset has 0 samples");
        return false;
    }

    // input_size is max_length (1D flat sequence of token IDs). The
    // Embedding layer (if present as the first model layer) will
    // project these to [batch, max_length, embed_dim]. Without an
    // Embedding layer, the first Dense layer sees raw token IDs as
    // features, which is unusual but allowed.
    config.input_size = static_cast<size_t>(batcher->GetMaxLength());
    config.input_shape = { static_cast<size_t>(batcher->GetMaxLength()) };

    spdlog::info("TrainingManager: Text dataset {} samples, input_size={} "
                 "(max_length), vocab_size={}, num_workers={}, seed={}",
                 batcher->GetNumSamples(), config.input_size,
                 batcher->GetVocabSize(), config.num_workers,
                 config.dataloader_seed);

    // Hand preprocessing / one-hot hints through to the batcher —
    // same pattern as the image/audio paths.
    if (UsesScalarBinaryTargets(config.loss_type)) {
        batcher->SetScalarLabelMode(true);
    } else if (config.preprocessing.has_onehot &&
               config.preprocessing.num_classes > 0) {
        batcher->SetOneHotEncoding(config.preprocessing.num_classes);
    }

    batcher->TryApplyBalancedClassWeights(config);

    auto executor = std::make_unique<TrainingExecutor>(std::move(config), std::move(batcher));
    return StartTrainingCommon(
        std::move(executor), epochs, batch_size, plot_panel,
        std::move(node_editor_callback),
        "Training Model (Text)", "Training from Text Dataset");
}
#endif

void TrainingManager::StopTraining() {
    if (!is_training_.load()) {
        return;
    }

    spdlog::info("TrainingManager: Stopping training...");
    stop_requested_.store(true);

    // Stop the executor - this is critical!
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (current_executor_) {
            current_executor_->Stop();
        }
    }

    // Cancel the async task
    uint64_t task_id = current_task_id_.load();
    if (task_id != 0) {
        AsyncTaskManager::Instance().Cancel(task_id);
    }
}

void TrainingManager::WaitForTrainingStop() {
    std::unique_ptr<std::thread> thread_to_join;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (training_thread_ && training_thread_->joinable() &&
            training_thread_->get_id() != std::this_thread::get_id()) {
            thread_to_join = std::move(training_thread_);
        }
    }

    if (thread_to_join) {
        spdlog::info("TrainingManager: Waiting for training thread to stop...");
        thread_to_join->join();
        spdlog::info("TrainingManager: Training thread stopped");
    }
}

void TrainingManager::PauseTraining() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (current_executor_) {
        current_executor_->Pause();
    }
}

void TrainingManager::ResumeTraining() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (current_executor_) {
        current_executor_->Resume();
    }
}

bool TrainingManager::IsPaused() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (current_executor_) {
        return current_executor_->IsPaused();
    }
    return false;
}

void TrainingManager::TrainingThreadFunc(
    std::unique_ptr<TrainingExecutor> executor,
    int epochs,
    int batch_size,
    std::weak_ptr<TrainingPlotPanel> plot_panel,
    std::function<void(bool)> node_editor_callback)
{
    spdlog::info("TrainingManager: Training thread started");

    // Store executor reference
    {
        std::lock_guard<std::mutex> lock(mutex_);
        current_executor_ = std::move(executor);
    }

    // Initialize cached metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        cached_metrics_ = TrainingMetrics();
        cached_metrics_.total_epochs = epochs;
        cached_metrics_.is_training = true;
    }

    // Track training start time for total duration
    auto training_start_time = std::chrono::steady_clock::now();

    TrainingExecutor* exec = nullptr;

    // Set up callbacks
    auto epoch_callback = [this, plot_panel, epochs, batch_size, &exec](int epoch, float train_loss, float train_acc,
                                               float val_loss, float val_acc, float epoch_time) {
        // Update cached metrics
        float samples_per_sec = 0.0f;
        TrainingMetrics seq_metrics;
        const bool sequence_mode =
            exec && exec->GetConfig().sequence_batch.enabled;
        const bool regression_mode =
            exec && UsesRegressionMetrics(exec->GetConfig());
        const TrainingMetrics objective_metrics = exec
            ? exec->GetMetrics()
            : TrainingMetrics{};
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            cached_metrics_.current_epoch = epoch;
            cached_metrics_.train_loss = train_loss;
            if (regression_mode) {
                cached_metrics_.train_mae = objective_metrics.train_mae;
                cached_metrics_.train_rmse = objective_metrics.train_rmse;
                cached_metrics_.mae_history.push_back(
                    objective_metrics.train_mae);
                cached_metrics_.rmse_history.push_back(
                    objective_metrics.train_rmse);
            } else {
                cached_metrics_.train_accuracy = train_acc;
                cached_metrics_.accuracy_history.push_back(train_acc);
            }
            cached_metrics_.epoch_time_seconds = epoch_time;
            cached_metrics_.loss_history.push_back(train_loss);
            if (val_loss >= 0.0f) {
                cached_metrics_.val_loss = val_loss;
                cached_metrics_.val_loss_history.push_back(val_loss);
                cached_metrics_.has_validation_metrics = true;
            }
            if (val_acc >= 0.0f) {
                if (regression_mode) {
                    cached_metrics_.val_mae = objective_metrics.val_mae;
                    cached_metrics_.val_rmse = objective_metrics.val_rmse;
                    cached_metrics_.val_mae_history.push_back(
                        objective_metrics.val_mae);
                    cached_metrics_.val_rmse_history.push_back(
                        objective_metrics.val_rmse);
                } else {
                    cached_metrics_.val_accuracy = val_acc;
                    cached_metrics_.val_accuracy_history.push_back(val_acc);
                }
                cached_metrics_.has_validation_metrics = true;
            }

            // Calculate samples/sec from cached metrics
            if (epoch_time > 0) {
                samples_per_sec = cached_metrics_.samples_per_second;
            }
        }

        if (sequence_mode) {
            seq_metrics = exec->GetMetrics();
            {
                std::lock_guard<std::mutex> lock(metrics_mutex_);
                cached_metrics_.train_token_accuracy = seq_metrics.train_token_accuracy;
                cached_metrics_.val_token_accuracy = seq_metrics.val_token_accuracy;
                cached_metrics_.train_entity_f1 = seq_metrics.train_entity_f1;
                cached_metrics_.val_entity_f1 = seq_metrics.val_entity_f1;
                cached_metrics_.train_token_count = seq_metrics.train_token_count;
                cached_metrics_.val_token_count = seq_metrics.val_token_count;
            }
        }

        // Update plot panel with metrics and training state
        if (auto panel = plot_panel.lock()) {
            CrashRunRecorder::Instance().MarkStage(
                TrainingTraceStage::UIPlotUpdate, epoch, 0, 0,
                train_loss, train_acc);
            panel->AddLossPoint(epoch, static_cast<double>(train_loss), static_cast<double>(val_loss));
            if (regression_mode) {
                panel->ShowAccuracyPlot(false);
                panel->AddCustomMetric("Train MAE", epoch,
                    objective_metrics.train_mae);
                panel->AddCustomMetric("Train RMSE", epoch,
                    objective_metrics.train_rmse);
                if (val_loss >= 0.0f) {
                    panel->AddCustomMetric("Val MAE", epoch,
                        objective_metrics.val_mae);
                    panel->AddCustomMetric("Val RMSE", epoch,
                        objective_metrics.val_rmse);
                }
            } else {
                panel->ShowAccuracyPlot(true);
                // Convert accuracy from fraction (0-1) to percentage (0-100).
                panel->AddAccuracyPoint(epoch,
                    static_cast<double>(train_acc) * 100.0,
                    static_cast<double>(val_acc) * 100.0);
            }
            // Update training state with timing info
            panel->SetTrainingState(true, epoch, epochs, epoch_time, samples_per_sec);
            if (sequence_mode) {
                panel->AddCustomMetric("Train Token Accuracy", epoch,
                    static_cast<double>(seq_metrics.train_token_accuracy) * 100.0);
                panel->AddCustomMetric("Train Entity F1", epoch,
                    static_cast<double>(seq_metrics.train_entity_f1) * 100.0);
                if (val_loss >= 0.0f || val_acc >= 0.0f) {
                    panel->AddCustomMetric("Val Token Accuracy", epoch,
                        static_cast<double>(seq_metrics.val_token_accuracy) * 100.0);
                    panel->AddCustomMetric("Val Entity F1", epoch,
                        static_cast<double>(seq_metrics.val_entity_f1) * 100.0);
                }
            }
            spdlog::info("TrainingPlotPanel: Updated state - epoch={}/{}, time={:.1f}s, sps={:.0f}",
                         epoch, epochs, epoch_time, samples_per_sec);
        } else {
            spdlog::warn("TrainingPlotPanel: panel expired or unavailable");
        }

        // Notify progress callback
        if (on_progress_) {
            on_progress_(epoch, train_loss, train_acc);
        }

        if (regression_mode) {
            spdlog::info(
                "Epoch {}: loss={:.4f}, mae={:.4f}, rmse={:.4f}, "
                "val_loss={:.4f}, val_mae={:.4f}, val_rmse={:.4f} ({:.1f}s)",
                epoch, train_loss, objective_metrics.train_mae,
                objective_metrics.train_rmse, val_loss,
                objective_metrics.val_mae, objective_metrics.val_rmse,
                epoch_time);
        } else {
            spdlog::info("Epoch {}: loss={:.4f}, acc={:.2f}%, val_loss={:.4f}, val_acc={:.2f}% ({:.1f}s)",
                         epoch, train_loss, train_acc * 100, val_loss,
                         val_acc * 100, epoch_time);
        }
    };

    TrainingMetrics final_metrics;

    // Check if stop was requested before training started
    if (stop_requested_.load()) {
        spdlog::info("TrainingManager: Training cancelled before start");
    } else {
        // Run training
        {
            std::lock_guard<std::mutex> lock(mutex_);
            exec = current_executor_.get();
        }

        if (exec) {
            if (auto panel = plot_panel.lock()) {
                panel->SetMetricReportingCadence(
                    exec->GetConfig().log_interval);
            }
            // Per-batch callback — keeps the Training Dashboard responsive during
            // the epoch. Without this, the dashboard stays on "Epoch 0/N" for the
            // full duration of the first epoch because epoch_callback only fires
            // at epoch boundaries. Also pushes a loss point with a fractional
            // epoch x-axis (epoch - 1 + batch/total) so the loss curve draws
            // smoothly during long epochs (e.g. audio at 8 min/epoch) instead
            // of staying empty until the first epoch finishes.
            auto batch_callback = [this, plot_panel, &exec](int epoch, int batch, int total_batches,
                                                       float batch_loss, float batch_acc) {
                const bool regression_mode =
                    exec && UsesRegressionMetrics(exec->GetConfig());
                const int report_interval =
                    exec ? std::max(0, exec->GetConfig().log_interval) : 0;
                const bool metrics_sampled =
                    batch <= 1 || batch >= total_batches ||
                    (report_interval > 0 && batch % report_interval == 0);
                {
                    std::lock_guard<std::mutex> lock(metrics_mutex_);
                    cached_metrics_.current_epoch = epoch;
                    cached_metrics_.current_batch = batch;
                    cached_metrics_.total_batches = total_batches;
                    cached_metrics_.train_loss = batch_loss;  // running estimate, overwritten each batch
                }
                if (auto panel = plot_panel.lock()) {
                    CrashRunRecorder::Instance().MarkStage(
                        TrainingTraceStage::UIPlotUpdate, epoch, batch,
                        total_batches, batch_loss, batch_acc);
                    panel->SetBatchProgress(epoch, batch, total_batches, batch_loss);

                    // Progressive curve: x = (epoch - 1) + batch / total_batches
                    // gives a smooth 0..N x-axis where each integer is an epoch
                    // boundary. The epoch_callback later writes the official
                    // x=epoch point with the averaged loss + val metrics.
                    if (metrics_sampled && total_batches > 0) {
                        double frac_epoch = static_cast<double>(epoch - 1) +
                                            static_cast<double>(batch) /
                                            static_cast<double>(total_batches);
                        panel->AddLossPoint(frac_epoch,
                                            static_cast<double>(batch_loss),
                                            -1.0);
                        if (!regression_mode) {
                            panel->AddAccuracyPoint(
                                frac_epoch,
                                static_cast<double>(batch_acc) * 100.0,
                                -1.0);
                        }
                    }
                }
            };

            try {
                exec->Train(
                    epochs,
                    batch_size,
                    batch_callback,
                    epoch_callback,
                    [this, &final_metrics](const TrainingMetrics& metrics) {
                        final_metrics = metrics;
                    }
                );
            } catch (const std::exception& error) {
                final_metrics = exec->GetMetrics();
                spdlog::error(
                    "TrainingManager: executor failed without escaping the training thread: {}",
                    error.what());
            } catch (...) {
                final_metrics = exec->GetMetrics();
                spdlog::error(
                    "TrainingManager: executor failed with an unknown exception without escaping the training thread");
            }
        }
    }

    // Calculate total training time
    auto training_end_time = std::chrono::steady_clock::now();
    float total_training_time = std::chrono::duration<float>(training_end_time - training_start_time).count();

    // Get final metrics
    const bool completed_regression_mode =
        exec && UsesRegressionMetrics(exec->GetConfig());
    const TrainingMetrics executor_metrics =
        exec ? exec->GetMetrics() : TrainingMetrics{};
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        // Preflight and setup failures terminate inside TrainingExecutor before
        // the normal completion callback runs. In that case, reconcile the
        // executor's authoritative terminal truth before publishing manager,
        // task, dashboard, callback, and active-model state.
        if (final_metrics.terminal_status.empty() &&
            !executor_metrics.terminal_status.empty()) {
            final_metrics = executor_metrics;
        }
        if (final_metrics.total_epochs == 0) {
            final_metrics = cached_metrics_;
        }
        if (final_metrics.terminal_status.empty() &&
            stop_requested_.load()) {
            final_metrics.terminal_status = "cancelled";
            final_metrics.terminal_reason = "user_cancelled";
            final_metrics.status_message = "Training cancelled";
            final_metrics.is_complete = true;
        }
        final_metrics.is_training = false;
        final_metrics.is_paused = false;
        cached_metrics_ = final_metrics;
    }

    // Update plot panel with completion status
    if (auto panel = plot_panel.lock()) {
        std::string run_id =
            "training-task-" + std::to_string(current_task_id_.load());
        TrainingConfiguration completed_config;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (current_executor_) {
                completed_config = current_executor_->GetConfig();
            }
        }
        const std::string run_status = final_metrics.terminal_status.empty()
            ? (stop_requested_.load() ? "stopped" : "complete")
            : final_metrics.terminal_status;
        const auto record = MakeTrainingRunComparisonRecord(
            run_id,
            completed_config,
            final_metrics,
            total_training_time,
            final_metrics.checkpoint_used,
            run_status);
        panel->AddRunComparisonRecord(record);
        panel->SetTrainingComplete(total_training_time, final_metrics);
    }

    // Cleanup
    const bool cancelled =
        stop_requested_.load() ||
        final_metrics.terminal_status == "cancelled";
    const bool failed = final_metrics.terminal_status == "failed";
    const bool success = !cancelled && !failed;
    is_training_.store(false);
    current_task_id_.store(0);

    // Deactivate node editor animation
    if (node_editor_callback) {
        node_editor_callback(false);
    }

    // Notify end callback
    if (on_training_end_) {
        on_training_end_(success, final_metrics);
    }

    // Preserve trained model for export before clearing executor
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (current_executor_ && (success || cancelled)) {
            // Transfer ownership of model and optimizer for later export
            last_trained_model_ = current_executor_->ReleaseModel();
            last_optimizer_ = current_executor_->ReleaseOptimizer();
            last_metrics_ = final_metrics;
            active_model_info_ = ActiveModelInfo{};
            active_model_info_.origin = ActiveModelOrigin::TrainedInSession;
            active_model_info_.checkpoint_path = final_metrics.checkpoint_used;
            const auto& completed_config = current_executor_->GetConfig();
            active_model_info_.effective_dataset_name =
                completed_config.dataset_name;
            active_model_info_.effective_label_column =
                completed_config.dataset_roles.train.label_column.empty()
                    ? completed_config.target.primary_column
                    : completed_config.dataset_roles.train.label_column;
            spdlog::info("TrainingManager: Preserved trained model for export (success={}, stopped={})",
                         success, stop_requested_.load());
        }
        current_executor_.reset();
    }

    if (success && completed_regression_mode) {
        spdlog::info(
            "TrainingManager: Training completed! Final MAE: {:.4f}, RMSE: {:.4f}",
            final_metrics.train_mae, final_metrics.train_rmse);
    } else if (success) {
        spdlog::info("TrainingManager: Training completed! Final acc: {:.2f}%",
                     final_metrics.train_accuracy * 100);
    } else {
        spdlog::info("TrainingManager: Training stopped");
    }
}

void TrainingManager::ClearTrainedModel() {
    std::lock_guard<std::mutex> lock(mutex_);
    last_trained_model_.reset();
    last_optimizer_.reset();
    last_metrics_ = TrainingMetrics();
    active_model_info_ = ActiveModelInfo{};
    spdlog::info("TrainingManager: Cleared preserved trained model");
}

bool TrainingManager::SaveModel(const std::string& path,
                                 const std::string& name,
                                 const std::string& description) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!last_trained_model_) {
        spdlog::error("TrainingManager::SaveModel: No trained model available");
        return false;
    }

    // Set metadata if provided
    if (!name.empty()) {
        last_trained_model_->SetName(name);
    }
    if (!description.empty()) {
        last_trained_model_->SetDescription(description);
    }

    // Save the model
    if (last_trained_model_->Save(path)) {
        spdlog::info("TrainingManager::SaveModel: Model saved to {}", path);
        return true;
    }

    spdlog::error("TrainingManager::SaveModel: Failed to save model to {}", path);
    return false;
}

} // namespace cyxwiz
