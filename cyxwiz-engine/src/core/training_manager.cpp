#include "training_progress_estimate.h"
#include <chrono>
#include "training_manager.h"
#include "sequence_arrow_batcher.h"
#include "classification_decision.h"
#include "model_builder.h"

#include <cstdint>
#include "crash_run_recorder.h"
#include "training_report_window.h"
#include "image_batchers.h"
#include "image_dataset_batcher.h"
#include <stdexcept>
#include "audio_dataset_batcher.h"
#include <cyxwiz/error_codes.h>
#include "training_batcher_setup.h"
#include "training_trace_collector.h"
#include "training_run_comparison.h"
#include "worker_defaults.h"
#include "../gui/panels/training_plot_panel.h"
#include <spdlog/spdlog.h>

#include <algorithm>
#include <optional>
#include <chrono>
#include <filesystem>

namespace cyxwiz {

namespace {
std::optional<ProcessDeviceSelection> CurrentThreadModelDevice() {
    if (const auto* device = Device::GetCurrentDevice()) {
        return ProcessDeviceSelection{device->GetType(), device->GetDeviceId()};
    }
    return std::nullopt;
}


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
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        cached_metrics_ = TrainingMetrics();
        cached_metrics_.total_epochs = epochs;
        cached_metrics_.is_training = true;
    }
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
            TrainingEtaEstimator eta;
            const auto clock_start = std::chrono::steady_clock::now();
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
                // Progress counts the batches of the epoch in progress; the
                // epoch counter alone reads 1/1 = 100% for a whole 1-epoch run.
                const double now_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - clock_start).count();
                eta.Observe(now_seconds, metrics.current_epoch, metrics.current_batch,
                            metrics.total_batches, metrics.total_epochs);
                const float progress = static_cast<float>(TrainingFractionComplete(
                    metrics.current_epoch, metrics.current_batch,
                    metrics.total_batches, metrics.total_epochs));
                std::string message = "Epoch " + std::to_string(metrics.current_epoch) + "/" +
                    std::to_string(metrics.total_epochs);
                if (metrics.total_batches > 0) {
                    message += " - batch " + std::to_string(metrics.current_batch) + "/" +
                        std::to_string(metrics.total_batches);
                }
                message += " - Loss: " + std::to_string(metrics.train_loss).substr(0, 6);
                if (const auto remaining = eta.RemainingSeconds()) {
                    message += " - ETA " + FormatTrainingDuration(*remaining);
                }
                task.ReportProgress(progress, message);
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
    ExternalBatcherFactory batcher_factory,
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

    auto executor = std::make_unique<TrainingExecutor>(
        std::move(config), std::move(batcher_factory));
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

std::shared_ptr<IExecutableModel> TrainingManager::GetActiveModel() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_trained_model_;
}

std::shared_ptr<SequentialModel> TrainingManager::GetActiveSequentialModel() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!last_trained_model_ || !last_trained_model_->AsSequentialModel()) return nullptr;
    return std::shared_ptr<SequentialModel>(last_trained_model_, last_trained_model_->AsSequentialModel());
}

ActiveModelInfo TrainingManager::GetActiveModelInfo() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_model_info_;
}

bool TrainingManager::HasTrainedModel() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_trained_model_ != nullptr;
}

CheckpointModel BuildModelFromCheckpoint(const TrainingConfiguration& config,
                                         std::shared_ptr<ArrowDataset> dataset,
                                         const std::string& checkpoint_path,
                                         const std::function<bool()>& cancel_requested) {
    CheckpointModel result;
    namespace fs = std::filesystem;

    // Build the model on the selected device, as training does; this runs
    // on a task worker, which otherwise starts on ArrayFire's default
    // backend (oneAPI on an Intel-only machine, not the selected route).
    if (const auto selected = Device::GetProcessDevice()) {
        const auto activation =
            Device(selected->type, selected->device_id).ActivateExact(false);
        if (!activation.success) {
            result.error_message = fmt::format(
                "Cannot load checkpoint on selected device (type={}, id={}): {}. "
                "Select an available device and retry; no fallback was attempted.",
                static_cast<int>(selected->type), selected->device_id, activation.message);
            spdlog::error("TrainingManager: {}", result.error_message);
            return result;
        }
    }

    const auto was_cancelled = [&cancel_requested]() {
        return cancel_requested && cancel_requested();
    };
    if (was_cancelled()) {
        result.error_message = "Checkpoint loading was cancelled.";
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

    // Generic for any causal-LM sequence graph: compiling a graph does not
    // produce the frozen token vocabulary. A checkpoint keeps the tokenizer
    // it was trained with (TOFIX159): use it. An older checkpoint rebuilds it
    // as training did, from the sequence batcher of the prepared dataset
    // (token-window metadata), so Run Test can verify that test data uses the
    // model's vocabulary.
    result.prepared = config;
    std::string tokenizer_error;
    const auto stored = ReadCheckpointTokenizer(resolved, tokenizer_error);
    if (!tokenizer_error.empty()) {
        result.error_message = "The checkpoint's tokenizer cannot be read: " + tokenizer_error;
        return result;
    }
    if (stored && config.sequence_batch.enabled) {
        ApplyCheckpointTokenizer(*stored, result.prepared);
        result.tokenizer_from_checkpoint = true;
    }
    if (!PrepareSequenceEvaluationVocabulary(result.prepared, dataset, result.error_message)) {
        return result;
    }

    // The graph's own model: a layer stack, or a graph with branches
    // (TOFIX157; its layers alone were tested as a stack before).
    auto built = BuildExecutableFromConfig(result.prepared);
    if (!built.ok() || !built.model) {
        result.error_message = built.error_message.empty()
            ? "The active graph could not build a model for this checkpoint."
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
    if (stored && !metadata->tokenizer_sha256.empty() && stored->Sha256() != metadata->tokenizer_sha256) {
        result.error_message = "The checkpoint's stored tokenizer is not the one its metadata names.";
        return result;
    }
    result.model = std::move(built.model);
    result.resolved_checkpoint_path = resolved.string();
    result.metadata = *metadata;
    return result;
}

CheckpointEvaluationLoadResult TrainingManager::LoadCheckpointForEvaluation(
    const TrainingConfiguration& config,
    std::shared_ptr<ArrowDataset> dataset,
    const std::string& checkpoint_path,
    const std::string& graph_fingerprint,
    std::function<bool()> cancel_requested) {
    CheckpointEvaluationLoadResult result;
    const auto was_cancelled = [&cancel_requested]() {
        return cancel_requested && cancel_requested();
    };
    if (IsTrainingActive()) {
        result.error_message =
            "Cannot load a checkpoint while training is active.";
        return result;
    }
    auto built = BuildModelFromCheckpoint(config, dataset, checkpoint_path, cancel_requested);
    if (!built.model) {
        result.error_message = built.error_message;
        return result;
    }
    const auto& prepared = built.prepared;
    const std::filesystem::path resolved(built.resolved_checkpoint_path);
    const std::optional<CheckpointMetadata> metadata = built.metadata;

    if (was_cancelled()) {
        result.error_message =
            "Checkpoint loading was cancelled; the active model was not replaced.";
        return result;
    }

    std::shared_ptr<IExecutableModel> loaded_model(std::move(built.model));
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
        // The model was built on this thread's active backend.
        active_model_info_.model_device = CurrentThreadModelDevice();
        active_model_info_.checkpoint_path = resolved.string();
        active_model_info_.graph_fingerprint = graph_fingerprint;
        active_model_info_.effective_dataset_name = config.dataset_name;
        active_model_info_.effective_label_column =
            config.dataset_roles.train.label_column.empty()
                ? config.target.primary_column
                : config.dataset_roles.train.label_column;
        active_model_info_.checkpoint_metadata = *metadata;
        active_model_info_.evaluation_config = prepared;
        // The Data Split the checkpoint trained with, when it recorded one:
        // Run Test reads the test part from it, not from today's graph.
        if (metadata->split.known) {
            auto& trained = *active_model_info_.evaluation_config;
            trained.train_ratio = metadata->split.train;
            trained.val_ratio = metadata->split.val;
            trained.test_ratio = metadata->split.test;
            trained.stratified = metadata->split.stratified;
            trained.split_seed = metadata->split.seed;
            trained.image_excluded_files = metadata->split.excluded_files;
        }
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
        ResolveTabularTrainingInputSize(config, arrow_dataset->GetSchema(), label_column);
    config.input_size = input_resolution.input_size;

    if (input_resolution.used_compiled_override) {
        spdlog::info("TrainingManager: Arrow dataset has {} rows, {} cols, "
                     "input_size={} (from GraphCompiler's TimeSeriesWindow override)",
                     num_rows, num_cols, config.input_size);
    } else if (input_resolution.has_separate_label_column) {
        spdlog::info("TrainingManager: Arrow dataset has {} rows, {} cols, input_size={} (label '{}')",
                     num_rows, num_cols, config.input_size, label_column);
    } else {
        spdlog::warn("TrainingManager: Arrow dataset has no label column '{}' ({} cols, input_size={})", label_column,
                     num_cols, config.input_size);
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
        ResolveTabularTrainingInputSize(config, parquet_dataset->GetSchema(), label_column);
    config.input_size = input_resolution.input_size;

    if (input_resolution.used_compiled_override) {
        spdlog::info("TrainingManager: Parquet dataset has {} rows, {} cols, "
                     "input_size={} (from GraphCompiler's TimeSeriesWindow override)",
                     num_rows, num_cols, config.input_size);
    } else if (input_resolution.has_separate_label_column) {
        spdlog::info("TrainingManager: Parquet dataset has {} rows, {} cols, input_size={} (label '{}')",
                     num_rows, num_cols, config.input_size, label_column);
    } else {
        spdlog::warn("TrainingManager: Parquet dataset has no label column '{}' ({} cols, input_size={})", label_column,
                     num_cols, config.input_size);
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

    // A Validation or Test image dataset wired to the Data Split (TOFIX178
    // 1.5): read in full as that part, through the same image steps.
    std::optional<DataRegistry::ImageDatasetEntry> dev_entry, test_entry;
    for (auto [role, entry] : {std::pair{&config.dataset_roles.dev, &dev_entry},
                               std::pair{&config.dataset_roles.test, &test_entry}}) {
        if (!role->IsSupplied() || role->joined_to_training) continue;
        const auto* found = DataRegistry::Instance().GetImageDatasetEntry(role->dataset_name);
        if (!found) {
            spdlog::error("TrainingManager: '{}' is wired as {} but is not a loaded image dataset", role->dataset_name,
                          role == &config.dataset_roles.dev ? "validation" : "test");
            return false;
        }
        *entry = *found;
    }

    // On the training thread: the images decoded once at the graph's Resize
    // size and mode (from the project's cache/images when they were before;
    // TOFIX178 1.3b), then the batcher gathers from them and runs the image
    // transforms and Normalize on the device. A Stop stops the decoding.
    auto stop_requested = std::make_shared<std::function<bool()>>();
    ExternalBatcherFactory factory = [image_entry, dev_entry, test_entry, stop_requested](
                                         const TrainingConfiguration& run, int run_batch_size) {
        const auto stop = [stop_requested] { return *stop_requested && (*stop_requested)(); };
        const auto make_batcher = [&](const DataRegistry::ImageDatasetEntry& entry, const SampleSplitSettings& split) {
            return std::shared_ptr<ImageDatasetBatcher>(MakeImageBatcher(entry, run, run_batch_size, split, stop));
        };
        // The Data Split's shares (a wired role's share is 0: the compiler
        // gave it to train) and its seed, by class when stratified.
        const SampleSplitSettings split = ImageSplitOf(run);
        auto batcher = make_batcher(image_entry, split);
        if (batcher->SamplesIn(BatcherPhase::Train) == 0)
            throw std::runtime_error("the image dataset has no images to train on");

        // The compiler set input_shape [H, W, C] from the same Resize size and
        // transforms (ApplyImageInputShape), so the model and the batches agree.
        spdlog::info("TrainingManager: Image dataset {} train / {} validation / {} test images, input_size={}, {} "
                     "image steps, split seed={}", batcher->SamplesIn(BatcherPhase::Train),
                     batcher->SamplesIn(BatcherPhase::Val), batcher->SamplesIn(BatcherPhase::Test), run.input_size,
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

    if (!config.audio_preprocessing.has_feature_node) {
        spdlog::error("TrainingManager: [{}] {}", errors::Compiler::AudioFeaturesMissing,
                      kAudioFeaturesMissingMessage);
        return false;
    }

    NormalizeTrainingNumWorkers(config, "TrainingManager");

    // AudioDatasetBatcher loads with the entry's settings, extracts the
    // features config.audio_preprocessing describes (Spectrogram /
    // MelSpectrogram / MFCC / AudioAugmentation nodes) and probes a sample
    // for the actual feature shape.
    auto batcher = std::make_shared<AudioDatasetBatcher>(
        audio_entry,
        config.audio_preprocessing,
        batch_size,
        SampleSplitSettings{config.train_ratio, config.val_ratio, config.test_ratio, config.stratified,
                            static_cast<uint32_t>(config.split_seed)},
        config.shuffle, config.num_workers, static_cast<uint32_t>(config.dataloader_seed));
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
    // [rows, cols] = [bins, frames] per sample, torch [N, C, L]: a Conv1D
    // convolves over the frames with one channel per bin.
    config.sequence_input_shape = {
        static_cast<size_t>(batcher->GetFeatureCols()),
        static_cast<size_t>(batcher->GetFeatureRows())
    };

    spdlog::info("TrainingManager: Audio dataset {} train / {} validation / {} test clips, input_size={} ({}x{}), "
                 "num_workers={}, split seed={}", batcher->SamplesIn(BatcherPhase::Train),
                 batcher->SamplesIn(BatcherPhase::Val), batcher->SamplesIn(BatcherPhase::Test), config.input_size,
                 batcher->GetFeatureRows(), batcher->GetFeatureCols(), config.num_workers, config.split_seed);

    // A Validation or Test audio dataset wired to the Data Split (the
    // compiler gave its share to train): read whole as that part, with the
    // same features. Its classes must be the training set's, in the same
    // order, and its features the same shape, or the model's inputs and
    // outputs would not line up.
    std::shared_ptr<AudioDatasetBatcher> wired_dev, wired_test;
    for (auto [role, wired] : {std::pair{&config.dataset_roles.dev, &wired_dev},
                               std::pair{&config.dataset_roles.test, &wired_test}}) {
        if (!role->IsSupplied() || role->joined_to_training) continue;
        const bool dev = role == &config.dataset_roles.dev;
        const char* part = dev ? "validation" : "test";
        const auto* entry = DataRegistry::Instance().GetAudioDatasetEntry(role->dataset_name);
        if (!entry) {
            spdlog::error("TrainingManager: '{}' is wired as {} but is not a loaded audio dataset",
                          role->dataset_name, part);
            return false;
        }
        if (!entry->class_names.empty() && !audio_entry.class_names.empty() &&
            entry->class_names != audio_entry.class_names) {
            spdlog::error("TrainingManager: '{}' (wired as {}) has other classes than the training dataset; "
                          "they must be the same, in the same order", role->dataset_name, part);
            return false;
        }
        auto set = std::make_shared<AudioDatasetBatcher>(
            *entry, config.audio_preprocessing, batch_size,
            SampleSplitSettings{0.0f, dev ? 1.0f : 0.0f, dev ? 0.0f : 1.0f, false,
                                static_cast<uint32_t>(config.split_seed)},
            config.shuffle, config.num_workers, static_cast<uint32_t>(config.dataloader_seed));
        set->SetDropLast(config.drop_last);
        if (set->SamplesIn(dev ? BatcherPhase::Val : BatcherPhase::Test) == 0) {
            spdlog::error("TrainingManager: '{}' (wired as {}) has no clips", role->dataset_name, part);
            return false;
        }
        if (set->GetFeatureRows() != batcher->GetFeatureRows() || set->GetFeatureCols() != batcher->GetFeatureCols()) {
            spdlog::error("TrainingManager: '{}' (wired as {}) gives {}x{} features, the training dataset {}x{}; "
                          "load it with the same sample rate and duration", role->dataset_name, part,
                          set->GetFeatureRows(), set->GetFeatureCols(), batcher->GetFeatureRows(),
                          batcher->GetFeatureCols());
            return false;
        }
        spdlog::info("TrainingManager: '{}' is the {} part: {} clips", role->dataset_name, part,
                     set->SamplesIn(dev ? BatcherPhase::Val : BatcherPhase::Test));
        *wired = std::move(set);
    }

    // Mirror image path: hand normalization / one-hot through to the batchers.
    for (AudioDatasetBatcher* b : {batcher.get(), wired_dev.get(), wired_test.get()}) {
        if (!b) continue;
        if (config.preprocessing.has_normalization) {
            b->SetNormalization(config.preprocessing.norm_mean, config.preprocessing.norm_std);
        }
        if (UsesScalarBinaryTargets(config.loss_type)) {
            b->SetScalarLabelMode(true);
        } else if (config.preprocessing.has_onehot && config.preprocessing.num_classes > 0) {
            b->SetOneHotEncoding(config.preprocessing.num_classes);
        }
    }

    // One batcher, three parts by phase (the held-out test runs at the end);
    // a wired role reads its own set.
    ResolvedExternalBatchers batchers;
    batchers.train = batcher;
    if (wired_dev) batchers.dev = wired_dev;
    else if (batcher->SamplesIn(BatcherPhase::Val) > 0) batchers.dev = batcher;
    if (wired_test) batchers.test = wired_test;
    else if (batcher->SamplesIn(BatcherPhase::Test) > 0) batchers.test = batcher;
    auto executor = std::make_unique<TrainingExecutor>(std::move(config), std::move(batchers));
    return StartTrainingCommon(
        std::move(executor), epochs, batch_size, plot_panel,
        std::move(node_editor_callback),
        "Training Model (Audio)", "Training from Audio Dataset");
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

    // Track training start time for total duration
    auto training_start_time = std::chrono::steady_clock::now();

    TrainingExecutor* exec = nullptr;

    // Set up callbacks
    auto epoch_callback = [this, plot_panel, epochs, &exec](int epoch, float train_loss, float train_acc,
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
            panel->AddEpochLoss(epoch, static_cast<double>(train_loss), static_cast<double>(val_loss));
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
                panel->AddEpochAccuracy(epoch,
                    static_cast<double>(train_acc) * 100.0,
                    static_cast<double>(val_acc) * 100.0);
            }
            // Pair / Retrieval Metrics (TOFIX140 A5), in percent like accuracy.
            {
                const TrainingMetrics latest = exec->GetMetrics();
                if (latest.val_pair_metrics && val_loss >= 0.0f) {
                    panel->AddCustomMetric("Val Pair Accuracy", epoch, latest.val_pair_metrics->accuracy * 100.0);
                }
                if (latest.val_retrieval_metrics && val_loss >= 0.0f) {
                    panel->AddCustomMetric("Val Recall@" + std::to_string(latest.val_retrieval_metrics->k), epoch,
                                           latest.val_retrieval_metrics->recall_at_k * 100.0);
                    panel->AddCustomMetric("Val MRR", epoch,
                                           latest.val_retrieval_metrics->mean_reciprocal_rank * 100.0);
                    panel->AddCustomMetric("Val 1-NN Agreement", epoch,
                                           latest.val_retrieval_metrics->nearest_neighbor_class_agreement * 100.0);
                }
            }
            // Update training state with timing info
            panel->SetTrainingState(true, epoch, epochs, epoch_time, samples_per_sec);
            if (sequence_mode) {
                panel->AddCustomMetric("Train Token Accuracy", epoch,
                    static_cast<double>(seq_metrics.train_token_accuracy) * 100.0);
                const bool entity_metrics =
                    !exec->GetConfig().sequence_batch.create_causal_lm_targets;
                if (entity_metrics) {
                    panel->AddCustomMetric("Train Entity F1", epoch,
                        static_cast<double>(seq_metrics.train_entity_f1) * 100.0);
                }
                if (val_loss >= 0.0f || val_acc >= 0.0f) {
                    panel->AddCustomMetric("Val Token Accuracy", epoch,
                        static_cast<double>(seq_metrics.val_token_accuracy) * 100.0);
                    if (entity_metrics) {
                        panel->AddCustomMetric("Val Entity F1", epoch,
                            static_cast<double>(seq_metrics.val_entity_f1) * 100.0);
                    }
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
                panel->SetBatchComposition(exec->GetConfig().batch_size,
                                           exec->GetConfig().grad_accum_steps);
                panel->SetGenerationPreviewFolder({});
            }
            // Per-batch callback — keeps the Training Dashboard responsive during
            // the epoch. Without this, the dashboard stays on "Epoch 0/N" for the
            // full duration of the first epoch because epoch_callback only fires
            // at epoch boundaries. Also pushes a loss point with a fractional
            // epoch x-axis (epoch - 1 + batch/total) so the loss curve draws
            // smoothly during long epochs (e.g. audio at 8 min/epoch) instead
            // of staying empty until the first epoch finishes.
            // The generation previews folder exists once Train has bound the
            // previews: the first batch tells the Dashboard (TOFIX159 A4).
            auto preview_folder_sent = std::make_shared<bool>(false);
            auto batch_callback = [this, plot_panel, &exec, preview_folder_sent](int epoch, int batch, int total_batches,
                                                       float batch_loss, float batch_acc) {
                const bool regression_mode =
                    exec && UsesRegressionMetrics(exec->GetConfig());
                const int report_interval =
                    exec ? std::max(0, exec->GetConfig().log_interval) : 0;
                // Each point is the mean since the previous one (the executor's
                // report windows); batch 1 is drawn in the first epoch only, so a
                // later epoch never starts with a one-batch point.
                const bool metrics_sampled =
                    ReportWindow::Ends(epoch, batch) &&
                    (batch <= 1 || batch >= total_batches ||
                     (report_interval > 0 && batch % report_interval == 0));
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
                    if (!*preview_folder_sent && exec) {
                        *preview_folder_sent = true;
                        if (const auto folder = exec->GenerationPreviewDir(); !folder.empty())
                            panel->SetGenerationPreviewFolder(folder);
                    }

                    // Progressive curve: x = (epoch - 1) + batch / total_batches
                    // gives a smooth 0..N x-axis where each integer is an epoch
                    // boundary. The epoch_callback later writes the official
                    // x=epoch point with the averaged loss + val metrics.
                    if (metrics_sampled && total_batches > 0) {
                        double frac_epoch = static_cast<double>(epoch - 1) +
                                            static_cast<double>(batch) /
                                            static_cast<double>(total_batches);
                        // The point is the mean since the previous one (the
                        // executor's report window), not the epoch's running mean.
                        const TrainingMetrics point = exec ? exec->GetMetrics() : TrainingMetrics{};
                        panel->AddLossPoint(frac_epoch,
                                            static_cast<double>(exec ? point.report_loss : batch_loss),
                                            -1.0);
                        if (!regression_mode) {
                            panel->AddAccuracyPoint(
                                frac_epoch,
                                static_cast<double>(exec ? point.report_accuracy : batch_acc) * 100.0,
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
                    [&final_metrics](const TrainingMetrics& metrics) {
                        final_metrics = metrics;
                    }
                );
            } catch (const std::exception& error) {
                final_metrics = exec->GetMetrics();
                final_metrics.terminal_status = "failed";
                if (final_metrics.terminal_reason.empty()) {
                    final_metrics.terminal_reason = error.what();
                }
                spdlog::error(
                    "TrainingManager: executor failed without escaping the training thread: {}",
                    error.what());
            } catch (...) {
                final_metrics = exec->GetMetrics();
                final_metrics.terminal_status = "failed";
                if (final_metrics.terminal_reason.empty()) {
                    final_metrics.terminal_reason = "unknown executor exception";
                }
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
            final_metrics.total_epochs = cached_metrics_.total_epochs;
        }
        if (final_metrics.terminal_status.empty() &&
            stop_requested_.load()) {
            final_metrics.terminal_status = "cancelled";
            final_metrics.terminal_reason = "user_cancelled";
            final_metrics.status_message = "Training cancelled";
            final_metrics.is_complete = true;
        }
        if (final_metrics.terminal_status.empty()) {
            final_metrics.terminal_status = "failed";
            final_metrics.terminal_reason = "Executor returned without a terminal result";
        }
        if (final_metrics.terminal_status == "failed") {
            final_metrics.is_complete = true;
            final_metrics.status_message = "Training failed: " + final_metrics.terminal_reason;
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
    // Keep admission closed until callbacks and owned cleanup finish.

    // Deactivate node editor animation
    if (node_editor_callback) {
        node_editor_callback(false);
    }

    // Notify end callback
    if (on_training_end_) {
        on_training_end_(success, final_metrics);
    }

    // Persist any placement evidence this run recorded (runtime GPU
    // fallbacks) so the next compile — including in a future session —
    // routes around the observed device/shape keys. Runs on the training
    // worker thread; the store snapshot is mutex-guarded.

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
            // Training worker thread: its active backend holds the weights.
            active_model_info_.model_device = CurrentThreadModelDevice();
            active_model_info_.checkpoint_path = final_metrics.checkpoint_used;
            const auto& completed_config = current_executor_->GetConfig();
            active_model_info_.evaluation_config = completed_config;
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

    current_task_id_.store(0);
    // The next start may join us while holding mutex_. No callbacks, manager
    // access or cleanup requiring that mutex may follow this publication.
    is_training_.store(false);
}

void TrainingManager::ClearTrainedModel() {
    std::lock_guard<std::mutex> lock(mutex_);
    last_trained_model_.reset();
    last_optimizer_.reset();
    last_metrics_ = TrainingMetrics();
    active_model_info_ = ActiveModelInfo{};
    spdlog::info("TrainingManager: Cleared preserved trained model");
}

} // namespace cyxwiz
