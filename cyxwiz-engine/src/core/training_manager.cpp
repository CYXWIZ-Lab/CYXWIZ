#include "training_manager.h"
#include "crash_run_recorder.h"
#include "image_dataset_batcher.h"
#include "audio_dataset_batcher.h"
#include "text_dataset_batcher.h"
#include "training_batcher_setup.h"
#include "worker_defaults.h"
#include "../gui/panels/training_plot_panel.h"
#include <spdlog/spdlog.h>

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

    if (node_editor_callback) {
        node_editor_callback(true);
    }

    if (auto panel = plot_panel.lock()) {
        panel->Clear();
        panel->SetTrainingState(true, 0, epochs, 0.0f, 0.0f);
        panel->SetVisible(true);
    }

    // Tasks-panel progress tracker. Poll loop reads TrainingManager
    // state — the actual training work lives in TrainingThreadFunc.
    current_task_id_.store(AsyncTaskManager::Instance().RunAsync(
        task_name,
        [](LambdaTask& task) {
            while (!task.ShouldStop()) {
                auto& mgr = TrainingManager::Instance();
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
            task.MarkCompleted();
        },
        nullptr,
        nullptr
    ));

    if (on_training_start_) {
        on_training_start_(start_msg);
    }

    // Join any still-running prior training thread before replacing
    // it. Done here (inside the mutex) so two concurrent starts can't
    // race over training_thread_.
    if (training_thread_ && training_thread_->joinable()) {
        training_thread_->join();
    }

    training_thread_ = std::make_unique<std::thread>(
        &TrainingManager::TrainingThreadFunc, this,
        std::move(executor), epochs, batch_size, plot_panel, node_editor_callback
    );

    spdlog::info("TrainingManager: Started {} ({} epochs, batch_size={})",
                 task_name, epochs, batch_size);
    return true;
}

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

    auto executor = std::make_unique<TrainingExecutor>(std::move(config), arrow_dataset, label_column);
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
        batch_size, config.train_ratio, config.shuffle, config.num_workers);

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

    spdlog::info("TrainingManager: Image dataset {} samples, input_size={} ({}x{}x{}), num_workers={}",
                 batcher->GetNumSamples(), config.input_size, tw, th, ch, config.num_workers);

    // Set up normalization / one-hot from the compiled graph config
    if (config.preprocessing.has_normalization) {
        batcher->SetNormalization(config.preprocessing.norm_mean,
                                  config.preprocessing.norm_std);
    }
    if (config.preprocessing.has_onehot && config.preprocessing.num_classes > 0) {
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
        batch_size, config.train_ratio, config.shuffle, config.num_workers);

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

    spdlog::info("TrainingManager: Audio dataset {} samples, input_size={} ({}x{}), num_workers={}",
                 batcher->GetNumSamples(), config.input_size,
                 batcher->GetFeatureRows(), batcher->GetFeatureCols(), config.num_workers);

    // Mirror image path: hand normalization / one-hot through to the batcher.
    if (config.preprocessing.has_normalization) {
        batcher->SetNormalization(config.preprocessing.norm_mean,
                                  config.preprocessing.norm_std);
    }
    if (config.preprocessing.has_onehot && config.preprocessing.num_classes > 0) {
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
        config.num_workers);

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
                 "(max_length), vocab_size={}, num_workers={}",
                 batcher->GetNumSamples(), config.input_size,
                 batcher->GetVocabSize(), config.num_workers);

    // Hand preprocessing / one-hot hints through to the batcher —
    // same pattern as the image/audio paths.
    if (config.preprocessing.has_onehot && config.preprocessing.num_classes > 0) {
        batcher->SetOneHotEncoding(config.preprocessing.num_classes);
    }

    auto executor = std::make_unique<TrainingExecutor>(std::move(config), std::move(batcher));
    return StartTrainingCommon(
        std::move(executor), epochs, batch_size, plot_panel,
        std::move(node_editor_callback),
        "Training Model (Text)", "Training from Text Dataset");
}

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

    // Set up callbacks
    auto epoch_callback = [this, plot_panel, epochs, batch_size](int epoch, float train_loss, float train_acc,
                                               float val_loss, float val_acc, float epoch_time) {
        // Update cached metrics
        float samples_per_sec = 0.0f;
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            cached_metrics_.current_epoch = epoch;
            cached_metrics_.train_loss = train_loss;
            cached_metrics_.train_accuracy = train_acc;
            cached_metrics_.val_loss = val_loss;
            cached_metrics_.val_accuracy = val_acc;
            cached_metrics_.epoch_time_seconds = epoch_time;
            cached_metrics_.loss_history.push_back(train_loss);
            cached_metrics_.accuracy_history.push_back(train_acc);

            // Calculate samples/sec from cached metrics
            if (epoch_time > 0) {
                samples_per_sec = cached_metrics_.samples_per_second;
            }
        }

        // Update plot panel with metrics and training state
        if (auto panel = plot_panel.lock()) {
            CrashRunRecorder::Instance().MarkStage(
                TrainingTraceStage::UIPlotUpdate, epoch, 0, 0,
                train_loss, train_acc);
            panel->AddLossPoint(epoch, static_cast<double>(train_loss), static_cast<double>(val_loss));
            // Convert accuracy from fraction (0-1) to percentage (0-100) for display
            panel->AddAccuracyPoint(epoch, static_cast<double>(train_acc) * 100.0, static_cast<double>(val_acc) * 100.0);
            // Update training state with timing info
            panel->SetTrainingState(true, epoch, epochs, epoch_time, samples_per_sec);
            spdlog::info("TrainingPlotPanel: Updated state - epoch={}/{}, time={:.1f}s, sps={:.0f}",
                         epoch, epochs, epoch_time, samples_per_sec);
        } else {
            spdlog::warn("TrainingPlotPanel: panel expired or unavailable");
        }

        // Notify progress callback
        if (on_progress_) {
            on_progress_(epoch, train_loss, train_acc);
        }

        spdlog::info("Epoch {}: loss={:.4f}, acc={:.2f}%, val_loss={:.4f}, val_acc={:.2f}% ({:.1f}s)",
                     epoch, train_loss, train_acc * 100, val_loss, val_acc * 100, epoch_time);
    };

    TrainingMetrics final_metrics;

    // Check if stop was requested before training started
    if (stop_requested_.load()) {
        spdlog::info("TrainingManager: Training cancelled before start");
    } else {
        // Run training
        TrainingExecutor* exec = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            exec = current_executor_.get();
        }

        if (exec) {
            // Per-batch callback — keeps the Training Dashboard responsive during
            // the epoch. Without this, the dashboard stays on "Epoch 0/N" for the
            // full duration of the first epoch because epoch_callback only fires
            // at epoch boundaries. Also pushes a loss point with a fractional
            // epoch x-axis (epoch - 1 + batch/total) so the loss curve draws
            // smoothly during long epochs (e.g. audio at 8 min/epoch) instead
            // of staying empty until the first epoch finishes.
            auto batch_callback = [this, plot_panel](int epoch, int batch, int total_batches,
                                                      float batch_loss, float batch_acc) {
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
                    if (total_batches > 0) {
                        double frac_epoch = static_cast<double>(epoch - 1) +
                                            static_cast<double>(batch) /
                                            static_cast<double>(total_batches);
                        panel->AddLossPoint(frac_epoch,
                                            static_cast<double>(batch_loss),
                                            -1.0);
                        panel->AddAccuracyPoint(frac_epoch,
                                                static_cast<double>(batch_acc) * 100.0,
                                                -1.0);
                    }
                }
            };

            exec->Train(
                epochs,
                batch_size,
                batch_callback,
                epoch_callback,
                [this, &final_metrics](const TrainingMetrics& metrics) {
                    final_metrics = metrics;
                }
            );
        }
    }

    // Calculate total training time
    auto training_end_time = std::chrono::steady_clock::now();
    float total_training_time = std::chrono::duration<float>(training_end_time - training_start_time).count();

    // Get final metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        final_metrics = cached_metrics_;
        cached_metrics_.is_training = false;
        cached_metrics_.is_complete = !stop_requested_.load();
    }

    // Update plot panel with completion status
    if (auto panel = plot_panel.lock()) {
        panel->SetTrainingComplete(total_training_time);
    }

    // Cleanup
    bool success = !stop_requested_.load();
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
        if (current_executor_ && (success || stop_requested_.load())) {
            // Transfer ownership of model and optimizer for later export
            last_trained_model_ = current_executor_->ReleaseModel();
            last_optimizer_ = current_executor_->ReleaseOptimizer();
            last_metrics_ = final_metrics;
            spdlog::info("TrainingManager: Preserved trained model for export (success={}, stopped={})",
                         success, stop_requested_.load());
        }
        current_executor_.reset();
    }

    if (success) {
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
