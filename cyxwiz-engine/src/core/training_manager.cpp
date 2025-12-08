#include "training_manager.h"
#include "../gui/panels/training_plot_panel.h"
#include <spdlog/spdlog.h>

namespace cyxwiz {

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

bool TrainingManager::StartTraining(
    TrainingConfiguration config,
    DatasetHandle dataset,
    int epochs,
    int batch_size,
    TrainingPlotPanel* plot_panel,
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

    // Create executor
    auto executor = std::make_unique<TrainingExecutor>(std::move(config), dataset);

    // Set state
    is_training_.store(true);
    stop_requested_.store(false);

    // Activate node editor animation
    if (node_editor_callback) {
        node_editor_callback(true);
    }

    // Clear plot panel
    if (plot_panel) {
        plot_panel->Clear();
        plot_panel->SetVisible(true);
    }

    // Create async task for visibility in Tasks panel
    std::string task_name = "Training Model";
    current_task_id_.store(AsyncTaskManager::Instance().RunAsync(
        task_name,
        [](LambdaTask& task) {
            // This task just tracks the training - actual work is in training thread
            while (!task.ShouldStop()) {
                auto& mgr = TrainingManager::Instance();
                if (!mgr.IsTrainingActive()) {
                    break;
                }
                auto metrics = mgr.GetCurrentMetrics();
                float progress = static_cast<float>(metrics.current_epoch) / std::max(1, metrics.total_epochs);
                task.ReportProgress(progress,
                    "Epoch " + std::to_string(metrics.current_epoch) + "/" +
                    std::to_string(metrics.total_epochs) +
                    " - Loss: " + std::to_string(metrics.train_loss).substr(0, 6));
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            task.MarkCompleted();
        },
        nullptr,  // progress callback
        nullptr   // completion callback
    ));

    // Notify start callback
    if (on_training_start_) {
        on_training_start_("Training from Node Graph");
    }

    // Wait for previous thread if any
    if (training_thread_ && training_thread_->joinable()) {
        training_thread_->join();
    }

    // Start training thread
    training_thread_ = std::make_unique<std::thread>(
        &TrainingManager::TrainingThreadFunc, this,
        std::move(executor), epochs, batch_size, plot_panel, node_editor_callback
    );

    spdlog::info("TrainingManager: Started training ({} epochs, batch_size={})", epochs, batch_size);
    return true;
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
    TrainingPlotPanel* plot_panel,
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

    // Set up callbacks
    auto epoch_callback = [this, plot_panel](int epoch, float train_loss, float train_acc,
                                               float val_loss, float val_acc, float epoch_time) {
        // Update cached metrics
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
        }

        // Update plot panel
        if (plot_panel) {
            plot_panel->AddLossPoint(epoch, static_cast<double>(train_loss), static_cast<double>(val_loss));
            // Convert accuracy from fraction (0-1) to percentage (0-100) for display
            plot_panel->AddAccuracyPoint(epoch, static_cast<double>(train_acc) * 100.0, static_cast<double>(val_acc) * 100.0);
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
            exec->Train(
                epochs,
                batch_size,
                nullptr,  // batch callback
                epoch_callback,
                [this, &final_metrics](const TrainingMetrics& metrics) {
                    final_metrics = metrics;
                }
            );
        }
    }

    // Get final metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        final_metrics = cached_metrics_;
        cached_metrics_.is_training = false;
        cached_metrics_.is_complete = !stop_requested_.load();
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
        if (current_executor_ && success) {
            // Transfer ownership of model and optimizer for later export
            last_trained_model_ = current_executor_->ReleaseModel();
            last_optimizer_ = current_executor_->ReleaseOptimizer();
            last_metrics_ = final_metrics;
            spdlog::info("TrainingManager: Preserved trained model for export");
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

} // namespace cyxwiz
