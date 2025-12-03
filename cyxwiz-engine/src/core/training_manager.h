#pragma once

#include "training_executor.h"
#include "graph_compiler.h"
#include "data_registry.h"
#include "async_task_manager.h"
#include <atomic>
#include <mutex>
#include <memory>
#include <functional>
#include <string>
#include <thread>

namespace cyxwiz {

class TrainingPlotPanel;

/**
 * TrainingManager - Centralized manager for ML training sessions
 *
 * This singleton ensures only one training session runs at a time.
 * All training uses the compiled node graph configuration.
 * Integrates with the async task system to show training progress in the Tasks panel.
 */
class TrainingManager {
public:
    // Singleton access
    static TrainingManager& Instance();

    // Check if training is currently active
    bool IsTrainingActive() const { return is_training_.load(); }

    // Get the current training task ID (0 if none)
    uint64_t GetCurrentTaskId() const { return current_task_id_.load(); }

    // Get current training metrics (thread-safe)
    TrainingMetrics GetCurrentMetrics() const;

    /**
     * Start training with compiled graph configuration
     * @param config Compiled training configuration from GraphCompiler
     * @param dataset Dataset handle
     * @param epochs Number of epochs
     * @param batch_size Batch size
     * @param plot_panel Optional plot panel for progress visualization
     * @param node_editor_callback Callback to update node editor training state
     * @return true if training started, false if already training
     */
    bool StartTraining(
        TrainingConfiguration config,
        DatasetHandle dataset,
        int epochs,
        int batch_size,
        TrainingPlotPanel* plot_panel = nullptr,
        std::function<void(bool)> node_editor_callback = nullptr
    );

    /**
     * Stop the current training session
     */
    void StopTraining();

    /**
     * Pause the current training session
     */
    void PauseTraining();

    /**
     * Resume a paused training session
     */
    void ResumeTraining();

    /**
     * Check if training is paused
     */
    bool IsPaused() const;

    // Callbacks for training events
    using TrainingStartCallback = std::function<void(const std::string& description)>;
    using TrainingEndCallback = std::function<void(bool success, const TrainingMetrics& metrics)>;
    using ProgressCallback = std::function<void(int epoch, float loss, float accuracy)>;

    void SetOnTrainingStart(TrainingStartCallback callback) { on_training_start_ = callback; }
    void SetOnTrainingEnd(TrainingEndCallback callback) { on_training_end_ = callback; }
    void SetOnProgress(ProgressCallback callback) { on_progress_ = callback; }

private:
    TrainingManager() = default;
    ~TrainingManager();

    TrainingManager(const TrainingManager&) = delete;
    TrainingManager& operator=(const TrainingManager&) = delete;

    void TrainingThreadFunc(
        std::unique_ptr<TrainingExecutor> executor,
        int epochs,
        int batch_size,
        TrainingPlotPanel* plot_panel,
        std::function<void(bool)> node_editor_callback
    );

    std::atomic<bool> is_training_{false};
    std::atomic<bool> stop_requested_{false};
    std::atomic<uint64_t> current_task_id_{0};

    mutable std::mutex mutex_;
    std::unique_ptr<TrainingExecutor> current_executor_;
    std::unique_ptr<std::thread> training_thread_;

    // Cached metrics for thread-safe access
    mutable std::mutex metrics_mutex_;
    TrainingMetrics cached_metrics_;

    // Callbacks
    TrainingStartCallback on_training_start_;
    TrainingEndCallback on_training_end_;
    ProgressCallback on_progress_;
};

} // namespace cyxwiz
