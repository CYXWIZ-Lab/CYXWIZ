#pragma once

#include "../panel.h"
#include "../../core/data_registry.h"
#include "../../core/async_task_manager.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>

namespace cyxwiz {
    class Tensor;
    class TrainingPlotPanel;
}

namespace network {
    class JobManager;
}

namespace gui {

class DatasetPanel : public cyxwiz::Panel {
public:
    DatasetPanel();
    ~DatasetPanel() override;

    void Render() override;

    // Dataset loading - now delegates to DataRegistry (blocking versions)
    bool LoadCSVDataset(const std::string& path);
    bool LoadImageDataset(const std::string& path);
    bool LoadMNISTDataset(const std::string& path);
    bool LoadCIFAR10Dataset(const std::string& path);
    bool LoadHuggingFaceDataset(const std::string& dataset_name);
    bool LoadKaggleDataset(const std::string& dataset_slug);

    // Load from path with auto-detection
    bool LoadDataset(const std::string& path);

    // Async dataset loading (non-blocking versions)
    void LoadCSVDatasetAsync(const std::string& path);
    void LoadImageDatasetAsync(const std::string& path);
    void LoadMNISTDatasetAsync(const std::string& path);
    void LoadCIFAR10DatasetAsync(const std::string& path);
    void LoadHuggingFaceDatasetAsync(const std::string& dataset_name);
    void LoadKaggleDatasetAsync(const std::string& dataset_slug);
    void LoadDatasetAsync(const std::string& path);

    // Async loading state
    bool IsLoading() const { return is_loading_.load(); }
    void CancelLoading();

    // Dataset access
    const cyxwiz::DatasetInfo& GetDatasetInfo() const;
    bool IsDatasetLoaded() const { return current_dataset_.IsValid(); }
    cyxwiz::DatasetHandle GetCurrentDataset() const { return current_dataset_; }

    // Get samples for preview
    bool GetPreviewSamples(int count, std::vector<float>& out_images, std::vector<int>& out_labels);

    // Clear loaded dataset
    void ClearDataset();

    // Job submission (P2P training)
    void SetJobManager(network::JobManager* job_manager) { job_manager_ = job_manager; }

    // Callback for when training starts (for UI updates)
    using TrainingStartCallback = std::function<void(const std::string& job_id)>;
    void SetTrainingStartCallback(TrainingStartCallback callback) { training_start_callback_ = callback; }

    // Local training support
    void SetTrainingPlotPanel(cyxwiz::TrainingPlotPanel* panel) { training_plot_panel_ = panel; }
    bool IsLocalTrainingRunning() const { return local_training_running_.load(); }
    void StopLocalTraining() { local_training_stop_requested_.store(true); }

    // Get raw data for job submission (from DataRegistry)
    const std::vector<size_t>& GetTrainIndices() const;

private:
    void RenderDatasetSelection();
    void RenderDatasetInfo();
    void RenderSplitConfiguration();
    void RenderDataPreview();
    void RenderStatistics();
    void RenderLoadedDatasets();

    // File browser dialog
    void ShowFileBrowser();

    // Data preprocessing
    void ApplySplit();
    void UpdateClassCounts();

    // Visualization
    void RenderImagePreview(const float* image_data, int width, int height, int channels);

    // Training
    void RenderTrainingSection();
    bool SubmitTrainingJob();      // P2P training
    void StartLocalTraining();     // Local training
    void LocalTrainingThread();    // Background thread for local training

    // Current active dataset (from DataRegistry)
    cyxwiz::DatasetHandle current_dataset_;
    cyxwiz::DatasetInfo cached_info_;  // Cached for quick access

    // Job manager for submitting training jobs
    network::JobManager* job_manager_ = nullptr;
    TrainingStartCallback training_start_callback_;

    // Training configuration
    int train_epochs_ = 10;
    int train_batch_size_ = 32;
    float train_learning_rate_ = 0.001f;
    int selected_optimizer_ = 0;  // 0=SGD, 1=Adam, 2=AdamW
    std::string last_submitted_job_id_;

    // Local training state
    cyxwiz::TrainingPlotPanel* training_plot_panel_ = nullptr;
    std::unique_ptr<std::thread> local_training_thread_;
    std::atomic<bool> local_training_running_{false};
    std::atomic<bool> local_training_stop_requested_{false};
    std::mutex local_training_mutex_;

    // Local training progress (for UI display)
    std::atomic<int> local_current_epoch_{0};
    std::atomic<float> local_current_loss_{0.0f};
    std::atomic<float> local_current_accuracy_{0.0f};

    // Split configuration (for UI)
    cyxwiz::SplitConfig split_config_;

    // UI state
    bool show_file_browser_ = false;
    char file_path_buffer_[512] = "";
    char hf_dataset_name_[256] = "mnist";  // HuggingFace dataset name
    char kaggle_dataset_slug_[256] = "titanic";  // Kaggle dataset slug
    cyxwiz::DatasetType selected_type_ = cyxwiz::DatasetType::None;
    int preview_sample_idx_ = 0;
    int selected_dataset_index_ = -1;  // For dataset list

    // Class names (for visualization)
    std::vector<std::string> class_names_;

    // Statistics cache
    std::vector<int> class_counts_;  // Per-class sample counts

    // Async loading state
    std::atomic<bool> is_loading_{false};
    uint64_t loading_task_id_ = 0;
    std::string loading_status_message_;
    std::atomic<float> loading_progress_{0.0f};
};

} // namespace gui
