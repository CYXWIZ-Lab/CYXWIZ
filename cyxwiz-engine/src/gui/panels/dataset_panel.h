#pragma once

#include "../panel.h"
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

enum class DatasetType {
    None,
    CSV,
    Images,
    MNIST,
    CIFAR10
};

struct DatasetInfo {
    DatasetType type = DatasetType::None;
    std::string path;
    std::string name;
    int num_samples = 0;
    int num_classes = 0;
    std::vector<int> input_shape;  // e.g., [28, 28, 1] for MNIST

    // Split configuration
    float train_ratio = 0.8f;
    float val_ratio = 0.1f;
    float test_ratio = 0.1f;

    // Loaded data counts
    int train_count = 0;
    int val_count = 0;
    int test_count = 0;
};

class DatasetPanel : public cyxwiz::Panel {
public:
    DatasetPanel();
    ~DatasetPanel() override;

    void Render() override;

    // Dataset loading
    bool LoadCSVDataset(const std::string& path);
    bool LoadImageDataset(const std::string& path);
    bool LoadMNISTDataset(const std::string& path);
    bool LoadCIFAR10Dataset(const std::string& path);

    // Dataset access
    const DatasetInfo& GetDatasetInfo() const { return dataset_info_; }
    bool IsDatasetLoaded() const { return dataset_info_.type != DatasetType::None; }

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

    // Get raw data for job submission
    const std::vector<std::vector<float>>& GetRawSamples() const { return raw_samples_; }
    const std::vector<int>& GetRawLabels() const { return raw_labels_; }
    const std::vector<int>& GetTrainIndices() const { return train_indices_; }

private:
    void RenderDatasetSelection();
    void RenderDatasetInfo();
    void RenderSplitConfiguration();
    void RenderDataPreview();
    void RenderStatistics();

    // File browser dialog
    void ShowFileBrowser();

    // Data preprocessing
    void ApplySplit();
    void ShuffleData();
    void NormalizeData();

    // Visualization
    void RenderImagePreview(const float* image_data, int width, int height, int channels);
    void RenderClassDistribution();

    // Training
    void RenderTrainingSection();
    bool SubmitTrainingJob();      // P2P training
    void StartLocalTraining();     // Local training
    void LocalTrainingThread();    // Background thread for local training

    DatasetInfo dataset_info_;

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

    // Raw data storage (will be moved to proper data manager later)
    std::vector<std::vector<float>> raw_samples_;  // All samples
    std::vector<int> raw_labels_;                   // All labels

    // Split indices
    std::vector<int> train_indices_;
    std::vector<int> val_indices_;
    std::vector<int> test_indices_;

    // UI state
    bool show_file_browser_ = false;
    char file_path_buffer_[512] = "";
    DatasetType selected_type_ = DatasetType::None;
    int preview_sample_idx_ = 0;

    // Class names (for visualization)
    std::vector<std::string> class_names_;

    // Statistics
    std::vector<int> class_counts_;  // Per-class sample counts
};

} // namespace gui
