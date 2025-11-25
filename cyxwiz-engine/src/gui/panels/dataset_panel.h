#pragma once

#include "../panel.h"
#include <string>
#include <vector>
#include <memory>

namespace cyxwiz {
    class Tensor;
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

    DatasetInfo dataset_info_;

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
