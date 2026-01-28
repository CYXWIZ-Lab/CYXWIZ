#pragma once

#include "../panel.h"
#include "../../core/data_registry.h"
#include "../../core/async_task_manager.h"
#include "../../core/training_executor.h"
#include "../../transforms/transforms.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <future>
#include "../../preprocessing/preprocessing_config.h"
#include "../../preprocessing/statistics_calculator.h"
#include "../../utils/dataset_analyzer.h"
#include "../../utils/image_quality_analyzer.h"
#include "../../utils/image_deduplicator.h"
#include "../../utils/hdf5_browser.h"

namespace cyxwiz {
    class Tensor;
    class TrainingPlotPanel;
}

namespace network {
    class JobManager;
}

namespace gui {

class NodeEditor;

// Content tabs for main area (DBGate-style layout)
enum class ContentTab {
    Preview = 0,
    Pipeline = 1,
    Training = 2,
    Details = 3
};

// Legacy tab indices (kept for compatibility)
enum class DatasetTab {
    LoadDataset = 0,
    LoadedDatasets = 1,
    Preview = 2,
    DataPipeline = 3,
    Interactive = 4,
    Training = 5
};

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
    void LoadHDF5DatasetAsync(const std::string& path);
    void LoadImageDatasetAsync(const std::string& path);
    void LoadMNISTDatasetAsync(const std::string& path);
    void LoadCIFAR10DatasetAsync(const std::string& path);
    void LoadHuggingFaceDatasetAsync(const std::string& dataset_name);
    void LoadKaggleDatasetAsync(const std::string& dataset_slug);
    void LoadCustomDatasetAsync(const cyxwiz::CustomConfig& config);
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
    void SetWalletPanel(class WalletPanel* wallet_panel) { wallet_panel_ = wallet_panel; }

    // Callback for when training starts (for UI updates)
    using TrainingStartCallback = std::function<void(const std::string& job_id)>;
    void SetTrainingStartCallback(TrainingStartCallback callback) { training_start_callback_ = callback; }

    // Local training support (delegates to TrainingManager)
    void SetTrainingPlotPanel(cyxwiz::TrainingPlotPanel* panel) { training_plot_panel_ = panel; }
    void SetNodeEditor(NodeEditor* editor) { node_editor_ = editor; }
    bool IsLocalTrainingRunning() const;  // Implemented in cpp
    void StopLocalTraining();             // Implemented in cpp

    // Training hyperparameters (for Node Editor training path)
    int GetTrainEpochs() const { return train_epochs_; }
    int GetTrainBatchSize() const { return train_batch_size_; }

    // Tab selection for menu navigation
    void SetActiveTab(DatasetTab tab) { pending_tab_ = static_cast<int>(tab); }

    // Get raw data for job submission (from DataRegistry)
    const std::vector<size_t>& GetTrainIndices() const;

private:
    int pending_tab_ = -1;  // -1 means no pending tab switch

    // ========== DBGate-Style Layout ==========
    // Layout state
    float sidebar_width_ = 200.0f;
    float footer_height_ = 24.0f;
    bool sidebar_collapsed_ = false;
    ContentTab active_content_tab_ = ContentTab::Preview;

    // Search buffer for sidebar filtering
    char search_buffer_[256] = "";

    // Layout rendering methods
    void RenderToolbar();
    void RenderSidebar();
    void RenderMainContent();
    void RenderStatusBar();
    void RenderSplitter(float height);
    void RenderDatasetTree();
    void RenderPreviewContent();
    void RenderPipelineContent();
    void RenderTrainingContent();
    void RenderDetailsContent();
    void RenderQuickLoadSection();
    const char* GetDatasetTypeIcon(cyxwiz::DatasetType type) const;
    
    // ========== Legacy Methods ==========
    void RenderDatasetSelection();
    void RenderDatasetInfo();
    void RenderSplitConfiguration();
    void RenderDataPreview();
    void RenderStatistics();
    void RenderLoadedDatasets();
    void RenderAugmentationTab();
    void RenderAugmentationPipeline();
    void RenderAugmentationPreview();

    // Data Pipeline tab (unified Augmentation + Preprocessing)
    void RenderDataPipelineTab();
    void RenderUnifiedPreview();

    // Interactive Tools tab (Phase 4)
    void RenderInteractiveToolsTab();

    // Preview display methods (extracted for unified preview)
    void RenderPreprocessingPreviewDisplay();
    void RenderAugmentationPreviewDisplay();
    void UpdateAugmentationPreview();

    // Preprocessing methods (used by Data Pipeline tab)
    void RenderPreprocessingTab();  // Deprecated - kept for reference
    void RenderDatasetStatistics();
    void RenderNormalizationSection();
    void RenderScalingSection();
    void RenderImagePreprocessingSection();
    void RenderPreprocessingPreview();

    // File browser dialogs
    void ShowFileBrowser();
    void ShowFolderBrowser(char* buffer, size_t buffer_size);
    void ShowCSVFileBrowser();
    void RenderHDF5ConfigDialog();
    void ScanHDF5File(const std::string& path);
    void UpdateHDF5Preview();

    // Data preprocessing
    void ApplySplit();
    void UpdateClassCounts();

    // Augmentation configuration
    void ApplyAugmentationConfig();

    // Visualization
    void RenderImagePreview(const float* image_data, int width, int height, int channels);
    void RenderSingleSamplePreview(size_t dataset_size, bool is_image_data);
    void RenderGridPreview(size_t dataset_size, bool is_image_data);
    void RenderTablePreview(size_t dataset_size);

    // Training
    void RenderTrainingSection();
    bool SubmitTrainingJob();      // P2P training
    void StartLocalTraining();     // Local training (delegates to TrainingManager)

    // Current active dataset (from DataRegistry)
    cyxwiz::DatasetHandle current_dataset_;
    cyxwiz::DatasetInfo cached_info_;  // Cached for quick access

    // Job manager for submitting training jobs
    network::JobManager* job_manager_ = nullptr;
    WalletPanel* wallet_panel_ = nullptr;
    TrainingStartCallback training_start_callback_;

    // Training configuration
    int train_epochs_ = 10;
    int train_batch_size_ = 32;
    float train_learning_rate_ = 0.001f;
    int selected_optimizer_ = 1;  // 0=SGD, 1=Adam, 2=AdamW, 3=RMSprop
    std::string last_submitted_job_id_;

    // Advanced training options
    float train_weight_decay_ = 0.0001f;
    float train_grad_clip_ = 1.0f;
    bool train_early_stopping_ = false;
    int train_early_stopping_patience_ = 5;
    bool train_lr_scheduler_ = false;
    int train_scheduler_type_ = 0;  // 0=StepLR, 1=CosineAnnealing, 2=ReduceOnPlateau

    // Training plot panel (for progress visualization)
    cyxwiz::TrainingPlotPanel* training_plot_panel_ = nullptr;

    // Node editor for getting graph architecture
    NodeEditor* node_editor_ = nullptr;

    // Split configuration (for UI)
    cyxwiz::SplitConfig split_config_;

    // UI state
    bool show_file_browser_ = false;
    char file_path_buffer_[512] = "";
    char csv_path_buffer_[512] = "";         // CSV file path for ImageCSV
    char hf_dataset_name_[256] = "mnist";    // HuggingFace dataset name
    char kaggle_dataset_slug_[256] = "titanic";  // Kaggle dataset slug
    char hdf5_data_path_[128] = "";          // HDF5 data dataset path (auto-detect if empty)
    char hdf5_label_path_[128] = "";         // HDF5 labels dataset path (auto-detect if empty)
    bool hdf5_normalize_ = true;             // Normalize uint8 to [0,1]
    bool show_hdf5_dialog_ = false;          // Show HDF5 configuration dialog
    bool hdf5_file_scanned_ = false;         // Has the file been scanned?
    std::string hdf5_dialog_path_;           // Path being configured in dialog
    int hdf5_selected_data_idx_ = -1;        // Selected data dataset in list
    int hdf5_selected_label_idx_ = -1;       // Selected label dataset in list
    std::vector<std::string> hdf5_dataset_paths_;  // List of dataset paths in file
    std::vector<std::string> hdf5_dataset_info_;   // Dataset info strings for display
    std::vector<float> hdf5_preview_data_;   // Preview data for selected dataset
    std::vector<size_t> hdf5_preview_shape_; // Shape of preview samples
    unsigned int hdf5_preview_texture_ = 0;  // Preview texture ID
    int hdf5_preview_tex_w_ = 0, hdf5_preview_tex_h_ = 0;
    cyxwiz::DatasetType selected_type_ = cyxwiz::DatasetType::None;
    int preview_sample_idx_ = 0;
    int selected_dataset_index_ = -1;  // For dataset list
    int image_target_width_ = 224;     // Target image width for ImageCSV
    int image_target_height_ = 224;    // Target image height for ImageCSV
    int image_cache_size_ = 100;       // LRU cache size for lazy loading

    // Popular datasets (unified HuggingFace/Kaggle)
    int popular_dataset_source_ = 0;   // 0=HuggingFace, 1=Kaggle
    int popular_dataset_index_ = 0;    // Selected dataset in dropdown
    char dataset_search_buffer_[256] = "";  // Search query for external search

    // Preview settings
    int preview_view_mode_ = 0;        // 0=Single, 1=Grid, 2=Table
    float preview_zoom_ = 1.0f;        // Image zoom level
    int preview_grid_cols_ = 4;        // Grid columns
    int preview_grid_rows_ = 4;        // Grid rows
    int preview_table_rows_ = 20;      // Rows per page in table view

    // Class names (for visualization)
    std::vector<std::string> class_names_;

    // Statistics cache
    std::vector<int> class_counts_;  // Per-class sample counts

    // Async loading state
    std::atomic<bool> is_loading_{false};
    uint64_t loading_task_id_ = 0;
    std::string loading_status_message_;
    std::atomic<float> loading_progress_{0.0f};

    // Augmentation pipeline
    std::shared_ptr<cyxwiz::transforms::Compose> augmentation_pipeline_;
    bool show_augmented_preview_ = false;
    int augmentation_preset_ = 0;  // 0=None, 1=ImageNet, 2=CIFAR10, 3=Medical, 4=Custom
    cyxwiz::transforms::Image preview_original_;
    cyxwiz::transforms::Image preview_augmented_;
    unsigned int preview_texture_original_ = 0;
    unsigned int preview_texture_augmented_ = 0;
    int preview_tex_orig_w_ = 0, preview_tex_orig_h_ = 0, preview_tex_orig_c_ = 0;
    int preview_tex_aug_w_ = 0, preview_tex_aug_h_ = 0, preview_tex_aug_c_ = 0;
    bool preview_needs_update_ = true;

    // Transform UI state
    struct TransformUIState {
        bool enabled = true;
        bool expanded = false;
    };
    std::vector<TransformUIState> transform_ui_states_;

    // Preprocessing state
    cyxwiz::PreprocessingConfig current_preprocessing_config_;
    cyxwiz::DatasetStatistics current_stats_;
    bool stats_computed_ = false;
    bool computing_stats_ = false;
    float stats_computation_progress_ = 0.0f;
    std::future<cyxwiz::DatasetStatistics> stats_future_;

    // Preprocessing preview
    bool show_preprocessing_preview_ = false;
    cyxwiz::Tensor preview_preprocessed_;
    unsigned int preview_texture_preprocessed_ = 0;
    int preview_tex_prep_w_ = 0, preview_tex_prep_h_ = 0, preview_tex_prep_c_ = 0;
    bool preprocessing_preview_needs_update_ = true;

    // Preprocessing preview textures (before/after)
    unsigned int preview_texture_before_ = 0;  // Original sample texture
    unsigned int preview_texture_after_ = 0;   // Preprocessed sample texture
    int preview_tex_before_w_ = 0, preview_tex_before_h_ = 0, preview_tex_before_c_ = 0;
    int preview_tex_after_w_ = 0, preview_tex_after_h_ = 0, preview_tex_after_c_ = 0;

    // Preprocessing helper methods
    void ComputeStatistics();
    void ApplyPreprocessingConfig();
    void UpdatePreprocessingPreview();  // Apply preprocessing to preview sample

    // Dataset Analytics
    void RenderDatasetAnalyticsSection();
    void StartAnalyticsComputation();
    std::atomic<bool> analytics_computing_{false};
    uint64_t analytics_task_id_ = 0;
    cyxwiz::DatasetAnalytics analytics_results_;
    bool show_analytics_outliers_popup_ = false;
    bool show_histogram_popup_ = false;

    // Image Quality Analysis
    void RenderQualityAnalysisSection();
    void StartQualityAnalysis();
    std::atomic<bool> quality_analysis_running_{false};
    uint64_t quality_analysis_task_id_ = 0;
    std::vector<cyxwiz::QualityMetrics> quality_results_;
    cyxwiz::ImageQualityAnalyzer::Thresholds quality_thresholds_;
    bool show_quality_issues_popup_ = false;

    // Duplicate Detection
    void RenderDuplicateDetectionSection();
    void StartDuplicateDetection();
    std::atomic<bool> duplicate_detection_running_{false};
    uint64_t duplicate_detection_task_id_ = 0;
    std::vector<cyxwiz::DuplicateGroup> duplicate_groups_;
    int duplicate_hash_type_ = 0;  // 0=pHash, 1=aHash, 2=dHash
    int duplicate_threshold_ = 5;   // Hamming distance threshold
    bool show_duplicate_groups_popup_ = false;

    // Notification state (for "Set as Active" feedback)
    bool show_notification_ = false;
    float notification_time_ = 0.0f;
    std::string notification_message_;
};

} // namespace gui
