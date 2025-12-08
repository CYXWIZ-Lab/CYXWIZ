// fine_tuning_panel.h - Model fine-tuning configuration UI
#pragma once
#include "gui/server_panel.h"
#include "ipc/daemon_client.h"
#include <vector>
#include <string>

namespace cyxwiz::servernode::gui {

// LR Schedule types
enum class LRScheduleType {
    Constant,
    Step,
    Exponential,
    Cosine,
    OneCycle,
    CosineWarmRestart
};

// Layer freezing mode
enum class FreezeMode {
    None,           // Train all layers
    ExceptLastN,    // Freeze all except last N layers
    UpToLayer,      // Freeze layers up to specified index
    Custom          // Custom per-layer selection
};

// Fine-tuning configuration
struct FineTuningConfig {
    // Model
    std::string model_path;
    std::string dataset_path;

    // Freeze settings
    FreezeMode freeze_mode = FreezeMode::None;
    int unfreeze_last_n = 1;
    int freeze_up_to_layer = 0;
    std::vector<char> layer_trainable;  // Using char because std::vector<bool> has issues with pointers

    // LR Schedule
    LRScheduleType lr_schedule = LRScheduleType::Constant;
    float initial_lr = 1e-4f;
    float final_lr = 1e-6f;
    float max_lr = 1e-3f;           // For OneCycle
    int warmup_epochs = 0;
    int step_size = 10;             // For Step schedule
    float step_gamma = 0.1f;        // LR multiplier per step
    float exp_gamma = 0.95f;        // For Exponential schedule
    int t_max = 50;                 // For Cosine schedule
    float eta_min = 1e-6f;          // Min LR for Cosine

    // Early stopping
    bool enable_early_stopping = true;
    int patience = 5;
    float min_delta = 1e-4f;
    std::string monitor_metric = "val_loss";  // "val_loss" or "val_accuracy"
    bool restore_best_weights = true;

    // Gradient clipping
    bool enable_grad_clipping = true;
    float max_grad_norm = 1.0f;

    // Training parameters
    int epochs = 10;
    int batch_size = 32;
    float weight_decay = 0.01f;
    bool mixed_precision = false;

    // Device selection
    int device_id = -1;  // -1 = auto from pool
};

// Layer info for display
struct LayerInfo {
    std::string name;
    std::string type;
    int64_t param_count = 0;
    bool trainable = true;
    bool has_weights = true;
};

class FineTuningPanel : public ServerPanel {
public:
    FineTuningPanel() : ServerPanel("Fine-tuning") {}
    void Render() override;
    void Update() override;

    // Set model for fine-tuning (called from model browser)
    void SetModelForFineTune(const std::string& model_path);

    // Get current config
    const FineTuningConfig& GetConfig() const { return config_; }

private:
    void RenderModelSection();
    void RenderFreezeSection();
    void RenderLRScheduleSection();
    void RenderEarlyStoppingSection();
    void RenderGradientClippingSection();
    void RenderTrainingSection();
    void RenderStartButton();
    void RenderLRPreviewPlot();

    // Load model and extract layer info
    void LoadModelInfo(const std::string& model_path);
    void RefreshModelList();

    // LR preview computation
    std::vector<float> ComputeLRSchedule(int total_epochs) const;

    // Validation
    bool ValidateConfig() const;
    std::string GetValidationError() const;

    // Start fine-tuning job
    void StartFineTuning();

    // Configuration state
    FineTuningConfig config_;

    // Model info
    std::vector<ipc::ModelInfo> available_models_;
    std::vector<LayerInfo> model_layers_;
    int selected_model_idx_ = -1;
    bool models_loaded_ = false;
    bool model_info_loaded_ = false;
    std::string model_load_error_;

    // Training state
    bool is_training_ = false;
    std::string training_job_id_;
    std::string training_error_;
    float training_progress_ = 0.0f;

    // LR preview data
    std::vector<float> lr_preview_data_;
    bool lr_preview_dirty_ = true;

    // UI state
    char dataset_path_input_[512] = "";
    int lr_schedule_combo_ = 0;
    int freeze_mode_combo_ = 0;
    int monitor_metric_combo_ = 0;  // 0=val_loss, 1=val_accuracy
};

} // namespace cyxwiz::servernode::gui
