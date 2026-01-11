#pragma once

#include "../panel.h"
#include <functional>
#include <string>

namespace cyxwiz {

enum class OptimizerTypeUI {
    SGD = 0,
    Adam,
    AdamW,
    RMSprop,
    NAdam
};

enum class LRSchedulerType {
    None = 0,
    StepLR,
    ExponentialLR,
    CosineAnnealing,
    ReduceOnPlateau,
    WarmupCosine
};

/**
 * Optimizer Settings Panel
 * Configure optimizer type, learning rate, and advanced hyperparameters
 */
class OptimizerSettingsPanel : public Panel {
public:
    OptimizerSettingsPanel();
    ~OptimizerSettingsPanel() override = default;

    void Render() override;

    // Getters for current settings
    OptimizerTypeUI GetOptimizerType() const { return optimizer_type_; }
    float GetLearningRate() const { return learning_rate_; }
    float GetWeightDecay() const { return weight_decay_; }
    float GetMomentum() const { return momentum_; }
    float GetBeta1() const { return beta1_; }
    float GetBeta2() const { return beta2_; }
    float GetEpsilon() const { return epsilon_; }
    float GetAlpha() const { return alpha_; }
    bool GetNesterov() const { return nesterov_; }
    float GetGradientClipNorm() const { return gradient_clip_norm_; }
    int GetGradientAccumulationSteps() const { return gradient_accumulation_steps_; }
    LRSchedulerType GetSchedulerType() const { return scheduler_type_; }

    // Callback when settings are applied
    using SettingsChangedCallback = std::function<void()>;
    void SetOnSettingsChanged(SettingsChangedCallback callback) { on_settings_changed_ = callback; }

private:
    void RenderOptimizerSelection();
    void RenderLearningRateSection();
    void RenderOptimizerSpecificSettings();
    void RenderSGDSettings();
    void RenderAdamSettings();
    void RenderAdamWSettings();
    void RenderRMSpropSettings();
    void RenderNAdamSettings();
    void RenderSchedulerSection();
    void RenderAdvancedSection();
    void RenderPresetButtons();

    // Optimizer type
    OptimizerTypeUI optimizer_type_ = OptimizerTypeUI::Adam;

    // Common settings
    float learning_rate_ = 0.001f;
    float weight_decay_ = 0.0f;

    // SGD specific
    float momentum_ = 0.9f;
    bool nesterov_ = false;

    // Adam/AdamW/NAdam specific
    float beta1_ = 0.9f;
    float beta2_ = 0.999f;
    float epsilon_ = 1e-8f;

    // RMSprop specific
    float alpha_ = 0.99f;
    float rmsprop_momentum_ = 0.0f;
    bool centered_ = false;

    // Learning rate scheduler
    LRSchedulerType scheduler_type_ = LRSchedulerType::None;
    int step_size_ = 10;           // StepLR
    float gamma_ = 0.1f;           // StepLR, ExponentialLR
    int t_max_ = 100;              // CosineAnnealing
    float eta_min_ = 0.0f;         // CosineAnnealing
    int patience_ = 10;            // ReduceOnPlateau
    float factor_ = 0.1f;          // ReduceOnPlateau
    int warmup_steps_ = 1000;      // WarmupCosine

    // Advanced settings
    float gradient_clip_norm_ = 0.0f;  // 0 = disabled
    int gradient_accumulation_steps_ = 1;
    bool use_mixed_precision_ = false;

    // Callback
    SettingsChangedCallback on_settings_changed_;
};

} // namespace cyxwiz
