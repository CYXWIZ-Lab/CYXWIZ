#include "optimizer_settings_panel.h"
#include "../icons.h"
#include <imgui.h>
#include <spdlog/spdlog.h>

namespace cyxwiz {

OptimizerSettingsPanel::OptimizerSettingsPanel()
    : Panel(ICON_FA_SLIDERS " Optimizer Settings", false) {
    spdlog::info("OptimizerSettingsPanel initialized");
}

void OptimizerSettingsPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(400, 550), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(name_.c_str(), &visible_)) {
        RenderPresetButtons();
        ImGui::Separator();

        RenderOptimizerSelection();
        ImGui::Separator();

        RenderLearningRateSection();
        ImGui::Separator();

        RenderOptimizerSpecificSettings();
        ImGui::Separator();

        RenderSchedulerSection();
        ImGui::Separator();

        RenderAdvancedSection();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Apply button
        if (ImGui::Button(ICON_FA_CHECK " Apply Settings", ImVec2(-1, 30))) {
            if (on_settings_changed_) {
                on_settings_changed_();
            }
            spdlog::info("Optimizer settings applied: {} with lr={}",
                static_cast<int>(optimizer_type_), learning_rate_);
        }
    }
    ImGui::End();
}

void OptimizerSettingsPanel::RenderPresetButtons() {
    ImGui::Text(ICON_FA_BOOKMARK " Quick Presets:");
    ImGui::SameLine();

    if (ImGui::SmallButton("Vision")) {
        optimizer_type_ = OptimizerTypeUI::SGD;
        learning_rate_ = 0.01f;
        momentum_ = 0.9f;
        weight_decay_ = 0.0001f;
        nesterov_ = true;
        scheduler_type_ = LRSchedulerType::StepLR;
        step_size_ = 30;
        gamma_ = 0.1f;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("SGD with momentum for image classification");

    ImGui::SameLine();
    if (ImGui::SmallButton("NLP")) {
        optimizer_type_ = OptimizerTypeUI::AdamW;
        learning_rate_ = 2e-5f;
        beta1_ = 0.9f;
        beta2_ = 0.999f;
        weight_decay_ = 0.01f;
        scheduler_type_ = LRSchedulerType::WarmupCosine;
        warmup_steps_ = 1000;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("AdamW with warmup for transformers");

    ImGui::SameLine();
    if (ImGui::SmallButton("GAN")) {
        optimizer_type_ = OptimizerTypeUI::Adam;
        learning_rate_ = 0.0002f;
        beta1_ = 0.5f;
        beta2_ = 0.999f;
        weight_decay_ = 0.0f;
        scheduler_type_ = LRSchedulerType::None;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Adam for GANs (DCGAN defaults)");

    ImGui::SameLine();
    if (ImGui::SmallButton("Default")) {
        optimizer_type_ = OptimizerTypeUI::Adam;
        learning_rate_ = 0.001f;
        beta1_ = 0.9f;
        beta2_ = 0.999f;
        epsilon_ = 1e-8f;
        weight_decay_ = 0.0f;
        scheduler_type_ = LRSchedulerType::None;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reset to default Adam settings");
}

void OptimizerSettingsPanel::RenderOptimizerSelection() {
    ImGui::Text(ICON_FA_COG " Optimizer Type");

    const char* optimizer_names[] = { "SGD", "Adam", "AdamW", "RMSprop", "NAdam" };
    int current = static_cast<int>(optimizer_type_);

    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##OptimizerType", &current, optimizer_names, IM_ARRAYSIZE(optimizer_names))) {
        optimizer_type_ = static_cast<OptimizerTypeUI>(current);
    }

    // Description
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
    switch (optimizer_type_) {
        case OptimizerTypeUI::SGD:
            ImGui::TextWrapped("Stochastic Gradient Descent with optional momentum. Good for vision tasks.");
            break;
        case OptimizerTypeUI::Adam:
            ImGui::TextWrapped("Adaptive Moment Estimation. Great default choice for most tasks.");
            break;
        case OptimizerTypeUI::AdamW:
            ImGui::TextWrapped("Adam with decoupled weight decay. Best for transformers and NLP.");
            break;
        case OptimizerTypeUI::RMSprop:
            ImGui::TextWrapped("Root Mean Square Propagation. Good for RNNs and non-stationary objectives.");
            break;
        case OptimizerTypeUI::NAdam:
            ImGui::TextWrapped("Nesterov-accelerated Adam. Combines Adam with Nesterov momentum.");
            break;
    }
    ImGui::PopStyleColor();
}

void OptimizerSettingsPanel::RenderLearningRateSection() {
    ImGui::Text(ICON_FA_GAUGE_HIGH " Learning Rate");

    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("##LR", &learning_rate_, 0.0f, 0.0f, "%.6f");

    ImGui::SameLine();
    if (ImGui::SmallButton("1e-2")) learning_rate_ = 0.01f;
    ImGui::SameLine();
    if (ImGui::SmallButton("1e-3")) learning_rate_ = 0.001f;
    ImGui::SameLine();
    if (ImGui::SmallButton("1e-4")) learning_rate_ = 0.0001f;
    ImGui::SameLine();
    if (ImGui::SmallButton("1e-5")) learning_rate_ = 0.00001f;

    // Slider for fine-tuning
    ImGui::SetNextItemWidth(-1);
    ImGui::SliderFloat("##LRSlider", &learning_rate_, 1e-7f, 1.0f, "%.7f", ImGuiSliderFlags_Logarithmic);
}

void OptimizerSettingsPanel::RenderOptimizerSpecificSettings() {
    if (ImGui::CollapsingHeader(ICON_FA_WRENCH " Optimizer Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
        switch (optimizer_type_) {
            case OptimizerTypeUI::SGD:
                RenderSGDSettings();
                break;
            case OptimizerTypeUI::Adam:
                RenderAdamSettings();
                break;
            case OptimizerTypeUI::AdamW:
                RenderAdamWSettings();
                break;
            case OptimizerTypeUI::RMSprop:
                RenderRMSpropSettings();
                break;
            case OptimizerTypeUI::NAdam:
                RenderNAdamSettings();
                break;
        }
    }
}

void OptimizerSettingsPanel::RenderSGDSettings() {
    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("Momentum", &momentum_, 0.01f, 0.1f, "%.3f");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Momentum factor (0.9 typical)");

    ImGui::Checkbox("Nesterov", &nesterov_);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Use Nesterov momentum (recommended)");

    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("Weight Decay", &weight_decay_, 0.0001f, 0.001f, "%.6f");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("L2 regularization (1e-4 typical)");
}

void OptimizerSettingsPanel::RenderAdamSettings() {
    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("Beta1", &beta1_, 0.01f, 0.1f, "%.3f");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Exponential decay rate for first moment (0.9 default)");

    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("Beta2", &beta2_, 0.001f, 0.01f, "%.4f");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Exponential decay rate for second moment (0.999 default)");

    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("Epsilon", &epsilon_, 0.0f, 0.0f, "%.2e");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Numerical stability term (1e-8 default)");

    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("Weight Decay", &weight_decay_, 0.0001f, 0.001f, "%.6f");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("L2 regularization");
}

void OptimizerSettingsPanel::RenderAdamWSettings() {
    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("Beta1", &beta1_, 0.01f, 0.1f, "%.3f");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Exponential decay rate for first moment (0.9 default)");

    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("Beta2", &beta2_, 0.001f, 0.01f, "%.4f");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Exponential decay rate for second moment (0.999 default)");

    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("Epsilon", &epsilon_, 0.0f, 0.0f, "%.2e");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Numerical stability term (1e-8 default)");

    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("Weight Decay", &weight_decay_, 0.001f, 0.01f, "%.4f");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Decoupled weight decay (0.01 typical for transformers)");

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
    ImGui::TextWrapped(ICON_FA_CIRCLE_INFO " AdamW uses decoupled weight decay which is more effective than L2 regularization");
    ImGui::PopStyleColor();
}

void OptimizerSettingsPanel::RenderRMSpropSettings() {
    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("Alpha", &alpha_, 0.01f, 0.1f, "%.3f");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Smoothing constant (0.99 default)");

    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("Momentum", &rmsprop_momentum_, 0.01f, 0.1f, "%.3f");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Momentum factor (0 default)");

    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("Epsilon", &epsilon_, 0.0f, 0.0f, "%.2e");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Numerical stability term (1e-8 default)");

    ImGui::Checkbox("Centered", &centered_);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Normalize gradient by variance estimate");

    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("Weight Decay", &weight_decay_, 0.0001f, 0.001f, "%.6f");
}

void OptimizerSettingsPanel::RenderNAdamSettings() {
    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("Beta1", &beta1_, 0.01f, 0.1f, "%.3f");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Exponential decay rate for first moment (0.9 default)");

    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("Beta2", &beta2_, 0.001f, 0.01f, "%.4f");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Exponential decay rate for second moment (0.999 default)");

    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("Epsilon", &epsilon_, 0.0f, 0.0f, "%.2e");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Numerical stability term (1e-8 default)");

    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("Weight Decay", &weight_decay_, 0.0001f, 0.001f, "%.6f");

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
    ImGui::TextWrapped(ICON_FA_CIRCLE_INFO " NAdam incorporates Nesterov momentum into Adam for potentially faster convergence");
    ImGui::PopStyleColor();
}

void OptimizerSettingsPanel::RenderSchedulerSection() {
    if (ImGui::CollapsingHeader(ICON_FA_CHART_LINE " Learning Rate Scheduler")) {
        const char* scheduler_names[] = {
            "None", "StepLR", "ExponentialLR", "CosineAnnealing",
            "ReduceOnPlateau", "Warmup + Cosine"
        };
        int current = static_cast<int>(scheduler_type_);

        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("##Scheduler", &current, scheduler_names, IM_ARRAYSIZE(scheduler_names))) {
            scheduler_type_ = static_cast<LRSchedulerType>(current);
        }

        ImGui::Spacing();

        switch (scheduler_type_) {
            case LRSchedulerType::StepLR:
                ImGui::SetNextItemWidth(150);
                ImGui::InputInt("Step Size", &step_size_);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Decay LR every N epochs");
                ImGui::SetNextItemWidth(150);
                ImGui::InputFloat("Gamma", &gamma_, 0.01f, 0.1f, "%.3f");
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Multiplicative factor (0.1 = 10x reduction)");
                break;

            case LRSchedulerType::ExponentialLR:
                ImGui::SetNextItemWidth(150);
                ImGui::InputFloat("Gamma", &gamma_, 0.001f, 0.01f, "%.4f");
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Exponential decay factor per epoch");
                break;

            case LRSchedulerType::CosineAnnealing:
                ImGui::SetNextItemWidth(150);
                ImGui::InputInt("T_max", &t_max_);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Maximum number of iterations");
                ImGui::SetNextItemWidth(150);
                ImGui::InputFloat("Eta Min", &eta_min_, 0.0f, 0.0f, "%.6f");
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Minimum learning rate");
                break;

            case LRSchedulerType::ReduceOnPlateau:
                ImGui::SetNextItemWidth(150);
                ImGui::InputInt("Patience", &patience_);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Number of epochs with no improvement before reducing LR");
                ImGui::SetNextItemWidth(150);
                ImGui::InputFloat("Factor", &factor_, 0.01f, 0.1f, "%.3f");
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Factor to reduce LR by");
                break;

            case LRSchedulerType::WarmupCosine:
                ImGui::SetNextItemWidth(150);
                ImGui::InputInt("Warmup Steps", &warmup_steps_);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Number of warmup steps");
                ImGui::SetNextItemWidth(150);
                ImGui::InputInt("T_max", &t_max_);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Total training steps");
                break;

            default:
                ImGui::TextDisabled("No scheduler - constant learning rate");
                break;
        }
    }
}

void OptimizerSettingsPanel::RenderAdvancedSection() {
    if (ImGui::CollapsingHeader(ICON_FA_FLASK " Advanced")) {
        ImGui::SetNextItemWidth(150);
        ImGui::InputFloat("Gradient Clip Norm", &gradient_clip_norm_, 0.1f, 1.0f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Max gradient norm (0 = disabled). Use 1.0-5.0 for transformers.");

        ImGui::SetNextItemWidth(150);
        ImGui::InputInt("Gradient Accumulation", &gradient_accumulation_steps_);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Accumulate gradients over N steps (effective batch = batch_size * N)");
        gradient_accumulation_steps_ =
            training_contract::ClampGradientAccumulationSteps(
                gradient_accumulation_steps_);

        ImGui::Checkbox("Mixed Precision (FP16)", &use_mixed_precision_);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Use automatic mixed precision for faster training (requires GPU)");
    }
}

} // namespace cyxwiz
