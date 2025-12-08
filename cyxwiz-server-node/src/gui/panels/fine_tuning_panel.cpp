// fine_tuning_panel.cpp - Model fine-tuning configuration UI implementation
#include "gui/panels/fine_tuning_panel.h"
#include "gui/icons.h"
#include "core/backend_manager.h"
#include <imgui.h>
#include <implot.h>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cyxwiz::servernode::gui {

void FineTuningPanel::Render() {
    ImGui::PushFont(GetSafeFont(FONT_LARGE));
    ImGui::Text("%s Fine-tuning", ICON_FA_BRAIN);
    ImGui::PopFont();
    ImGui::Separator();

    // Connection status
    if (IsDaemonConnected()) {
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s Daemon Connected", ICON_FA_LINK);
    } else {
        ImGui::TextColored(ImVec4(0.8f, 0.5f, 0.3f, 1.0f), "%s Local Mode", ICON_FA_FOLDER);
    }
    ImGui::Spacing();

    // Training in progress indicator
    if (is_training_) {
        ImGui::TextColored(ImVec4(0.3f, 0.7f, 1.0f, 1.0f), "%s Training in progress...", ICON_FA_SPINNER);
        ImGui::ProgressBar(training_progress_, ImVec2(-1, 0), "");
        ImGui::Spacing();

        if (ImGui::Button(ICON_FA_STOP " Stop Training")) {
            // Cancel training via daemon
            auto* client = GetDaemonClient();
            if (client && client->IsConnected() && !training_job_id_.empty()) {
                std::string error;
                if (client->CancelJob(training_job_id_, error)) {
                    is_training_ = false;
                    spdlog::info("Fine-tuning job cancelled: {}", training_job_id_);
                }
            }
        }
        ImGui::Separator();
        ImGui::Spacing();
    }

    // Main content in tabs
    if (ImGui::BeginTabBar("FineTuningTabs")) {
        if (ImGui::BeginTabItem("Configuration")) {
            RenderModelSection();
            ImGui::Spacing();

            if (ImGui::CollapsingHeader("Layer Freezing", ImGuiTreeNodeFlags_DefaultOpen)) {
                RenderFreezeSection();
            }
            ImGui::Spacing();

            if (ImGui::CollapsingHeader("Learning Rate Schedule", ImGuiTreeNodeFlags_DefaultOpen)) {
                RenderLRScheduleSection();
            }
            ImGui::Spacing();

            if (ImGui::CollapsingHeader("Early Stopping")) {
                RenderEarlyStoppingSection();
            }
            ImGui::Spacing();

            if (ImGui::CollapsingHeader("Gradient Clipping")) {
                RenderGradientClippingSection();
            }
            ImGui::Spacing();

            if (ImGui::CollapsingHeader("Training Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
                RenderTrainingSection();
            }
            ImGui::Spacing();

            ImGui::Separator();
            RenderStartButton();

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("LR Preview")) {
            RenderLRPreviewPlot();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }
}

void FineTuningPanel::Update() {
    // Refresh model list periodically
    if (!models_loaded_) {
        RefreshModelList();
    }

    // Update training progress if job is running
    if (is_training_ && !training_job_id_.empty()) {
        auto* client = GetDaemonClient();
        if (client && client->IsConnected()) {
            ipc::JobInfo job_info;
            if (client->GetJob(training_job_id_, job_info)) {
                training_progress_ = job_info.progress;
                // Status: 4=completed, 5=failed, 6=cancelled
                if (job_info.status == 4 || job_info.status == 5 || job_info.status == 6) {
                    is_training_ = false;
                    if (job_info.status == 5) {
                        training_error_ = "Job failed";
                    }
                }
            }
        }
    }
}

void FineTuningPanel::SetModelForFineTune(const std::string& model_path) {
    config_.model_path = model_path;
    LoadModelInfo(model_path);
}

void FineTuningPanel::RenderModelSection() {
    ImGui::Text("%s Model & Dataset", ICON_FA_CUBE);
    ImGui::Spacing();

    // Model selection
    ImGui::Text("Base Model:");
    ImGui::SetNextItemWidth(400);

    // Build combo preview
    std::string preview = config_.model_path.empty() ?
        "Select a model..." :
        std::filesystem::path(config_.model_path).filename().string();

    if (ImGui::BeginCombo("##ModelCombo", preview.c_str())) {
        for (size_t i = 0; i < available_models_.size(); ++i) {
            bool is_selected = (selected_model_idx_ == static_cast<int>(i));
            if (ImGui::Selectable(available_models_[i].name.c_str(), is_selected)) {
                selected_model_idx_ = static_cast<int>(i);
                config_.model_path = available_models_[i].path;
                LoadModelInfo(config_.model_path);
            }
            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ROTATE " Refresh")) {
        RefreshModelList();
    }

    // Show model info
    if (model_info_loaded_ && !model_layers_.empty()) {
        ImGui::TextDisabled("%zu layers, %s parameters",
            model_layers_.size(),
            [&]() {
                int64_t total = 0;
                for (const auto& layer : model_layers_) {
                    total += layer.param_count;
                }
                if (total > 1e9) return std::string(std::to_string(total / 1000000000) + "B");
                if (total > 1e6) return std::string(std::to_string(total / 1000000) + "M");
                if (total > 1e3) return std::string(std::to_string(total / 1000) + "K");
                return std::to_string(total);
            }().c_str());
    }

    if (!model_load_error_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s %s",
            ICON_FA_TRIANGLE_EXCLAMATION, model_load_error_.c_str());
    }

    ImGui::Spacing();

    // Dataset selection
    ImGui::Text("Training Dataset:");
    ImGui::SetNextItemWidth(400);
    if (ImGui::InputTextWithHint("##DatasetPath", "Path to training data...",
                                  dataset_path_input_, sizeof(dataset_path_input_))) {
        config_.dataset_path = dataset_path_input_;
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_FOLDER_OPEN "##BrowseDataset")) {
        // TODO: File browser dialog
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Browse for dataset");
    }
}

void FineTuningPanel::RenderFreezeSection() {
    ImGui::Text("Control which layers are trainable during fine-tuning.");
    ImGui::Spacing();

    // Freeze mode combo
    const char* freeze_modes[] = {
        "Train All Layers",
        "Freeze All Except Last N",
        "Freeze Up To Layer",
        "Custom Selection"
    };

    ImGui::SetNextItemWidth(200);
    if (ImGui::Combo("Freeze Mode", &freeze_mode_combo_, freeze_modes, IM_ARRAYSIZE(freeze_modes))) {
        config_.freeze_mode = static_cast<FreezeMode>(freeze_mode_combo_);
        lr_preview_dirty_ = true;
    }

    ImGui::Spacing();

    switch (config_.freeze_mode) {
        case FreezeMode::None:
            ImGui::TextDisabled("All layers will be trained.");
            break;

        case FreezeMode::ExceptLastN:
            ImGui::SetNextItemWidth(100);
            if (ImGui::InputInt("Unfreeze Last N Layers", &config_.unfreeze_last_n)) {
                config_.unfreeze_last_n = std::max(1, config_.unfreeze_last_n);
            }
            ImGui::TextDisabled("Recommended: 1-3 layers for classification head changes.");
            break;

        case FreezeMode::UpToLayer:
            ImGui::SetNextItemWidth(100);
            if (ImGui::InputInt("Freeze Up To Layer", &config_.freeze_up_to_layer)) {
                config_.freeze_up_to_layer = std::max(0, config_.freeze_up_to_layer);
            }
            ImGui::TextDisabled("Freeze all layers up to and including this index.");
            break;

        case FreezeMode::Custom:
            // Custom per-layer selection
            if (model_info_loaded_ && !model_layers_.empty()) {
                ImGui::Text("Layer Selection:");

                if (config_.layer_trainable.size() != model_layers_.size()) {
                    config_.layer_trainable.resize(model_layers_.size(), 1);  // 1 = trainable
                }

                // Quick actions
                if (ImGui::Button("Select All")) {
                    std::fill(config_.layer_trainable.begin(), config_.layer_trainable.end(), static_cast<char>(1));
                }
                ImGui::SameLine();
                if (ImGui::Button("Deselect All")) {
                    std::fill(config_.layer_trainable.begin(), config_.layer_trainable.end(), static_cast<char>(0));
                }
                ImGui::SameLine();
                if (ImGui::Button("Invert")) {
                    for (size_t i = 0; i < config_.layer_trainable.size(); ++i) {
                        config_.layer_trainable[i] = config_.layer_trainable[i] ? 0 : 1;
                    }
                }

                ImGui::Spacing();

                // Layer list with checkboxes
                if (ImGui::BeginChild("LayerList", ImVec2(0, 200), true)) {
                    for (size_t i = 0; i < model_layers_.size(); ++i) {
                        const auto& layer = model_layers_[i];
                        if (!layer.has_weights) continue;

                        ImGui::PushID(static_cast<int>(i));
                        bool trainable = config_.layer_trainable[i] != 0;
                        if (ImGui::Checkbox("##trainable", &trainable)) {
                            config_.layer_trainable[i] = trainable ? 1 : 0;
                        }
                        ImGui::SameLine();
                        ImGui::Text("%s [%s] - %s params",
                            layer.name.c_str(),
                            layer.type.c_str(),
                            [](int64_t n) {
                                if (n > 1e6) return std::to_string(n / 1000000) + "M";
                                if (n > 1e3) return std::to_string(n / 1000) + "K";
                                return std::to_string(n);
                            }(layer.param_count).c_str());
                        ImGui::PopID();
                    }
                    ImGui::EndChild();
                }

                // Summary
                int trainable_count = 0;
                int64_t trainable_params = 0;
                for (size_t i = 0; i < model_layers_.size(); ++i) {
                    if (config_.layer_trainable[i] != 0 && model_layers_[i].has_weights) {
                        trainable_count++;
                        trainable_params += model_layers_[i].param_count;
                    }
                }
                ImGui::TextDisabled("Training %d layers, %s parameters",
                    trainable_count,
                    [](int64_t n) {
                        if (n > 1e6) return std::to_string(n / 1000000) + "M";
                        if (n > 1e3) return std::to_string(n / 1000) + "K";
                        return std::to_string(n);
                    }(trainable_params).c_str());
            } else {
                ImGui::TextDisabled("Load a model to configure layer selection.");
            }
            break;
    }
}

void FineTuningPanel::RenderLRScheduleSection() {
    // Schedule type
    const char* schedule_types[] = {
        "Constant",
        "Step Decay",
        "Exponential Decay",
        "Cosine Annealing",
        "One Cycle",
        "Cosine Warm Restart"
    };

    ImGui::SetNextItemWidth(200);
    if (ImGui::Combo("Schedule Type", &lr_schedule_combo_, schedule_types, IM_ARRAYSIZE(schedule_types))) {
        config_.lr_schedule = static_cast<LRScheduleType>(lr_schedule_combo_);
        lr_preview_dirty_ = true;
    }

    ImGui::Spacing();

    // Common: Initial LR
    ImGui::SetNextItemWidth(150);
    if (ImGui::InputFloat("Initial Learning Rate", &config_.initial_lr, 0, 0, "%.2e")) {
        config_.initial_lr = std::max(1e-8f, std::min(1.0f, config_.initial_lr));
        lr_preview_dirty_ = true;
    }

    // Schedule-specific parameters
    switch (config_.lr_schedule) {
        case LRScheduleType::Constant:
            ImGui::TextDisabled("Learning rate remains constant throughout training.");
            break;

        case LRScheduleType::Step:
            ImGui::SetNextItemWidth(100);
            if (ImGui::InputInt("Step Size (epochs)", &config_.step_size)) {
                config_.step_size = std::max(1, config_.step_size);
                lr_preview_dirty_ = true;
            }
            ImGui::SetNextItemWidth(150);
            if (ImGui::InputFloat("Gamma (decay factor)", &config_.step_gamma, 0, 0, "%.3f")) {
                config_.step_gamma = std::max(0.01f, std::min(1.0f, config_.step_gamma));
                lr_preview_dirty_ = true;
            }
            ImGui::TextDisabled("LR multiplied by gamma every step_size epochs.");
            break;

        case LRScheduleType::Exponential:
            ImGui::SetNextItemWidth(150);
            if (ImGui::InputFloat("Gamma (per epoch)", &config_.exp_gamma, 0, 0, "%.4f")) {
                config_.exp_gamma = std::max(0.01f, std::min(1.0f, config_.exp_gamma));
                lr_preview_dirty_ = true;
            }
            ImGui::TextDisabled("LR = initial_lr * gamma^epoch");
            break;

        case LRScheduleType::Cosine:
            ImGui::SetNextItemWidth(100);
            if (ImGui::InputInt("T_max (period)", &config_.t_max)) {
                config_.t_max = std::max(1, config_.t_max);
                lr_preview_dirty_ = true;
            }
            ImGui::SetNextItemWidth(150);
            if (ImGui::InputFloat("Min LR (eta_min)", &config_.eta_min, 0, 0, "%.2e")) {
                config_.eta_min = std::max(0.0f, config_.eta_min);
                lr_preview_dirty_ = true;
            }
            ImGui::TextDisabled("Cosine annealing from initial_lr to eta_min.");
            break;

        case LRScheduleType::OneCycle:
            ImGui::SetNextItemWidth(150);
            if (ImGui::InputFloat("Max Learning Rate", &config_.max_lr, 0, 0, "%.2e")) {
                config_.max_lr = std::max(config_.initial_lr, config_.max_lr);
                lr_preview_dirty_ = true;
            }
            ImGui::SetNextItemWidth(150);
            if (ImGui::InputFloat("Final Learning Rate", &config_.final_lr, 0, 0, "%.2e")) {
                config_.final_lr = std::max(1e-10f, config_.final_lr);
                lr_preview_dirty_ = true;
            }
            ImGui::TextDisabled("Ramps up to max_lr, then anneals down to final_lr.");
            break;

        case LRScheduleType::CosineWarmRestart:
            ImGui::SetNextItemWidth(100);
            if (ImGui::InputInt("T_0 (initial period)", &config_.t_max)) {
                config_.t_max = std::max(1, config_.t_max);
                lr_preview_dirty_ = true;
            }
            ImGui::SetNextItemWidth(150);
            if (ImGui::InputFloat("Min LR (eta_min)", &config_.eta_min, 0, 0, "%.2e")) {
                config_.eta_min = std::max(0.0f, config_.eta_min);
                lr_preview_dirty_ = true;
            }
            ImGui::TextDisabled("Cosine annealing with warm restarts.");
            break;
    }

    // Warmup epochs (common for most schedules)
    if (config_.lr_schedule != LRScheduleType::OneCycle) {
        ImGui::Spacing();
        ImGui::SetNextItemWidth(100);
        if (ImGui::InputInt("Warmup Epochs", &config_.warmup_epochs)) {
            config_.warmup_epochs = std::max(0, config_.warmup_epochs);
            lr_preview_dirty_ = true;
        }
        ImGui::TextDisabled("Linearly warm up LR for this many epochs.");
    }
}

void FineTuningPanel::RenderEarlyStoppingSection() {
    ImGui::Checkbox("Enable Early Stopping", &config_.enable_early_stopping);

    if (config_.enable_early_stopping) {
        ImGui::Spacing();

        // Monitor metric
        const char* metrics[] = { "Validation Loss", "Validation Accuracy" };
        ImGui::SetNextItemWidth(200);
        if (ImGui::Combo("Monitor", &monitor_metric_combo_, metrics, IM_ARRAYSIZE(metrics))) {
            config_.monitor_metric = (monitor_metric_combo_ == 0) ? "val_loss" : "val_accuracy";
        }

        ImGui::SetNextItemWidth(100);
        ImGui::InputInt("Patience (epochs)", &config_.patience);
        config_.patience = std::max(1, config_.patience);

        ImGui::SetNextItemWidth(150);
        ImGui::InputFloat("Min Delta", &config_.min_delta, 0, 0, "%.2e");
        config_.min_delta = std::max(0.0f, config_.min_delta);

        ImGui::Checkbox("Restore Best Weights", &config_.restore_best_weights);

        ImGui::TextDisabled("Stop training if metric doesn't improve for %d epochs.", config_.patience);
    }
}

void FineTuningPanel::RenderGradientClippingSection() {
    ImGui::Checkbox("Enable Gradient Clipping", &config_.enable_grad_clipping);

    if (config_.enable_grad_clipping) {
        ImGui::Spacing();
        ImGui::SetNextItemWidth(150);
        ImGui::InputFloat("Max Gradient Norm", &config_.max_grad_norm, 0, 0, "%.2f");
        config_.max_grad_norm = std::max(0.01f, config_.max_grad_norm);

        ImGui::TextDisabled("Clip gradients with norm exceeding this value.");
        ImGui::TextDisabled("Recommended: 1.0 for transformers, 5.0 for CNNs.");
    }
}

void FineTuningPanel::RenderTrainingSection() {
    // Epochs
    ImGui::SetNextItemWidth(100);
    if (ImGui::InputInt("Epochs", &config_.epochs)) {
        config_.epochs = std::max(1, config_.epochs);
        lr_preview_dirty_ = true;
    }

    // Batch size
    ImGui::SetNextItemWidth(100);
    ImGui::InputInt("Batch Size", &config_.batch_size);
    config_.batch_size = std::max(1, config_.batch_size);

    // Weight decay
    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("Weight Decay", &config_.weight_decay, 0, 0, "%.4f");
    config_.weight_decay = std::max(0.0f, config_.weight_decay);

    // Mixed precision
    ImGui::Checkbox("Mixed Precision (FP16)", &config_.mixed_precision);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Use FP16 for faster training (requires GPU support)");
    }

    // Device selection
    ImGui::Spacing();
    ImGui::SetNextItemWidth(150);
    if (ImGui::InputInt("Device ID", &config_.device_id)) {
        // -1 means auto selection from pool
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(-1 = auto from pool)");
}

void FineTuningPanel::RenderStartButton() {
    // Validation
    std::string error = GetValidationError();
    bool valid = error.empty();

    if (!valid) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s %s",
            ICON_FA_TRIANGLE_EXCLAMATION, error.c_str());
        ImGui::Spacing();
    }

    // Training error display
    if (!training_error_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s %s",
            ICON_FA_TRIANGLE_EXCLAMATION, training_error_.c_str());
        ImGui::Spacing();
    }

    // Start button
    ImGui::BeginDisabled(!valid || is_training_);
    if (ImGui::Button(ICON_FA_PLAY " Start Fine-tuning", ImVec2(200, 40))) {
        StartFineTuning();
    }
    ImGui::EndDisabled();

    // Config summary
    ImGui::SameLine();
    ImGui::BeginGroup();
    ImGui::TextDisabled("Schedule: %s",
        [](LRScheduleType t) {
            switch (t) {
                case LRScheduleType::Constant: return "Constant";
                case LRScheduleType::Step: return "Step";
                case LRScheduleType::Exponential: return "Exponential";
                case LRScheduleType::Cosine: return "Cosine";
                case LRScheduleType::OneCycle: return "OneCycle";
                case LRScheduleType::CosineWarmRestart: return "CosineWarmRestart";
            }
            return "Unknown";
        }(config_.lr_schedule));
    ImGui::TextDisabled("Early Stop: %s", config_.enable_early_stopping ? "Yes" : "No");
    ImGui::EndGroup();
}

void FineTuningPanel::RenderLRPreviewPlot() {
    ImGui::Text("Learning Rate Schedule Preview");
    ImGui::Spacing();

    // Recompute if dirty
    if (lr_preview_dirty_) {
        lr_preview_data_ = ComputeLRSchedule(config_.epochs);
        lr_preview_dirty_ = false;
    }

    if (lr_preview_data_.empty()) {
        ImGui::TextDisabled("Configure epochs to preview schedule.");
        return;
    }

    // Plot
    if (ImPlot::BeginPlot("##LRPreview", ImVec2(-1, 300))) {
        ImPlot::SetupAxes("Epoch", "Learning Rate");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, static_cast<double>(config_.epochs));

        // Find Y range
        float min_lr = *std::min_element(lr_preview_data_.begin(), lr_preview_data_.end());
        float max_lr = *std::max_element(lr_preview_data_.begin(), lr_preview_data_.end());
        float range = max_lr - min_lr;
        ImPlot::SetupAxisLimits(ImAxis_Y1, min_lr - range * 0.1, max_lr + range * 0.1);

        // Create x values
        std::vector<float> x_data(lr_preview_data_.size());
        for (size_t i = 0; i < x_data.size(); ++i) {
            x_data[i] = static_cast<float>(i);
        }

        ImPlot::PlotLine("LR", x_data.data(), lr_preview_data_.data(),
                         static_cast<int>(lr_preview_data_.size()));

        // Warmup marker
        if (config_.warmup_epochs > 0 && config_.lr_schedule != LRScheduleType::OneCycle) {
            double warmup_x = static_cast<double>(config_.warmup_epochs);
            ImPlot::PlotInfLines("Warmup", &warmup_x, 1);
        }

        ImPlot::EndPlot();
    }

    // Info
    ImGui::TextDisabled("Initial LR: %.2e, Final LR: %.2e",
        lr_preview_data_.front(),
        lr_preview_data_.back());
}

void FineTuningPanel::LoadModelInfo(const std::string& model_path) {
    model_layers_.clear();
    model_info_loaded_ = false;
    model_load_error_.clear();

    if (model_path.empty()) return;

    // Try loading via daemon
    auto* client = GetDaemonClient();
    if (client && client->IsConnected()) {
        // TODO: Add GetModelLayers RPC to daemon
        // For now, use placeholder data
    }

    // Placeholder: simulate layer info
    // In a real implementation, this would parse the model file
    model_layers_ = {
        {"conv1", "Conv2d", 9408, true, true},
        {"bn1", "BatchNorm2d", 128, true, true},
        {"layer1.0.conv1", "Conv2d", 36864, true, true},
        {"layer1.0.bn1", "BatchNorm2d", 128, true, true},
        {"layer1.0.conv2", "Conv2d", 36864, true, true},
        {"layer1.0.bn2", "BatchNorm2d", 128, true, true},
        {"layer2.0.conv1", "Conv2d", 73728, true, true},
        {"layer2.0.bn1", "BatchNorm2d", 256, true, true},
        {"layer2.0.conv2", "Conv2d", 147456, true, true},
        {"layer2.0.bn2", "BatchNorm2d", 256, true, true},
        {"fc", "Linear", 512000, true, true},
    };

    config_.layer_trainable.resize(model_layers_.size(), 1);  // 1 = trainable
    model_info_loaded_ = true;

    spdlog::info("Loaded model info for fine-tuning: {}", model_path);
}

void FineTuningPanel::RefreshModelList() {
    available_models_.clear();
    models_loaded_ = true;

    auto* client = GetDaemonClient();
    if (client && client->IsConnected()) {
        if (client->ListLocalModels(available_models_)) {
            spdlog::debug("Loaded {} models for fine-tuning", available_models_.size());
            return;
        }
    }

    // Fallback: local scan
    auto& backend_config = GetBackend().GetConfig();
    std::string models_dir = backend_config.models_directory;

    if (!std::filesystem::exists(models_dir)) return;

    for (const auto& entry : std::filesystem::directory_iterator(models_dir)) {
        if (!entry.is_regular_file()) continue;

        std::string ext = entry.path().extension().string();
        if (ext != ".gguf" && ext != ".onnx" && ext != ".safetensors" &&
            ext != ".pt" && ext != ".pth") {
            continue;
        }

        ipc::ModelInfo info;
        info.name = entry.path().stem().string();
        info.path = entry.path().string();
        info.size_bytes = entry.file_size();
        available_models_.push_back(info);
    }
}

std::vector<float> FineTuningPanel::ComputeLRSchedule(int total_epochs) const {
    if (total_epochs <= 0) return {};

    std::vector<float> schedule(total_epochs);
    int warmup = config_.warmup_epochs;

    for (int epoch = 0; epoch < total_epochs; ++epoch) {
        float lr = 0.0f;

        // Warmup phase (except for OneCycle which has its own warmup)
        if (epoch < warmup && config_.lr_schedule != LRScheduleType::OneCycle) {
            lr = config_.initial_lr * (static_cast<float>(epoch + 1) / static_cast<float>(warmup));
        } else {
            int effective_epoch = epoch - warmup;
            int effective_total = total_epochs - warmup;

            switch (config_.lr_schedule) {
                case LRScheduleType::Constant:
                    lr = config_.initial_lr;
                    break;

                case LRScheduleType::Step: {
                    int steps = effective_epoch / config_.step_size;
                    lr = config_.initial_lr * std::pow(config_.step_gamma, static_cast<float>(steps));
                    break;
                }

                case LRScheduleType::Exponential:
                    lr = config_.initial_lr * std::pow(config_.exp_gamma, static_cast<float>(effective_epoch));
                    break;

                case LRScheduleType::Cosine: {
                    float progress = static_cast<float>(effective_epoch) / static_cast<float>(config_.t_max);
                    lr = config_.eta_min + (config_.initial_lr - config_.eta_min) *
                         (1.0f + std::cos(static_cast<float>(M_PI) * progress)) / 2.0f;
                    break;
                }

                case LRScheduleType::OneCycle: {
                    // Phase 1: warmup to max_lr (first 30%)
                    // Phase 2: anneal to initial_lr (next 40%)
                    // Phase 3: final anneal to final_lr (last 30%)
                    float progress = static_cast<float>(epoch) / static_cast<float>(total_epochs);
                    if (progress < 0.3f) {
                        // Warmup
                        float phase_progress = progress / 0.3f;
                        lr = config_.initial_lr + (config_.max_lr - config_.initial_lr) * phase_progress;
                    } else if (progress < 0.7f) {
                        // Anneal to initial
                        float phase_progress = (progress - 0.3f) / 0.4f;
                        lr = config_.max_lr - (config_.max_lr - config_.initial_lr) * phase_progress;
                    } else {
                        // Final anneal
                        float phase_progress = (progress - 0.7f) / 0.3f;
                        lr = config_.initial_lr - (config_.initial_lr - config_.final_lr) * phase_progress;
                    }
                    break;
                }

                case LRScheduleType::CosineWarmRestart: {
                    // SGDR: Stochastic Gradient Descent with Warm Restarts
                    int t_cur = effective_epoch % config_.t_max;
                    float progress = static_cast<float>(t_cur) / static_cast<float>(config_.t_max);
                    lr = config_.eta_min + (config_.initial_lr - config_.eta_min) *
                         (1.0f + std::cos(static_cast<float>(M_PI) * progress)) / 2.0f;
                    break;
                }
            }
        }

        schedule[epoch] = lr;
    }

    return schedule;
}

bool FineTuningPanel::ValidateConfig() const {
    return GetValidationError().empty();
}

std::string FineTuningPanel::GetValidationError() const {
    if (config_.model_path.empty()) {
        return "Select a model";
    }
    if (config_.dataset_path.empty()) {
        return "Specify training dataset";
    }
    if (config_.epochs <= 0) {
        return "Epochs must be positive";
    }
    if (config_.batch_size <= 0) {
        return "Batch size must be positive";
    }
    if (config_.initial_lr <= 0) {
        return "Learning rate must be positive";
    }
    return "";
}

void FineTuningPanel::StartFineTuning() {
    training_error_.clear();

    auto* client = GetDaemonClient();
    if (!client || !client->IsConnected()) {
        training_error_ = "Daemon not connected";
        return;
    }

    // Build fine-tuning job config and submit
    // TODO: Add SubmitFineTuningJob RPC to daemon

    // Placeholder implementation
    spdlog::info("Starting fine-tuning job: model={}, dataset={}, epochs={}, lr={:.2e}",
        config_.model_path, config_.dataset_path, config_.epochs, config_.initial_lr);

    // For now, simulate job submission
    is_training_ = true;
    training_progress_ = 0.0f;
    training_job_id_ = "finetune-" + std::to_string(std::time(nullptr));

    // In a real implementation:
    // std::string error;
    // if (!client->SubmitFineTuningJob(config_, training_job_id_, error)) {
    //     training_error_ = error;
    //     is_training_ = false;
    // }
}

} // namespace cyxwiz::servernode::gui
