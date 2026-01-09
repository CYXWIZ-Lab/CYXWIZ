#include "dataset_panel.h"
#include "../../core/texture_manager.h"
#include "../icons.h"
#include <imgui.h>
#include <spdlog/spdlog.h>

namespace gui {

void DatasetPanel::RenderDataPipelineTab() {
    // Main layout: 60% config (left), 40% preview (right)
    float total_width = ImGui::GetContentRegionAvail().x;
    float config_width = total_width * 0.6f;

    // Left column: Configuration sections
    ImGui::BeginChild("PipelineConfig", ImVec2(config_width, 0), true);

    // ========================================================================
    // SECTION 1: PREPROCESSING (Deterministic)
    // ========================================================================
    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.3f, 0.5f, 0.7f, 1.0f));
    if (ImGui::CollapsingHeader(ICON_FA_SLIDERS " Preprocessing (Deterministic)", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PopStyleColor();

        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Deterministic transforms applied once and cached.\n"
                            "Use for data cleanup, enhancement, and standardization.\n"
                            "Pipeline: Original -> Image Enhancement -> Normalization -> Scaling");
        }

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
        ImGui::TextWrapped("Applied once before training. Same result every time.");
        ImGui::PopStyleColor();
        ImGui::Spacing();

        // Dataset statistics
        if (ImGui::TreeNodeEx("Dataset Statistics", ImGuiTreeNodeFlags_DefaultOpen)) {
            RenderDatasetStatistics();
            ImGui::TreePop();
        }
        ImGui::Spacing();

        // Image enhancement
        if (ImGui::TreeNodeEx("Image Enhancement", ImGuiTreeNodeFlags_DefaultOpen)) {
            RenderImagePreprocessingSection();
            ImGui::TreePop();
        }
        ImGui::Spacing();

        // Normalization
        if (ImGui::TreeNodeEx("Normalization", ImGuiTreeNodeFlags_DefaultOpen)) {
            RenderNormalizationSection();
            ImGui::TreePop();
        }
        ImGui::Spacing();

        // Scaling
        if (ImGui::TreeNodeEx("Scaling", ImGuiTreeNodeFlags_DefaultOpen)) {
            RenderScalingSection();
            ImGui::TreePop();
        }
        ImGui::Spacing();

        // Apply button
        if (ImGui::Button("Apply Preprocessing to Dataset##PreprocessBtn", ImVec2(-1, 0))) {
            ApplyPreprocessingConfig();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Apply preprocessing transforms to all dataset samples");
        }
    } else {
        ImGui::PopStyleColor();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ========================================================================
    // SECTION 2: AUGMENTATION (Random, Training-time)
    // ========================================================================
    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.7f, 0.5f, 0.3f, 1.0f));
    if (ImGui::CollapsingHeader(ICON_FA_SHUFFLE " Training Augmentation (Random)", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PopStyleColor();

        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Random transforms applied per-batch during training.\n"
                            "Use to increase effective dataset size and prevent overfitting.\n"
                            "Each epoch sees different variations of the same images.");
        }

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
        ImGui::TextWrapped("Applied during training with randomness. Different every epoch.");
        ImGui::PopStyleColor();
        ImGui::Spacing();

        // Augmentation pipeline (includes Apply button)
        RenderAugmentationPipeline();
    } else {
        ImGui::PopStyleColor();
    }

    ImGui::EndChild();

    // ========================================================================
    // RIGHT COLUMN: UNIFIED PREVIEW
    // ========================================================================
    ImGui::SameLine();
    ImGui::BeginChild("PipelinePreview", ImVec2(0, 0), true);

    RenderUnifiedPreview();

    ImGui::EndChild();
}

void DatasetPanel::RenderUnifiedPreview() {
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Preview");
    ImGui::Spacing();

    if (!IsDatasetLoaded()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No dataset loaded");
        return;
    }

    // Tab selector for preview mode
    static int preview_mode = 0;  // 0=Preprocessing, 1=Augmentation

    ImGui::Text("Preview Mode:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-1);
    const char* modes[] = { "Preprocessing", "Augmentation" };
    if (ImGui::Combo("##PipelinePreviewMode", &preview_mode, modes, 2)) {
        preprocessing_preview_needs_update_ = true;
        preview_needs_update_ = true;
    }

    ImGui::Spacing();

    // Sample navigation (shared controls)
    ImGui::Text("Sample:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    int max_preview = static_cast<int>(current_dataset_.Size()) - 1;
    if (ImGui::SliderInt("##PipelinePreviewSample", &preview_sample_idx_, 0, max_preview)) {
        preprocessing_preview_needs_update_ = true;
        preview_needs_update_ = true;
    }

    ImGui::SameLine();
    if (ImGui::Button("<##PipelinePrevSample", ImVec2(30, 0))) {
        preview_sample_idx_ = std::max(0, preview_sample_idx_ - 1);
        preprocessing_preview_needs_update_ = true;
        preview_needs_update_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button(">##PipelineNextSample", ImVec2(30, 0))) {
        preview_sample_idx_ = std::min(max_preview, preview_sample_idx_ + 1);
        preprocessing_preview_needs_update_ = true;
        preview_needs_update_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Random##PipelineRandomSample", ImVec2(-1, 0))) {
        preview_sample_idx_ = rand() % current_dataset_.Size();
        preprocessing_preview_needs_update_ = true;
        preview_needs_update_ = true;
    }

    ImGui::Spacing();

    // Auto-refresh toggle (for augmentation only)
    if (preview_mode == 1) {
        ImGui::Checkbox("Auto-refresh##PipelineAutoRefresh", &show_augmented_preview_);
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Continuously apply random augmentation");
        }
    }

    // Update button
    if (ImGui::Button("Update Preview##PipelineUpdatePreview", ImVec2(-1, 0))) {
        if (preview_mode == 0) {
            // Preprocessing only
            UpdatePreprocessingPreview();
            show_preprocessing_preview_ = true;
        } else {
            // Augmentation only
            UpdateAugmentationPreview();
            show_augmented_preview_ = true;
        }
    }

    // Auto-refresh for augmentation
    if (preview_mode == 1 && show_augmented_preview_) {
        UpdateAugmentationPreview();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Display preview based on mode
    if (preview_mode == 0 && show_preprocessing_preview_) {
        // Preprocessing preview - display only
        RenderPreprocessingPreviewDisplay();
    } else if (preview_mode == 1 && show_augmented_preview_) {
        // Augmentation preview - display only
        RenderAugmentationPreviewDisplay();
    } else {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                         "Click 'Update Preview' to see results");
    }
}

// ============================================================================
// Preview Display Methods (extracted for unified preview)
// ============================================================================

void DatasetPanel::RenderPreprocessingPreviewDisplay() {
    if (preview_texture_before_ == 0 || preview_texture_after_ == 0) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No preview available");
        return;
    }

    // Calculate display sizes (scale up small images)
    float scale = 1.0f;
    if (preview_tex_before_w_ < 64 || preview_tex_before_h_ < 64) {
        scale = std::max(2.0f, 128.0f / std::max(static_cast<float>(preview_tex_before_w_),
                                                  static_cast<float>(preview_tex_before_h_)));
    }

    float display_w = preview_tex_before_w_ * scale;
    float display_h = preview_tex_before_h_ * scale;

    // BEFORE image
    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "Before (Original):");
    ImGui::Image((ImTextureID)(intptr_t)preview_texture_before_,
                 ImVec2(display_w, display_h));

    ImGui::Spacing();

    // AFTER image (may have different dimensions)
    float after_display_w = preview_tex_after_w_ * scale;
    float after_display_h = preview_tex_after_h_ * scale;

    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "After (Preprocessed):");
    ImGui::Image((ImTextureID)(intptr_t)preview_texture_after_,
                 ImVec2(after_display_w, after_display_h));

    ImGui::Spacing();
    ImGui::Text("Dimensions: %dx%dx%d → %dx%dx%d",
               preview_tex_before_w_, preview_tex_before_h_, preview_tex_before_c_,
               preview_tex_after_w_, preview_tex_after_h_, preview_tex_after_c_);
}

void DatasetPanel::RenderAugmentationPreviewDisplay() {
    using namespace cyxwiz::transforms;

    if (preview_texture_original_ == 0 || preview_texture_augmented_ == 0) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No preview available");
        return;
    }

    float preview_size = 150.0f;

    // Calculate aspect ratio preserved display size for original
    float aspect = static_cast<float>(preview_tex_orig_w_) / static_cast<float>(preview_tex_orig_h_);
    float display_w = preview_size;
    float display_h = preview_size;
    if (aspect > 1.0f) {
        display_h = preview_size / aspect;
    } else {
        display_w = preview_size * aspect;
    }

    // Calculate for augmented (might have different size)
    float aug_aspect = static_cast<float>(preview_tex_aug_w_) / static_cast<float>(preview_tex_aug_h_);
    float aug_w = preview_size;
    float aug_h = preview_size;
    if (aug_aspect > 1.0f) {
        aug_h = preview_size / aug_aspect;
    } else {
        aug_w = preview_size * aug_aspect;
    }

    // Side-by-side layout: Original on left, Augmented on right
    ImGui::BeginGroup();
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "ORIGINAL (%dx%d)",
                      preview_tex_orig_w_, preview_tex_orig_h_);
    ImGui::Image((ImTextureID)(intptr_t)preview_texture_original_, ImVec2(display_w, display_h));
    ImGui::EndGroup();

    ImGui::SameLine(0.0f, 20.0f);  // 20px spacing between images

    ImGui::BeginGroup();
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "AUGMENTED (%dx%d)",
                      preview_tex_aug_w_, preview_tex_aug_h_);
    ImGui::Image((ImTextureID)(intptr_t)preview_texture_augmented_, ImVec2(aug_w, aug_h));
    ImGui::EndGroup();

    // Show label
    ImGui::Spacing();
    auto [_, label] = current_dataset_.GetSample(preview_sample_idx_);
    auto info = cached_info_;
    if (label >= 0 && label < static_cast<int>(info.class_names.size())) {
        ImGui::Text("Label: %d (%s)", label, info.class_names[label].c_str());
    } else {
        ImGui::Text("Label: %d", label);
    }

    // Show pipeline info
    if (augmentation_pipeline_ && augmentation_pipeline_->size() > 0) {
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f),
                         "Pipeline: %zu transforms active", augmentation_pipeline_->size());
    } else {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No augmentation pipeline");
    }
}

void DatasetPanel::UpdateAugmentationPreview() {
    using namespace cyxwiz::transforms;

    if (!IsDatasetLoaded()) return;

    auto info = cached_info_;
    if (info.num_samples == 0) return;

    try {
        auto [sample_data, label] = current_dataset_.GetSample(preview_sample_idx_);

        if (!sample_data.empty() && info.shape.size() >= 2) {
            int width = static_cast<int>(info.shape[0]);
            int height = static_cast<int>(info.shape[1]);
            int channels = info.shape.size() >= 3 ? static_cast<int>(info.shape[2]) : 1;

            // Create original image
            preview_original_ = Image(sample_data, width, height, channels);

            // Apply augmentation
            if (augmentation_pipeline_) {
                preview_augmented_ = augmentation_pipeline_->apply(preview_original_);
            } else {
                preview_augmented_ = preview_original_;
            }

            // Create/update textures
            auto& tm = cyxwiz::TextureManager::Instance();

            // Original texture
            bool orig_size_changed = (preview_original_.width != preview_tex_orig_w_ ||
                                       preview_original_.height != preview_tex_orig_h_ ||
                                       preview_original_.channels != preview_tex_orig_c_);

            if (preview_texture_original_ == 0 || orig_size_changed) {
                if (preview_texture_original_ != 0) {
                    tm.DeleteTexture(preview_texture_original_);
                }
                preview_texture_original_ = tm.CreateTextureFromFloatData(
                    preview_original_.data.data(),
                    preview_original_.width,
                    preview_original_.height,
                    preview_original_.channels
                );
                preview_tex_orig_w_ = preview_original_.width;
                preview_tex_orig_h_ = preview_original_.height;
                preview_tex_orig_c_ = preview_original_.channels;
            } else {
                tm.UpdateTexture(preview_texture_original_,
                    preview_original_.data.data(),
                    preview_original_.width,
                    preview_original_.height,
                    preview_original_.channels
                );
            }

            // Augmented texture
            bool aug_size_changed = (preview_augmented_.width != preview_tex_aug_w_ ||
                                      preview_augmented_.height != preview_tex_aug_h_ ||
                                      preview_augmented_.channels != preview_tex_aug_c_);

            if (preview_texture_augmented_ == 0 || aug_size_changed) {
                if (preview_texture_augmented_ != 0) {
                    tm.DeleteTexture(preview_texture_augmented_);
                }
                preview_texture_augmented_ = tm.CreateTextureFromFloatData(
                    preview_augmented_.data.data(),
                    preview_augmented_.width,
                    preview_augmented_.height,
                    preview_augmented_.channels
                );
                preview_tex_aug_w_ = preview_augmented_.width;
                preview_tex_aug_h_ = preview_augmented_.height;
                preview_tex_aug_c_ = preview_augmented_.channels;
            } else {
                tm.UpdateTexture(preview_texture_augmented_,
                    preview_augmented_.data.data(),
                    preview_augmented_.width,
                    preview_augmented_.height,
                    preview_augmented_.channels
                );
            }
        }
    } catch (const std::exception& e) {
        spdlog::error("Failed to update augmentation preview: {}", e.what());
    }
}

} // namespace gui
