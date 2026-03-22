// Preprocessing tab implementation for Dataset Panel
// This file contains all preprocessing UI rendering methods

#include "dataset_panel.h"
#include "../../preprocessing/preprocessing_config.h"
#include "../../preprocessing/statistics_calculator.h"
#include "../../preprocessing/normalization_transform.h"
#include "../../preprocessing/scaling_transform.h"
#include "../../preprocessing/image_transform.h"
#include "../../core/texture_manager.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <future>
#include <cmath>

namespace gui {

// ============================================================================
// Main Preprocessing Tab
// ============================================================================

void TrainingEvaluationPanel::RenderPreprocessingTab() {
    ImGui::BeginChild("PreprocessingPanel", ImVec2(0, 0), false);

    // Header
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Data Preprocessing");
    ImGui::Text("Configure normalization, scaling, and image transformations");
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Main layout: Configuration on left (60%), Preview on right (40%)
    ImGui::Columns(2, "PreprocessingColumns", true);
    ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() * 0.60f);

    // Left column: Configuration sections
    {
        RenderDatasetStatistics();
        ImGui::Spacing();

        RenderNormalizationSection();
        ImGui::Spacing();

        RenderScalingSection();
        ImGui::Spacing();

        RenderImagePreprocessingSection();
        ImGui::Spacing();

        // Action buttons
        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button("Apply to Dataset", ImVec2(-1, 0))) {
            ApplyPreprocessingConfig();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Apply preprocessing configuration to the dataset.\n"
                            "This will be used during training.");
        }

        ImGui::Spacing();

        if (ImGui::Button("Save Preset", ImVec2(-1, 0))) {
            // TODO: Implement preset saving
            spdlog::info("Save preprocessing preset not yet implemented");
        }

        if (ImGui::Button("Load Preset", ImVec2(-1, 0))) {
            // TODO: Implement preset loading
            spdlog::info("Load preprocessing preset not yet implemented");
        }
    }

    ImGui::NextColumn();

    // Right column: Preview
    {
        RenderPreprocessingPreview();
    }

    ImGui::Columns(1);
    ImGui::EndChild();
}

// ============================================================================
// Dataset Statistics Section
// ============================================================================

void TrainingEvaluationPanel::RenderDatasetStatistics() {
    // Collapsing header for statistics
    if (ImGui::CollapsingHeader("Dataset Statistics", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent(10.0f);

        if (!stats_computed_ && !computing_stats_) {
            // Not computed yet - show compute button
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                             "Statistics not computed yet");
            ImGui::Spacing();

            if (ImGui::Button("Compute Statistics", ImVec2(200, 0))) {
                ComputeStatistics();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Compute mean, std, min, max for the dataset.\n"
                                "Required for normalization and scaling.");
            }

        } else if (computing_stats_) {
            // Computing in progress
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f),
                             "Computing statistics...");
            ImGui::ProgressBar(stats_computation_progress_, ImVec2(-1, 0));

            // Check if computation is complete
            if (stats_future_.valid() &&
                stats_future_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                current_stats_ = stats_future_.get();
                stats_computed_ = current_stats_.is_valid;
                computing_stats_ = false;

                if (stats_computed_) {
                    spdlog::info("Dataset statistics computed successfully");
                } else {
                    spdlog::error("Failed to compute dataset statistics");
                }
            }

        } else if (stats_computed_) {
            // Statistics computed - display them
            auto& stats = current_stats_;

            // Basic info
            ImGui::Text("Shape:");
            ImGui::SameLine(100);
            ImGui::TextColored(ImVec4(0.7f, 1.0f, 0.7f, 1.0f), "[");
            for (size_t i = 0; i < stats.shape.size(); ++i) {
                if (i > 0) {
                    ImGui::SameLine(0, 0);
                    ImGui::TextColored(ImVec4(0.7f, 1.0f, 0.7f, 1.0f), ", ");
                }
                ImGui::SameLine(0, 0);
                ImGui::TextColored(ImVec4(0.7f, 1.0f, 0.7f, 1.0f), "%zu", stats.shape[i]);
            }
            ImGui::SameLine(0, 0);
            ImGui::TextColored(ImVec4(0.7f, 1.0f, 0.7f, 1.0f), "]");

            ImGui::Text("Samples:");
            ImGui::SameLine(100);
            ImGui::TextColored(ImVec4(0.7f, 1.0f, 0.7f, 1.0f), "%zu", stats.num_samples);

            ImGui::Text("Channels:");
            ImGui::SameLine(100);
            ImGui::TextColored(ImVec4(0.7f, 1.0f, 0.7f, 1.0f), "%zu", stats.GetNumChannels());

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Per-channel statistics
            for (size_t ch = 0; ch < stats.GetNumChannels(); ++ch) {
                if (stats.GetNumChannels() > 1) {
                    ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "Channel %zu:", ch);
                }

                ImGui::Text("  Mean:");
                ImGui::SameLine(100);
                ImGui::Text("%.4f", stats.mean[ch]);

                ImGui::Text("  Std:");
                ImGui::SameLine(100);
                ImGui::Text("%.4f", stats.std[ch]);

                ImGui::Text("  Range:");
                ImGui::SameLine(100);
                ImGui::Text("[%.2f, %.2f]", stats.min[ch], stats.max[ch]);

                if (ch < stats.GetNumChannels() - 1) {
                    ImGui::Spacing();
                }
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Helper indicators
            if (stats.IsNormalized()) {
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                                 "✓ Data appears normalized [0, 1]");
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f),
                                 "⚠ Data not normalized");
            }

            ImGui::Spacing();

            // Action buttons
            if (ImGui::Button("Recompute", ImVec2(120, 0))) {
                ComputeStatistics();
            }
            ImGui::SameLine();
            if (ImGui::Button("Export CSV", ImVec2(120, 0))) {
                // TODO: Export statistics to CSV
                spdlog::info("Export statistics not yet implemented");
            }
        }

        ImGui::Unindent(10.0f);
    }
}

// ============================================================================
// Normalization Section
// ============================================================================

void TrainingEvaluationPanel::RenderNormalizationSection() {
    if (ImGui::CollapsingHeader("Normalization", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent(10.0f);

        auto& config = current_preprocessing_config_.normalization_config;

        // Strategy selection
        ImGui::Text("Strategy:");
        ImGui::SameLine(120);
        ImGui::SetNextItemWidth(-1);

        const char* strategies[] = {
            "None",
            "Auto-Detect",
            "MNIST (0.1307, 0.3081)",
            "CIFAR-10 (per-channel)",
            "ImageNet (per-channel)",
            "Custom (use computed stats)"
        };

        int current_strategy = static_cast<int>(config.strategy);
        if (ImGui::Combo("##NormStrategy", &current_strategy, strategies, IM_ARRAYSIZE(strategies))) {
            config.strategy = static_cast<cyxwiz::NormalizationStrategy>(current_strategy);
        }

        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("Normalization strategies:");
            ImGui::BulletText("None: No normalization");
            ImGui::BulletText("Auto-Detect: Detect based on dataset name");
            ImGui::BulletText("MNIST: mean=0.1307, std=0.3081");
            ImGui::BulletText("CIFAR-10: Per-channel ImageNet-like stats");
            ImGui::BulletText("ImageNet: RGB mean/std for pretrained models");
            ImGui::BulletText("Custom: Use computed dataset statistics");
            ImGui::EndTooltip();
        }

        // Per-channel option
        ImGui::Checkbox("Per-channel normalization", &config.per_channel);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Apply normalization independently to each channel\n"
                            "(e.g., R, G, B for color images)");
        }

        // Show current values for reference
        if (config.strategy != cyxwiz::NormalizationStrategy::None) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Preview values:");

            // Get preview values based on strategy
            std::vector<float> preview_mean, preview_std;
            switch (config.strategy) {
                case cyxwiz::NormalizationStrategy::MNIST:
                    preview_mean = {0.1307f};
                    preview_std = {0.3081f};
                    break;
                case cyxwiz::NormalizationStrategy::CIFAR10:
                    preview_mean = {0.4914f, 0.4822f, 0.4465f};
                    preview_std = {0.2470f, 0.2435f, 0.2616f};
                    break;
                case cyxwiz::NormalizationStrategy::ImageNet:
                    preview_mean = {0.485f, 0.456f, 0.406f};
                    preview_std = {0.229f, 0.224f, 0.225f};
                    break;
                case cyxwiz::NormalizationStrategy::Custom:
                    if (stats_computed_) {
                        preview_mean = current_stats_.mean;
                        preview_std = current_stats_.std;
                    }
                    break;
                default:
                    break;
            }

            if (!preview_mean.empty()) {
                ImGui::Text("  Mean: ");
                ImGui::SameLine();
                for (size_t i = 0; i < preview_mean.size(); ++i) {
                    if (i > 0) ImGui::SameLine(0, 5);
                    ImGui::Text("%.4f", preview_mean[i]);
                }

                ImGui::Text("  Std:  ");
                ImGui::SameLine();
                for (size_t i = 0; i < preview_std.size(); ++i) {
                    if (i > 0) ImGui::SameLine(0, 5);
                    ImGui::Text("%.4f", preview_std[i]);
                }
            }
        }

        // Warning for Custom without computed stats
        if (config.strategy == cyxwiz::NormalizationStrategy::Custom && !stats_computed_) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.4f, 1.0f),
                             "⚠ Compute statistics first for Custom strategy");
        }

        ImGui::Unindent(10.0f);
    }
}

// ============================================================================
// Scaling Section
// ============================================================================

void TrainingEvaluationPanel::RenderScalingSection() {
    if (ImGui::CollapsingHeader("Scaling", ImGuiTreeNodeFlags_None)) {
        ImGui::Indent(10.0f);

        auto& config = current_preprocessing_config_.scaling_config;

        // Strategy selection
        ImGui::Text("Strategy:");
        ImGui::SameLine(120);
        ImGui::SetNextItemWidth(-1);

        const char* strategies[] = {
            "None",
            "MinMax [0, 1]",
            "Standard (z-score)",
            "Robust (IQR-based)",
            "MaxAbs [-1, 1]",
            "Quantile Transform",
            "PCA Whitening"
        };

        int current_strategy = static_cast<int>(config.strategy);
        if (ImGui::Combo("##ScalingStrategy", &current_strategy, strategies, IM_ARRAYSIZE(strategies))) {
            config.strategy = static_cast<cyxwiz::ScalingStrategy>(current_strategy);
            preprocessing_preview_needs_update_ = true;
        }

        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("Scaling strategies:");
            ImGui::BulletText("None: No scaling");
            ImGui::BulletText("MinMax: Scale to [min, max] range");
            ImGui::BulletText("Standard: (x - mean) / std");
            ImGui::BulletText("Robust: (x - median) / IQR (outlier-resistant)");
            ImGui::BulletText("MaxAbs: x / max(|x|) to [-1, 1]");
            ImGui::BulletText("Quantile: Map to uniform or normal distribution");
            ImGui::BulletText("PCA Whitening: Decorrelate and normalize");
            ImGui::EndTooltip();
        }

        // MinMax specific options
        if (config.strategy == cyxwiz::ScalingStrategy::MinMax) {
            ImGui::Spacing();
            ImGui::Text("Target Range:");

            ImGui::Text("  Min:");
            ImGui::SameLine(120);
            ImGui::SetNextItemWidth(80);
            ImGui::InputFloat("##MinValue", &config.min_value, 0.0f, 0.0f, "%.2f");

            ImGui::Text("  Max:");
            ImGui::SameLine(120);
            ImGui::SetNextItemWidth(80);
            ImGui::InputFloat("##MaxValue", &config.max_value, 0.0f, 0.0f, "%.2f");
        }

        // MaxAbs specific info
        if (config.strategy == cyxwiz::ScalingStrategy::MaxAbs) {
            ImGui::Spacing();
            ImGui::TextWrapped("Scales data by maximum absolute value to [-1, 1] range.");
            ImGui::BulletText("Formula: x_scaled = x / max(|x|)");
            ImGui::BulletText("Preserves zero values");
            ImGui::BulletText("Useful for sparse data");
        }

        // Quantile specific options
        if (config.strategy == cyxwiz::ScalingStrategy::Quantile) {
            ImGui::Spacing();
            ImGui::Text("Output Distribution:");

            if (ImGui::RadioButton("Uniform [0, 1]", !config.quantile_use_normal)) {
                config.quantile_use_normal = false;
                preprocessing_preview_needs_update_ = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Maps data to uniform distribution in [0, 1] range");
            }

            if (ImGui::RadioButton("Normal (Gaussian)", config.quantile_use_normal)) {
                config.quantile_use_normal = true;
                preprocessing_preview_needs_update_ = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Maps data to standard normal distribution N(0, 1)");
            }

            ImGui::Spacing();
            ImGui::TextWrapped("Quantile transform is non-linear and robust to outliers.");
            ImGui::BulletText("Preserves rank order of values");
            ImGui::BulletText("Handles outliers well");
        }

        // Warning for strategies requiring stats
        if (config.strategy != cyxwiz::ScalingStrategy::None && !stats_computed_) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.4f, 1.0f),
                             "⚠ Compute statistics first for scaling");
        }

        ImGui::Unindent(10.0f);
    }
}

// ============================================================================
// Image Preprocessing Section
// ============================================================================

void TrainingEvaluationPanel::RenderImagePreprocessingSection() {
    if (ImGui::CollapsingHeader("Image Preprocessing", ImGuiTreeNodeFlags_None)) {
        ImGui::Indent(10.0f);

        auto& config = current_preprocessing_config_.image_config;

        // Resize mode
        ImGui::Text("Resize Mode:");
        ImGui::SameLine(140);
        ImGui::SetNextItemWidth(-1);

        const char* resize_modes[] = {
            "None",
            "Exact (may distort)",
            "Aspect Fit (pad)",
            "Aspect Fill (crop)",
            "Center Crop"
        };

        int current_mode = static_cast<int>(config.resize_mode);
        if (ImGui::Combo("##ResizeMode", &current_mode, resize_modes, IM_ARRAYSIZE(resize_modes))) {
            config.resize_mode = static_cast<cyxwiz::ResizeMode>(current_mode);
        }

        // Target size (if resizing)
        if (config.resize_mode != cyxwiz::ResizeMode::None) {
            ImGui::Spacing();
            ImGui::Text("Target Size:");

            ImGui::Text("  Width:");
            ImGui::SameLine(140);
            ImGui::SetNextItemWidth(100);
            ImGui::InputInt("##TargetWidth", &config.target_width, 1, 10);
            if (config.target_width < 1) config.target_width = 1;

            ImGui::Text("  Height:");
            ImGui::SameLine(140);
            ImGui::SetNextItemWidth(100);
            ImGui::InputInt("##TargetHeight", &config.target_height, 1, 10);
            if (config.target_height < 1) config.target_height = 1;

            // Quick size presets
            ImGui::Spacing();
            ImGui::Text("Quick Sizes:");
            ImGui::SameLine(140);
            if (ImGui::SmallButton("28x28")) {
                config.target_width = 28;
                config.target_height = 28;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("32x32")) {
                config.target_width = 32;
                config.target_height = 32;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("224x224")) {
                config.target_width = 224;
                config.target_height = 224;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("256x256")) {
                config.target_width = 256;
                config.target_height = 256;
            }
        }

        ImGui::Spacing();

        // Color conversion
        ImGui::Checkbox("Convert to Grayscale", &config.convert_to_grayscale);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Convert RGB images to single-channel grayscale");
        }

        ImGui::Checkbox("Convert to RGB", &config.convert_to_rgb);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Convert grayscale images to 3-channel RGB\n"
                            "(duplicates the grayscale values)");
        }

        // Mutual exclusion warning
        if (config.convert_to_grayscale && config.convert_to_rgb) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                             "⚠ Cannot convert to both grayscale and RGB");
        }

        // Image Enhancement Section
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Image Enhancement");
        ImGui::Spacing();

        // CLAHE (Contrast Limited Adaptive Histogram Equalization)
        ImGui::Checkbox("Apply CLAHE", &config.enable_clahe);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Contrast Limited Adaptive Histogram Equalization\n"
                            "Improves local contrast in low-contrast images");
        }

        if (config.enable_clahe) {
            ImGui::Indent(20.0f);

            ImGui::Text("Clip Limit:");
            ImGui::SameLine(140);
            ImGui::SetNextItemWidth(200);
            ImGui::SliderFloat("##CLAHEClipLimit", &config.clahe_clip_limit, 1.0f, 10.0f, "%.1f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Threshold for contrast limiting (1.0-10.0)\n"
                                "Higher = more contrast enhancement");
            }

            ImGui::Text("Tile Size:");
            ImGui::SameLine(140);
            ImGui::SetNextItemWidth(200);
            ImGui::SliderInt("##CLAHETileSize", &config.clahe_tile_size, 4, 16);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Grid size for histogram equalization\n"
                                "Smaller = more local contrast (but may introduce artifacts)");
            }

            ImGui::Unindent(20.0f);
        }

        ImGui::Spacing();

        // Denoise
        ImGui::Checkbox("Apply Denoising", &config.enable_denoise);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Non-local Means Denoising\n"
                            "Removes Gaussian noise while preserving edges");
        }

        if (config.enable_denoise) {
            ImGui::Indent(20.0f);

            ImGui::Text("Strength:");
            ImGui::SameLine(140);
            ImGui::SetNextItemWidth(200);
            ImGui::SliderFloat("##DenoiseStrength", &config.denoise_strength, 1.0f, 30.0f, "%.1f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Filter strength (1.0-30.0)\n"
                                "Recommended: 10.0 for moderate noise");
            }

            ImGui::Unindent(20.0f);
        }

        ImGui::Spacing();

        // Sharpen
        ImGui::Checkbox("Apply Sharpening", &config.enable_sharpen);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Unsharp Masking\n"
                            "Enhances edges and fine details");
        }

        if (config.enable_sharpen) {
            ImGui::Indent(20.0f);

            ImGui::Text("Amount:");
            ImGui::SameLine(140);
            ImGui::SetNextItemWidth(200);
            ImGui::SliderFloat("##SharpenAmount", &config.sharpen_amount, 0.1f, 3.0f, "%.1f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Sharpening strength (0.1-3.0)\n"
                                "Recommended: 1.0 for balanced sharpening");
            }

            ImGui::Unindent(20.0f);
        }

        ImGui::Spacing();

        // Edge Detection
        ImGui::Checkbox("Apply Edge Detection", &config.enable_edge_detection);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Detect edges in images\n"
                            "Useful for computer vision tasks");
        }

        if (config.enable_edge_detection) {
            ImGui::Indent(20.0f);

            ImGui::Text("Algorithm:");
            ImGui::SameLine(140);
            ImGui::SetNextItemWidth(200);
            const char* edge_types[] = {"Canny", "Sobel", "Laplacian", "Scharr"};
            int current_type = static_cast<int>(config.edge_detector_type);
            if (ImGui::Combo("##EdgeType", &current_type, edge_types, 4)) {
                config.edge_detector_type = static_cast<cyxwiz::EdgeDetectorType>(current_type);
                preprocessing_preview_needs_update_ = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Canny: Best general-purpose\n"
                                "Sobel: Gradient-based\n"
                                "Laplacian: Second derivative\n"
                                "Scharr: High-precision gradient");
            }

            // Show parameters based on algorithm type
            if (config.edge_detector_type == cyxwiz::EdgeDetectorType::Canny) {
                ImGui::Text("Low Threshold:");
                ImGui::SameLine(140);
                ImGui::SetNextItemWidth(200);
                ImGui::SliderFloat("##EdgeThresh1", &config.edge_threshold1, 10.0f, 200.0f, "%.0f");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Lower threshold for edge detection\n"
                                    "Recommended: 50");
                }

                ImGui::Text("High Threshold:");
                ImGui::SameLine(140);
                ImGui::SetNextItemWidth(200);
                ImGui::SliderFloat("##EdgeThresh2", &config.edge_threshold2, 10.0f, 300.0f, "%.0f");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Upper threshold for edge detection\n"
                                    "Recommended: 150 (3x low threshold)");
                }
            } else {
                // Sobel, Laplacian, Scharr use kernel size
                ImGui::Text("Kernel Size:");
                ImGui::SameLine(140);
                ImGui::SetNextItemWidth(200);
                int kernel_size = config.edge_kernel_size;
                if (ImGui::SliderInt("##EdgeKernel", &kernel_size, 1, 7)) {
                    // Ensure odd values only
                    if (kernel_size % 2 == 0) kernel_size++;
                    config.edge_kernel_size = kernel_size;
                    preprocessing_preview_needs_update_ = true;
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Kernel size (must be odd: 1, 3, 5, 7)\n"
                                    "Recommended: 3");
                }
            }

            ImGui::Unindent(20.0f);
        }

        ImGui::Unindent(10.0f);
    }
}

// ============================================================================
// Preview Section
// ============================================================================

void TrainingEvaluationPanel::RenderPreprocessingPreview() {
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Preview");
    ImGui::Spacing();

    if (!IsDatasetLoaded()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No dataset loaded");
        return;
    }

    // Preview controls
    ImGui::Text("Sample:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    int max_preview = static_cast<int>(current_dataset_.Size()) - 1;
    if (ImGui::SliderInt("##PreviewSample", &preview_sample_idx_, 0, max_preview)) {
        preprocessing_preview_needs_update_ = true;
    }

    ImGui::SameLine();
    if (ImGui::Button("◄", ImVec2(30, 0))) {
        preview_sample_idx_ = std::max(0, preview_sample_idx_ - 1);
        preprocessing_preview_needs_update_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("►", ImVec2(30, 0))) {
        preview_sample_idx_ = std::min(max_preview, preview_sample_idx_ + 1);
        preprocessing_preview_needs_update_ = true;
    }

    ImGui::Spacing();

    if (ImGui::Button("Update Preview", ImVec2(-1, 0))) {
        show_preprocessing_preview_ = true;
        UpdatePreprocessingPreview();  // Actually update the preview
        preprocessing_preview_needs_update_ = false;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Show preview images
    if (show_preprocessing_preview_ && preview_texture_before_ != 0 && preview_texture_after_ != 0) {
        // Calculate display sizes (scale up small images)
        float scale = 1.0f;
        if (preview_tex_before_w_ < 64 || preview_tex_before_h_ < 64) {
            scale = std::max(2.0f, 128.0f / std::max(preview_tex_before_w_, preview_tex_before_h_));
        }

        float display_w = preview_tex_before_w_ * scale;
        float display_h = preview_tex_before_h_ * scale;

        // BEFORE image
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "Before:");
        ImGui::Image((ImTextureID)(intptr_t)preview_texture_before_,
                     ImVec2(display_w, display_h));

        ImGui::Spacing();

        // AFTER image (may have different dimensions)
        float after_display_w = preview_tex_after_w_ * scale;
        float after_display_h = preview_tex_after_h_ * scale;

        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "After:");
        ImGui::Image((ImTextureID)(intptr_t)preview_texture_after_,
                     ImVec2(after_display_w, after_display_h));

        ImGui::Spacing();
        ImGui::Text("Dimensions: %dx%dx%d → %dx%dx%d",
                   preview_tex_before_w_, preview_tex_before_h_, preview_tex_before_c_,
                   preview_tex_after_w_, preview_tex_after_h_, preview_tex_after_c_);
    } else {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                         "Click 'Update Preview' to see\n"
                         "preprocessing results");
    }
}

// ============================================================================
// Helper Methods
// ============================================================================

void TrainingEvaluationPanel::ComputeStatistics() {
    if (!IsDatasetLoaded()) {
        spdlog::warn("No dataset loaded to compute statistics");
        return;
    }

    if (computing_stats_) {
        spdlog::warn("Statistics computation already in progress");
        return;
    }

    spdlog::info("Starting statistics computation for dataset '{}'",
                 current_dataset_.GetName());

    computing_stats_ = true;
    stats_computation_progress_ = 0.0f;

    // Launch async computation
    stats_future_ = std::async(std::launch::async, [this]() {
        auto progress_callback = [this](float progress) {
            stats_computation_progress_ = progress;
        };

        return cyxwiz::StatisticsCalculator::Compute(
            current_dataset_.GetName(),
            &cyxwiz::DataRegistry::Instance(),
            progress_callback
        );
    });
}

void TrainingEvaluationPanel::ApplyPreprocessingConfig() {
    if (!IsDatasetLoaded()) {
        spdlog::warn("No dataset loaded to apply preprocessing");
        return;
    }

    // Validation
    if (current_preprocessing_config_.normalization_config.strategy ==
        cyxwiz::NormalizationStrategy::Custom && !stats_computed_) {
        spdlog::error("Cannot apply Custom normalization without computed statistics");
        // TODO: Show error message to user
        return;
    }

    if (current_preprocessing_config_.scaling_config.strategy !=
        cyxwiz::ScalingStrategy::None && !stats_computed_) {
        spdlog::error("Cannot apply scaling without computed statistics");
        // TODO: Show error message to user
        return;
    }

    // Validate image preprocessing
    auto& img_config = current_preprocessing_config_.image_config;
    if (img_config.convert_to_grayscale && img_config.convert_to_rgb) {
        spdlog::error("Cannot convert to both grayscale and RGB");
        // TODO: Show error message to user
        return;
    }

    // Set the config in DataRegistry
    current_preprocessing_config_.enabled = true;
    current_preprocessing_config_.dataset_id = current_dataset_.GetName();

    cyxwiz::DataRegistry::Instance().SetPreprocessingConfig(
        current_dataset_.GetName(),
        current_preprocessing_config_
    );

    spdlog::info("Preprocessing configuration applied to dataset '{}'",
                 current_dataset_.GetName());

    // Show success notification
    notification_message_ = "Preprocessing configuration applied successfully!";
    notification_time_ = static_cast<float>(ImGui::GetTime());
    show_notification_ = true;
}

void TrainingEvaluationPanel::UpdatePreprocessingPreview() {
    if (!IsDatasetLoaded()) {
        spdlog::warn("No dataset loaded for preprocessing preview");
        return;
    }

    try {
        // 1. Get sample from dataset
        auto [sample_data, label] = current_dataset_.GetSample(preview_sample_idx_);

        // 2. Get dataset shape info
        auto shape_info = current_dataset_.GetInfo().shape;  // e.g., [28, 28, 1]
        int height = shape_info[0];
        int width = shape_info[1];
        int channels = shape_info.size() >= 3 ? shape_info[2] : 1;

        // 3. Create BEFORE tensor (original sample)
        std::vector<size_t> tensor_shape = {static_cast<size_t>(height),
                                              static_cast<size_t>(width),
                                              static_cast<size_t>(channels)};
        cyxwiz::Tensor before_tensor(tensor_shape, sample_data.data(), cyxwiz::DataType::Float32);

        // 4. Apply preprocessing pipeline to create AFTER tensor
        cyxwiz::Tensor after_tensor = before_tensor;

        // Apply Image Preprocessing
        if (current_preprocessing_config_.image_config.resize_mode != cyxwiz::ResizeMode::None ||
            current_preprocessing_config_.image_config.enable_clahe ||
            current_preprocessing_config_.image_config.enable_denoise ||
            current_preprocessing_config_.image_config.enable_sharpen ||
            current_preprocessing_config_.image_config.enable_edge_detection) {

            cyxwiz::ImageTransform img_transform(current_preprocessing_config_.image_config);
            after_tensor = img_transform.Apply(after_tensor);
        }

        // Apply Normalization
        if (current_preprocessing_config_.normalization_config.strategy !=
            cyxwiz::NormalizationStrategy::None && stats_computed_) {

            cyxwiz::NormalizationTransform norm_transform(
                current_preprocessing_config_.normalization_config
            );
            norm_transform.Initialize(current_stats_);
            after_tensor = norm_transform.Apply(after_tensor);
        }

        // Apply Scaling
        if (current_preprocessing_config_.scaling_config.strategy !=
            cyxwiz::ScalingStrategy::None && stats_computed_) {

            cyxwiz::ScalingTransform scale_transform(
                current_preprocessing_config_.scaling_config
            );
            scale_transform.Initialize(current_stats_);
            after_tensor = scale_transform.Apply(after_tensor);
        }

        // 5. Create textures using TextureManager
        auto& tm = cyxwiz::TextureManager::Instance();

        // BEFORE texture
        preview_tex_before_w_ = width;
        preview_tex_before_h_ = height;
        preview_tex_before_c_ = channels;

        if (preview_texture_before_ != 0) {
            tm.UpdateTexture(preview_texture_before_,
                           before_tensor.Data<float>(),
                           width, height, channels);
        } else {
            preview_texture_before_ = tm.CreateTextureFromFloatData(
                before_tensor.Data<float>(),
                width, height, channels
            );
        }

        // AFTER texture (might have different dimensions after resize)
        auto after_shape = after_tensor.Shape();
        int after_h = after_shape[0];
        int after_w = after_shape[1];
        int after_c = after_shape.size() >= 3 ? after_shape[2] : 1;

        preview_tex_after_w_ = after_w;
        preview_tex_after_h_ = after_h;
        preview_tex_after_c_ = after_c;

        if (preview_texture_after_ != 0) {
            tm.UpdateTexture(preview_texture_after_,
                           after_tensor.Data<float>(),
                           after_w, after_h, after_c);
        } else {
            preview_texture_after_ = tm.CreateTextureFromFloatData(
                after_tensor.Data<float>(),
                after_w, after_h, after_c
            );
        }

        spdlog::info("Updated preprocessing preview (sample {}, label {})",
                     preview_sample_idx_, label);

    } catch (const std::exception& e) {
        spdlog::error("Failed to update preprocessing preview: {}", e.what());
    }
}

} // namespace gui
