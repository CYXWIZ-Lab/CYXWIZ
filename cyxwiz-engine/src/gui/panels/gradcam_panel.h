#pragma once

#include <cyxwiz/model_interpretability.h>
#include <imgui.h>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>

namespace cyxwiz {

class SequentialModel;

/**
 * GradCAMPanel - CNN Interpretability Visualization Tool
 *
 * Features:
 * - Grad-CAM heatmap computation
 * - Saliency map visualization
 * - SmoothGrad for noise reduction
 * - Layer selector for target layer
 * - Class selector for target class
 * - Multiple colormap options
 * - Overlay mode (heatmap on input)
 * - Export heatmap image
 */
class GradCAMPanel {
public:
    GradCAMPanel();
    ~GradCAMPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

    // Set the model to analyze
    void SetModel(std::shared_ptr<SequentialModel> model);

    // Set input image data (flattened NCHW format)
    void SetInputImage(const std::vector<float>& data, int channels, int height, int width);

private:
    void RenderToolbar();
    void RenderModelInfo();
    void RenderConfiguration();
    void RenderLoadingIndicator();
    void RenderResults();
    void RenderHeatmapVisualization();
    void RenderSaliencyVisualization();
    void RenderLayerActivations();
    void RenderExportOptions();

    void RunGradCAM();
    void RunSaliency();

    // Convert heatmap values to colors
    ImU32 GetHeatmapColor(float value) const;

    bool visible_ = false;

    // Model
    std::shared_ptr<SequentialModel> model_;
    std::vector<std::string> layer_names_;
    int selected_layer_ = -1;

    // Input image
    std::vector<float> input_data_;
    int input_channels_ = 0;
    int input_height_ = 0;
    int input_width_ = 0;
    bool has_input_ = false;

    // Configuration
    int target_class_ = -1;  // -1 = predicted class
    int colormap_ = 0;       // 0=Jet, 1=Viridis, 2=Hot, 3=Cool
    float overlay_alpha_ = 0.5f;
    bool show_overlay_ = true;

    // Saliency options
    int saliency_method_ = 0;  // 0=Gradient, 1=SmoothGrad
    int smoothgrad_samples_ = 50;
    float smoothgrad_noise_ = 0.1f;

    // Results
    GradCAMResult gradcam_result_;
    SaliencyMap saliency_result_;
    LayerActivations activations_result_;

    bool has_gradcam_ = false;
    bool has_saliency_ = false;
    bool has_activations_ = false;

    // Async computation
    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;

    // Export
    char export_path_[256] = "";
};

} // namespace cyxwiz
