#include "gradcam_panel.h"
#include <cyxwiz/sequential.h>
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <fstream>
#include <cmath>

namespace cyxwiz {

GradCAMPanel::GradCAMPanel() {
    std::memset(export_path_, 0, sizeof(export_path_));
}

GradCAMPanel::~GradCAMPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        is_computing_ = false;
        compute_thread_->join();
    }
}

void GradCAMPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(650, 700), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_EYE " Grad-CAM Visualization###GradCAM", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else {
            RenderModelInfo();
            ImGui::Spacing();
            RenderConfiguration();

            if (has_gradcam_ || has_saliency_) {
                ImGui::Separator();
                RenderResults();
            }
        }
    }
    ImGui::End();
}

void GradCAMPanel::RenderToolbar() {
    if (!has_gradcam_ && !has_saliency_) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_FILE_EXPORT " Export")) {
        ImGui::OpenPopup("ExportHeatmap");
    }

    if (!has_gradcam_ && !has_saliency_) ImGui::EndDisabled();

    RenderExportOptions();
}

void GradCAMPanel::RenderModelInfo() {
    ImGui::Text("%s Model Information", ICON_FA_BRAIN);
    ImGui::Spacing();

    if (!model_) {
        ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "No model loaded");
        ImGui::TextDisabled("Use the Training panel to train a model first");
        return;
    }

    ImGui::Text("Layers: %zu", model_->Size());

    // Layer names list
    if (layer_names_.empty()) {
        layer_names_ = ModelInterpretability::GetLayerNames(*model_);
    }

    ImGui::Text("Available layers:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(200);

    std::string layer_preview = selected_layer_ >= 0 && selected_layer_ < static_cast<int>(layer_names_.size())
        ? layer_names_[selected_layer_] : "Select layer...";

    if (ImGui::BeginCombo("##LayerSelect", layer_preview.c_str())) {
        for (size_t i = 0; i < layer_names_.size(); i++) {
            bool is_selected = (selected_layer_ == static_cast<int>(i));
            if (ImGui::Selectable(layer_names_[i].c_str(), is_selected)) {
                selected_layer_ = static_cast<int>(i);
                has_gradcam_ = false;
            }
        }
        ImGui::EndCombo();
    }

    // Input status
    ImGui::Spacing();
    if (has_input_) {
        ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1), "Input image loaded: %dx%dx%d",
                          input_channels_, input_height_, input_width_);
    } else {
        ImGui::TextColored(ImVec4(1, 0.7f, 0.5f, 1), "No input image set");
        ImGui::TextDisabled("Call SetInputImage() with image data");
    }
}

void GradCAMPanel::RenderConfiguration() {
    if (!model_ || !has_input_) {
        ImGui::TextDisabled("Load a model and input image to continue");
        return;
    }

    ImGui::Text("%s Configuration", ICON_FA_GEAR);
    ImGui::Spacing();

    // Target class
    ImGui::Text("Target Class:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::InputInt("##TargetClass", &target_class_)) {
        if (target_class_ < -1) target_class_ = -1;
        has_gradcam_ = false;
        has_saliency_ = false;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(-1 = predicted)");

    // Colormap
    const char* colormaps[] = {"Jet", "Viridis", "Hot", "Cool"};
    ImGui::Text("Colormap:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    ImGui::Combo("##Colormap", &colormap_, colormaps, IM_ARRAYSIZE(colormaps));

    // Overlay settings
    ImGui::Checkbox("Show Overlay", &show_overlay_);
    if (show_overlay_) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("Alpha", &overlay_alpha_, 0.1f, 0.9f);
    }

    ImGui::Spacing();

    // Saliency method
    const char* saliency_methods[] = {"Gradient", "SmoothGrad"};
    ImGui::Text("Saliency Method:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    ImGui::Combo("##SaliencyMethod", &saliency_method_, saliency_methods, IM_ARRAYSIZE(saliency_methods));

    if (saliency_method_ == 1) {
        ImGui::Text("Samples:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        ImGui::SliderInt("##SGSamples", &smoothgrad_samples_, 10, 100);

        ImGui::Text("Noise:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        ImGui::SliderFloat("##SGNoise", &smoothgrad_noise_, 0.01f, 0.3f);
    }

    ImGui::Spacing();

    // Run buttons
    bool can_run = selected_layer_ >= 0;
    if (!can_run) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_FIRE " Compute Grad-CAM", ImVec2(150, 0))) {
        RunGradCAM();
    }

    if (!can_run) ImGui::EndDisabled();

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_CHART_AREA " Compute Saliency", ImVec2(150, 0))) {
        RunSaliency();
    }
}

void GradCAMPanel::RenderLoadingIndicator() {
    ImGui::Spacing();
    ImGui::Text("%s Computing visualization...", ICON_FA_SPINNER);
}

void GradCAMPanel::RenderResults() {
    if (ImGui::BeginTabBar("GradCAMTabs")) {
        if (has_gradcam_ && ImGui::BeginTabItem(ICON_FA_FIRE " Grad-CAM")) {
            RenderHeatmapVisualization();
            ImGui::EndTabItem();
        }

        if (has_saliency_ && ImGui::BeginTabItem(ICON_FA_CHART_AREA " Saliency")) {
            RenderSaliencyVisualization();
            ImGui::EndTabItem();
        }

        if (has_gradcam_ && ImGui::BeginTabItem(ICON_FA_LAYER_GROUP " Activations")) {
            RenderLayerActivations();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }
}

void GradCAMPanel::RenderHeatmapVisualization() {
    std::lock_guard<std::mutex> lock(result_mutex_);

    if (!gradcam_result_.success) {
        ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "Error: %s",
                          gradcam_result_.error_message.c_str());
        return;
    }

    ImGui::Text("Layer: %s", gradcam_result_.layer_name.c_str());
    ImGui::Text("Target Class: %d", gradcam_result_.target_class);
    ImGui::Text("Class Score: %.4f", gradcam_result_.class_score);
    ImGui::Text("Heatmap Size: %zux%zu",
               gradcam_result_.heatmap_shape.size() > 1 ? gradcam_result_.heatmap_shape[1] : 0,
               gradcam_result_.heatmap_shape.size() > 0 ? gradcam_result_.heatmap_shape[0] : 0);

    ImGui::Spacing();

    // Draw heatmap as texture-like grid
    if (!gradcam_result_.heatmap.empty() && gradcam_result_.heatmap_shape.size() >= 2) {
        size_t h = gradcam_result_.heatmap_shape[0];
        size_t w = gradcam_result_.heatmap_shape[1];

        ImVec2 canvas_size(400, 400);
        float cell_w = canvas_size.x / w;
        float cell_h = canvas_size.y / h;

        ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        for (size_t y = 0; y < h; y++) {
            for (size_t x = 0; x < w; x++) {
                float value = gradcam_result_.heatmap[y * w + x];
                ImU32 color = GetHeatmapColor(value);

                ImVec2 p_min(canvas_pos.x + x * cell_w, canvas_pos.y + y * cell_h);
                ImVec2 p_max(p_min.x + cell_w, p_min.y + cell_h);

                draw_list->AddRectFilled(p_min, p_max, color);
            }
        }

        // Invisible button to reserve space
        ImGui::InvisibleButton("##HeatmapCanvas", canvas_size);
    }

    // Colorbar
    ImGui::Spacing();
    ImVec2 bar_pos = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    float bar_width = 200;
    float bar_height = 20;

    for (int i = 0; i < 100; i++) {
        float t = i / 99.0f;
        ImU32 color = GetHeatmapColor(t);

        ImVec2 p_min(bar_pos.x + i * bar_width / 100, bar_pos.y);
        ImVec2 p_max(p_min.x + bar_width / 100 + 1, p_min.y + bar_height);

        draw_list->AddRectFilled(p_min, p_max, color);
    }

    ImGui::InvisibleButton("##Colorbar", ImVec2(bar_width, bar_height));
    ImGui::Text("0.0                    0.5                    1.0");
}

void GradCAMPanel::RenderSaliencyVisualization() {
    std::lock_guard<std::mutex> lock(result_mutex_);

    if (!saliency_result_.success) {
        ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "Error: %s",
                          saliency_result_.error_message.c_str());
        return;
    }

    ImGui::Text("Target Class: %d", saliency_result_.target_class);

    // Choose which gradients to display
    const std::vector<float>* grads = saliency_method_ == 1 && !saliency_result_.smoothgrad.empty()
        ? &saliency_result_.smoothgrad : &saliency_result_.absolute_gradients;

    if (grads->empty()) {
        ImGui::TextDisabled("No gradient data available");
        return;
    }

    ImGui::Spacing();

    // If image-shaped, display as heatmap
    if (input_height_ > 1 && input_width_ > 1) {
        size_t spatial_size = input_height_ * input_width_;

        // Average across channels if multi-channel
        std::vector<float> avg_grads(spatial_size, 0.0f);

        if (grads->size() == static_cast<size_t>(input_channels_) * spatial_size) {
            for (int c = 0; c < input_channels_; c++) {
                for (size_t s = 0; s < spatial_size; s++) {
                    avg_grads[s] += (*grads)[c * spatial_size + s];
                }
            }
            for (auto& g : avg_grads) g /= input_channels_;
        } else if (grads->size() == spatial_size) {
            avg_grads = *grads;
        }

        // Normalize
        float max_val = *std::max_element(avg_grads.begin(), avg_grads.end());
        if (max_val > 1e-6f) {
            for (auto& g : avg_grads) g /= max_val;
        }

        // Draw
        ImVec2 canvas_size(300, 300);
        float cell_w = canvas_size.x / input_width_;
        float cell_h = canvas_size.y / input_height_;

        ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        for (int y = 0; y < input_height_; y++) {
            for (int x = 0; x < input_width_; x++) {
                float value = avg_grads[y * input_width_ + x];
                ImU32 color = GetHeatmapColor(value);

                ImVec2 p_min(canvas_pos.x + x * cell_w, canvas_pos.y + y * cell_h);
                ImVec2 p_max(p_min.x + cell_w, p_min.y + cell_h);

                draw_list->AddRectFilled(p_min, p_max, color);
            }
        }

        ImGui::InvisibleButton("##SaliencyCanvas", canvas_size);
    } else {
        // Display as bar chart for 1D
        if (ImPlot::BeginPlot("Saliency", ImVec2(-1, 300))) {
            ImPlot::SetupAxes("Feature", "Importance");

            std::vector<double> indices(grads->size());
            std::vector<double> values(grads->size());
            for (size_t i = 0; i < grads->size(); i++) {
                indices[i] = static_cast<double>(i);
                values[i] = static_cast<double>((*grads)[i]);
            }

            ImPlot::PlotBars("Gradient", indices.data(), values.data(),
                            static_cast<int>(indices.size()), 0.8);

            ImPlot::EndPlot();
        }
    }
}

void GradCAMPanel::RenderLayerActivations() {
    std::lock_guard<std::mutex> lock(result_mutex_);

    ImGui::TextDisabled("Layer activation visualization");
    ImGui::TextDisabled("(Shows feature maps from target layer)");

    // This would show the activation maps from the selected layer
    // For now, display info about what would be shown
    if (selected_layer_ >= 0 && selected_layer_ < static_cast<int>(layer_names_.size())) {
        ImGui::Text("Layer: %s", layer_names_[selected_layer_].c_str());
    }
}

void GradCAMPanel::RenderExportOptions() {
    if (ImGui::BeginPopup("ExportHeatmap")) {
        ImGui::Text("Export Heatmap");
        ImGui::Separator();

        ImGui::InputText("File Path", export_path_, sizeof(export_path_));

        if (ImGui::Button("Save CSV")) {
            std::lock_guard<std::mutex> lock(result_mutex_);

            std::ofstream file(export_path_);
            if (file) {
                if (has_gradcam_ && !gradcam_result_.heatmap.empty()) {
                    size_t h = gradcam_result_.heatmap_shape.size() > 0 ? gradcam_result_.heatmap_shape[0] : 1;
                    size_t w = gradcam_result_.heatmap_shape.size() > 1 ? gradcam_result_.heatmap_shape[1] : gradcam_result_.heatmap.size();

                    for (size_t y = 0; y < h; y++) {
                        for (size_t x = 0; x < w; x++) {
                            file << gradcam_result_.heatmap[y * w + x];
                            if (x < w - 1) file << ",";
                        }
                        file << "\n";
                    }
                }
                spdlog::info("Exported heatmap to: {}", export_path_);
            }

            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

void GradCAMPanel::SetModel(std::shared_ptr<SequentialModel> model) {
    model_ = model;
    layer_names_.clear();
    selected_layer_ = -1;
    has_gradcam_ = false;
    has_saliency_ = false;

    if (model_) {
        layer_names_ = ModelInterpretability::GetLayerNames(*model_);
        // Select last conv layer by default
        for (int i = static_cast<int>(layer_names_.size()) - 1; i >= 0; i--) {
            if (layer_names_[i].find("conv") != std::string::npos ||
                layer_names_[i].find("Conv") != std::string::npos) {
                selected_layer_ = i;
                break;
            }
        }
        if (selected_layer_ < 0 && !layer_names_.empty()) {
            selected_layer_ = static_cast<int>(layer_names_.size()) - 2;  // Second to last
            if (selected_layer_ < 0) selected_layer_ = 0;
        }
    }
}

void GradCAMPanel::SetInputImage(const std::vector<float>& data, int channels, int height, int width) {
    input_data_ = data;
    input_channels_ = channels;
    input_height_ = height;
    input_width_ = width;
    has_input_ = !data.empty();
    has_gradcam_ = false;
    has_saliency_ = false;
}

ImU32 GradCAMPanel::GetHeatmapColor(float value) const {
    value = std::max(0.0f, std::min(1.0f, value));

    float r = 0, g = 0, b = 0;

    switch (colormap_) {
        case 0:  // Jet
            if (value < 0.25f) {
                r = 0; g = 4 * value; b = 1;
            } else if (value < 0.5f) {
                r = 0; g = 1; b = 1 - 4 * (value - 0.25f);
            } else if (value < 0.75f) {
                r = 4 * (value - 0.5f); g = 1; b = 0;
            } else {
                r = 1; g = 1 - 4 * (value - 0.75f); b = 0;
            }
            break;

        case 1:  // Viridis
            r = 0.267f + 0.329f * value + 0.404f * value * value;
            g = 0.004f + 0.873f * value - 0.377f * value * value;
            b = 0.329f + 0.424f * value - 0.753f * value * value;
            break;

        case 2:  // Hot
            r = std::min(1.0f, 3 * value);
            g = std::max(0.0f, std::min(1.0f, 3 * value - 1));
            b = std::max(0.0f, 3 * value - 2);
            break;

        case 3:  // Cool
            r = value;
            g = 1 - value;
            b = 1;
            break;
    }

    return IM_COL32(static_cast<int>(r * 255), static_cast<int>(g * 255),
                    static_cast<int>(b * 255), 255);
}

void GradCAMPanel::RunGradCAM() {
    if (is_computing_.load() || !model_ || !has_input_ || selected_layer_ < 0) return;

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;

    auto model = model_;
    auto input_data = input_data_;
    int layer = selected_layer_;
    int target = target_class_;
    std::vector<size_t> shape = {1, static_cast<size_t>(input_channels_),
                                  static_cast<size_t>(input_height_),
                                  static_cast<size_t>(input_width_)};

    compute_thread_ = std::make_unique<std::thread>([this, model, input_data, shape, layer, target]() {
        try {
            // Create input tensor
            Tensor input(shape, DataType::Float32);
            float* data = input.Data<float>();
            if (data) {
                std::copy(input_data.begin(), input_data.end(), data);
            }

            auto result = ModelInterpretability::ComputeGradCAM(*model, input, layer, target);

            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                gradcam_result_ = std::move(result);
                has_gradcam_ = gradcam_result_.success;
            }

            if (has_gradcam_) {
                spdlog::info("Grad-CAM complete for class {}", gradcam_result_.target_class);
            } else {
                spdlog::error("Grad-CAM failed: {}", gradcam_result_.error_message);
            }

        } catch (const std::exception& e) {
            spdlog::error("Grad-CAM error: {}", e.what());
        }

        is_computing_ = false;
    });
}

void GradCAMPanel::RunSaliency() {
    if (is_computing_.load() || !model_ || !has_input_) return;

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;

    auto model = model_;
    auto input_data = input_data_;
    int target = target_class_;
    int method = saliency_method_;
    int samples = smoothgrad_samples_;
    float noise = smoothgrad_noise_;
    std::vector<size_t> shape = {1, static_cast<size_t>(input_channels_),
                                  static_cast<size_t>(input_height_),
                                  static_cast<size_t>(input_width_)};

    compute_thread_ = std::make_unique<std::thread>([this, model, input_data, shape, target, method, samples, noise]() {
        try {
            // Create input tensor
            Tensor input(shape, DataType::Float32);
            float* data = input.Data<float>();
            if (data) {
                std::copy(input_data.begin(), input_data.end(), data);
            }

            SaliencyMap result;
            if (method == 0) {
                result = ModelInterpretability::ComputeSaliencyMap(*model, input, target);
            } else {
                result = ModelInterpretability::ComputeSmoothGrad(*model, input, target, samples, noise);
            }

            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                saliency_result_ = std::move(result);
                has_saliency_ = saliency_result_.success;
            }

            if (has_saliency_) {
                spdlog::info("Saliency map complete for class {}", saliency_result_.target_class);
            } else {
                spdlog::error("Saliency failed: {}", saliency_result_.error_message);
            }

        } catch (const std::exception& e) {
            spdlog::error("Saliency error: {}", e.what());
        }

        is_computing_ = false;
    });
}

} // namespace cyxwiz
