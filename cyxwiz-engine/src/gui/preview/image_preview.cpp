#include "image_preview.h"
#include "../../core/texture_manager.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace gui {

void ImagePreviewRenderer::RenderFile(const std::string& filepath, float zoom) {
    if (filepath != file_texture_path_) {
        // Load new file
        auto& tm = cyxwiz::TextureManager::Instance();
        file_texture_ = tm.LoadTextureFromFile(filepath, &file_tex_w_, &file_tex_h_);
        file_texture_path_ = filepath;
    }

    if (file_texture_ == 0) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Failed to load image.");
        return;
    }

    // Info
    RenderImageInfo(file_tex_w_, file_tex_h_, 0);

    ImGui::Spacing();

    // Scrollable image display
    float display_w = file_tex_w_ * zoom;
    float display_h = file_tex_h_ * zoom;

    ImGui::BeginChild("##ImagePreviewContent", ImVec2(0, 0), true,
                       ImGuiWindowFlags_HorizontalScrollbar);
    ImGui::Image((ImTextureID)(intptr_t)file_texture_,
                  ImVec2(display_w, display_h));
    ImGui::EndChild();
}

void ImagePreviewRenderer::RenderSample(const std::vector<float>& data,
                                          int width, int height, int channels,
                                          float zoom) {
    if (data.empty() || width <= 0 || height <= 0) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No image data.");
        return;
    }

    RenderImageInfo(width, height, channels, data.data(), data.size());

    float display_w = width * zoom;
    float display_h = height * zoom;

    ImGui::BeginChild("##ImageSampleContent", ImVec2(0, 0), true,
                       ImGuiWindowFlags_HorizontalScrollbar);
    cyxwiz::RenderImageWithTexture(data.data(), width, height, channels, display_w, display_h);
    ImGui::EndChild();
}

void ImagePreviewRenderer::RenderGrid(const cyxwiz::DatasetHandle& handle,
                                        const cyxwiz::DatasetInfo& info,
                                        int start_idx, int cols, int rows,
                                        const std::vector<std::string>& class_names) {
    if (!handle.IsValid() || info.shape.size() < 2) return;

    int width = static_cast<int>(info.shape[0]);
    int height = static_cast<int>(info.shape[1]);
    int channels = info.shape.size() > 2 ? static_cast<int>(info.shape[2]) : 1;
    size_t dataset_size = handle.Size();

    int item_size = 80;
    int idx = start_idx;

    for (int row = 0; row < rows && idx < static_cast<int>(dataset_size); ++row) {
        for (int col = 0; col < cols && idx < static_cast<int>(dataset_size); ++col) {
            if (col > 0) ImGui::SameLine();

            auto [sample, label] = handle.GetSample(idx);

            ImGui::BeginGroup();
            ImGui::PushID(idx);

            if (sample.size() == static_cast<size_t>(width * height * channels)) {
                auto& tm = cyxwiz::TextureManager::Instance();
                std::string cache_key = "grid_" + std::to_string(idx) + "_" +
                                       handle.GetName() + "_" +
                                       std::to_string(width) + "x" + std::to_string(height);
                uint32_t tex_id = tm.GetOrCreateCachedTexture(cache_key, sample.data(),
                                                               width, height, channels);
                if (tex_id) {
                    ImGui::Image((ImTextureID)(intptr_t)tex_id,
                                  ImVec2(static_cast<float>(item_size), static_cast<float>(item_size)));
                }
            }

            // Label
            if (label >= 0 && label < static_cast<int>(class_names.size())) {
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "%s", class_names[label].c_str());
            } else {
                ImGui::Text("Label: %d", label);
            }

            ImGui::PopID();
            ImGui::EndGroup();

            idx++;
        }
    }
}

void ImagePreviewRenderer::RenderImageInfo(int width, int height, int channels,
                                             const float* data, size_t data_size) {
    ImGui::Text("%dx%d", width, height);
    if (channels > 0) {
        ImGui::SameLine();
        ImGui::Text("(%d ch)", channels);
    }

    if (data && data_size > 0) {
        float min_v = *std::min_element(data, data + data_size);
        float max_v = *std::max_element(data, data + data_size);
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "[%.3f, %.3f]", min_v, max_v);
    }
}

} // namespace gui
