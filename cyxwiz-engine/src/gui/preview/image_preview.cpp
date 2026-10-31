#include "image_preview.h"
#include "../../core/texture_manager.h"
#include "../ui_tokens.h"
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
        ImGui::TextColored(cyxwiz::ui::CurrentTokens().error, "Failed to load image.");
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
        ImGui::TextColored(cyxwiz::ui::CurrentTokens().text_dim, "No image data.");
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
        ImGui::TextColored(cyxwiz::ui::CurrentTokens().text_dim, "[%.3f, %.3f]", min_v, max_v);
    }
}

} // namespace gui
