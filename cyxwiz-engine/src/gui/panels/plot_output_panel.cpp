#include "plot_output_panel.h"
#include "output_renderer.h"
#include "../icons.h"
#include "../../core/file_dialogs.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <stb_image.h>
#include <fstream>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#endif

namespace cyxwiz {

PlotOutputPanel::PlotOutputPanel()
    : Panel("Plot Output", true) {
    spdlog::info("PlotOutputPanel initialized");
}

PlotOutputPanel::~PlotOutputPanel() {
    // Clean up textures
    std::lock_guard<std::mutex> lock(plots_mutex_);
    for (auto& plot : plots_) {
        if (plot.texture_id != 0) {
            glDeleteTextures(1, &plot.texture_id);
        }
    }
    plots_.clear();
}

void PlotOutputPanel::SetScriptingEngine(std::shared_ptr<scripting::ScriptingEngine> engine) {
    scripting_engine_ = engine;
}

void PlotOutputPanel::AddPlot(const scripting::CapturedPlot& plot) {
    PlotEntry entry;
    entry.png_data = plot.png_data;
    entry.label = plot.label.empty() ? "Figure " + std::to_string(plots_.size() + 1) : plot.label;

    // Create texture from PNG data
    entry.texture_id = CreateTextureFromPNG(plot.png_data, entry.width, entry.height);

    if (entry.texture_id != 0) {
        std::lock_guard<std::mutex> lock(plots_mutex_);
        plots_.push_back(std::move(entry));

        // Auto-select new plot
        selected_plot_index_ = static_cast<int>(plots_.size()) - 1;

        spdlog::info("Added plot to PlotOutputPanel: {}x{}", entry.width, entry.height);
    } else {
        spdlog::error("Failed to create texture for plot");
    }
}

void PlotOutputPanel::ClearPlots() {
    std::lock_guard<std::mutex> lock(plots_mutex_);
    for (auto& plot : plots_) {
        if (plot.texture_id != 0) {
            glDeleteTextures(1, &plot.texture_id);
        }
    }
    plots_.clear();
    selected_plot_index_ = -1;
}

void PlotOutputPanel::Render() {
    if (!visible_) return;

    // Poll for new plots from script execution
    PollForNewPlots();

    ImGui::Begin(GetName(), &visible_, ImGuiWindowFlags_MenuBar);
    focused_ = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);

    // Menu bar
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Show Thumbnails", nullptr, &show_thumbnails_);
            ImGui::MenuItem("Auto-scroll", nullptr, &auto_scroll_);
            ImGui::Separator();
            if (ImGui::MenuItem("Clear All", nullptr, false, !plots_.empty())) {
                ClearPlots();
            }
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    RenderToolbar();

    std::lock_guard<std::mutex> lock(plots_mutex_);

    if (plots_.empty()) {
        // Empty state
        ImVec2 avail = ImGui::GetContentRegionAvail();
        float text_height = ImGui::GetTextLineHeightWithSpacing() * 3;
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + (avail.y - text_height) / 2);

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
        ImGui::TextWrapped("No plots to display.\n\nRun a script with matplotlib to see plots here.\nPlots will appear automatically when scripts complete.");
        ImGui::PopStyleColor();
    } else {
        // Layout: thumbnails on left (if enabled), main view on right
        if (show_thumbnails_ && plots_.size() > 1) {
            // Left panel: thumbnails
            ImGui::BeginChild("##thumbnails", ImVec2(thumbnail_size_ + 20, 0), ImGuiChildFlags_Border);
            RenderThumbnails();
            ImGui::EndChild();

            ImGui::SameLine();
        }

        // Right panel: selected plot
        ImGui::BeginChild("##main_plot", ImVec2(0, 0), ImGuiChildFlags_Border);
        RenderSelectedPlot();
        ImGui::EndChild();
    }

    ImGui::End();
}

void PlotOutputPanel::RenderToolbar() {
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));

    // Plot navigation
    if (!plots_.empty()) {
        ImGui::BeginDisabled(selected_plot_index_ <= 0);
        if (ImGui::Button(ICON_FA_CHEVRON_LEFT "##prev")) {
            selected_plot_index_--;
            ResetZoom();  // Reset zoom/pan when switching plots
        }
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Previous plot");

        ImGui::SameLine();

        // Current plot indicator
        ImGui::Text("%d / %zu", selected_plot_index_ + 1, plots_.size());

        ImGui::SameLine();

        ImGui::BeginDisabled(selected_plot_index_ >= static_cast<int>(plots_.size()) - 1);
        if (ImGui::Button(ICON_FA_CHEVRON_RIGHT "##next")) {
            selected_plot_index_++;
            ResetZoom();  // Reset zoom/pan when switching plots
        }
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Next plot");

        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();

        // Zoom controls
        ImGui::BeginDisabled(zoom_level_ <= min_zoom_);
        if (ImGui::Button(ICON_FA_MINUS "##zoom_out")) {
            ZoomOut();
        }
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Zoom out");

        ImGui::SameLine();

        // Zoom level display
        ImGui::Text("%.0f%%", zoom_level_ * 100.0f);

        ImGui::SameLine();

        ImGui::BeginDisabled(zoom_level_ >= max_zoom_);
        if (ImGui::Button(ICON_FA_PLUS "##zoom_in")) {
            ZoomIn();
        }
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Zoom in");

        ImGui::SameLine();

        if (ImGui::Button("Fit##fit")) {
            FitToWindow();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Fit to window");

        ImGui::SameLine();

        if (ImGui::Button("100%##actual")) {
            ActualSize();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Actual size (100%)");

        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();

        // Copy button
        if (ImGui::Button(ICON_FA_COPY "##copy")) {
            CopyToClipboard(selected_plot_index_);
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Copy to clipboard");

        ImGui::SameLine();

        // Save button
        if (ImGui::Button(ICON_FA_FLOPPY_DISK "##save")) {
            SaveToFile(selected_plot_index_);
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Save as PNG");

        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();

        // Clear current
        if (ImGui::Button(ICON_FA_XMARK "##close")) {
            if (selected_plot_index_ >= 0 && selected_plot_index_ < static_cast<int>(plots_.size())) {
                if (plots_[selected_plot_index_].texture_id != 0) {
                    glDeleteTextures(1, &plots_[selected_plot_index_].texture_id);
                }
                plots_.erase(plots_.begin() + selected_plot_index_);
                if (selected_plot_index_ >= static_cast<int>(plots_.size())) {
                    selected_plot_index_ = static_cast<int>(plots_.size()) - 1;
                }
            }
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Close this plot");

        ImGui::SameLine();

        // Clear all
        if (ImGui::Button(ICON_FA_TRASH "##clear_all")) {
            ClearPlots();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Clear all plots");
    }

    ImGui::PopStyleColor();
    ImGui::Separator();
}

void PlotOutputPanel::RenderSelectedPlot() {
    if (selected_plot_index_ < 0 || selected_plot_index_ >= static_cast<int>(plots_.size())) {
        ImGui::TextDisabled("No plot selected");
        return;
    }

    auto& plot = plots_[selected_plot_index_];

    // Show label if present
    if (!plot.label.empty()) {
        ImGui::TextColored(ImVec4(0.7f, 0.8f, 1.0f, 1.0f), "%s", plot.label.c_str());
        ImGui::Separator();
    }

    if (plot.texture_id == 0) {
        ImGui::TextDisabled("Failed to load plot");
        return;
    }

    // Calculate display size to fit in available space while maintaining aspect ratio
    ImVec2 avail = ImGui::GetContentRegionAvail();
    float aspect_ratio = static_cast<float>(plot.width) / static_cast<float>(plot.height);

    float display_width = avail.x;
    float display_height = display_width / aspect_ratio;

    if (display_height > avail.y) {
        display_height = avail.y;
        display_width = display_height * aspect_ratio;
    }

    // Center the image
    float x_offset = (avail.x - display_width) / 2;
    float y_offset = (avail.y - display_height) / 2;
    ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPosX() + x_offset, ImGui::GetCursorPosY() + y_offset));

    // Calculate UV coordinates based on zoom and pan
    float visible_w = 1.0f / zoom_level_;
    float visible_h = 1.0f / zoom_level_;

    // Clamp pan offset to valid range
    float max_pan_x = std::max(0.0f, 1.0f - visible_w);
    float max_pan_y = std::max(0.0f, 1.0f - visible_h);
    pan_offset_.x = std::clamp(pan_offset_.x, 0.0f, max_pan_x);
    pan_offset_.y = std::clamp(pan_offset_.y, 0.0f, max_pan_y);

    ImVec2 uv0(pan_offset_.x, pan_offset_.y);
    ImVec2 uv1(pan_offset_.x + visible_w, pan_offset_.y + visible_h);

    // Render the image with zoom/pan UV coordinates
    ImGui::Image((ImTextureID)(uintptr_t)plot.texture_id,
                 ImVec2(display_width, display_height),
                 uv0, uv1);

    // Handle zoom/pan mouse interactions
    HandleZoomPan();

    // Context menu
    if (ImGui::BeginPopupContextItem("##plot_context")) {
        RenderPlotContextMenu(selected_plot_index_);
        ImGui::EndPopup();
    }

    // Show zoom indicator overlay when zoomed
    if (zoom_level_ != 1.0f) {
        ImVec2 image_max = ImGui::GetItemRectMax();
        char zoom_text[16];
        snprintf(zoom_text, sizeof(zoom_text), "%.0f%%", zoom_level_ * 100.0f);
        ImVec2 text_size = ImGui::CalcTextSize(zoom_text);
        ImVec2 text_pos(image_max.x - text_size.x - 8, image_max.y - text_size.y - 8);

        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        // Background
        draw_list->AddRectFilled(
            ImVec2(text_pos.x - 4, text_pos.y - 2),
            ImVec2(text_pos.x + text_size.x + 4, text_pos.y + text_size.y + 2),
            IM_COL32(0, 0, 0, 150), 4.0f);
        // Text
        draw_list->AddText(text_pos, IM_COL32(255, 255, 255, 200), zoom_text);
    }
}

void PlotOutputPanel::RenderThumbnails() {
    for (int i = 0; i < static_cast<int>(plots_.size()); i++) {
        auto& plot = plots_[i];

        ImGui::PushID(i);

        // Highlight selected thumbnail
        bool is_selected = (i == selected_plot_index_);
        if (is_selected) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.3f, 0.4f, 0.6f, 1.0f));
        }

        // Render thumbnail
        if (plot.texture_id != 0) {
            // Calculate thumbnail size maintaining aspect ratio
            float aspect_ratio = static_cast<float>(plot.width) / static_cast<float>(plot.height);
            float thumb_w = thumbnail_size_;
            float thumb_h = thumbnail_size_ / aspect_ratio;
            if (thumb_h > thumbnail_size_) {
                thumb_h = thumbnail_size_;
                thumb_w = thumbnail_size_ * aspect_ratio;
            }

            if (ImGui::ImageButton("##thumb",
                                   (ImTextureID)(uintptr_t)plot.texture_id,
                                   ImVec2(thumb_w, thumb_h))) {
                if (selected_plot_index_ != i) {
                    selected_plot_index_ = i;
                    ResetZoom();  // Reset zoom/pan when switching plots
                }
            }
        } else {
            if (ImGui::Button("?##thumb", ImVec2(thumbnail_size_, thumbnail_size_))) {
                if (selected_plot_index_ != i) {
                    selected_plot_index_ = i;
                    ResetZoom();  // Reset zoom/pan when switching plots
                }
            }
        }

        if (is_selected) {
            ImGui::PopStyleColor();
        }

        // Tooltip with plot info
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("%s", plot.label.c_str());
            ImGui::Text("%dx%d", plot.width, plot.height);
            ImGui::EndTooltip();
        }

        // Context menu for thumbnail
        if (ImGui::BeginPopupContextItem("##thumb_context")) {
            RenderPlotContextMenu(i);
            ImGui::EndPopup();
        }

        ImGui::PopID();
    }
}

void PlotOutputPanel::RenderPlotContextMenu(int plot_index) {
    if (plot_index < 0 || plot_index >= static_cast<int>(plots_.size())) return;

    auto& plot = plots_[plot_index];

    if (ImGui::MenuItem(ICON_FA_COPY " Copy to Clipboard")) {
        CopyToClipboard(plot_index);
    }

    if (ImGui::MenuItem(ICON_FA_FLOPPY_DISK " Save as PNG...")) {
        SaveToFile(plot_index);
    }

    ImGui::Separator();

    if (ImGui::MenuItem(ICON_FA_XMARK " Close")) {
        if (plot.texture_id != 0) {
            glDeleteTextures(1, &plot.texture_id);
        }
        plots_.erase(plots_.begin() + plot_index);
        if (selected_plot_index_ >= static_cast<int>(plots_.size())) {
            selected_plot_index_ = static_cast<int>(plots_.size()) - 1;
        }
    }
}

bool PlotOutputPanel::CopyToClipboard(int plot_index) {
    if (plot_index < 0 || plot_index >= static_cast<int>(plots_.size())) return false;

    auto& plot = plots_[plot_index];
    if (plot.png_data.empty()) {
        spdlog::warn("No PNG data for clipboard copy");
        return false;
    }

    // Use OutputRenderer's clipboard functionality
    bool success = OutputRenderer::CopyImageToClipboard(plot.png_data);
    if (success) {
        spdlog::info("Plot copied to clipboard");
    }
    return success;
}

bool PlotOutputPanel::SaveToFile(int plot_index) {
    if (plot_index < 0 || plot_index >= static_cast<int>(plots_.size())) return false;

    auto& plot = plots_[plot_index];
    if (plot.png_data.empty()) {
        spdlog::warn("No PNG data for save");
        return false;
    }

    // Use OutputRenderer's save functionality
    bool success = OutputRenderer::SaveImageToFile(plot.png_data, plot.label);
    if (success) {
        spdlog::info("Plot saved to file");
    }
    return success;
}

void PlotOutputPanel::PollForNewPlots() {
    if (!scripting_engine_) return;

    bool is_running = scripting_engine_->IsScriptRunning();

    // Check if script just finished
    if (was_script_running_ && !is_running) {
        // Script finished - check for plots in the result
        auto result = scripting_engine_->GetAsyncResult();
        if (result.has_value()) {
            auto& r = result.value();
            if (!r.plots.empty()) {
                spdlog::info("PlotOutputPanel: Received {} plots from script execution", r.plots.size());
                for (const auto& plot : r.plots) {
                    AddPlot(plot);
                }

                // Auto-scroll: ensure panel is visible
                if (auto_scroll_ && !plots_.empty()) {
                    visible_ = true;
                }
            }
        }
    }

    was_script_running_ = is_running;
}

GLuint PlotOutputPanel::CreateTextureFromPNG(const std::vector<unsigned char>& png_data, int& out_width, int& out_height) {
    if (png_data.empty()) return 0;

    // Decode PNG using stb_image
    int width, height, channels;
    unsigned char* pixels = stbi_load_from_memory(
        png_data.data(),
        static_cast<int>(png_data.size()),
        &width, &height, &channels, 4);  // Force RGBA

    if (!pixels) {
        spdlog::error("Failed to decode PNG: {}", stbi_failure_reason());
        return 0;
    }

    out_width = width;
    out_height = height;

    // Create OpenGL texture
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    glBindTexture(GL_TEXTURE_2D, 0);

    stbi_image_free(pixels);

    spdlog::debug("Created plot texture {} ({}x{})", texture, width, height);
    return texture;
}

void PlotOutputPanel::DeleteTexture(GLuint texture_id) {
    if (texture_id != 0) {
        glDeleteTextures(1, &texture_id);
    }
}

void PlotOutputPanel::HandleZoomPan() {
    // Only handle if mouse is over the image
    if (!ImGui::IsItemHovered()) {
        is_panning_ = false;
        return;
    }

    // Mouse scroll for zoom (centered on mouse position)
    float scroll = ImGui::GetIO().MouseWheel;
    if (scroll != 0.0f) {
        ImVec2 mouse_pos = ImGui::GetMousePos();
        ImVec2 image_min = ImGui::GetItemRectMin();
        ImVec2 image_size = ImGui::GetItemRectSize();

        // Mouse position relative to image (0-1 normalized)
        float mx = (mouse_pos.x - image_min.x) / image_size.x;
        float my = (mouse_pos.y - image_min.y) / image_size.y;

        // Clamp mouse position to image bounds
        mx = std::clamp(mx, 0.0f, 1.0f);
        my = std::clamp(my, 0.0f, 1.0f);

        // Calculate the texture coordinate under the mouse before zoom
        float visible_w = 1.0f / zoom_level_;
        float visible_h = 1.0f / zoom_level_;
        float tex_x = pan_offset_.x + mx * visible_w;
        float tex_y = pan_offset_.y + my * visible_h;

        // Apply zoom
        zoom_level_ *= (scroll > 0) ? 1.15f : 0.87f;  // ~15% per scroll tick
        zoom_level_ = std::clamp(zoom_level_, min_zoom_, max_zoom_);

        // Adjust pan to keep the same texture point under the mouse
        float new_visible_w = 1.0f / zoom_level_;
        float new_visible_h = 1.0f / zoom_level_;
        pan_offset_.x = tex_x - mx * new_visible_w;
        pan_offset_.y = tex_y - my * new_visible_h;

        // Clamp pan to valid range
        float max_pan_x = std::max(0.0f, 1.0f - new_visible_w);
        float max_pan_y = std::max(0.0f, 1.0f - new_visible_h);
        pan_offset_.x = std::clamp(pan_offset_.x, 0.0f, max_pan_x);
        pan_offset_.y = std::clamp(pan_offset_.y, 0.0f, max_pan_y);
    }

    // Middle mouse button or left mouse button for panning when zoomed
    bool pan_button = ImGui::IsMouseDown(ImGuiMouseButton_Middle) ||
                      (zoom_level_ > 1.0f && ImGui::IsMouseDown(ImGuiMouseButton_Left));

    if (pan_button) {
        ImVec2 delta = ImGui::GetIO().MouseDelta;
        if (delta.x != 0.0f || delta.y != 0.0f) {
            ImVec2 image_size = ImGui::GetItemRectSize();

            // Convert pixel delta to normalized texture coords
            // Negative because dragging right should show more to the left
            float visible_w = 1.0f / zoom_level_;
            float visible_h = 1.0f / zoom_level_;
            pan_offset_.x -= (delta.x / image_size.x) * visible_w;
            pan_offset_.y -= (delta.y / image_size.y) * visible_h;

            // Clamp pan to valid range
            float max_pan_x = std::max(0.0f, 1.0f - visible_w);
            float max_pan_y = std::max(0.0f, 1.0f - visible_h);
            pan_offset_.x = std::clamp(pan_offset_.x, 0.0f, max_pan_x);
            pan_offset_.y = std::clamp(pan_offset_.y, 0.0f, max_pan_y);

            is_panning_ = true;
        }
    } else {
        is_panning_ = false;
    }

    // Show cursor hint when zoomed
    if (zoom_level_ > 1.0f) {
        ImGui::SetMouseCursor(is_panning_ ? ImGuiMouseCursor_ResizeAll : ImGuiMouseCursor_Hand);
    }
}

void PlotOutputPanel::ResetZoom() {
    zoom_level_ = 1.0f;
    pan_offset_ = ImVec2(0.0f, 0.0f);
}

void PlotOutputPanel::ZoomIn() {
    zoom_level_ *= 1.25f;  // 25% increase
    zoom_level_ = std::clamp(zoom_level_, min_zoom_, max_zoom_);
}

void PlotOutputPanel::ZoomOut() {
    zoom_level_ *= 0.8f;  // 20% decrease
    zoom_level_ = std::clamp(zoom_level_, min_zoom_, max_zoom_);

    // Adjust pan if we've zoomed out past valid pan range
    float visible_w = 1.0f / zoom_level_;
    float visible_h = 1.0f / zoom_level_;
    float max_pan_x = std::max(0.0f, 1.0f - visible_w);
    float max_pan_y = std::max(0.0f, 1.0f - visible_h);
    pan_offset_.x = std::clamp(pan_offset_.x, 0.0f, max_pan_x);
    pan_offset_.y = std::clamp(pan_offset_.y, 0.0f, max_pan_y);
}

void PlotOutputPanel::FitToWindow() {
    zoom_level_ = 1.0f;
    pan_offset_ = ImVec2(0.0f, 0.0f);
}

void PlotOutputPanel::ActualSize() {
    // Set zoom so 1 image pixel = 1 screen pixel
    if (selected_plot_index_ < 0 || selected_plot_index_ >= static_cast<int>(plots_.size())) {
        return;
    }

    auto& plot = plots_[selected_plot_index_];

    // Get the display size that would be used at 1.0 zoom
    ImVec2 avail = ImGui::GetContentRegionAvail();
    float aspect_ratio = static_cast<float>(plot.width) / static_cast<float>(plot.height);

    float display_width = avail.x;
    float display_height = display_width / aspect_ratio;

    if (display_height > avail.y) {
        display_height = avail.y;
        display_width = display_height * aspect_ratio;
    }

    // Calculate zoom needed for 1:1 pixel mapping
    // At zoom_level_ = X, we show 1/X of the image
    // We want: display_width = plot.width / zoom_level_
    // So: zoom_level_ = plot.width / display_width
    zoom_level_ = static_cast<float>(plot.width) / display_width;
    zoom_level_ = std::clamp(zoom_level_, min_zoom_, max_zoom_);

    // Center the view
    float visible_w = 1.0f / zoom_level_;
    float visible_h = 1.0f / zoom_level_;
    pan_offset_.x = (1.0f - visible_w) / 2.0f;
    pan_offset_.y = (1.0f - visible_h) / 2.0f;

    // Clamp pan to valid range
    float max_pan_x = std::max(0.0f, 1.0f - visible_w);
    float max_pan_y = std::max(0.0f, 1.0f - visible_h);
    pan_offset_.x = std::clamp(pan_offset_.x, 0.0f, max_pan_x);
    pan_offset_.y = std::clamp(pan_offset_.y, 0.0f, max_pan_y);
}

} // namespace cyxwiz
