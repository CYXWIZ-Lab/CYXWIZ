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
        }
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Next plot");

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

    // Render the image
    ImGui::Image((ImTextureID)(uintptr_t)plot.texture_id,
                 ImVec2(display_width, display_height));

    // Context menu
    if (ImGui::BeginPopupContextItem("##plot_context")) {
        RenderPlotContextMenu(selected_plot_index_);
        ImGui::EndPopup();
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
                selected_plot_index_ = i;
            }
        } else {
            if (ImGui::Button("?##thumb", ImVec2(thumbnail_size_, thumbnail_size_))) {
                selected_plot_index_ = i;
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

} // namespace cyxwiz
