#include "plot_output_panel.h"
#include "output_renderer.h"
#include "../icons.h"
#include "../plot/plot_window.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"
#include "../../core/file_dialogs.h"
#include <arrow/api.h>
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
        spdlog::info("Added plot to PlotOutputPanel: {}x{}", entry.width, entry.height);
        plots_.push_back(std::move(entry));
        Select(static_cast<int>(plots_.size()) - 1);  // the new figure
    } else {
        spdlog::error("Failed to create texture for plot");
    }
}

void PlotOutputPanel::ShowImage(const std::vector<unsigned char>& png_data, const std::string& title) {
    scripting::CapturedPlot plot;
    plot.png_data = png_data;
    plot.label = title;
    AddPlot(plot);
    if (filter_ == 2) filter_ = 0;  // the image must be in the list
    visible_ = true;
    focus_next_ = true;
}

void PlotOutputPanel::ClearPlots() {
    for (auto& plot : plots_) {
        if (plot.texture_id != 0) {
            glDeleteTextures(1, &plot.texture_id);
        }
    }
    plots_.clear();
    selected_plot_index_ = -1;
}

std::vector<int> PlotOutputPanel::Shown() const {
    std::vector<int> shown;
    for (int i = 0; i < static_cast<int>(plots_.size()); ++i)
        if (filter_ == 0 || (filter_ == 1) == !plots_[static_cast<size_t>(i)].python) shown.push_back(i);
    return shown;
}

void PlotOutputPanel::Select(int index) {
    if (index != selected_plot_index_) ResetZoom();  // zoom and pan are per figure
    selected_plot_index_ = index;
}

void PlotOutputPanel::RemoveEntry(int index) {
    if (index < 0 || index >= static_cast<int>(plots_.size())) return;
    if (plots_[static_cast<size_t>(index)].texture_id != 0) glDeleteTextures(1, &plots_[static_cast<size_t>(index)].texture_id);
    plots_.erase(plots_.begin() + index);
    if (selected_plot_index_ >= static_cast<int>(plots_.size())) selected_plot_index_ = static_cast<int>(plots_.size()) - 1;
}

void PlotOutputPanel::AddPythonPlot(scripting::PythonPlot plot) {
    const auto& request = plot.request;
    const std::string title = request.title.empty() ? std::string("Python plot") : request.title;
    const size_t rows = plot.table ? static_cast<size_t>(plot.table->num_rows()) : 0;
    auto found = std::find_if(plots_.begin(), plots_.end(), [&](const PlotEntry& e) { return e.python && e.label == title; });
    if (found == plots_.end()) {
        PlotEntry entry;
        entry.python = true;
        entry.label = title;
        entry.source_text = std::make_shared<std::string>();
        entry.window = std::make_unique<plot::PlotWindow>("python_plot_" + std::to_string(next_window_id_));
        // Cascade: each new window a step down and right of the last.
        const ImVec2 origin = ImGui::GetMainViewport()->WorkPos;
        const float step = 32.0f * static_cast<float>((next_window_id_ - 1) % 8);
        entry.window->first_position = ImVec2(origin.x + 120.0f + step, origin.y + 80.0f + step);
        ++next_window_id_;
        // The window's source line: where the data came from, and how to update it.
        entry.window->draw_header = [text = entry.source_text]() {
            const ui::Tokens& t = ui::CurrentTokens();
            ImGui::PushStyleColor(ImGuiCol_Text, t.success);
            ImGui::Bullet();
            ImGui::PopStyleColor();
            ImGui::SameLine();
            ImGui::TextUnformatted(text->c_str());
            ImGui::SameLine();
            ImGui::TextColored(t.text_faint, "\xC2\xB7 run the script again to update it");
        };
        plots_.push_back(std::move(entry));
        found = plots_.end() - 1;
    }
    found->kind_label = plot::Info(request.spec.kind).label;
    *found->source_text = plot::PythonPlotSourceText(request, rows);
    // The spec first: with its columns chosen, the table keeps them.
    found->window->SetSpec(request.spec);
    found->window->SetArrowTable(title, plot.table, 0, rows);
    found->window->Focus();
    Select(static_cast<int>(found - plots_.begin()));
    if (filter_ == 1) filter_ = 0;
}

void PlotOutputPanel::ClosePythonPlot(const std::string& title) {
    for (int i = 0; i < static_cast<int>(plots_.size()); ++i)
        if (plots_[static_cast<size_t>(i)].python && plots_[static_cast<size_t>(i)].label == title) {
            RemoveEntry(i);
            return;
        }
}

void PlotOutputPanel::Render() {
    // Before the visibility check: a figure published while the window is
    // closed still arrives (it used to be lost), and Python plots keep their
    // own windows.
    PollForNewPlots();
    for (auto& entry : plots_)
        if (entry.window) entry.window->Render();
    if (!visible_) return;

    if (focus_next_) {
        ImGui::SetNextWindowFocus();
        focus_next_ = false;
    }
    // Collapsed or behind another dock tab: skip the body (TOFIX129 0.6).
    if (!ImGui::Begin(GetName(), &visible_, ImGuiWindowFlags_MenuBar)) {
        focused_ = false;
        ImGui::End();
        return;
    }
    focused_ = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);
    const ui::Tokens& t = ui::CurrentTokens();

    // Menu bar
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Show the list", nullptr, &show_thumbnails_);
            ImGui::MenuItem("Show each new plot", nullptr, &auto_scroll_);
            ImGui::Separator();
            if (ImGui::MenuItem("Clear All", nullptr, false, !plots_.empty())) {
                ClearPlots();
            }
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    RenderToolbar();

    const std::vector<int> shown = Shown();
    if (plots_.empty() || shown.empty()) {
        // Empty state
        ImVec2 avail = ImGui::GetContentRegionAvail();
        float text_height = ImGui::GetTextLineHeightWithSpacing() * 3;
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + (avail.y - text_height) / 2);
        ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
        ImGui::TextWrapped("%s", plots_.empty() ? "No plots yet.\n\nplt.show() figures and plots made with the cyxwiz module (import cyxwiz) "
                                                  "from scripts, notebooks and the Console appear here."
                                                : "Nothing of this kind: choose All.");
        ImGui::PopStyleColor();
    } else {
        if (std::find(shown.begin(), shown.end(), selected_plot_index_) == shown.end()) Select(shown.back());
        // Layout: the list on the left (if enabled), the selected one on the right
        bool any_python = false;
        for (const auto& e : plots_) any_python = any_python || e.python;
        if (show_thumbnails_ && (shown.size() > 1 || any_python)) {
            ImGui::BeginChild("##thumbnails", ImVec2(230, 0));
            RenderThumbnails();
            ImGui::EndChild();
            ImGui::SameLine();
        }
        ImGui::BeginChild("##main_plot", ImVec2(0, 0));
        RenderSelectedPlot();
        ImGui::EndChild();
    }

    ImGui::End();
}

void PlotOutputPanel::RenderToolbar() {
    const ui::Tokens& t = ui::CurrentTokens();
    const std::vector<int> shown = Shown();
    const auto at = std::find(shown.begin(), shown.end(), selected_plot_index_);
    const int position = at == shown.end() ? -1 : static_cast<int>(at - shown.begin());
    const bool has = position >= 0;
    const bool figure = has && !plots_[static_cast<size_t>(selected_plot_index_)].python;
    const char* not_image = "A Python plot: zoom, copy and export are in its Plot window";

    if (ui::GhostButton(ICON_FA_CHEVRON_LEFT "##prev", position > 0)) Select(shown[static_cast<size_t>(position - 1)]);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Previous");
    ImGui::SameLine();
    ImGui::TextColored(t.text_dim, "%d / %zu", position + 1, shown.size());
    ImGui::SameLine();
    if (ui::GhostButton(ICON_FA_CHEVRON_RIGHT "##next", has && position + 1 < static_cast<int>(shown.size())))
        Select(shown[static_cast<size_t>(position + 1)]);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Next");

    ImGui::SameLine(0, t.space_lg);
    if (ui::GhostButton(ICON_FA_MINUS "##zoom_out", figure && zoom_level_ > min_zoom_, figure ? nullptr : not_image)) ZoomOut();
    if (figure && ImGui::IsItemHovered()) ImGui::SetTooltip("Zoom out");
    ImGui::SameLine();
    ImGui::TextColored(t.text_dim, "%.0f%%", zoom_level_ * 100.0f);
    ImGui::SameLine();
    if (ui::GhostButton(ICON_FA_PLUS "##zoom_in", figure && zoom_level_ < max_zoom_, figure ? nullptr : not_image)) ZoomIn();
    if (figure && ImGui::IsItemHovered()) ImGui::SetTooltip("Zoom in");
    ImGui::SameLine();
    if (ui::GhostButton("Fit##fit", figure, figure ? nullptr : not_image)) FitToWindow();
    ImGui::SameLine();
    if (ui::GhostButton("100%##actual", figure, figure ? nullptr : not_image)) ActualSize();

    ImGui::SameLine(0, t.space_lg);
    if (ui::GhostButton(ICON_FA_COPY "##copy", figure, figure ? nullptr : not_image)) CopyToClipboard(selected_plot_index_);
    if (figure && ImGui::IsItemHovered()) ImGui::SetTooltip("Copy to clipboard");
    ImGui::SameLine();
    if (ui::GhostButton(ICON_FA_FLOPPY_DISK "##save", figure, figure ? nullptr : not_image)) SaveToFile(selected_plot_index_);
    if (figure && ImGui::IsItemHovered()) ImGui::SetTooltip("Save as PNG");

    ImGui::SameLine(0, t.space_lg);
    if (ui::GhostButton(ICON_FA_XMARK "##close", has)) RemoveEntry(selected_plot_index_);
    if (has && ImGui::IsItemHovered()) ImGui::SetTooltip("Close this one");
    ImGui::SameLine();
    if (ui::GhostButton(ICON_FA_TRASH "##clear_all", !plots_.empty())) ClearPlots();
    if (!plots_.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("Clear all");

    static const char* const kFilter[] = {"All", "Figures", "Python plots"};
    float filter_w = 0;
    for (const char* f : kFilter) filter_w += ImGui::CalcTextSize(f).x + 2 * t.button_padding_small.x + 2;
    if (ui::SameLineRight(filter_w)) ui::SegmentedControl("##plot_filter", kFilter, 3, &filter_);
    ImGui::Dummy(ImVec2(0, t.space_xs));
}

void PlotOutputPanel::RenderSelectedPlot() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (selected_plot_index_ < 0 || selected_plot_index_ >= static_cast<int>(plots_.size())) {
        ImGui::TextColored(t.text_dim, "Nothing selected");
        return;
    }

    auto& plot = plots_[selected_plot_index_];
    if (plot.python) {
        RenderPythonCard(plot);
        return;
    }

    // Show label if present
    if (!plot.label.empty()) ImGui::TextColored(t.accent_text, "%s", plot.label.c_str());

    if (plot.texture_id == 0) {
        ImGui::TextColored(t.text_dim, "The image could not be shown.");
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
        draw_list->AddRectFilled(ImVec2(text_pos.x - 4, text_pos.y - 2), ImVec2(text_pos.x + text_size.x + 4, text_pos.y + text_size.y + 2),
                                 ui::ToU32(ui::WithAlpha(t.bg_panel, 0.85f)), 4.0f);
        draw_list->AddText(text_pos, ui::ToU32(t.text), zoom_text);
    }
}

void PlotOutputPanel::RenderPythonCard(PlotEntry& entry) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::Dummy(ImVec2(0, t.space_lg));
    ImGui::Indent(t.space_lg);
    ImGui::TextColored(t.text_bright, "%s", entry.label.c_str());
    ImGui::TextColored(t.text_dim, "Python plot \xC2\xB7 %s", entry.kind_label.c_str());
    if (entry.source_text) ImGui::TextColored(t.text_dim, "%s", entry.source_text->c_str());
    ImGui::Dummy(ImVec2(0, t.space_sm));
    if (ui::PrimaryButton(entry.window && entry.window->visible ? "Show window" : "Open window")) entry.window->Focus();
    ImGui::SameLine();
    if (ui::SecondaryButton("Close plot")) {
        RemoveEntry(selected_plot_index_);
        ImGui::Unindent(t.space_lg);
        return;
    }
    ImGui::Dummy(ImVec2(0, t.space_sm));
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint,
                       "A live plot: its Plot window changes the type, columns and colours, and exports it. Calling the same "
                       "function with the same title again updates it.");
    ImGui::PopTextWrapPos();
    ImGui::Unindent(t.space_lg);
}

void PlotOutputPanel::RenderThumbnails() {
    const ui::Tokens& t = ui::CurrentTokens();
    const float h = 40.0f, thumb_w = 54.0f, thumb_h = 34.0f;
    for (int i : Shown()) {
        auto& plot = plots_[static_cast<size_t>(i)];
        ImGui::PushID(i);
        const bool is_selected = i == selected_plot_index_;
        if (ImGui::Selectable("##row", is_selected, ImGuiSelectableFlags_AllowOverlap, ImVec2(0, h))) Select(i);
        const ImVec2 lo = ImGui::GetItemRectMin();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 t0(lo.x + 4, lo.y + (h - thumb_h) * 0.5f), t1(t0.x + thumb_w, t0.y + thumb_h);
        if (plot.python) {
            // A live plot: a tile in the accent tone with the chart icon.
            dl->AddRectFilled(t0, t1, ui::ToU32(ui::Mix(t.bg_panel, t.accent, 0.35f)), 4.0f);
            const ImVec2 icon = ImGui::CalcTextSize(ICON_FA_CHART_LINE);
            dl->AddText(ImVec2((t0.x + t1.x - icon.x) * 0.5f, (t0.y + t1.y - icon.y) * 0.5f), ui::ToU32(t.accent_text), ICON_FA_CHART_LINE);
        } else if (plot.texture_id != 0) {
            // The figure, fitted into the tile.
            const float s = std::min(thumb_w / std::max(1, plot.width), thumb_h / std::max(1, plot.height));
            const float w = plot.width * s, hh = plot.height * s;
            const ImVec2 a(t0.x + (thumb_w - w) * 0.5f, t0.y + (thumb_h - hh) * 0.5f);
            dl->AddImage((ImTextureID)(uintptr_t)plot.texture_id, a, ImVec2(a.x + w, a.y + hh));
        }
        const float text_x = t1.x + 8;
        const float line = ImGui::GetTextLineHeight();
        const std::string sub = plot.python ? "Python plot \xC2\xB7 " + plot.kind_label + " \xC2\xB7 live" : "figure";
        dl->PushClipRect(ImVec2(text_x, lo.y), ImVec2(ImGui::GetItemRectMax().x - 4, lo.y + h), true);
        dl->AddText(ImVec2(text_x, lo.y + h * 0.5f - line), ui::ToU32(t.text), plot.label.c_str());
        dl->AddText(ImVec2(text_x, lo.y + h * 0.5f), ui::ToU32(t.text_dim), sub.c_str());
        dl->PopClipRect();
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::TextUnformatted(plot.label.c_str());
            if (plot.python && plot.source_text) ImGui::TextColored(t.text_dim, "%s", plot.source_text->c_str());
            else ImGui::TextColored(t.text_dim, "%d x %d", plot.width, plot.height);
            ImGui::EndTooltip();
        }
        if (ImGui::BeginPopupContextItem("##thumb_context")) {
            RenderPlotContextMenu(i);
            ImGui::EndPopup();
        }
        ImGui::PopID();
    }
}

void PlotOutputPanel::RenderPlotContextMenu(int plot_index) {
    if (plot_index < 0 || plot_index >= static_cast<int>(plots_.size())) return;
    auto& plot = plots_[static_cast<size_t>(plot_index)];
    if (plot.python) {
        if (ImGui::MenuItem(ICON_FA_CHART_LINE " Show window")) plot.window->Focus();
    } else {
        if (ImGui::MenuItem(ICON_FA_COPY " Copy to Clipboard")) CopyToClipboard(plot_index);
        if (ImGui::MenuItem(ICON_FA_FLOPPY_DISK " Save as PNG...")) SaveToFile(plot_index);
    }
    ImGui::Separator();
    if (ImGui::MenuItem(ICON_FA_XMARK " Close")) RemoveEntry(plot_index);
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

    const auto plots = scripting_engine_->TakePublishedPlots();
    auto python = scripting_engine_->TakePythonPlots();
    if (plots.empty() && python.empty()) return;
    for (const auto& plot : plots) AddPlot(plot);
    for (auto& p : python) {
        if (!p.close.empty()) ClosePythonPlot(p.close);
        else AddPythonPlot(std::move(p));
    }
    // Show the window with the new figure or plot
    if (auto_scroll_ && !plots_.empty()) visible_ = true;
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
