// Application side of the plot view hooks (TOFIX134 P1 step 1.4): PNG read
// back from the framebuffer after the frame is drawn, the save dialog and
// the clipboard image. Only the Engine application compiles this file.
#include "plot_capture_gl.h"

#include "plot_view.h"
#include "../panels/output_renderer.h"
#include "../../core/file_dialogs.h"

#include <glad/glad.h>
#include <imgui.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <functional>
#include <vector>

namespace cyxwiz::plot {

namespace {

struct Request {
    ImVec2 min, max;
    ImGuiID viewport;  // the window the plot is drawn in
    std::function<void(std::vector<unsigned char>)> done;
};

void (*g_backend_render_window)(ImGuiViewport*, void*) = nullptr;

void RenderWindowThenCapture(ImGuiViewport* viewport, void* render_arg) {
    g_backend_render_window(viewport, render_arg);
    CompletePlotCaptures(viewport);
}

std::vector<Request>& Queue() {
    static std::vector<Request> queue;
    return queue;
}

void AppendBytes(void* context, void* data, int size) {
    auto* out = static_cast<std::vector<unsigned char>*>(context);
    const auto* bytes = static_cast<const unsigned char*>(data);
    out->insert(out->end(), bytes, bytes + size);
}

}  // namespace

void InstallPlotViewHooks() {
    ViewHooks hooks;
    hooks.capture_png = [](ImVec2 min, ImVec2 max, std::function<void(std::vector<unsigned char>)> done) {
        Queue().push_back({min, max, ImGui::GetWindowViewport()->ID, std::move(done)});
    };
    hooks.save_path = [](const char* title, const char* ext, const std::string& name) -> std::optional<std::string> {
        const std::string upper = std::string(ext) == "png" ? "PNG image" : std::string(ext) == "svg" ? "SVG image" : "CSV file";
        return FileDialogs::SaveFile(title, {{upper.c_str(), ext}}, nullptr, name.c_str());
    };
    hooks.copy_png = [](const std::vector<unsigned char>& png) { return OutputRenderer::CopyImageToClipboard(png); };
    SetViewHooks(std::move(hooks));
}

void CapturePlotsInSeparateWindows() {
    ImGuiPlatformIO& pio = ImGui::GetPlatformIO();
    if (!pio.Renderer_RenderWindow || pio.Renderer_RenderWindow == RenderWindowThenCapture) return;
    g_backend_render_window = pio.Renderer_RenderWindow;
    pio.Renderer_RenderWindow = RenderWindowThenCapture;
}

void CompletePlotCaptures(ImGuiViewport* viewport) {
    auto& queue = Queue();
    if (queue.empty()) return;
    const ImGuiIO& io = ImGui::GetIO();
    const float sx = io.DisplayFramebufferScale.x, sy = io.DisplayFramebufferScale.y;
    const int fb_h = static_cast<int>(viewport->Size.y * sy);
    // This window's requests; one whose window is gone is read from the main window.
    const bool main = viewport == ImGui::GetMainViewport();
    std::vector<Request> requests;
    for (auto it = queue.begin(); it != queue.end();) {
        const bool mine = it->viewport == viewport->ID || (main && !ImGui::FindViewportByID(it->viewport));
        if (mine) {
            requests.push_back(std::move(*it));
            it = queue.erase(it);
        } else {
            ++it;
        }
    }
    for (auto& r : requests) {
        // Plot rectangles are in screen coordinates; the framebuffer starts at the window's corner.
        const int x = std::max(0, static_cast<int>((r.min.x - viewport->Pos.x) * sx));
        const int w = std::max(1, static_cast<int>((r.max.x - r.min.x) * sx));
        const int h = std::max(1, static_cast<int>((r.max.y - r.min.y) * sy));
        const int y = std::max(0, fb_h - static_cast<int>((r.max.y - viewport->Pos.y) * sy));  // GL rows start at the bottom
        std::vector<unsigned char> pixels(static_cast<size_t>(w) * static_cast<size_t>(h) * 4);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(x, y, w, h, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        // Flip to top-down rows and make it opaque (the plot is drawn on the window).
        std::vector<unsigned char> flipped(pixels.size());
        const size_t row = static_cast<size_t>(w) * 4;
        for (int i = 0; i < h; ++i)
            std::copy(pixels.begin() + static_cast<std::ptrdiff_t>(row * static_cast<size_t>(h - 1 - i)),
                      pixels.begin() + static_cast<std::ptrdiff_t>(row * static_cast<size_t>(h - i)),
                      flipped.begin() + static_cast<std::ptrdiff_t>(row * static_cast<size_t>(i)));
        for (size_t i = 3; i < flipped.size(); i += 4) flipped[i] = 255;
        std::vector<unsigned char> png;
        stbi_write_png_to_func(AppendBytes, &png, w, h, 4, flipped.data(), static_cast<int>(row));
        r.done(std::move(png));
    }
}

}  // namespace cyxwiz::plot
