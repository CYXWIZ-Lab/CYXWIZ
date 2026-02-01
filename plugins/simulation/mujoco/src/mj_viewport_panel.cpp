#include "mj_viewport_panel.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

namespace cyxwiz::plugin::mujoco {

void MjViewportPanel::Render(MjEnvManager& env, MjRenderer& renderer, bool* visible) {
    if (!visible || !*visible) return;

    ImGui::SetNextWindowSize(ImVec2(680, 540), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("MuJoCo Viewport", visible,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
        ImGui::End();
        return;
    }

    if (!env.IsLoaded()) {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                           "No environment loaded. Load an MJCF model to begin.");
        ImGui::End();
        return;
    }

    if (!renderer.IsInitialized()) {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                           "Renderer not initialized.");
        ImGui::End();
        return;
    }

    RenderToolbar(env, renderer);
    ImGui::Separator();
    RenderViewport(env, renderer);

    if (show_settings_) {
        ImGui::Separator();
        RenderSettings(renderer);
    }

    ImGui::End();
}

// =============================================================================
// Toolbar
// =============================================================================

void MjViewportPanel::RenderToolbar(MjEnvManager& env, MjRenderer& renderer) {
    // Play/Pause
    if (playing_) {
        if (ImGui::Button("Pause")) {
            playing_ = false;
        }
    } else {
        if (ImGui::Button("Play")) {
            playing_ = true;
        }
    }

    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        env.Reset();
        renderer.ResetCamera(env.GetModel());
    }

    ImGui::SameLine();
    if (ImGui::Button("Step")) {
        // Single step with zero action
        std::vector<float> zero_action(env.GetActionDim(), 0.0f);
        env.Step(zero_action);
    }

    ImGui::SameLine();
    ImGui::Text("|");
    ImGui::SameLine();

    // Toggle render options
    bool enabled = renderer.IsEnabled();
    if (ImGui::Checkbox("Render", &enabled)) {
        renderer.SetEnabled(enabled);
    }

    ImGui::SameLine();
    if (ImGui::Button("Settings")) {
        show_settings_ = !show_settings_;
    }

    // Status line
    ImGui::SameLine(ImGui::GetWindowWidth() - 280);
    ImGui::Text("Step: %d | Reward: %.2f | Ep: %d",
                env.GetCurrentStep(),
                env.GetEpisodeReward(),
                env.GetEpisodeCount());
}

// =============================================================================
// Viewport (3D rendered image)
// =============================================================================

void MjViewportPanel::RenderViewport(MjEnvManager& env, MjRenderer& renderer) {
    // Auto-step if playing
    if (playing_ && env.IsLoaded()) {
        std::vector<float> zero_action(env.GetActionDim(), 0.0f);
        for (int i = 0; i < steps_per_frame_; ++i) {
            auto result = env.Step(zero_action);
            if (result.terminated || result.truncated) {
                env.Reset();
            }
        }
    }

    // Get available viewport size
    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (show_settings_) {
        avail.y -= 100.0f;  // Reserve space for settings
    }
    if (avail.x < 10.0f || avail.y < 10.0f) return;

    // Update renderer resolution if viewport size changed
    int new_w = static_cast<int>(avail.x);
    int new_h = static_cast<int>(avail.y);
    if (new_w != renderer.GetWidth() || new_h != renderer.GetHeight()) {
        renderer.SetResolution(new_w, new_h);
    }

    // Render frame
    unsigned int tex_id = renderer.RenderFrame(env.GetModel(), env.GetData());
    if (tex_id == 0) {
        ImGui::TextDisabled("Rendering disabled or failed.");
        return;
    }

    // Display texture (flip V because MuJoCo outputs bottom-up)
    ImVec2 uv0(0.0f, 1.0f);  // Bottom-left of texture
    ImVec2 uv1(1.0f, 0.0f);  // Top-right of texture
    ImGui::Image(static_cast<ImTextureID>(tex_id),
                 avail, uv0, uv1);

    // Mouse camera controls when hovering viewport
    if (ImGui::IsItemHovered()) {
        ImGuiIO& io = ImGui::GetIO();

        // Left mouse button: rotate
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            ImVec2 delta = io.MouseDelta;
            renderer.RotateCamera(delta.x, delta.y);
        }

        // Right mouse button: pan
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
            ImVec2 delta = io.MouseDelta;
            renderer.PanCamera(delta.x, delta.y);
        }

        // Scroll wheel: zoom
        if (io.MouseWheel != 0.0f) {
            renderer.ZoomCamera(io.MouseWheel);
        }
    }
}

// =============================================================================
// Settings
// =============================================================================

void MjViewportPanel::RenderSettings(MjRenderer& renderer) {
    ImGui::Text("Render Options:");

    if (ImGui::Button("Contacts")) renderer.ToggleContactPoints();
    ImGui::SameLine();
    if (ImGui::Button("Forces")) renderer.ToggleContactForces();
    ImGui::SameLine();
    if (ImGui::Button("Wireframe")) renderer.ToggleWireframe();
    ImGui::SameLine();
    if (ImGui::Button("Joints")) renderer.ToggleJoints();
    ImGui::SameLine();
    if (ImGui::Button("Actuators")) renderer.ToggleActuators();

    ImGui::SliderFloat("Speed", &playback_speed_, 0.1f, 10.0f, "%.1fx");
    steps_per_frame_ = static_cast<int>(playback_speed_);
    if (steps_per_frame_ < 1) steps_per_frame_ = 1;

    ImGui::Text("Resolution: %d x %d", renderer.GetWidth(), renderer.GetHeight());
}

} // namespace cyxwiz::plugin::mujoco
