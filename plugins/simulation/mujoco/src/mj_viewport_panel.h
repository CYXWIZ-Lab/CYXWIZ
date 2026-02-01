#pragma once

// MjViewportPanel — ImGui panel displaying the MuJoCo 3D viewport.
//
// Renders MuJoCo scene as a texture via MjRenderer and displays it
// with mouse camera controls, playback buttons, and render toggles.

#include "mj_renderer.h"
#include "mj_env_manager.h"

namespace cyxwiz::plugin::mujoco {

class MjViewportPanel {
public:
    MjViewportPanel() = default;
    ~MjViewportPanel() = default;

    // Render the panel. Call each frame from IPanelProvider::RenderPanel.
    // |env| and |renderer| must be valid. |visible| controls panel open state.
    void Render(MjEnvManager& env, MjRenderer& renderer, bool* visible);

    // Playback state
    bool IsPlaying() const { return playing_; }
    void SetPlaying(bool playing) { playing_ = playing; }

private:
    void RenderToolbar(MjEnvManager& env, MjRenderer& renderer);
    void RenderViewport(MjEnvManager& env, MjRenderer& renderer);
    void RenderSettings(MjRenderer& renderer);

    bool playing_ = false;
    bool show_settings_ = false;
    float playback_speed_ = 1.0f;
    int steps_per_frame_ = 1;

    // Resize debouncing — only apply after user stops resizing
    int pending_w_ = 0;
    int pending_h_ = 0;
    int stable_frames_ = 0;  // Frames since last size change
    static constexpr int kResizeDebounceFrames = 10;
};

} // namespace cyxwiz::plugin::mujoco
