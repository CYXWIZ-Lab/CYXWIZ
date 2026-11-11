#pragma once

// Policy Playback (TOFIX195 S5.3): plays a trained PPO policy in one copy of
// its environment. The copy is shown in the plugin's own view when it has one
// (the MuJoCo Viewport); the built-in CartPole is drawn here; the words come
// from core/rl_playback_presentation.

#include "../../core/rl/policy_playback.h"

#include <memory>
#include <string>

namespace cyxwiz {

class PolicyPlaybackPanel {
public:
    // Plays `policy` from the Preferences defaults and shows the panel.
    // file_name: where it was loaded from (empty: this session's run).
    bool Play(std::shared_ptr<rl::TrainedPolicy> policy, std::string file_name, std::string& error);
    void Stop();
    // The policy shown (null before the first Play)
    std::shared_ptr<rl::TrainedPolicy> Policy() const { return playback_.Policy(); }

    void Render();
    bool IsVisible() const { return visible_; }
    void SetVisible(bool visible) { visible_ = visible; }
    bool* VisiblePtr() { return &visible_; }  // View > Panels and the sidebar

private:
    void RenderCartPole(const rl::PolicyPlayback::Snapshot& s, float width, float height);
    void ShowPluginView();

    rl::PolicyPlayback playback_;
    std::string file_name_;
    bool visible_ = false;
    bool was_visible_ = false;  // closing the window stops the playback
    std::string start_error_;   // why Play policy could not start again
    std::string view_title_;    // the plugin view's window title, once shown
    bool view_opened_ = false;  // the plugin's view was opened for this playback
};

}  // namespace cyxwiz
