#pragma once

// Policy Playback (TOFIX195 S5.3): plays a trained PPO policy in one copy of
// its environment. The copy is shown in the plugin's own view when it has one
// (the MuJoCo Viewport); the built-in CartPole is drawn here; the words come
// from core/rl_playback_presentation.

#include "../../core/rl/policy_playback.h"
#include "../../core/rl/policy_stress_test.h"

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
    void RenderPlay(const std::shared_ptr<rl::TrainedPolicy>& policy);
    // The Stress test tab (TOFIX212 P4 board 6): N episodes at each difficulty
    // of one disturbance, the policy against a second saved one
    void RenderStressTest(const std::shared_ptr<rl::TrainedPolicy>& policy);
    void WatchLevel(const std::shared_ptr<rl::TrainedPolicy>& policy, int level);
    void RenderCartPole(const rl::PolicyPlayback::Snapshot& s, float width, float height);
    void ShowPluginView();

    rl::PolicyPlayback playback_;
    std::string file_name_;
    bool visible_ = false;
    bool was_visible_ = false;  // closing the window stops the playback
    std::string start_error_;   // why Play policy could not start again
    // The plugin view's window title for the policy's environment (empty: it
    // has none), kept once found
    std::string view_title_;
    std::string view_title_key_;
    void LookUpViewTitle(const std::string& environment_key);
    bool view_opened_ = false;  // the plugin's view was opened for this playback

    // Stress test
    rl::PolicyStressTest stress_;
    std::shared_ptr<rl::TrainedPolicy> stress_policy_;  // the one under test (kept: playback may switch)
    std::string stress_policy_name_;
    char compare_path_[512] = "";
    std::shared_ptr<rl::TrainedPolicy> compare_policy_;
    std::string compare_name_;
    std::string compare_error_;
    int stress_kind_ = 0;
    int stress_levels_ = 5;
    int stress_episodes_ = 10;
    std::string stress_note_;  // where the CSV went, or why the test did not start
    bool stress_note_failed_ = false;
};

}  // namespace cyxwiz
