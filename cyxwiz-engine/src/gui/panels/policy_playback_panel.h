#pragma once

// Policy Playback (TOFIX195 S5.3): plays a trained PPO policy in one copy of
// its environment. The copy is shown in the plugin's own view when it has one
// (the MuJoCo Viewport); the built-in CartPole is drawn here; the words come
// from core/rl_playback_presentation.

#include "../../core/rl/policy_playback.h"
#include "../../core/rl/policy_stress_test.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

class PolicyPlaybackPanel {
public:
    // Plays `policy` from the Preferences defaults and shows the panel.
    // file_name: where it was loaded from (empty: this session's run).
    bool Play(std::shared_ptr<rl::TrainedPolicy> policy, std::string file_name, std::string& error);
    // Stops the playback and a stress test that runs
    void Stop();
    // Stops whichever of them steps an environment `uses` owns (its plugin is stopping)
    void StopIfUses(const std::function<bool(const std::string& environment_key)>& uses);
    // The policy shown (null before the first Play)
    std::shared_ptr<rl::TrainedPolicy> Policy() const { return playback_.Policy(); }

    void Render();
    bool IsVisible() const { return visible_; }
    void SetVisible(bool visible) { visible_ = visible; }
    bool* VisiblePtr() { return &visible_; }  // View > Panels and the sidebar

    // The plugin view that shows an environment's copies: its title (empty:
    // none, or its plugin is not loaded), and opening it
    static std::string ViewTitle(const std::string& environment_key);
    static void ShowView(const std::string& environment_key);
    // Called after every playback this panel starts (it takes the plugin's view)
    void SetOnPlay(std::function<void()> on_play) { on_play_ = std::move(on_play); }

private:
    void RenderPlay(const std::shared_ptr<rl::TrainedPolicy>& policy);
    // The Stress test tab (TOFIX212 P4 board 6): N episodes at each difficulty
    // of one disturbance, the policy against a second saved one
    void RenderStressTest(const std::shared_ptr<rl::TrainedPolicy>& policy);
    void WatchLevel(const std::shared_ptr<rl::TrainedPolicy>& policy, int level);
    void RenderCartPole(const rl::PolicyPlayback::Snapshot& s, float width, float height);
    void ShowPluginView();

    std::function<void()> on_play_;  // SetOnPlay
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
    // The policies of the last test, in its columns' order (Watch plays these,
    // whatever the Compare with field holds now)
    std::vector<std::pair<std::string, std::shared_ptr<rl::TrainedPolicy>>> tested_;
    char compare_path_[512] = "";
    std::shared_ptr<rl::TrainedPolicy> compare_policy_;
    std::string compare_name_;
    std::string compare_error_;
    // From Preferences > Simulation, read when the tab is first drawn
    bool stress_defaults_read_ = false;
    int stress_kind_ = 0;
    int stress_levels_ = 5;
    int stress_episodes_ = 10;
    bool select_play_ = false;  // Watch shows the Play tab
    std::string stress_note_;  // where the CSV went, or why the test did not start
    bool stress_note_failed_ = false;
};

}  // namespace cyxwiz
