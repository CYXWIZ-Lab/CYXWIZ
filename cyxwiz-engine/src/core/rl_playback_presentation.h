#pragma once

// What the Policy Playback panel, the dashboard and the PPO node's
// Properties say about a trained policy and its playback (TOFIX195 S5.3).
// Pure data in and out (no ImGui).

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::rl_playback {

enum class Status { None, Playing, Paused, Stopped, Failed };
// Where the played copy is seen
enum class ViewKind { PluginView, CartPole, None };

struct Inputs {
    Status status = Status::None;
    std::string message;           // why it failed
    std::string environment_key;   // "<provider>:<id>"
    std::string environment_id;    // "CartPole-v1"
    bool discrete = false;
    std::string source;            // PolicySource(...)
    std::string device;            // "CPU": where the policy acts (empty: not said)
    int episode = 0;
    int step = 0;
    double episode_return = 0.0;
    std::vector<double> returns;   // finished episodes, oldest first
    std::vector<float> action;
    bool shown_in_view = false;
    // the plugin view's window title ("MuJoCo Viewport"); empty: the environment has no view
    std::string view_title;
};

struct View {
    Status status = Status::None;
    std::string status_text;       // "Playing"
    std::string title;             // "CartPole-v1"
    std::string source_line;       // "policy from this session's run (88,320 steps)"
    std::string primary_label;     // "Pause", "Resume" or "Play policy"
    bool can_new_episode = false;
    bool can_stop = false;
    std::vector<std::pair<std::string, std::string>> facts;  // Episode, Step, Return, Last returns, Action
    ViewKind view = ViewKind::None;
    std::string view_note;
    std::string view_link;         // "Show the MuJoCo Viewport" (PluginView only)
};

View Build(const Inputs& in);

// The playback's choices, as the panel and Preferences > Simulation show them
// (indices: 0 best / 1 sampled; the PlaybackSpeed order)
inline constexpr const char* kActionChoices[] = {"Best", "Sampled"};
inline constexpr const char* kSpeedChoices[] = {"Real time", "x4", "As fast as possible"};
inline constexpr const char* kActionChoicesHint = "Best: the most likely action; Sampled: drawn as in training";

// Where a policy came from: "this session's run (88,320 steps)" or
// "ppo_cartpole.cyxmodel (88,320 steps)"
std::string PolicySource(const std::string& file_name, int64_t steps);
// The PPO node's and the dashboard's one line: "From this session's run ·
// 88,320 steps · mean return 475.1"; "None yet: Train, or load a saved policy"
std::string PolicyLine(bool has_policy, const std::string& file_name, int64_t steps, float mean_return);
// The action in words: CartPole's "push left" / "push right", another
// discrete choice "choice 2", continuous "[-0.083, 0.512]"
std::string ActionText(const std::string& environment_id, bool discrete, const std::vector<float>& action);

}  // namespace cyxwiz::rl_playback
