// The Policy Playback panel's words (TOFIX195 S5.3): status, buttons,
// facts, where the copy is seen, the action in words, the policy's line.
#include "../src/core/rl_playback_presentation.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

using namespace cyxwiz::rl_playback;

std::string Fact(const View& v, const std::string& key) {
    for (const auto& [k, value] : v.facts)
        if (k == key) return value;
    return "<missing>";
}

void CheckCartPole() {
    Inputs in;
    in.status = Status::Playing;
    in.environment_key = "cyxwiz.builtin:CartPole-v1";
    in.environment_id = "CartPole-v1";
    in.discrete = true;
    in.source = PolicySource("", 88320);
    in.episode = 3;
    in.step = 412;
    in.episode_return = 412.0;
    in.returns = {500.0, 500.0, 487.0};
    in.action = {1.0f};
    const View v = Build(in);
    Check(v.status_text == "Playing" && v.primary_label == "Pause" && v.can_new_episode && v.can_stop,
          "playing: Pause, New episode and Stop");
    Check(v.title == "CartPole-v1" && v.source_line == "policy from this session's run (88,320 steps)",
          "title and source: " + v.source_line);
    Check(Fact(v, "Episode") == "3" && Fact(v, "Step") == "412" && Fact(v, "Return") == "412", "episode facts");
    Check(Fact(v, "Last returns") == "487 \xC2\xB7 500 \xC2\xB7 500", "last returns, newest first: " + Fact(v, "Last returns"));
    Check(Fact(v, "Action") == "push right", "CartPole's action in words");
    Check(v.view == ViewKind::CartPole, "the built-in CartPole is drawn by the Engine");

    in.device = "CPU";
    Check(Build(in).source_line == "policy from this session's run (88,320 steps) \xC2\xB7 on CPU",
          "the source line says where it acts");
    in.device.clear();
    in.status = Status::Paused;
    Check(Build(in).primary_label == "Resume", "paused: Resume");
    in.status = Status::Stopped;
    const View s = Build(in);
    Check(s.primary_label == "Play policy" && !s.can_stop && !s.can_new_episode, "stopped: Play policy only");
    in.status = Status::None;
    in.returns.clear();
    const View n = Build(in);
    Check(n.facts.empty() && n.status_text == "Not playing", "before playing: no facts");
}

void CheckMuJoCo() {
    Inputs in;
    in.status = Status::Playing;
    in.environment_key = "com.cyxwiz.simulation.mujoco:InvertedPendulum-v5";
    in.environment_id = "InvertedPendulum-v5";
    in.source = PolicySource("InvertedPendulum.cyxmodel", 122880);
    in.action = {-0.0831f};
    in.shown_in_view = true;
    in.view_title = "MuJoCo Viewport";
    in.episode_return = 412.0;
    const View v = Build(in);
    Check(v.view == ViewKind::PluginView && v.view_note == "The MuJoCo Viewport shows the copy being played." &&
              v.view_link == "Show the MuJoCo Viewport",
          "a plugin's copy is in its view, named by the view's title");
    Inputs untitled = in;
    untitled.view_title.clear();
    Check(Build(untitled).view_link == "Show the plugin's view", "a view without a known title");
    Check(Fact(v, "Action") == "[-0.083]" && Fact(v, "Return") == "412.0" && Fact(v, "Last returns") == "None finished yet",
          "continuous facts");
    Check(v.source_line == "policy from InvertedPendulum.cyxmodel (122,880 steps)", "a file's source line");

    in.shown_in_view = false;
    Check(Build(in).view == ViewKind::None, "a copy with no picture: no view");
    in.status = Status::Failed;
    in.message = "the environment is gone";
    const View f = Build(in);
    Check(f.status_text == "Failed" && f.view_note == "the environment is gone", "a failure says why");
}

void CheckLines() {
    Check(PolicyLine(false, "", 0, 0.0f) == "None yet: Train, or load a saved policy", "no policy yet");
    Check(PolicyLine(true, "", 88320, 475.1f) == "From this session's run \xC2\xB7 88,320 steps \xC2\xB7 mean return 475.1",
          "this session's policy");
    Check(PolicyLine(true, "ppo.cyxmodel", 1000, std::numeric_limits<float>::quiet_NaN()) ==
              "From ppo.cyxmodel \xC2\xB7 1,000 steps",
          "a loaded policy without a return");
    Check(ActionText("Acrobot-v1", true, {2.0f}) == "choice 2" && ActionText("x", false, {}) == "None yet" &&
              ActionText("HalfCheetah-v5", false, {0.5f, -1.0f}) == "[0.500, -1.000]" &&
              ActionText("InvertedPendulum-v5", false, {-0.0001f}) == "[0.000]",
          "actions in words");
}

}  // namespace

int main() {
    CheckCartPole();
    CheckMuJoCo();
    CheckLines();
    if (failures == 0) std::printf("test_rl_playback_presentation: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
