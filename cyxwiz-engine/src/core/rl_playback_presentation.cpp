#include "rl_playback_presentation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cyxwiz::rl_playback {

namespace {

constexpr const char* kBuiltinProvider = "cyxwiz.builtin:";

std::string Thousands(int64_t n) {
    std::string digits = std::to_string(n < 0 ? -n : n);
    for (int i = static_cast<int>(digits.size()) - 3; i > 0; i -= 3) digits.insert(static_cast<size_t>(i), ",");
    return n < 0 ? "-" + digits : digits;
}

std::string Number(double v, const char* format) {
    char text[64];
    std::snprintf(text, sizeof(text), format, v);
    return text;
}

bool StartsWith(const std::string& s, const char* prefix) { return s.rfind(prefix, 0) == 0; }

}  // namespace

std::string PolicySource(const std::string& file_name, int64_t steps) {
    return (file_name.empty() ? std::string("this session's run") : file_name) + " (" + Thousands(steps) + " steps)";
}

std::string PolicyLine(bool has_policy, const std::string& file_name, int64_t steps, float mean_return) {
    if (!has_policy) return "None yet: Train, or load a saved policy";
    std::string line = file_name.empty() ? "From this session's run" : "From " + file_name;
    line += " \xC2\xB7 " + Thousands(steps) + " steps";
    if (std::isfinite(mean_return)) line += " \xC2\xB7 mean return " + Number(mean_return, "%.1f");
    return line;
}

std::string ActionText(const std::string& environment_id, bool discrete, const std::vector<float>& action) {
    if (action.empty()) return "None yet";
    if (discrete) {
        const int choice = static_cast<int>(std::lround(action[0]));
        if (environment_id == "CartPole-v1") return choice == 1 ? "push right" : "push left";
        return "choice " + std::to_string(choice);
    }
    std::string text = "[";
    for (size_t i = 0; i < action.size(); ++i)  // no "-0.000"
        text += (i ? ", " : "") + Number(std::fabs(action[i]) < 0.0005f ? 0.0 : action[i], "%.3f");
    return text + "]";
}

View Build(const Inputs& in) {
    View v;
    v.status = in.status;
    v.title = in.environment_id;
    v.source_line = "policy from " + in.source + (in.device.empty() ? "" : " \xC2\xB7 on " + in.device);
    switch (in.status) {
        case Status::Playing: v.status_text = "Playing"; v.primary_label = "Pause"; break;
        case Status::Paused: v.status_text = "Paused"; v.primary_label = "Resume"; break;
        case Status::Stopped: v.status_text = "Stopped"; v.primary_label = "Play policy"; break;
        case Status::Failed: v.status_text = "Failed"; v.primary_label = "Play policy"; break;
        case Status::None: v.status_text = "Not playing"; v.primary_label = "Play policy"; break;
    }
    const bool live = in.status == Status::Playing || in.status == Status::Paused;
    v.can_new_episode = live;
    v.can_stop = live;

    if (in.status != Status::None) {
        v.facts.emplace_back("Episode", std::to_string(std::max(in.episode, 1)));
        v.facts.emplace_back("Step", Thousands(in.step));
        v.facts.emplace_back("Return", Number(in.episode_return, in.discrete ? "%.0f" : "%.1f"));
        std::string last;
        for (size_t n = 0; n < in.returns.size() && n < 5; ++n)
            last += (n ? " \xC2\xB7 " : "") + Number(in.returns[in.returns.size() - 1 - n], in.discrete ? "%.0f" : "%.1f");
        v.facts.emplace_back("Last returns", last.empty() ? "None finished yet" : last);
        v.facts.emplace_back("Action", ActionText(in.environment_id, in.discrete, in.action));
    }

    if (in.shown_in_view) {
        // Any plugin's view, named by its own window title
        v.view = ViewKind::PluginView;
        const std::string name = in.view_title.empty() ? std::string("plugin's view") : in.view_title;
        v.view_note = "The " + name + " shows the copy being played.";
        v.view_link = "Show the " + name;
    } else if (StartsWith(in.environment_key, kBuiltinProvider) && in.environment_id == "CartPole-v1") {
        v.view = ViewKind::CartPole;
        v.view_note = "Drawn from the copy's state: cart position and pole angle.";
    } else {
        v.view = ViewKind::None;
        v.view_note = "This environment has no view: the numbers and the returns show how the policy does.";
    }
    if (in.status == Status::Failed && !in.message.empty()) v.view_note = in.message;
    return v;
}

}  // namespace cyxwiz::rl_playback
