#include "policy_playback_panel.h"

#include "../../core/engine_config.h"
#include "../../core/execution_device_preferences.h"
#include "../../core/rl_device_presentation.h"
#include "../../core/rl_playback_presentation.h"
#include "../../plugin/engine_plugin_context.h"
#include "../../plugin/registries/plugin_environment_registry.h"
#include "../../plugin/registries/plugin_panel_registry.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>

namespace cyxwiz {

namespace {

namespace pb = rl_playback;

pb::Status ToStatus(rl::PolicyPlayback::State s) {
    switch (s) {
        case rl::PolicyPlayback::State::Playing: return pb::Status::Playing;
        case rl::PolicyPlayback::State::Paused: return pb::Status::Paused;
        case rl::PolicyPlayback::State::Stopped: return pb::Status::Stopped;
        case rl::PolicyPlayback::State::Failed: return pb::Status::Failed;
        default: return pb::Status::None;
    }
}

ImU32 Colour(const ImVec4& c) { return ImGui::GetColorU32(c); }

}  // namespace

bool PolicyPlaybackPanel::Play(std::shared_ptr<rl::TrainedPolicy> policy, std::string file_name, std::string& error) {
    const auto& config = core::EngineConfig::Instance();
    if (!playback_.Start(std::move(policy), static_cast<rl::PlaybackSpeed>(config.PolicyPlaybackSpeed()),
                         config.PolicyPlaybackBestActions(), error))
        return false;
    file_name_ = std::move(file_name);
    view_opened_ = false;
    visible_ = true;
    ImGui::SetWindowFocus("Policy Playback");
    return true;
}

void PolicyPlaybackPanel::Stop() { playback_.Stop(); }

void PolicyPlaybackPanel::LookUpViewTitle(const std::string& environment_key) {
    // Kept once found; an environment without one (yet: its plugin may load later) is asked again
    if (environment_key == view_title_key_ && !view_title_.empty()) return;
    view_title_key_ = environment_key;
    view_title_.clear();
    plugin::PluginEnvironmentRegistry::Environment env;
    if (!plugin::PluginEnvironmentRegistry::Instance().Find(environment_key, env) || env.info.view_panel_id.empty())
        return;
    // The view's own title, from the plugin that registered it
    for (const auto& panel : plugin::PluginPanelRegistry::Instance().GetAllPanels())
        if (panel.panel_id == env.info.view_panel_id) view_title_ = panel.title;
}

void PolicyPlaybackPanel::ShowPluginView() {
    const auto policy = playback_.Policy();
    if (!policy) return;
    plugin::PluginEnvironmentRegistry::Environment env;
    if (!plugin::PluginEnvironmentRegistry::Instance().Find(policy->config.environment_key, env) ||
        env.info.view_panel_id.empty())
        return;
    plugin::PluginPanelRegistry::Instance().ShowPanel(env.plugin_id, env.info.view_panel_id);
}

void PolicyPlaybackPanel::RenderCartPole(const rl::PolicyPlayback::Snapshot& s, float width, float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const ImVec2 p1(p0.x + width, p0.y + height);
    dl->AddRectFilled(p0, p1, Colour(t.bg_input), 6.0f);
    ImGui::Dummy(ImVec2(width, height));
    // Gymnasium's CartPole: the track is +-2.4 m, the pole 1 m (2 x half length)
    const float scale = width / 5.6f;
    const float base = p1.y - 44.0f;
    const float cx = p0.x + width * 0.5f;
    dl->AddRectFilled(ImVec2(p0.x + 8.0f, base), ImVec2(p1.x - 8.0f, base + 2.0f), Colour(t.border_soft));
    for (const float x : {-2.4f, 0.0f, 2.4f}) {
        const float px = cx + x * scale;
        dl->AddRectFilled(ImVec2(px - 1.0f, base + 2.0f), ImVec2(px + 1.0f, base + 9.0f), Colour(t.text_faint));
        char label[16];
        std::snprintf(label, sizeof(label), "%.1f", x);
        const ImVec2 size = ImGui::CalcTextSize(label);
        dl->AddText(ImVec2(px - size.x * 0.5f, base + 12.0f), Colour(t.text_dim), label);
    }
    if (s.observation.size() < 4) return;
    const float x = std::clamp(s.observation[0], -2.6f, 2.6f);
    const float angle = s.observation[2];
    const float px = cx + x * scale;
    const float cart_w = 0.5f * scale, cart_h = 0.3f * scale;
    dl->AddRectFilled(ImVec2(px - cart_w * 0.5f, base - cart_h), ImVec2(px + cart_w * 0.5f, base), Colour(t.bg_raised),
                      4.0f);
    const ImVec2 hinge(px, base - cart_h * 0.6f);
    const float pole = 1.0f * scale;
    dl->AddLine(hinge, ImVec2(hinge.x + std::sin(angle) * pole, hinge.y - std::cos(angle) * pole), Colour(t.accent_text),
                std::max(4.0f, scale * 0.08f));
    dl->AddCircleFilled(hinge, std::max(3.0f, scale * 0.04f), Colour(t.text_bright));
}

void PolicyPlaybackPanel::Render() {
    if (!visible_) {
        // Closed (its X): nothing plays unseen; Play opens it again
        if (was_visible_) playback_.Stop();
        was_visible_ = false;
        return;
    }
    was_visible_ = true;
    ImGui::SetNextWindowSize(ImVec2(900, 480), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Policy Playback", &visible_)) {
        ImGui::End();
        return;
    }
    const ui::Tokens& t = ui::CurrentTokens();
    const auto policy = playback_.Policy();
    if (!policy) {
        ImGui::TextColored(t.text_dim, "No policy is playing. Train an RL graph, or load a saved policy: "
                                       "the PPO node's Properties, the dashboard after a run, or File > Open Model File.");
        ImGui::End();
        return;
    }
    const auto s = playback_.Get();
    pb::Inputs in;
    in.status = ToStatus(s.state);
    in.message = s.message;
    in.environment_key = policy->config.environment_key;
    in.environment_id = policy->config.environment.id;
    in.discrete = policy->config.environment.discrete_actions;
    in.source = pb::PolicySource(file_name_, policy->steps);
    in.device = rl_device::BackendWord(ExecutionDeviceSelectionBackendName(policy->device_type));
    in.episode = s.episode;
    in.step = s.step;
    in.episode_return = s.episode_return;
    in.returns = s.returns;
    in.action = s.action;
    in.shown_in_view = s.shown_in_view;
    LookUpViewTitle(policy->config.environment_key);
    in.view_title = view_title_;
    const pb::View v = pb::Build(in);
    // The copy moves: draw every frame (the plugin's view too)
    if (v.status == pb::Status::Playing) plugin::EnginePluginContext::KeepFullFrameRate();
    if (s.shown_in_view && !view_opened_) {
        view_opened_ = true;
        ShowPluginView();
    }

    // Header: status, title, source; Pause/Resume, New episode, Stop at the right
    const ImVec4 tone = v.status == pb::Status::Playing ? t.success
                        : v.status == pb::Status::Failed ? t.error
                        : v.status == pb::Status::Paused ? t.accent_text
                                                         : t.pending;
    ui::StatusPill("##playback_status", v.status_text.c_str(), tone);
    ImGui::SameLine();
    ImGui::TextColored(t.text_bright, "%s", v.title.c_str());
    ImGui::SameLine();
    ImGui::TextColored(t.text_dim, "%s", v.source_line.c_str());
    const float gap = ImGui::GetStyle().ItemSpacing.x;
    const std::string primary = std::string(v.primary_label == "Pause"    ? ICON_FA_PAUSE " "
                                            : v.primary_label == "Resume" ? ICON_FA_PLAY " "
                                                                          : ICON_FA_PLAY " ") +
                                v.primary_label;
    const char* new_episode = "New episode";
    const char* stop = ICON_FA_STOP " Stop";
    float right = ui::ButtonWidth(primary.c_str(), ui::ButtonSize::Regular);
    if (v.can_new_episode) right += gap + ui::ButtonWidth(new_episode, ui::ButtonSize::Regular);
    if (v.can_stop) right += gap + ui::ButtonWidth(stop, ui::ButtonSize::Regular);
    ImGui::SameLine();
    if (ImGui::GetContentRegionAvail().x < right) ImGui::NewLine();  // narrow: the buttons on a line of their own
    ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), ImGui::GetWindowContentRegionMax().x - right));
    if (ui::PrimaryButton(primary.c_str(), true, nullptr, ui::ButtonSize::Regular)) {
        if (v.status == pb::Status::Playing) playback_.Pause();
        else if (v.status == pb::Status::Paused) playback_.Resume();
        else {
            start_error_.clear();
            if (!playback_.Start(policy, playback_.Speed(), playback_.Best(), start_error_))
                spdlog::error("Policy playback: {}", start_error_);
            view_opened_ = false;
        }
    }
    if (v.can_new_episode) {
        ImGui::SameLine();
        if (ui::SecondaryButton(new_episode, true, nullptr, ui::ButtonSize::Regular)) playback_.NewEpisode();
    }
    if (v.can_stop) {
        ImGui::SameLine();
        if (ui::DangerButton(stop, true, nullptr, ui::ButtonSize::Regular)) playback_.Stop();
    }

    // The view at the left, the facts and choices at the right (wide enough
    // for the speed choice)
    const auto& kActions = pb::kActionChoices;
    const auto& kSpeeds = pb::kSpeedChoices;
    const float label_w = 70.0f;
    const float side_wanted = std::max(300.0f, label_w + ui::SegmentedControlWidth(kSpeeds, 3) + gap);
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    // Narrow: the facts and choices go under the view instead of beside it
    const bool stacked = avail.x < side_wanted + 200.0f + gap;
    const float side = stacked ? avail.x : side_wanted;
    const float view_w = stacked ? avail.x : avail.x - side - gap;
    ImGui::BeginGroup();
    if (v.view == pb::ViewKind::CartPole) {
        RenderCartPole(s, view_w, std::max(160.0f, std::min(avail.y - 60.0f, view_w * 0.4f)));
    }
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + view_w);
    if (!start_error_.empty()) ImGui::TextColored(t.error, "%s", start_error_.c_str());
    ImGui::TextColored(v.status == pb::Status::Failed ? t.error : t.text_dim, "%s", v.view_note.c_str());
    ImGui::PopTextWrapPos();
    if (v.view == pb::ViewKind::PluginView && ui::LinkButton(v.view_link.c_str())) ShowPluginView();
    if (v.view == pb::ViewKind::None && !s.returns.empty()) {
        std::vector<float> returns;
        for (const double r : s.returns) returns.push_back(static_cast<float>(r));
        ImGui::PushStyleColor(ImGuiCol_FrameBg, t.bg_input);
        ImGui::PushStyleColor(ImGuiCol_PlotLines, t.accent_text);
        ImGui::PlotLines("##returns", returns.data(), static_cast<int>(returns.size()), 0, "return per episode", FLT_MAX,
                         FLT_MAX, ImVec2(view_w, 90.0f));
        ImGui::PopStyleColor(2);
    }
    ImGui::EndGroup();
    if (!stacked) ImGui::SameLine();
    ImGui::BeginGroup();
    if (ImGui::BeginTable("##playback_facts", 2, ImGuiTableFlags_SizingFixedFit, ImVec2(side, 0.0f))) {
        for (const auto& [label, value] : v.facts) {
            ImGui::TableNextColumn();
            ImGui::TextColored(t.text_dim, "%s", label.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(value.c_str());
        }
        ImGui::EndTable();
    }
    ImGui::Spacing();
    int actions = playback_.Best() ? 0 : 1;
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Actions");
    ImGui::SameLine(label_w + ImGui::GetCursorStartPos().x);
    if (ui::SegmentedControl("##playback_actions", kActions, 2, &actions)) playback_.SetBest(actions == 0);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Best: the policy's most likely action (argmax, or the Gaussian mean). "
                          "Sampled: drawn as in training.");
    int speed = static_cast<int>(playback_.Speed());
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Speed");
    ImGui::SameLine(label_w + ImGui::GetCursorStartPos().x);
    if (ui::SegmentedControl("##playback_speed", kSpeeds, 3, &speed))
        playback_.SetSpeed(static_cast<rl::PlaybackSpeed>(speed));
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Real time: the environment's own step time. The defaults are in Preferences > Simulation.");
    ImGui::EndGroup();
    ImGui::End();
}

}  // namespace cyxwiz
