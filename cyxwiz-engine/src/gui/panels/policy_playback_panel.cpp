#include "policy_playback_panel.h"

#include "../../core/engine_config.h"
#include "../../core/project_manager.h"
#include "../../core/execution_device_preferences.h"
#include "../../core/rl_device_presentation.h"
#include "../../core/rl_playback_presentation.h"
#include "../../plugin/engine_plugin_context.h"
#include "../../plugin/registries/plugin_environment_registry.h"
#include "../../plugin/registries/plugin_panel_registry.h"
#include "../icons.h"
#include "../plot/plot_style.h"
#include "../ui_buttons.h"
#include "../ui_widgets.h"
#include "../ui_tokens.h"

#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>

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
    ArmGhost(policy, file_name);
    if (!playback_.Start(std::move(policy), static_cast<rl::PlaybackSpeed>(config.PolicyPlaybackSpeed()),
                         config.PolicyPlaybackBestActions(), error))
        return false;
    if (on_play_) on_play_();
    file_name_ = std::move(file_name);
    view_opened_ = false;
    visible_ = true;
    ImGui::SetWindowFocus("Policy Playback");
    return true;
}

void PolicyPlaybackPanel::Stop() {
    playback_.Stop();
    stress_.Stop();
}

void PolicyPlaybackPanel::StopIfUses(const std::function<bool(const std::string& environment_key)>& uses) {
    if (const auto policy = playback_.Policy(); policy && uses(policy->config.environment_key)) playback_.Stop();
    for (const auto& [name, policy] : tested_) {
        if (policy && uses(policy->config.environment_key)) {
            stress_.Stop();
            break;
        }
    }
}

void PolicyPlaybackPanel::LookUpViewTitle(const std::string& environment_key) {
    // Kept once found; an environment without one (yet: its plugin may load later) is asked again
    if (environment_key == view_title_key_ && !view_title_.empty()) return;
    view_title_key_ = environment_key;
    view_title_ = ViewTitle(environment_key);
}

std::string PolicyPlaybackPanel::ViewTitle(const std::string& environment_key) {
    plugin::PluginEnvironmentRegistry::Environment env;
    if (!plugin::PluginEnvironmentRegistry::Instance().Find(environment_key, env) || env.info.view_panel_id.empty())
        return {};
    // The view's own title, from the plugin that registered it
    for (const auto& panel : plugin::PluginPanelRegistry::Instance().GetAllPanels())
        if (panel.panel_id == env.info.view_panel_id) return panel.title;
    return {};
}

void PolicyPlaybackPanel::ShowView(const std::string& environment_key) {
    plugin::PluginEnvironmentRegistry::Environment env;
    if (plugin::PluginEnvironmentRegistry::Instance().Find(environment_key, env) && !env.info.view_panel_id.empty())
        plugin::PluginPanelRegistry::Instance().ShowPanel(env.plugin_id, env.info.view_panel_id);
}

void PolicyPlaybackPanel::ShowPluginView() {
    if (const auto policy = playback_.Policy()) ShowView(policy->config.environment_key);
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
    if (ImGui::BeginTabBar("##playback_tabs")) {
        const ImGuiTabItemFlags play_flags = select_play_ ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;
        select_play_ = false;
        if (ImGui::BeginTabItem("Play", nullptr, play_flags)) {
            RenderPlay(policy);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Stress test")) {
            RenderStressTest(policy);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

void PolicyPlaybackPanel::RenderPlay(const std::shared_ptr<rl::TrainedPolicy>& policy) {
    const ui::Tokens& t = ui::CurrentTokens();
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
        else
            StartAgain(policy);
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

    ImGui::Spacing();
    RenderGhost(policy, s, std::max(label_w, ImGui::CalcTextSize("Compare with").x + gap * 2.0f));
}

void PolicyPlaybackPanel::StartAgain(const std::shared_ptr<rl::TrainedPolicy>& policy) {
    start_error_.clear();
    ArmGhost(policy, file_name_);
    if (!playback_.Start(policy, playback_.Speed(), playback_.Best(), start_error_))
        spdlog::error("Policy playback: {}", start_error_);
    else if (on_play_)
        on_play_();
    view_opened_ = false;
}

void PolicyPlaybackPanel::ArmGhost(const std::shared_ptr<rl::TrainedPolicy>& policy, const std::string& file_name) {
    if (!ghost_policy_ || !policy) return;
    if (ghost_policy_->config.environment_key != policy->config.environment_key) {
        ghost_policy_.reset();  // another environment: it cannot play beside this one
        return;
    }
    playback_.SetGhost(ghost_policy_, pb::PolicyName(ghost_file_), pb::PolicyName(file_name));
}

void PolicyPlaybackPanel::RenderGhost(const std::shared_ptr<rl::TrainedPolicy>& policy,
                                      const rl::PolicyPlayback::Snapshot& s, float label_w) {
    const ui::Tokens& t = ui::CurrentTokens();
    RenderCompareWith(policy, label_w, true);
    const bool ghost_on = ghost_policy_ != nullptr;
    ImGui::SetCursorPosX(label_w + ImGui::GetCursorStartPos().x);
    if (ghost_on) {
        if (ui::SecondaryButton("Remove the ghost", true, nullptr, ui::ButtonSize::Regular)) {
            ghost_policy_.reset();
            playback_.RemoveGhost();
        }
    } else if (ui::SecondaryButton("Show as a ghost", compare_policy_ != nullptr,
                                   "Load a saved policy of the same environment in Compare with first.",
                                   ui::ButtonSize::Regular)) {
        ghost_policy_ = compare_policy_;
        ghost_file_ = compare_name_;
        StartAgain(policy);  // both start on one seed
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Plays the second policy beside this one, on the same ground and seed, drawn see-through "
                          "in the plugin's view. Starts the playback again.");
    const std::string line =
        pb::GhostLine(s.ghost, s.ghost_episode, s.ghost_step, s.ghost_return, s.ghost_returns, s.ghost_note);
    if (!line.empty() && (ghost_on || !s.ghost_note.empty())) {
        ImGui::SameLine();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(s.ghost ? t.text_dim : t.caution, "%s", line.c_str());
        ImGui::PopTextWrapPos();
    }
}

void PolicyPlaybackPanel::RenderCompareWith(const std::shared_ptr<rl::TrainedPolicy>& policy, float label_w,
                                            bool enabled) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Compare with");
    ImGui::SameLine(label_w + ImGui::GetCursorStartPos().x);
    ImGui::SetNextItemWidth(std::max(160.0f, ImGui::GetContentRegionAvail().x -
                                                 2.0f * ui::ButtonWidth("Clear", ui::ButtonSize::Regular) - 20.0f));
    ImGui::InputTextWithHint("##compare_path", "a saved .cyxmodel (optional)", compare_path_, sizeof(compare_path_));
    ImGui::SameLine();
    if (ui::SecondaryButton("Load", compare_path_[0] != '\0' && enabled, "Type the path of a saved policy first.",
                            ui::ButtonSize::Regular)) {
        compare_error_.clear();
        std::string error;
        std::unique_ptr<rl::TrainedPolicy> loaded = rl::LoadTrainedPolicy(compare_path_, error);
        if (!loaded) {
            compare_error_ = error;
            compare_policy_.reset();
        } else if (loaded->config.environment_key != policy->config.environment_key) {
            compare_error_ = "It plays " + loaded->config.environment.id + ", not " + policy->config.environment.id + ".";
            compare_policy_.reset();
        } else {
            compare_policy_ = std::move(loaded);
            compare_name_ = std::filesystem::path(compare_path_).filename().string();
        }
    }
    if (compare_policy_) {
        ImGui::SameLine();
        if (ui::LinkButton("Clear")) compare_policy_.reset();
    }
    if (!compare_error_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.error, "%s", compare_error_.c_str());
        ImGui::PopTextWrapPos();
    } else if (compare_policy_) {
        ImGui::SetCursorPosX(label_w + ImGui::GetCursorStartPos().x);
        ImGui::TextColored(t.text_dim, "%s", pb::PolicySource(compare_name_, compare_policy_->steps).c_str());
    }
}

void PolicyPlaybackPanel::WatchLevel(const std::shared_ptr<rl::TrainedPolicy>& policy, int level) {
    // The level's first episode again: copy 0 of its batch draws from (seed, 0) alone
    const auto& spec = stress_.Spec();
    playback_.SetConditions(
        rl::StressSettings(policy->config.environment_settings, spec.kind, rl::StressDifficulty(spec, level)),
        rl::StressSeed(spec, level));
    start_error_.clear();
    std::string played_file = file_name_;
    for (const auto& [name, tested] : tested_)
        if (tested == policy) played_file = name;
    ArmGhost(policy, played_file);
    if (!playback_.Start(policy, playback_.Speed(), true, start_error_)) spdlog::error("Policy playback: {}", start_error_);
    else if (on_play_)
        on_play_();
    view_opened_ = false;
    select_play_ = true;  // the Play tab, which opens the plugin's view
    for (const auto& [name, tested] : tested_)
        if (tested == policy) file_name_ = name;
}

void PolicyPlaybackPanel::RenderStressTest(const std::shared_ptr<rl::TrainedPolicy>& policy) {
    const ui::Tokens& t = ui::CurrentTokens();
    const auto snap = stress_.Get();
    const bool running = snap.state == rl::PolicyStressTest::State::Running;
    const float label_w = 120.0f;
    const auto label = [&](const char* text) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, "%s", text);
        ImGui::SameLine(label_w + ImGui::GetCursorStartPos().x);
    };
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim,
                       "Plays N episodes at each difficulty of one disturbance, with the policy's best actions, and "
                       "compares it with a second saved policy. Each level has its own seed, so both policies meet the "
                       "same grounds and the same pushes.");
    ImGui::PopTextWrapPos();
    ImGui::Spacing();

    label("Policy");
    ImGui::TextUnformatted(stress_policy_ == policy || !stress_policy_ ? pb::PolicySource(file_name_, policy->steps).c_str()
                                                                       : stress_policy_name_.c_str());
    RenderCompareWith(policy, label_w, !running);

    if (!stress_defaults_read_) {
        stress_defaults_read_ = true;
        const auto d = core::EngineConfig::Instance().PolicyStressTestDefaults();
        stress_kind_ = d.kind;
        stress_levels_ = d.levels;
        stress_episodes_ = d.episodes;
    }
    label("Disturbance");
    ui::SegmentedControl("##stress_kind", rl::StressKindNames(), rl::kStressKindCount, &stress_kind_);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("A ground at the level's difficulty, or pushes every 1.5 s of up to 4 m/s times the difficulty");
    label("Levels");
    ui::NumberField("##stress_levels", &stress_levels_, 2, 21, 1, 60.0f, "Difficulty 0 to 1 in this many levels");
    ImGui::SameLine();
    ImGui::TextColored(t.text_dim, "episodes a level");
    ImGui::SameLine();
    ui::NumberField("##stress_episodes", &stress_episodes_, 1, 100, 1, 60.0f);
    ImGui::SameLine();
    if (ui::LinkButton("Keep these as the default")) {
        auto& config = core::EngineConfig::Instance();
        config.SetPolicyStressTestDefaults({stress_kind_, stress_levels_, stress_episodes_});
        config.Save();
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Preferences > Simulation keeps them too");
    ImGui::Spacing();
    if (running) {
        if (ui::DangerButton(ICON_FA_STOP " Stop test", true, nullptr, ui::ButtonSize::Regular)) stress_.Stop();
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "%d of %d levels", snap.done, snap.total);
        plugin::EnginePluginContext::KeepFullFrameRate();
    } else if (ui::PrimaryButton("Run test", true, nullptr, ui::ButtonSize::Regular)) {
        std::vector<std::pair<std::string, std::shared_ptr<rl::TrainedPolicy>>> policies;
        stress_policy_ = policy;
        stress_policy_name_ = file_name_.empty() ? std::string("this session's run") : file_name_;
        policies.push_back({stress_policy_name_, policy});
        if (compare_policy_) policies.push_back({compare_name_, compare_policy_});
        tested_ = policies;
        rl::StressTestSpec spec;
        spec.kind = static_cast<rl::StressKind>(stress_kind_);
        spec.levels = stress_levels_;
        spec.episodes = stress_episodes_;
        std::string error;
        playback_.Stop();  // one user of the networks at a time
        stress_note_failed_ = !stress_.Start(std::move(policies), spec, error);
        stress_note_ = stress_note_failed_ ? error : std::string();
    }
    if (snap.state == rl::PolicyStressTest::State::Failed) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.error, "%s", snap.message.c_str());
        ImGui::PopTextWrapPos();
    }
    if (!stress_note_.empty()) ImGui::TextColored(stress_note_failed_ ? t.error : t.success, "%s", stress_note_.c_str());
    if (snap.results.empty() || snap.results[0].levels.empty()) return;

    // The chart: mean return by difficulty, one line a policy
    const auto& spec = stress_.Spec();
    ImGui::Spacing();
    ImGui::Text("Mean return by difficulty");
    ImGui::SameLine();
    ImGui::TextColored(t.text_dim, "\xC2\xB7 %s \xC2\xB7 %d episodes a level", rl::StressKindName(spec.kind), spec.episodes);
    const float chart_h = std::max(180.0f, ImGui::GetContentRegionAvail().y * 0.5f);
    ImPlot::PushColormap(plot::SeriesColormap());
    if (ImPlot::BeginPlot("##stress_chart", ImVec2(-1, chart_h))) {
        // The return axis follows the results as levels come in
        ImPlot::SetupAxes("difficulty", "mean return", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, 1.0, ImPlotCond_Always);
        for (size_t p = 0; p < snap.results.size(); ++p) {
            const auto& r = snap.results[p];
            std::vector<double> x, y;
            for (const auto& l : r.levels) {
                x.push_back(l.difficulty);
                y.push_back(l.MeanReturn());
            }
            if (x.empty()) continue;
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 3.0f);
            // The column's index keeps two policies of the same file name apart
            const std::string series = r.name + "##policy" + std::to_string(p);
            ImPlot::PlotLine(series.c_str(), x.data(), y.data(), static_cast<int>(x.size()));
        }
        ImPlot::EndPlot();
    }
    ImPlot::PopColormap();

    // The table: each level, each policy's mean return and falls, and Watch
    const int columns = 1 + 3 * static_cast<int>(snap.results.size());
    if (ImGui::BeginTable("##stress_table", columns, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Difficulty");
        for (const auto& r : snap.results) {
            ImGui::TableSetupColumn((r.name + " return").c_str());
            ImGui::TableSetupColumn("falls");
            ImGui::TableSetupColumn("##watch");
        }
        ImGui::TableHeadersRow();
        size_t levels = 0;
        for (const auto& r : snap.results) levels = std::max(levels, r.levels.size());
        for (size_t l = 0; l < levels; ++l) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%.2f", rl::StressDifficulty(spec, static_cast<int>(l)));
            for (size_t p = 0; p < snap.results.size(); ++p) {
                const auto& r = snap.results[p];
                ImGui::TableNextColumn();
                if (l < r.levels.size()) ImGui::Text("%.1f", r.levels[l].MeanReturn());
                ImGui::TableNextColumn();
                if (l < r.levels.size()) {
                    const auto& lv = r.levels[l];
                    ImGui::TextColored(lv.falls > 0 ? t.caution : t.text, "%d / %zu", lv.falls, lv.returns.size());
                }
                ImGui::TableNextColumn();
                if (l < r.levels.size() && !running) {
                    ImGui::PushID(static_cast<int>(p * 100 + l));
                    if (ui::LinkButton("Watch") && p < tested_.size() && tested_[p].second)
                        WatchLevel(tested_[p].second, static_cast<int>(l));
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Play this level's first episode again in the Play tab");
                    ImGui::PopID();
                }
            }
        }
        ImGui::EndTable();
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "A fall: the episode ended early, or the robot ended upside down. Watch plays the "
                                   "level's first episode again (Play tab).");
    ImGui::PopTextWrapPos();
    if (!running && ui::SecondaryButton("Export CSV", true, nullptr, ui::ButtonSize::Regular)) {
        const std::string root = ProjectManager::Instance().GetProjectRoot();
        if (root.empty()) {
            stress_note_failed_ = true;
            stress_note_ = "Open a project first: the CSV goes into its exports folder.";
        } else {
            const auto folder = std::filesystem::path(root) / "exports";
            std::error_code ec;
            std::filesystem::create_directories(folder, ec);
            char stamp[32];
            const std::time_t now = std::time(nullptr);
            std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", std::localtime(&now));
            std::string kind = rl::StressKindName(spec.kind);
            for (char& c : kind) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            const auto path = folder / ("stress_" + kind + "_" + stamp + ".csv");
            std::ofstream out(path, std::ios::binary);
            out << rl::StressResultsCsv(snap.results, spec.episodes);
            stress_note_failed_ = !out.good();
            stress_note_ = stress_note_failed_ ? "The CSV could not be written to " + path.string()
                                               : "Saved " + path.filename().string() + " in the project's exports";
        }
    }
}

}  // namespace cyxwiz
