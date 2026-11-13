// The MuJoCo Viewport's timeline and plots (TOFIX212 P3b, P3c, approved
// board 8): the steps the plugin's history kept (the Viewport's own run: the
// last 30 s; a played copy: its episode), a track to step back through them,
// and the last 10 s of a few signals drawn under the scene.

#include "mj_viewport_panel.h"
#include "mj_simulation_executor.h"
#include "mj_ui_helpers.h"
#include "env/mj_history.h"

#include "plugin/plugin_context.h"
#include "plugin/plugin_ui.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace cyxwiz::plugin::mujoco {
namespace {

constexpr float kTrackHeight = 18.0f;
constexpr float kPlotHeight = 70.0f;
constexpr double kPlotSeconds = 10.0;

struct Series {
    std::string name;
    std::vector<float> values;
};

}  // namespace

float MjViewportPanel::StripsHeight() const {
    if (!history_ || !has_history_) return 0.0f;
    const ImGuiStyle& style = ImGui::GetStyle();
    float h = ImGui::GetFrameHeightWithSpacing() + kTrackHeight + style.ItemSpacing.y +
              ImGui::GetTextLineHeightWithSpacing();
    if (timeline_.plots_open)
        h += ImGui::GetFrameHeightWithSpacing() + kPlotHeight + style.ItemSpacing.y + ImGui::GetTextLineHeightWithSpacing();
    return h;
}

void MjViewportPanel::ShowHistoryEntry(MjEnvManager& env, size_t index) {
    if (!history_) return;
    if (!Scrubbing()) {  // a step back pauses the Viewport's own run; Back to live resumes it
        resume_after_scrub_ = !playback_ && sim_executor_ && sim_executor_->GetState() == SimState::Running;
        if (resume_after_scrub_) sim_executor_->Pause();
    }
    std::lock_guard phys_lock(env.GetPhysicsMutex());
    if (index >= history_->Size()) return;
    scrub_index_ = static_cast<long long>(index);
    MjHistory::Apply(env.GetModel(), env.GetMutableData(), history_->At(index));
}

void MjViewportPanel::BackToLive(MjEnvManager& env) {
    if (history_) {
        std::lock_guard phys_lock(env.GetPhysicsMutex());
        if (!history_->Empty()) MjHistory::Apply(env.GetModel(), env.GetMutableData(), history_->Newest());
    }
    scrub_index_ = -1;
    if (resume_after_scrub_ && sim_executor_) sim_executor_->Resume();
    resume_after_scrub_ = false;
}

void MjViewportPanel::GoOnFromShownStep(MjEnvManager& env) {
    if (!Scrubbing() || !history_) return;
    {
        std::lock_guard phys_lock(env.GetPhysicsMutex());
        history_->TruncateAfter(static_cast<size_t>(scrub_index_));
    }
    scrub_index_ = -1;
    resume_after_scrub_ = false;
}

void MjViewportPanel::RenderTimeline(MjEnvManager& env) {
    if (!history_ || !has_history_) return;
    PluginUi* ui = ctx_->GetUi();
    const UiColors colors = ui->Colors();
    const float spacing = ImGui::GetStyle().ItemSpacing.x;

    // What to draw, read under the physics lock
    size_t count = 0;
    int shown_step = 0;
    double shown_time = 0.0;
    std::vector<float> falls;  // 0..1 along the track
    {
        std::lock_guard phys_lock(env.GetPhysicsMutex());
        count = history_->Size();
        if (count == 0) return;
        if (scrub_index_ >= static_cast<long long>(count)) scrub_index_ = static_cast<long long>(count) - 1;
        const size_t shown = Scrubbing() ? static_cast<size_t>(scrub_index_) : count - 1;
        shown_step = history_->At(shown).step;
        shown_time = history_->At(shown).time;
        // The Stress panel's falls, at the kept step nearest to them
        for (const auto& event : stress_log_.Events()) {
            if (event.outcome != MjStressLog::Outcome::Fell && event.outcome != MjStressLog::Outcome::Flipped) continue;
            size_t best = count;
            double best_gap = 1.0e9;
            for (size_t i = 0; i < count; ++i) {
                const auto& e = history_->At(i);
                if (e.episode != event.episode) continue;
                const double gap = std::abs(e.time - event.outcome_time);
                if (gap < best_gap) {
                    best_gap = gap;
                    best = i;
                }
            }
            if (best < count && count > 1) falls.push_back(static_cast<float>(best) / static_cast<float>(count - 1));
        }
    }

    // The row: the name and what is kept, the step buttons, Back to live, the
    // plots. Its own ID scope: the control row has a Step button too.
    ImGui::PushID("timeline");
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(ToImVec4(colors.text_dim), "Timeline");
    ImGui::SameLine();
    ImGui::TextColored(ToImVec4(colors.text_dim), playback_ ? "\xC2\xB7 the played episode" : "\xC2\xB7 the last 30 s");
    const char* plots = timeline_.plots_open ? "Hide plots" : "Plots";
    // Back to live only while a step back is shown
    const float back_w = Scrubbing() ? ui->ButtonWidth("Back to live", UiButtonSize::Small) + spacing : 0.0f;
    const float right = ui->ButtonWidth("Step back", UiButtonSize::Small) + ui->ButtonWidth("Step", UiButtonSize::Small) +
                        back_w + ui->ButtonWidth(plots, UiButtonSize::Small) + spacing * 2;
    ImGui::SameLine(std::max(ImGui::GetCursorPosX() + spacing, ImGui::GetWindowContentRegionMax().x - right));
    const size_t shown_index = Scrubbing() ? static_cast<size_t>(scrub_index_) : count - 1;
    if (ui->SecondaryButton("Step back", shown_index > 0, "The first step kept", UiButtonSize::Small))
        ShowHistoryEntry(env, shown_index - 1);
    ImGui::SameLine();
    if (ui->SecondaryButton("Step", Scrubbing() && shown_index + 1 < count, "Live already", UiButtonSize::Small))
        ShowHistoryEntry(env, shown_index + 1);
    ImGui::SameLine();
    if (Scrubbing()) {
        if (ui->PrimaryButton("Back to live", true, nullptr, UiButtonSize::Small)) BackToLive(env);
        ImGui::SameLine();
    }
    if (ui->SecondaryButton(plots, true, nullptr, UiButtonSize::Small)) timeline_.plots_open = !timeline_.plots_open;

    // The track: the kept steps, the shown one, the falls
    const float width = std::max(10.0f, ImGui::GetContentRegionAvail().x);
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##timeline_track", ImVec2(width, kTrackHeight));
    if (ImGui::IsItemActive() && count > 1) {
        const float f = std::clamp((ImGui::GetIO().MousePos.x - p0.x) / width, 0.0f, 1.0f);
        const size_t index = static_cast<size_t>(std::lround(f * static_cast<float>(count - 1)));
        if (!Scrubbing() || index != static_cast<size_t>(scrub_index_)) ShowHistoryEntry(env, index);
    }
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float mid = p0.y + kTrackHeight * 0.5f;
    const float shown_f = count > 1 ? static_cast<float>(shown_index) / static_cast<float>(count - 1) : 1.0f;
    dl->AddRectFilled(ImVec2(p0.x, mid - 2.0f), ImVec2(p0.x + width, mid + 2.0f), ToU32(colors.bg_input), 2.0f);
    dl->AddRectFilled(ImVec2(p0.x, mid - 2.0f), ImVec2(p0.x + width * shown_f, mid + 2.0f), ToU32(colors.accent), 2.0f);
    for (const float f : falls) {
        const float x = p0.x + width * f;
        dl->AddTriangleFilled(ImVec2(x - 4.0f, p0.y), ImVec2(x + 4.0f, p0.y), ImVec2(x, p0.y + 6.0f), ToU32(colors.error));
    }
    const float hx = p0.x + width * shown_f;
    dl->AddRectFilled(ImVec2(hx - 5.0f, p0.y + 1.0f), ImVec2(hx + 5.0f, p0.y + kTrackHeight - 1.0f),
                      ToU32(colors.text_bright), 3.0f);

    char line[160];
    if (Scrubbing())
        std::snprintf(line, sizeof(line),
                      "Step %d shown \xC2\xB7 %.2f s \xC2\xB7 %zu steps kept. Run goes on from here; Back to live from "
                      "the newest step.",
                      shown_step, shown_time, count);
    else
        std::snprintf(line, sizeof(line), "%zu steps kept \xC2\xB7 live. Drag along it, or Step back, to look back.",
                      count);
    ImGui::TextColored(ToImVec4(colors.text_dim), "%s", line);
    ImGui::PopID();
}

void MjViewportPanel::RenderPlots(MjEnvManager& env) {
    if (!history_ || !has_history_ || !timeline_.plots_open) return;
    PluginUi* ui = ctx_->GetUi();
    const UiColors colors = ui->Colors();

    // The selected body's first joint that has one value (a hinge or slide)
    std::string joint_name;
    int joint_qpos = -1;
    bool sensors_exist = false;
    std::vector<Series> series;
    long long marker = -1;  // the shown step within the window
    {
        std::lock_guard phys_lock(env.GetPhysicsMutex());
        const mjModel* m = env.GetModel();
        if (!m || history_->Empty()) return;
        sensors_exist = m->nsensordata > 0;
        if (selected_body_ > 0 && selected_body_ < m->nbody) {
            for (int j = m->body_jntadr[selected_body_];
                 j >= 0 && j < m->body_jntadr[selected_body_] + m->body_jntnum[selected_body_]; ++j) {
                if (m->jnt_type[j] != mjJNT_HINGE && m->jnt_type[j] != mjJNT_SLIDE) continue;
                const char* name = mj_id2name(m, mjOBJ_JOINT, j);
                joint_name = name && *name ? name : "joint " + std::to_string(j);
                joint_qpos = m->jnt_qposadr[j];
                break;
            }
        }
        const size_t n = history_->CountWithin(kPlotSeconds);
        const size_t first = history_->Size() - n;
        if (Scrubbing() && static_cast<size_t>(scrub_index_) >= first) marker = scrub_index_ - static_cast<long long>(first);
        auto add = [&](std::string name, auto value) {
            Series s;
            s.name = std::move(name);
            for (size_t i = first; i < history_->Size(); ++i) s.values.push_back(static_cast<float>(value(history_->At(i))));
            series.push_back(std::move(s));
        };
        if (timeline_.signals & kPlotReward) add("reward", [](const MjHistoryEntry& e) { return e.reward; });
        if ((timeline_.signals & kPlotJoint) && joint_qpos >= 0)
            add(joint_name + " angle", [joint_qpos](const MjHistoryEntry& e) {  // the state starts with the time
                return e.state.size() > static_cast<size_t>(joint_qpos) + 1 ? e.state[joint_qpos + 1] : 0.0;
            });
        if (timeline_.signals & kPlotSpeed) add("root speed", [](const MjHistoryEntry& e) { return e.root_speed; });
        if (timeline_.signals & kPlotControls)
            for (int a = 0; a < m->nu; ++a) {
                const char* name = mj_id2name(m, mjOBJ_ACTUATOR, a);
                add(name && *name ? std::string(name) : "control " + std::to_string(a),
                    [a](const MjHistoryEntry& e) { return static_cast<size_t>(a) < e.controls.size() ? e.controls[a] : 0.0; });
            }
        if ((timeline_.signals & kPlotSensors) && sensors_exist)
            for (size_t k = 0; k < std::min(kHistorySensors, static_cast<size_t>(m->nsensordata)); ++k)
                add("sensor " + std::to_string(k),
                    [k](const MjHistoryEntry& e) { return k < e.sensors.size() ? e.sensors[k] : 0.0; });
    }

    // The row: the name, the window, the signals as chips
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(ToImVec4(colors.text_dim), "Plots \xC2\xB7 the last 10 s");
    struct Chip {
        const char* label;
        unsigned bit;
        bool shown;
    };
    const std::string joint_label = joint_name.empty() ? std::string("Selected joint") : joint_name + " angle";
    const Chip chips[] = {{"Reward", kPlotReward, true},
                          {joint_label.c_str(), kPlotJoint, true},
                          {"Root speed", kPlotSpeed, true},
                          {"Controls", kPlotControls, true},
                          {"Sensors", kPlotSensors, sensors_exist}};
    for (const auto& chip : chips) {
        if (!chip.shown) continue;
        ImGui::SameLine();
        bool on = (timeline_.signals & chip.bit) != 0;
        ImGui::PushID(static_cast<int>(chip.bit));
        if (ui->ToggleChip("##plot_chip", chip.label, &on))
            timeline_.signals = on ? (timeline_.signals | chip.bit) : (timeline_.signals & ~chip.bit);
        ImGui::PopID();
    }

    // The chart: every signal on one scale, the shown step marked
    const UiColor palette[] = {colors.accent, colors.success, colors.warning, colors.running, colors.caution,
                               colors.accent_text, colors.error};
    const float width = std::max(10.0f, ImGui::GetContentRegionAvail().x);
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(width, kPlotHeight));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p0, ImVec2(p0.x + width, p0.y + kPlotHeight), ToU32(colors.bg_input), 4.0f);
    float low = 1.0e30f, high = -1.0e30f;
    size_t length = 0;
    for (const auto& s : series)
        for (const float v : s.values) {
            if (!std::isfinite(v)) continue;
            low = std::min(low, v);
            high = std::max(high, v);
            length = std::max(length, s.values.size());
        }
    if (length >= 2 && high >= low) {
        if (high - low < 1e-6f) {
            low -= 0.5f;
            high += 0.5f;
        }
        const float pad = 4.0f;
        auto at = [&](size_t i, float v) {
            const float x = p0.x + pad + (width - 2 * pad) * static_cast<float>(i) / static_cast<float>(length - 1);
            const float y = p0.y + kPlotHeight - pad - (kPlotHeight - 2 * pad) * (v - low) / (high - low);
            return ImVec2(x, y);
        };
        std::vector<ImVec2> points;
        for (size_t k = 0; k < series.size(); ++k) {
            points.clear();
            for (size_t i = 0; i < series[k].values.size(); ++i) points.push_back(at(i, series[k].values[i]));
            dl->AddPolyline(points.data(), static_cast<int>(points.size()), ToU32(palette[k % 7]), ImDrawFlags_None,
                            1.5f);
        }
        if (marker >= 0) {
            const float x = at(static_cast<size_t>(marker), low).x;
            dl->AddLine(ImVec2(x, p0.y), ImVec2(x, p0.y + kPlotHeight), ToU32(colors.text_bright, 0.6f));
        }
    }
    // The legend: each signal in its colour, the scale's ends
    if (series.empty()) {
        ImGui::TextColored(ToImVec4(colors.text_dim), joint_name.empty() && (timeline_.signals & kPlotJoint)
                                                           ? "Select a body (double-click it) to plot its joint."
                                                           : "Choose what to plot above.");
        return;
    }
    for (size_t k = 0; k < series.size(); ++k) {
        if (k) ImGui::SameLine();
        const float size = ImGui::GetTextLineHeight() * 0.6f;
        const ImVec2 at = ImGui::GetCursorScreenPos();
        const float top = at.y + (ImGui::GetTextLineHeight() - size) * 0.5f;
        ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(at.x, top), ImVec2(at.x + size, top + size),
                                                  ToU32(palette[k % 7]), 2.0f);
        ImGui::Dummy(ImVec2(size, ImGui::GetTextLineHeight()));
        ImGui::SameLine();
        ImGui::TextColored(ToImVec4(palette[k % 7]), "%s", series[k].name.c_str());
    }
    ImGui::SameLine();
    ImGui::TextColored(ToImVec4(colors.text_dim), "\xC2\xB7 %.2f to %.2f", low, high);
}

}  // namespace cyxwiz::plugin::mujoco
