// The MuJoCo Viewport's Stress panel (TOFIX212 P4): kick the robot, throw
// objects at it, change the ground, pull a body with Ctrl+drag, and see what
// each disturbance did. The panel only asks (MjStressRequest); the plugin
// sends the requests to the played copy or applies them to the Viewport's own
// run (env/mj_stress.h).

#include "mj_viewport_panel.h"
#include "mj_ui_helpers.h"

#include "plugin/plugin_context.h"
#include "plugin/plugin_ui.h"

#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <mutex>
#include <string>

namespace cyxwiz::plugin::mujoco {
namespace {

std::string LowerFirst(std::string text) {
    if (!text.empty()) text[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(text[0])));
    return text;
}

}  // namespace

void MjViewportPanel::ResetStress() {
    pushing_ = false;
    stress_requests_.clear();
    stress_log_.Clear();
    ground_asked_ = false;
    shown_ground_known_ = false;
    next_auto_kick_ = -1.0;
}

std::string MjViewportPanel::StressBlocked(MjEnvManager& env) const {
    if (graph_sim_active_) return kGraphReason;
    if (!env.IsLoaded()) return "Open a model first.";
    return {};
}

MjViewportPanel::StressClock MjViewportPanel::ShownClock(MjEnvManager& env) const {
    StressClock clock;
    {
        std::lock_guard phys_lock(env.GetPhysicsMutex());
        if (const mjData* d = env.GetData()) clock.time = d->time;
    }
    clock.episode = playback_ ? numbers_.episode : env.GetEpisodeCount() + 1;
    return clock;
}

void MjViewportPanel::Queue(MjStressRequest request, MjEnvManager& env, std::string what) {
    stress_requests_.push_back(request);
    LogStress(env, std::move(what));
}

void MjViewportPanel::LogStress(MjEnvManager& env, std::string what) {
    const StressClock clock = ShownClock(env);
    stress_log_.Add(clock.time, clock.episode, std::move(what));
}

void MjViewportPanel::WatchStress(MjEnvManager& env) {
    if (!env.IsLoaded()) return;
    bool flipped = false;
    double time = 0.0;
    {
        std::lock_guard phys_lock(env.GetPhysicsMutex());
        const mjModel* m = env.GetModel();
        const mjData* d = env.GetData();
        if (!m || !d) return;
        flipped = RobotFlipped(m, d, env.Robot());
        time = d->time;
    }
    const int episode = playback_ ? numbers_.episode : env.GetEpisodeCount() + 1;
    const bool ended_by_task = playback_ ? numbers_.last_ended_by_task : env.LastEpisodeTerminated();
    stress_log_.Observe(time, episode, flipped, ended_by_task);

    // A new ground came in with an episode: say so in the events
    const MjGround& ground = env.Scene().ground;
    if (!shown_ground_known_ || ground != shown_ground_) {
        if (shown_ground_known_) stress_log_.Note(time, episode, "Episode " + std::to_string(episode) + " on " +
                                                                     DescribeGround(ground));
        shown_ground_ = ground;
        shown_ground_known_ = true;
    }
    if (ground_asked_ && asked_ground_ == ground) ground_asked_ = false;

    // Automatic kicks while the scene moves, in simulated time
    const bool moving = playback_ || GetRunState().running;
    const bool restarted = time < last_watch_time_;  // an episode's reset sends the clock back
    last_watch_time_ = time;
    if (!stress_.auto_disturb || !moving || !StressBlocked(env).empty() ||
        RobotRootBody(env.GetModel(), env.Robot()) < 0) {
        next_auto_kick_ = -1.0;
        return;
    }
    if (next_auto_kick_ < 0.0 || restarted) next_auto_kick_ = time + stress_.auto_seconds;
    if (time >= next_auto_kick_) {
        const MjKick kick = static_cast<MjKick>(auto_random_.Next() % 3);
        Queue(MjStressRequest::MakeKick(kick, stress_.kick_speed), env,
              "Auto " + LowerFirst(DescribeKick(kick, stress_.kick_speed)));
        next_auto_kick_ = time + stress_.auto_seconds;
    }
}

void MjViewportPanel::HandlePush(MjEnvManager& env, MjRenderer& renderer, bool scene_active, float x, float y,
                                 float height) {
    ImGuiIO& io = ImGui::GetIO();
    const mjModel* m = env.GetModel();
    mjData* d = env.GetMutableData();
    if (!pushing_) {
        // Ctrl + left button down on the scene: hold the body under the mouse
        if (!(ImGui::IsItemActivated() && io.KeyCtrl && ImGui::IsMouseDown(ImGuiMouseButton_Left))) return;
        if (!StressBlocked(env).empty() || !m || !d) return;
        if (!renderer.StartPush(m, d, x, y)) return;
        pushing_ = true;
        const char* name = mj_id2name(m, mjOBJ_BODY, renderer.PushedBody());
        LogStress(env, std::string("Push on ") + (name && *name ? name : "a body"));
    }
    if (!scene_active || !ImGui::IsMouseDown(ImGuiMouseButton_Left) || !m || !d) {
        renderer.EndPush();
        stress_requests_.push_back(MjStressRequest::MakeEndPush());
        pushing_ = false;
        return;
    }
    // MuJoCo's mouse moves are fractions of the view's height, up positive
    if (io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f)
        renderer.MovePush(m, d, io.MouseDelta.x / height, -io.MouseDelta.y / height);
    mjtNum force[6];
    const int body = renderer.PushForce(m, d, force);
    if (body > 0) stress_requests_.push_back(MjStressRequest::MakePush(body, force));
}

void MjViewportPanel::RenderStressPanel(MjEnvManager& env) {
    PluginUi* ui = ctx_->GetUi();
    const UiColors colors = ui->Colors();
    using L = StressLimits;
    const std::string blocked = StressBlocked(env);
    const bool can = blocked.empty();
    const char* why = can ? nullptr : blocked.c_str();
    const bool has_root = env.IsLoaded() && RobotRootBody(env.GetModel(), env.Robot()) >= 0;
    const bool can_kick = can && has_root;
    const char* kick_why = !can ? why : "This model has no body that moves freely: nothing to kick.";
    constexpr float kLabel = 84.0f;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;

    auto section = [&](const char* title) {
        ImGui::Spacing();
        ImGui::TextColored(ToImVec4(colors.text_dim), "%s", title);
    };
    auto label = [&](const char* text) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(text);
        ImGui::SameLine(kLabel);
    };
    auto hint = [&](const char* text) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(ToImVec4(colors.text_dim), "%s", text);
        ImGui::PopTextWrapPos();
    };

    ImGui::TextColored(ToImVec4(colors.text_bright), "Stress");
    hint(playback_ ? "Acts on the copy the policy plays." : "Acts on this Viewport's run.");

    // Push: a sudden kick, or a pull while the mouse is held
    section("Push");
    label("Kick");
    ui->NumberField("##kick_speed", &stress_.kick_speed, L::kKickMin, L::kKickMax, 0.5, "%.2f m/s", 80.0f,
                    "How much the robot's velocity changes");
    struct KickButton {
        const char* label;
        MjKick kick;
    };
    constexpr KickButton kKicks[] = {{"Kick back", MjKick::Back}, {"Kick forward", MjKick::Forward}, {"Up", MjKick::Up}};
    bool first = true;
    for (const auto& button : kKicks) {
        if (!first) SameLineIfFits(ui->ButtonWidth(button.label, UiButtonSize::Regular) + spacing);
        first = false;
        if (ui->SecondaryButton(button.label, can_kick, kick_why))
            Queue(MjStressRequest::MakeKick(button.kick, stress_.kick_speed), env,
                  DescribeKick(button.kick, stress_.kick_speed));
    }
    hint("Or Ctrl+drag a body in the scene: a spring pull while the mouse is down.");

    // Throw an object at the robot
    section("Throw");
    label("Object");
    static const char* kObjects[] = {"Ball", "Box"};
    int object = stress_.object.ball ? 0 : 1;
    if (ui->SegmentedControl("##object", kObjects, 2, &object)) stress_.object.ball = object == 0;
    label("Size");
    ui->NumberField("##size", &stress_.object.size, L::kSizeMin, L::kSizeMax, 0.05, "%.2f m", 80.0f,
                    "A ball's diameter or a box's edge");
    label("Mass");
    ui->NumberField("##mass", &stress_.object.mass, L::kMassMin, L::kMassMax, 0.5, "%.1f kg", 80.0f);
    label("Speed");
    ui->NumberField("##throw_speed", &stress_.throw_speed, L::kSpeedMin, L::kSpeedMax, 1.0, "%.1f m/s", 80.0f);
    label("From");
    static const char* kFrom[] = {"Ahead", "Behind", "Above"};
    int from = static_cast<int>(stress_.from);
    if (ui->SegmentedControl("##from", kFrom, 3, &from)) stress_.from = static_cast<MjThrowFrom>(from);
    if (ui->PrimaryButton("Throw", can, why))
        Queue(MjStressRequest::MakeThrow(stress_.object, stress_.from, stress_.throw_speed), env,
              DescribeThrow(stress_.object, stress_.from, stress_.throw_speed));
    hint("Aimed at where the robot will be when it arrives. Thrown objects stay in the scene until the episode "
         "ends; four can be out at once.");

    // Random kicks at an interval
    section("Auto disturb");
    ImGui::BeginDisabled(!can_kick);
    ImGui::AlignTextToFramePadding();
    ImGui::Checkbox("A random kick every", &stress_.auto_disturb);
    ImGui::SameLine();
    ui->NumberField("##auto_seconds", &stress_.auto_seconds, L::kAutoMin, L::kAutoMax, 0.5, "%.1f s", 64.0f,
                    "Simulated seconds between kicks");
    ImGui::EndDisabled();
    hint("Back, forward or up at the kick speed above, while the scene moves.");

    // The ground
    section("Ground");
    static const char* kGrounds[] = {"Flat", "Rough", "Slopes", "Steps", "Obstacles"};
    int ground = static_cast<int>(stress_.ground.kind);
    if (ui->SegmentedControl("##ground", kGrounds, 5, &ground)) stress_.ground.kind = static_cast<MjGroundKind>(ground);
    if (stress_.ground.kind != MjGroundKind::Flat) {
        label("Difficulty");
        ui->NumberField("##difficulty", &stress_.ground.difficulty, 0.0, 1.0, 0.1, "%.2f", 80.0f,
                        "0 is gentle, 1 is hard: the heights and slopes grow with it");
        label("Seed");
        double seed = static_cast<double>(stress_.ground.seed);
        if (ui->NumberField("##seed", &seed, 0.0, 1.0e9, 1.0, "%.0f", 80.0f, "The same seed gives the same ground"))
            stress_.ground.seed = static_cast<uint64_t>(seed);
    }
    const auto ask_ground = [&](bool now) {
        stress_requests_.push_back(MjStressRequest::MakeGround(stress_.ground, now));
        asked_ground_ = stress_.ground;
        ground_asked_ = true;
        const StressClock clock = ShownClock(env);
        if (now) stress_log_.EndEpisodeByUser();
        stress_log_.Note(clock.time, clock.episode,
                         DescribeGround(stress_.ground) + (now ? ": a new episode now" : ": from the next episode"));
    };
    if (ui->SecondaryButton("Next episode", can, why)) ask_ground(false);
    SameLineIfFits(ui->ButtonWidth("Now (new episode)", UiButtonSize::Regular) + spacing);
    if (ui->SecondaryButton("Now (new episode)", can, why)) ask_ground(true);
    const MjGround& in_use = env.Scene().ground;
    ImGui::TextColored(ToImVec4(colors.text_dim), "This episode:");
    ImGui::SameLine();
    ImGui::TextUnformatted(DescribeGround(in_use).c_str());
    if (ground_asked_ && asked_ground_ != in_use) {
        ImGui::TextColored(ToImVec4(colors.text_dim), "Next episode:");
        ImGui::SameLine();
        ImGui::TextColored(ToImVec4(colors.accent_text), "%s", DescribeGround(asked_ground_).c_str());
    }
    hint("Along the run, from 2 m ahead of the start, for 300 m.");

    // What happened
    section("Events");
    const auto& events = stress_log_.Events();
    if (events.empty()) {
        hint("Nothing yet. Kicks, throws and pushes show here with what followed.");
    } else if (ImGui::BeginTable("##stress_events", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("##time", ImGuiTableColumnFlags_WidthFixed, 60.0f);
        ImGui::TableSetupColumn("##what", ImGuiTableColumnFlags_WidthStretch);
        char time[32];
        for (const auto& event : events) {
            ImGui::TableNextColumn();
            std::snprintf(time, sizeof(time), "%.2f s", event.time);
            ImGui::TextColored(ToImVec4(colors.text_dim), "%s", time);
            ImGui::TableNextColumn();
            ImGui::PushTextWrapPos(0.0f);
            const std::string outcome = MjStressLog::OutcomeText(event);
            if (outcome.empty()) {
                ImGui::TextUnformatted(event.what.c_str());
            } else {
                const UiColor& tone = event.outcome == MjStressLog::Outcome::StayedUp ? colors.success
                                      : event.outcome == MjStressLog::Outcome::Fell ||
                                              event.outcome == MjStressLog::Outcome::Flipped
                                          ? colors.caution
                                          : colors.text_dim;
                // What was done, and below it what followed
                ImGui::TextUnformatted(event.what.c_str());
                ImGui::TextColored(ToImVec4(tone), "%s", outcome.c_str());
            }
            ImGui::PopTextWrapPos();
        }
        ImGui::EndTable();
    }

    ImGui::Spacing();
    if (!events.empty()) {
        if (ui->LinkButton("Clear events")) stress_log_.Clear();
        ImGui::SameLine();
    }
    if (ui->LinkButton("Keep these as the default")) {
        defaults_ = LoadViewportDefaults(*ctx_);
        defaults_.stress = stress_;
        SaveViewportDefaults(*ctx_, defaults_);
        Notify("The Stress panel's choices are kept for next time (Preferences > Simulation)");
    }
}

}  // namespace cyxwiz::plugin::mujoco
