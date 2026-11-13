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
    next_auto_throw_ = -1.0;
    stress_episode_ = -1;
}

std::string MjViewportPanel::StressBlocked(MjEnvManager& env) const {
    if (graph_sim_active_) return kGraphReason;
    if (playback_ && numbers_.watch_only)
        return "This is the training's own copy: a disturbance would change what it learns. Watch the live "
               "preview to disturb a copy of the policy.";
    if (!env.IsLoaded()) return "Open a model first.";
    return {};
}

MjViewportPanel::StressClock MjViewportPanel::ShownClock(MjEnvManager& env) const {
    StressClock clock;
    {
        std::lock_guard phys_lock(env.GetPhysicsMutex());
        if (const mjData* d = env.GetData()) clock.time = d->time;
        clock.episode = playback_ ? numbers_.episode : env.GetEpisodeCount() + 1;
    }
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
    int episode = 0;
    bool ended_by_task = false;
    {
        std::lock_guard phys_lock(env.GetPhysicsMutex());
        const mjModel* m = env.GetModel();
        const mjData* d = env.GetData();
        if (!m || !d) return;
        flipped = RobotFlipped(m, d, env.Robot());
        time = d->time;
        episode = playback_ ? numbers_.episode : env.GetEpisodeCount() + 1;
        ended_by_task = playback_ ? numbers_.last_ended_by_task : env.LastEpisodeTerminated();
    }
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

    // Every episode draws its random kicks and throws from the same stream (the
    // ground's seed), so the same choices give the same disturbances again
    const bool restarted = time < last_watch_time_;  // an episode's reset sends the clock back
    last_watch_time_ = time;
    if (episode != stress_episode_ || restarted) {
        stress_episode_ = episode;
        stress_random_ = MjRandom(stress_.ground.seed);
        next_auto_kick_ = -1.0;
        next_auto_throw_ = -1.0;
    }

    // Automatic kicks and throws while the scene moves, in simulated time
    const bool moving = playback_ || GetRunState().running;
    const bool can = moving && StressBlocked(env).empty();
    const bool has_root = RobotRootBody(env.GetModel(), env.Robot()) >= 0;
    if (!stress_.auto_disturb || !can || !has_root) {
        next_auto_kick_ = -1.0;
    } else {
        if (next_auto_kick_ < 0.0) next_auto_kick_ = time + stress_.auto_seconds;
        if (time >= next_auto_kick_) {
            const int way = static_cast<int>(stress_random_.Next() % 4);
            const std::array<double, 3> dv = KickVelocity(way);
            Queue(MjStressRequest::MakeKick(dv, stress_.kick_seconds), env,
                  "Auto " + LowerFirst(DescribeKick(dv, stress_.kick_seconds)));
            next_auto_kick_ = time + stress_.auto_seconds;
        }
    }
    if (!stress_.auto_throw || !can) {
        next_auto_throw_ = -1.0;
    } else {
        if (next_auto_throw_ < 0.0) next_auto_throw_ = time + stress_.auto_throw_seconds;
        if (time >= next_auto_throw_) {
            QueueThrow(env, true);
            next_auto_throw_ = time + stress_.auto_throw_seconds;
        }
    }
}

std::array<double, 3> MjViewportPanel::KickVelocity(int way) {
    const double s = stress_.kick_speed;
    switch (way) {
        case 0: return {-s, 0.0, 0.0};
        case 1: return {s, 0.0, 0.0};
        case 2: return {0.0, 0.0, s};
        default: {
            using L = StressLimits;
            const std::array<double, 3> d =
                Direction(stress_random_.Uniform(0.0, 360.0), stress_random_.Uniform(0.0, L::kRandomKickUpMax));
            return {s * d[0], s * d[1], s * d[2]};
        }
    }
}

void MjViewportPanel::QueueThrow(MjEnvManager& env, bool random) {
    using L = StressLimits;
    MjThrownObject object = stress_.object;
    MjThrow how = stress_.how;
    if (stress_.any_shape)
        object.shape = static_cast<MjShape>(stress_random_.Next() % kShapeCount);
    if (random) {
        // Vary the object and come from anywhere around, level to steep
        const double size = stress_random_.Uniform(stress_.size_low, stress_.size_high);
        for (double& s : object.size) s = std::clamp(s * size, L::kSizeMin, L::kSizeMax);
        object.mass = std::clamp(object.mass * stress_random_.Uniform(stress_.mass_low, stress_.mass_high),
                                 L::kMassMin, L::kMassMax);
        how.around = stress_random_.Uniform(0.0, 360.0);
        how.up = stress_random_.Uniform(0.0, L::kRandomUpMax);
    }
    if (stress_.tumble) {
        // A uniformly random turn (Shoemake) and a spin of 2-8 rad/s about a random axis
        const double u1 = stress_random_.Uniform(), u2 = stress_random_.Uniform(), u3 = stress_random_.Uniform();
        constexpr double kTwoPi = 6.28318530717958647692;
        how.quat = {std::sqrt(u1) * std::cos(kTwoPi * u3), std::sqrt(1.0 - u1) * std::sin(kTwoPi * u2),
                    std::sqrt(1.0 - u1) * std::cos(kTwoPi * u2), std::sqrt(u1) * std::sin(kTwoPi * u3)};
        const std::array<double, 3> axis =
            Direction(stress_random_.Uniform(0.0, 360.0), stress_random_.Uniform(-90.0, 90.0));
        const double rate = stress_random_.Uniform(2.0, 8.0);
        how.spin = {axis[0] * rate, axis[1] * rate, axis[2] * rate};
    }
    Queue(MjStressRequest::MakeThrow(object, how), env, (random ? "Auto " : "") + DescribeThrow(object, how));
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

void MjViewportPanel::RenderThrowDial(float right, float top) {
    const UiColors colors = ctx_->GetUi()->Colors();
    constexpr float kRadius = 52.0f;
    constexpr float kPad = 10.0f;
    constexpr float kMargin = 12.0f;
    const float line_h = ImGui::GetTextLineHeightWithSpacing();
    const ImVec2 box0(right - kMargin - kPad * 2 - kRadius * 2, top + kMargin);
    const ImVec2 box1(right - kMargin, box0.y + kPad * 2 + line_h + kRadius * 2 + line_h);
    const ImVec2 centre(box0.x + kPad + kRadius, box0.y + kPad + line_h + kRadius);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(box0, box1, ToU32(colors.bg_window, 0.85f), 6.0f);
    dl->AddText(ImVec2(box0.x + kPad, box0.y + kPad), ToU32(colors.text_dim), "Throw from");
    dl->AddCircle(centre, kRadius, ToU32(colors.text_dim, 0.6f), 48, 1.5f);
    dl->AddCircle(centre, kRadius * 0.5f, ToU32(colors.text_dim, 0.3f), 32, 1.0f);
    // Top-down: the robot runs to the right (+x); +y (left of the run) is up
    dl->AddLine(ImVec2(centre.x - 14.0f, centre.y), ImVec2(centre.x + 14.0f, centre.y), ToU32(colors.text), 5.0f);
    const std::array<double, 3> from = Direction(stress_.how.around, stress_.how.up);
    const float reach = kRadius * static_cast<float>(std::hypot(from[0], from[1]));
    const float angle = static_cast<float>(stress_.how.around * 3.14159265358979323846 / 180.0);
    const ImVec2 dot(centre.x + std::cos(angle) * reach, centre.y - std::sin(angle) * reach);
    dl->AddLine(dot, centre, ToU32(colors.accent), 2.0f);
    dl->AddCircleFilled(dot, 6.0f, ToU32(colors.accent));
    char text[48];
    std::snprintf(text, sizeof(text), "%.0f\xC2\xB0, up %.0f\xC2\xB0", stress_.how.around, stress_.how.up);
    dl->AddText(ImVec2(box0.x + kPad, centre.y + kRadius + 4.0f), ToU32(colors.text_dim), text);

    // Click or drag on the dial: the direction around (the angle up stays typed)
    const ImVec2 saved = ImGui::GetCursorScreenPos();
    ImGui::SetCursorScreenPos(ImVec2(centre.x - kRadius, centre.y - kRadius));
    ImGui::InvisibleButton("##throw_dial", ImVec2(kRadius * 2, kRadius * 2));
    if (ImGui::IsItemActive()) {
        const ImVec2 mouse = ImGui::GetIO().MousePos;
        const double degrees = std::atan2(-(mouse.y - centre.y), mouse.x - centre.x) * 180.0 / 3.14159265358979323846;
        stress_.how.around = std::round(std::fmod(degrees + 360.0, 360.0));
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Click or drag to throw from that side");
    ImGui::SetCursorScreenPos(saved);
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
    auto between = [&](const char* text) {  // a word between fields on one line
        ImGui::SameLine();
        ImGui::TextColored(ToImVec4(colors.text_dim), "%s", text);
        ImGui::SameLine();
    };
    auto hint = [&](const char* text) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(ToImVec4(colors.text_dim), "%s", text);
        ImGui::PopTextWrapPos();
    };

    ImGui::TextColored(ToImVec4(colors.text_bright), "Stress");
    hint(playback_ ? "Acts on the copy the policy plays." : "Acts on this Viewport's run.");

    // Push: a sudden kick, a push held for a time, or a pull while the mouse is held
    section("Push");
    label("Kick");
    ui->NumberField("##kick_speed", &stress_.kick_speed, L::kKickMin, L::kKickMax, 0.5, "%.2f m/s", 64.0f,
                    "How much the robot's velocity changes");
    between("over");
    ui->NumberField("##kick_seconds", &stress_.kick_seconds, L::kSecondsMin, L::kSecondsMax, 0.1, "%.2f s", 52.0f,
                    "0: a sudden kick. More: a push held that long (0.2 s in Geijtenbeek et al. 2013)");
    static const char* kKicks[] = {"Kick back", "Kick forward", "Up", "Any way"};
    for (int way = 0; way < 4; ++way) {
        if (way) SameLineIfFits(ui->ButtonWidth(kKicks[way], UiButtonSize::Regular) + spacing);
        if (ui->SecondaryButton(kKicks[way], can_kick, kick_why)) {
            const std::array<double, 3> dv = KickVelocity(way);
            Queue(MjStressRequest::MakeKick(dv, stress_.kick_seconds), env, DescribeKick(dv, stress_.kick_seconds));
        }
    }
    hint("Any way: a random direction. Or Ctrl+drag a body in the scene: a spring pull while the mouse is down.");

    // Throw an object at the robot
    section("Throw");
    label("Object");
    static const char* kShapes[] = {"Ball", "Box", "Capsule", "Cylinder", "Any"};
    int shape = stress_.any_shape ? kShapeCount : static_cast<int>(stress_.object.shape);
    if (ui->SegmentedControl("##shape", kShapes, kShapeCount + 1, &shape)) {
        stress_.any_shape = shape == kShapeCount;
        if (!stress_.any_shape) stress_.object.shape = static_cast<MjShape>(shape);
    }
    const int sizes = stress_.any_shape ? 3 : ShapeSizes(stress_.object.shape);
    label("Size");
    static const char* kSizeIds[] = {"##size0", "##size1", "##size2"};
    for (int i = 0; i < sizes; ++i) {
        if (i) between("\xC3\x97");
        ui->NumberField(kSizeIds[i], &stress_.object.size[i], L::kSizeMin, L::kSizeMax, 0.0, "%.2f", 48.0f);
    }
    between("m");
    ImGui::NewLine();
    const char* size_hint = stress_.any_shape                                 ? "Ball: the first as diameter. Box: length \xC3\x97 width \xC3\x97 height. Capsule, cylinder: length \xC3\x97 diameter."
                            : stress_.object.shape == MjShape::Ball           ? "Diameter."
                            : stress_.object.shape == MjShape::Box            ? "Length \xC3\x97 width \xC3\x97 height."
                                                                              : "Length (end to end) \xC3\x97 diameter.";
    hint(size_hint);
    label("Mass");
    ui->NumberField("##mass", &stress_.object.mass, L::kMassMin, L::kMassMax, 0.5, "%.1f kg", 80.0f);
    label("Speed");
    ui->NumberField("##throw_speed", &stress_.how.speed, L::kSpeedMin, L::kSpeedMax, 1.0, "%.1f m/s", 80.0f);
    label("From");
    struct Preset {
        double around, up;
    };
    static const char* kFrom[] = {"Ahead", "Behind", "Left", "Right", "Above"};
    constexpr Preset kPresets[] = {{0, 0}, {180, 0}, {90, 0}, {270, 0}, {0, 90}};
    int from = -1;
    for (int i = 0; i < 5; ++i) {
        const bool above = kPresets[i].up >= 90.0 && stress_.how.up >= 90.0;
        if (above || (stress_.how.around == kPresets[i].around && stress_.how.up == kPresets[i].up)) from = i;
    }
    if (ui->SegmentedControl("##from", kFrom, 5, &from) && from >= 0) {
        stress_.how.around = kPresets[from].around;
        stress_.how.up = kPresets[from].up;
    }
    label("Angle");
    ui->NumberField("##around", &stress_.how.around, L::kAroundMin, L::kAroundMax, 15.0, "%.0f\xC2\xB0", 48.0f,
                    "Around the robot: 0 ahead, 90 left, 180 behind, 270 right (or drag the dial on the scene)");
    between("up");
    ui->NumberField("##up", &stress_.how.up, L::kUpMin, L::kUpMax, 15.0, "%.0f\xC2\xB0", 48.0f,
                    "Above level: 90 is straight above");
    label("Distance");
    ui->NumberField("##distance", &stress_.how.distance, L::kDistanceMin, L::kDistanceMax, 0.5, "%.1f m", 80.0f);
    ImGui::Checkbox("Tumble: start turned and spinning", &stress_.tumble);
    if (ui->PrimaryButton("Throw", can, why)) QueueThrow(env, false);
    hint("Aimed at where the robot will be when it arrives. Thrown objects stay in the scene until the episode "
         "ends; eight can be out at once.");

    // Random kicks and throws at an interval
    section("Auto disturb");
    ImGui::BeginDisabled(!can_kick);
    ImGui::AlignTextToFramePadding();
    ImGui::Checkbox("A random kick every", &stress_.auto_disturb);
    ImGui::SameLine();
    ui->NumberField("##auto_seconds", &stress_.auto_seconds, L::kAutoMin, L::kAutoMax, 0.5, "%.1f s", 56.0f,
                    "Simulated seconds between kicks");
    ImGui::EndDisabled();
    ImGui::BeginDisabled(!can);
    ImGui::AlignTextToFramePadding();
    ImGui::Checkbox("A random throw every", &stress_.auto_throw);
    ImGui::SameLine();
    ui->NumberField("##auto_throw_seconds", &stress_.auto_throw_seconds, L::kAutoMin, L::kAutoMax, 0.5, "%.1f s",
                    56.0f, "Simulated seconds between throws");
    label("Size \xC3\x97");
    ui->NumberField("##size_low", &stress_.size_low, L::kFactorMin, L::kFactorMax, 0.0, "%.2f", 48.0f);
    between("to");
    ui->NumberField("##size_high", &stress_.size_high, L::kFactorMin, L::kFactorMax, 0.0, "%.2f", 48.0f);
    label("Mass \xC3\x97");
    ui->NumberField("##mass_low", &stress_.mass_low, L::kFactorMin, L::kFactorMax, 0.0, "%.2f", 48.0f);
    between("to");
    ui->NumberField("##mass_high", &stress_.mass_high, L::kFactorMin, L::kFactorMax, 0.0, "%.2f", 48.0f);
    ImGui::EndDisabled();
    stress_.size_high = std::max(stress_.size_high, stress_.size_low);
    stress_.mass_high = std::max(stress_.mass_high, stress_.mass_low);
    hint("Kicks: back, forward, up or any way, at the kick above. Throws: the object above scaled by these factors "
         "(any shape when Any), from any side, level to 60\xC2\xB0 up. The ground's seed repeats them each episode.");

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
