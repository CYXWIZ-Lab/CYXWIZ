// The MuJoCo Viewport's Camera, View and Record menus, the Record dialog and
// the camera a model starts with (mj_viewport_panel.h); the Preferences
// section of its saved choices.

#include "mj_viewport_panel.h"
#include "mj_ui_helpers.h"

#include "plugin/plugin_context.h"
#include "plugin/plugin_ui.h"

#include <imgui.h>
#include <algorithm>
#include <iterator>
#include <cctype>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <string>
#include <system_error>

namespace cyxwiz::plugin::mujoco {
namespace {

std::string ObjectName(const mjModel* m, int type, int id, const char* fallback) {
    const char* name = m ? mj_id2name(m, type, id) : nullptr;
    if (name && *name) return name;
    return std::string(fallback) + " " + std::to_string(id);
}

// The model's root: the first body attached to the world
int RootBody(const mjModel* m) {
    for (int b = 1; b < m->nbody; ++b)
        if (m->body_parentid[b] == 0) return b;
    return -1;
}

const char* PlaybackCameraText(PlaybackCamera camera) {
    switch (camera) {
        case PlaybackCamera::FollowRoot: return "following the root body";
        case PlaybackCamera::Free: return "the free camera";
        case PlaybackCamera::ModelTrack: break;
    }
    return "the model's track camera, else following the root body";
}

void Wrapped(const char* text) {
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 320.0f);
    ImGui::TextDisabled("%s", text);
    ImGui::PopTextWrapPos();
}

}  // namespace

// =============================================================================
// Camera
// =============================================================================

std::string MjViewportPanel::CameraLabel(const mjModel* model, const MjRenderer& renderer) const {
    switch (renderer.GetCameraMode()) {
        case MjRenderer::CameraMode::FollowBody:
            return "Camera: Follow \xC2\xB7 " + ObjectName(model, mjOBJ_BODY, renderer.CameraTarget(), "body");
        case MjRenderer::CameraMode::ModelCamera:
            return "Camera: " + ObjectName(model, mjOBJ_CAMERA, renderer.CameraTarget(), "camera");
        case MjRenderer::CameraMode::Free: break;
    }
    return "Camera: Free";
}

void MjViewportPanel::ApplyStartCamera(const mjModel* model, const mjData* data, MjRenderer& renderer) {
    if (!model) return;
    renderer.ResetCamera(model);  // the free camera, aimed at the whole model
    if (!playback_) return;       // building a model starts free
    switch (defaults_.playback_camera) {
        case PlaybackCamera::ModelTrack: {
            const int track = mj_name2id(model, mjOBJ_CAMERA, "track");
            if (track >= 0) {
                renderer.UseModelCamera(track);
                return;
            }
            [[fallthrough]];
        }
        case PlaybackCamera::FollowRoot: {
            const int root = RootBody(model);
            if (root > 0) renderer.FollowBody(model, data, root);
            return;
        }
        case PlaybackCamera::Free: return;
    }
}

void MjViewportPanel::RenderCameraMenu(MjEnvManager& env, MjRenderer& renderer) {
    if (!ImGui::BeginPopup("mujoco_camera_menu")) return;
    const mjModel* m = env.GetModel();
    const MjRenderer::CameraMode mode = renderer.GetCameraMode();
    const int target = renderer.CameraTarget();
    ImGui::TextDisabled("Camera");
    if (ImGui::MenuItem("Free (drag to orbit)", nullptr, mode == MjRenderer::CameraMode::Free))
        renderer.UseFreeCamera();
    if (m && m->nbody > 1 && ImGui::BeginMenu("Follow a body")) {
        for (int b = 1; b < m->nbody; ++b) {
            ImGui::PushID(b);
            const bool on = mode == MjRenderer::CameraMode::FollowBody && target == b;
            if (ImGui::MenuItem(ObjectName(m, mjOBJ_BODY, b, "body").c_str(), nullptr, on))
                renderer.FollowBody(m, env.GetData(), b);
            ImGui::PopID();
        }
        ImGui::EndMenu();
    }
    if (m && m->ncam > 0) {
        ImGui::Spacing();
        ImGui::TextDisabled("The model's cameras");
        for (int c = 0; c < m->ncam; ++c) {
            ImGui::PushID(1000 + c);
            const bool on = mode == MjRenderer::CameraMode::ModelCamera && target == c;
            if (ImGui::MenuItem(ObjectName(m, mjOBJ_CAMERA, c, "camera").c_str(), nullptr, on))
                renderer.UseModelCamera(c);
            ImGui::PopID();
        }
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Reset view")) renderer.ResetCamera(m);
    const std::string start = std::string("Playback starts with ") + PlaybackCameraText(defaults_.playback_camera) +
                              " (Preferences > Simulation).";
    Wrapped(start.c_str());
    ImGui::EndPopup();
}

// =============================================================================
// View
// =============================================================================

void MjViewportPanel::RenderViewMenu(MjEnvManager& env, MjRenderer& renderer) {
    if (!ImGui::BeginPopup("mujoco_view_menu")) return;
    // Ticking an option keeps the menu open, to set several
    ImGui::PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);

    ImGui::TextDisabled("Show in the scene");
    struct VisItem {
        const char* label;
        int flag;
    };
    static constexpr VisItem kShow[] = {
        {"Contact points", mjVIS_CONTACTPOINT}, {"Contact forces", mjVIS_CONTACTFORCE},
        {"Joints", mjVIS_JOINT},                {"Actuators", mjVIS_ACTUATOR},
        {"Centre of mass", mjVIS_COM},          {"Inertia boxes", mjVIS_INERTIA},
        {"Transparent", mjVIS_TRANSPARENT},
    };
    for (const auto& item : kShow) {
        const bool on = renderer.VisFlag(item.flag);
        if (ImGui::MenuItem(item.label, nullptr, on)) renderer.SetVisFlag(item.flag, !on);
    }
    const bool wireframe = renderer.RenderFlag(mjRND_WIREFRAME);
    if (ImGui::MenuItem("Wireframe", nullptr, wireframe)) renderer.SetRenderFlag(mjRND_WIREFRAME, !wireframe);
    const bool hulls = renderer.VisFlag(mjVIS_CONVEXHULL);
    if (ImGui::MenuItem("Convex hulls", nullptr, hulls)) renderer.SetVisFlag(mjVIS_CONVEXHULL, !hulls);

    ImGui::Separator();
    struct Choice {
        const char* label;
        int value;
    };
    static constexpr Choice kFrames[] = {
        {"None", mjFRAME_NONE}, {"Body", mjFRAME_BODY}, {"Geom", mjFRAME_GEOM},
        {"Site", mjFRAME_SITE}, {"World", mjFRAME_WORLD},
    };
    static constexpr Choice kLabels[] = {
        {"None", mjLABEL_NONE}, {"Body", mjLABEL_BODY},         {"Joint", mjLABEL_JOINT},
        {"Geom", mjLABEL_GEOM}, {"Site", mjLABEL_SITE},         {"Actuator", mjLABEL_ACTUATOR},
        {"Contact", mjLABEL_CONTACTPOINT},
    };
    const auto choices = [](const char* menu, const Choice* items, size_t count, int current, auto set) {
        const char* shown = "Other";
        for (size_t i = 0; i < count; ++i)
            if (items[i].value == current) shown = items[i].label;
        const std::string title = std::string(menu) + ": " + shown;
        if (!ImGui::BeginMenu((title + "###" + menu).c_str())) return;
        for (size_t i = 0; i < count; ++i)
            if (ImGui::MenuItem(items[i].label, nullptr, items[i].value == current)) set(items[i].value);
        ImGui::EndMenu();
    };
    choices("Frames", kFrames, std::size(kFrames), renderer.Frames(), [&](int v) { renderer.SetFrames(v); });
    choices("Labels", kLabels, std::size(kLabels), renderer.Labels(), [&](int v) { renderer.SetLabels(v); });

    if (ImGui::BeginMenu("Groups")) {
        ImGui::TextDisabled("The groups the model puts its parts in");
        const auto row = [&](const char* label, bool geoms) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(label);
            for (int g = 0; g < 6; ++g) {
                ImGui::SameLine(g == 0 ? 64.0f : 0.0f);
                ImGui::PushID(geoms ? g : 10 + g);
                const bool on = geoms ? renderer.GeomGroup(g) : renderer.SiteGroup(g);
                char text[4];
                std::snprintf(text, sizeof(text), "%d", g);
                if (ImGui::Selectable(text, on, ImGuiSelectableFlags_None, ImVec2(ImGui::GetFrameHeight(), 0.0f))) {
                    if (geoms) renderer.SetGeomGroup(g, !on);
                    else renderer.SetSiteGroup(g, !on);
                }
                ImGui::PopID();
            }
        };
        row("Geoms", true);
        row("Sites", false);
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Rendering")) {
        static constexpr VisItem kRender[] = {
            {"Shadows", mjRND_SHADOW}, {"Reflections", mjRND_REFLECTION}, {"Sky", mjRND_SKYBOX}, {"Fog", mjRND_FOG},
        };
        for (const auto& item : kRender) {
            const bool on = renderer.RenderFlag(item.flag);
            if (ImGui::MenuItem(item.label, nullptr, on)) renderer.SetRenderFlag(item.flag, !on);
        }
        const bool endless = renderer.EndlessFloor();
        if (ImGui::MenuItem("Endless floor", nullptr, endless)) renderer.SetEndlessFloor(!endless);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("The model's floor is drawn wherever the camera looks. It is endless for the physics "
                              "already; off draws only the size the model gives it.");
        ImGui::EndMenu();
    }

    ImGui::Separator();
    if (ImGui::MenuItem("Live numbers during playback", nullptr, live_numbers_)) live_numbers_ = !live_numbers_;
    const bool drawing = renderer.IsEnabled();
    if (ImGui::MenuItem("Draw the scene", nullptr, drawing)) renderer.SetEnabled(!drawing);
    ImGui::PopItemFlag();
    if (ImGui::MenuItem("Reset camera")) renderer.ResetCamera(env.GetModel());

    ImGui::Separator();
    const SceneOptions scene = ReadScene(renderer);
    const bool is_default = scene == defaults_.scene && live_numbers_ == defaults_.live_numbers;
    if (ImGui::MenuItem("Keep these as the default", nullptr, false, !is_default)) {
        defaults_ = LoadViewportDefaults(*ctx_);
        defaults_.scene = scene;
        defaults_.live_numbers = live_numbers_;
        SaveViewportDefaults(*ctx_, defaults_);
        Notify("The View options are now the default (Preferences > Simulation)");
    }
    if (ImGui::MenuItem("MuJoCo's options", nullptr, false, !(scene == SceneOptions{}))) ApplyScene(SceneOptions{}, renderer);
    ImGui::TextDisabled("Render size %d x %d", renderer.GetWidth(), renderer.GetHeight());
    ImGui::EndPopup();
}

// =============================================================================
// Record
// =============================================================================

std::filesystem::path MjViewportPanel::ExportsFolder() const {
    const std::filesystem::path project = ctx_ ? ctx_->GetProjectDir() : std::filesystem::path{};
    return project.empty() ? project : project / "exports";
}

std::string MjViewportPanel::ExportName() const {
    std::string slug;
    for (const char c : model_title_.empty() ? std::string("model") : model_title_) {
        if (std::isalnum(static_cast<unsigned char>(c))) slug += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        else if (!slug.empty() && slug.back() != '_') slug += '_';
    }
    while (!slug.empty() && slug.back() == '_') slug.pop_back();
    if (slug.empty()) slug = "model";
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &local);
    std::string name = slug + "_" + stamp;
    // Two in the same second get a number
    const std::filesystem::path folder = ExportsFolder();
    std::error_code ec;
    for (int n = 2; !folder.empty() && (std::filesystem::exists(folder / (name + ".png"), ec) ||
                                         std::filesystem::exists(folder / (name + ".gif"), ec) ||
                                         std::filesystem::exists(folder / (name + "_frames"), ec));
         ++n)
        name = slug + "_" + stamp + "-" + std::to_string(n);
    return name;
}

void MjViewportPanel::Notify(std::string text, bool failed) {
    notice_ = std::move(text);
    notice_failed_ = failed;
    notice_until_ = std::chrono::steady_clock::now() + std::chrono::seconds(8);
}

void MjViewportPanel::TakeScreenshot(MjRenderer& renderer) {
    const std::filesystem::path folder = ExportsFolder();
    std::error_code ec;
    std::filesystem::create_directories(folder, ec);
    const std::filesystem::path file = folder / (ExportName() + ".png");
    std::string error;
    if (MjRecorder::SavePng(file, renderer.LastFrame(), renderer.GetWidth(), renderer.GetHeight(), error)) {
        ctx_->LogInfo("Viewport screenshot saved: " + file.string());
        Notify("Saved " + file.filename().string() + " in the project's exports");
    } else {
        ctx_->LogWarn("Viewport screenshot failed: " + error);
        Notify(error, true);
    }
}

int MjViewportPanel::CurrentEpisode(MjEnvManager& env) const {
    if (playback_) return numbers_.episode;
    std::lock_guard phys_lock(env.GetPhysicsMutex());
    return env.GetEpisodeCount() + 1;
}

void MjViewportPanel::StartRecording(MjEnvManager& env) {
    const std::filesystem::path folder = ExportsFolder();
    std::error_code ec;
    std::filesystem::create_directories(folder, ec);
    const auto format = record_format_choice_ == 1 ? MjRecorder::Format::PngFrames : MjRecorder::Format::Gif;
    const std::string name = ExportName();
    const std::filesystem::path target =
        format == MjRecorder::Format::Gif ? folder / (name + ".gif") : folder / (name + "_frames");
    std::string error;
    if (!recorder_.Start(target, format, error)) {
        Notify(error, true);
        return;
    }
    record_until_stop_ = record_length_choice_ == 1;
    record_episode_ = CurrentEpisode(env);
    ctx_->LogInfo("Viewport recording started: " + target.string());
}

void MjViewportPanel::RecordFrame(MjEnvManager& env, MjRenderer& renderer) {
    if (!recorder_.IsRecording()) return;
    if (!record_until_stop_ && CurrentEpisode(env) != record_episode_) {  // the episode ended
        recorder_.Stop();
        return;
    }
    double time = 0.0;
    {
        std::lock_guard phys_lock(env.GetPhysicsMutex());
        if (const mjData* d = env.GetData()) time = d->time;
    }
    recorder_.AddFrame(renderer.LastFrame(), renderer.GetWidth(), renderer.GetHeight(), time);
}

void MjViewportPanel::RenderRecordMenu(MjEnvManager& env, MjRenderer& renderer) {
    (void)env;
    if (!ImGui::BeginPopup("mujoco_record_menu")) return;
    const bool can_save = !ExportsFolder().empty();
    ImGui::TextDisabled("Record");
    if (ImGui::MenuItem("Screenshot (PNG)", nullptr, false, can_save)) TakeScreenshot(renderer);
    if (ImGui::MenuItem("Record video...", nullptr, false, can_save)) open_record_dialog_ = true;
    ImGui::TextDisabled("%s", can_save ? "Saved in the project's exports folder"
                                       : "Open a project to save pictures and videos in its exports folder");
    ImGui::EndPopup();
}

void MjViewportPanel::RenderRecordDialog(MjEnvManager& env) {
    constexpr const char* kTitle = "Record video##mujoco_record";
    if (open_record_dialog_) {
        open_record_dialog_ = false;
        record_format_choice_ = defaults_.record_format == MjRecorder::Format::PngFrames ? 1 : 0;
        record_length_choice_ = defaults_.record_until_stop ? 1 : 0;
        ImGui::OpenPopup(kTitle);
    }
    if (!ImGui::BeginPopupModal(kTitle, nullptr, ImGuiWindowFlags_AlwaysAutoResize)) return;
    PluginUi* ui = ctx_->GetUi();
    static const char* kFormats[] = {"GIF", "PNG frames"};
    static const char* kLengths[] = {"Until the episode ends", "Until Stop"};
    const float label_w = 90.0f;

    ImGui::AlignTextToFramePadding();
    ImGui::TextDisabled("Format");
    ImGui::SameLine(label_w);
    ui->SegmentedControl("##record_format", kFormats, 2, &record_format_choice_);
    ImGui::AlignTextToFramePadding();
    ImGui::TextDisabled("Length");
    ImGui::SameLine(label_w);
    ui->SegmentedControl("##record_length", kLengths, 2, &record_length_choice_);

    // A playback shows the played task's steps (its frame skip), else the Viewport's own
    double step = playback_ ? numbers_.step_seconds : 0.0;
    if (step <= 0.0) {
        std::lock_guard phys_lock(env.GetPhysicsMutex());
        if (const mjModel* m = env.GetModel()) step = m->opt.timestep * env.GetFrameSkip();
    }
    char frames[160];
    if (step > 0.0)
        std::snprintf(frames, sizeof(frames), "One frame a simulation step: %.0f a second for this model, played back at real speed.",
                      1.0 / step);
    else
        std::snprintf(frames, sizeof(frames), "One frame a simulation step, played back at real speed.");
    ImGui::Spacing();
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 460.0f);
    ImGui::TextDisabled("%s", frames);
    ImGui::TextDisabled("%s", record_format_choice_ == 0
                                  ? "An animated GIF at most 640 pixels wide, for a README or a chat."
                                  : "One PNG a frame at the Viewport's size, in a folder: for a video editor or ffmpeg.");
    ImGui::PopTextWrapPos();
    ImGui::Spacing();

    const float cancel_w = ui->ButtonWidth("Cancel", UiButtonSize::Regular);
    const float record_w = ui->ButtonWidth("Record", UiButtonSize::Regular);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() +
                         std::max(0.0f, ImGui::GetContentRegionAvail().x - cancel_w - record_w -
                                            ImGui::GetStyle().ItemSpacing.x));
    if (ui->SecondaryButton("Cancel")) ImGui::CloseCurrentPopup();
    ImGui::SameLine();
    if (ui->PrimaryButton("Record")) {
        StartRecording(env);
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

// =============================================================================
// Preferences > Simulation
// =============================================================================

void RenderViewportPreferences(PluginContext& ctx) {
    PluginUi* ui = ctx.GetUi();
    if (!ui) return;
    ViewportDefaults d = LoadViewportDefaults(ctx);
    bool changed = false;
    if (ImGui::BeginTable("##mujoco_viewport_prefs", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("##label", ImGuiTableColumnFlags_WidthFixed, 200.0f);
        ImGui::TableSetupColumn("##value", ImGuiTableColumnFlags_WidthStretch);

        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Camera for playback");
        ImGui::TableNextColumn();
        static const char* kCameras[] = {"Model's track camera", "Follow the root body", "Free"};
        int camera = static_cast<int>(d.playback_camera);
        if (ui->SegmentedControl("##playback_camera", kCameras, 3, &camera)) {
            d.playback_camera = static_cast<PlaybackCamera>(camera);
            changed = true;
        }
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("The track camera is Gymnasium's (HalfCheetah, Hopper, Walker2d); models without one follow "
                            "their root body. Building a model starts with the free camera.");
        ImGui::PopTextWrapPos();

        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Live numbers");
        ImGui::TableNextColumn();
        if (ImGui::Checkbox("Episode, step, return and action on the scene during playback (on by default)",
                            &d.live_numbers))
            changed = true;

        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Record video");
        ImGui::TableNextColumn();
        static const char* kFormats[] = {"GIF", "PNG frames"};
        int format = d.record_format == MjRecorder::Format::PngFrames ? 1 : 0;
        if (ui->SegmentedControl("##record_format", kFormats, 2, &format)) {
            d.record_format = format == 1 ? MjRecorder::Format::PngFrames : MjRecorder::Format::Gif;
            changed = true;
        }
        ImGui::SameLine();
        static const char* kLengths[] = {"Until the episode ends", "Until Stop"};
        int length = d.record_until_stop ? 1 : 0;
        if (ui->SegmentedControl("##record_length", kLengths, 2, &length)) {
            d.record_until_stop = length == 1;
            changed = true;
        }

        // The Stress panel: shown or not, and what it offers first
        using L = StressLimits;
        StressOptions& st = d.stress;
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Stress panel");
        ImGui::TableNextColumn();
        if (ImGui::Checkbox("Show it beside the scene when the Viewport starts", &st.panel_open)) changed = true;
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("Kick");
        ImGui::SameLine();
        changed |= ui->NumberField("##pref_kick", &st.kick_speed, L::kKickMin, L::kKickMax, 0.5, "%.2f m/s", 80.0f);
        ImGui::SameLine();
        if (ImGui::Checkbox("A random kick every", &st.auto_disturb)) changed = true;
        ImGui::SameLine();
        changed |= ui->NumberField("##pref_auto", &st.auto_seconds, L::kAutoMin, L::kAutoMax, 0.5, "%.1f s", 64.0f);
        static const char* kObjects[] = {"Ball", "Box"};
        int object = st.object.ball ? 0 : 1;
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("Throw");
        ImGui::SameLine();
        if (ui->SegmentedControl("##pref_object", kObjects, 2, &object)) {
            st.object.ball = object == 0;
            changed = true;
        }
        ImGui::SameLine();
        changed |= ui->NumberField("##pref_size", &st.object.size, L::kSizeMin, L::kSizeMax, 0.05, "%.2f m", 72.0f);
        ImGui::SameLine();
        changed |= ui->NumberField("##pref_mass", &st.object.mass, L::kMassMin, L::kMassMax, 0.5, "%.1f kg", 72.0f);
        ImGui::SameLine();
        changed |= ui->NumberField("##pref_speed", &st.throw_speed, L::kSpeedMin, L::kSpeedMax, 1.0, "%.1f m/s", 72.0f);
        static const char* kFrom[] = {"Ahead", "Behind", "Above"};
        int from = static_cast<int>(st.from);
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("From");
        ImGui::SameLine();
        if (ui->SegmentedControl("##pref_from", kFrom, 3, &from)) {
            st.from = static_cast<MjThrowFrom>(from);
            changed = true;
        }
        static const char* kGrounds[] = {"Flat", "Rough", "Slopes", "Steps", "Obstacles"};
        int ground = static_cast<int>(st.ground.kind);
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("Ground offered");
        ImGui::SameLine();
        if (ui->SegmentedControl("##pref_ground", kGrounds, 5, &ground)) {
            st.ground.kind = static_cast<MjGroundKind>(ground);
            changed = true;
        }
        ImGui::SameLine();
        changed |= ui->NumberField("##pref_difficulty", &st.ground.difficulty, 0.0, 1.0, 0.1, "%.2f", 64.0f,
                                   "Difficulty, 0 to 1");
        double seed = static_cast<double>(st.ground.seed);
        ImGui::SameLine();
        if (ui->NumberField("##pref_seed", &seed, 0.0, 1.0e9, 1.0, "%.0f", 64.0f, "Seed")) {
            st.ground.seed = static_cast<uint64_t>(seed);
            changed = true;
        }
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("Every model starts on its own floor; the panel offers this ground for an episode.");
        ImGui::PopTextWrapPos();

        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Scene options");
        ImGui::TableNextColumn();
        const bool own = !(d.scene == SceneOptions{});
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("%s", own ? "Your own, kept from the View menu." : "MuJoCo's.");
        if (own) {
            ImGui::SameLine();
            if (ui->LinkButton("Use MuJoCo's")) {
                d.scene = SceneOptions{};
                changed = true;
            }
        }
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("Set them in the Viewport's View menu, then choose Keep these as the default.");
        ImGui::PopTextWrapPos();
        ImGui::EndTable();
    }
    if (changed) SaveViewportDefaults(ctx, d);
}

}  // namespace cyxwiz::plugin::mujoco
