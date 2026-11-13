#include "mj_viewport_panel.h"
#include "mj_simulation_executor.h"
#include "mj_ui_helpers.h"

#include "plugin/plugin_context.h"
#include "plugin/plugin_ui.h"

#include <imgui.h>
#include <algorithm>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace cyxwiz::plugin::mujoco {
namespace {

constexpr const char* kLibraryPanelId = "mujoco_env_browser";

// What the panel shows of the model, read in one physics transaction.
struct SceneSnapshot {
    double time = 0.0;
    double model_step = 0.0;  // timestep x frame skip
    int episode = 1;
    int episode_step = 0;
    int max_episode_steps = 0;
    float episode_reward = 0.0f;
};

// `shown`: the timeline's step back (-1: live), whose own numbers replace
// the live run's
SceneSnapshot Snapshot(MjEnvManager& env, const MjHistory* history, long long shown) {
    SceneSnapshot s;
    std::lock_guard phys_lock(env.GetPhysicsMutex());
    if (const mjData* d = env.GetData()) s.time = d->time;
    if (const mjModel* m = env.GetModel()) s.model_step = m->opt.timestep * env.GetFrameSkip();
    s.episode = env.GetEpisodeCount() + 1;
    s.episode_step = env.GetCurrentStep();
    s.max_episode_steps = env.GetMaxEpisodeSteps();
    s.episode_reward = env.GetEpisodeReward();
    if (history && shown >= 0 && static_cast<size_t>(shown) < history->Size()) {
        const MjHistoryEntry& e = history->At(static_cast<size_t>(shown));
        s.episode = e.episode;
        s.episode_step = e.step;
        s.episode_reward = static_cast<float>(e.episode_return);
    }
    return s;
}

}  // namespace

void MjViewportPanel::Render(MjEnvManager& env, MjRenderer& renderer, bool* visible) {
    if (!visible || !*visible) return;
    current_env_ = &env;

    // A model file the user picked: loaded before this frame's ImGui calls,
    // since loading switches GL contexts.
    if (!pending_open_path_.empty()) {
        const std::string path = std::move(pending_open_path_);
        pending_open_path_.clear();
        if (open_model_) open_model_(path);
    }

    ImGui::SetNextWindowSize(ImVec2(900, 640), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("MuJoCo Viewport", visible,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
        ImGui::End();
        return;
    }
    // The Engine's widgets: always there in an Engine with plugin API 1.4
    if (!ctx_ || !ctx_->GetUi()) {
        ImGui::End();
        return;
    }
    if (!defaults_loaded_) {  // the saved choices, once per session
        defaults_ = LoadViewportDefaults(*ctx_);
        live_numbers_ = defaults_.live_numbers;
        stress_ = defaults_.stress;
        inspect_ = defaults_.inspect;
        timeline_ = defaults_.timeline;
        if (inspect_.panel_open && stress_.panel_open) inspect_.panel_open = false;  // one side panel
        defaults_loaded_ = true;
    }

    // A finished recording: say where it went
    MjRecorder::Result result;
    if (recorder_.TakeResult(result)) {
        if (result.ok) {
            std::string text = "Saved " + result.path.filename().string() + " in the project's exports (" +
                               std::to_string(result.frames) + " frames";
            if (result.dropped > 0) text += ", " + std::to_string(result.dropped) + " skipped";
            text += ")";
            ctx_->LogInfo("Viewport recording saved: " + result.path.string());
            Notify(text);
        } else {
            ctx_->LogWarn("Viewport recording failed: " + result.error);
            Notify(result.error, true);
        }
    }

    if (!env.IsLoaded()) {
        RenderEmpty();
    } else if (!renderer.IsInitialized()) {
        ctx_->GetUi()->EmptyState("The 3D view could not start",
                                  "The model is loaded, but its renderer did not start. The log has the reason.");
    } else {
        if (scene_pending_ || camera_default_pending_) defaults_ = LoadViewportDefaults(*ctx_);
        if (scene_pending_) {
            scene_pending_ = false;
            ApplyScene(defaults_.scene, renderer);
        }
        if (camera_default_pending_) {
            camera_default_pending_ = false;
            ApplyStartCamera(env.GetModel(), env.GetData(), renderer);
        }
        RenderControls(env, renderer);
        WatchStress(env);
        if (history_) {  // the strips under the scene take room only when there is something kept
            std::lock_guard phys_lock(env.GetPhysicsMutex());
            has_history_ = !history_->Empty();
        }
        if (stress_.panel_open || inspect_.panel_open) {
            // The scene and the Stress or Inspect panel side by side
            constexpr float kStressWidth = 390.0f;
            const ImVec2 avail = ImGui::GetContentRegionAvail();
            const float spacing = ImGui::GetStyle().ItemSpacing.x;
            const float scene_w = std::max(10.0f, avail.x - kStressWidth - spacing);
            constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
            ImGui::BeginChild("##scene_area", ImVec2(scene_w, avail.y), ImGuiChildFlags_None, kFlags);
            RenderScene(env, renderer);
            RenderTimeline(env);
            RenderPlots(env);
            RenderEpisodeStrip(env);
            ImGui::EndChild();
            ImGui::SameLine();
            const UiColors colors = ctx_->GetUi()->Colors();
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ToImVec4(colors.bg_panel));
            ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 0.0f);
            ImGui::BeginChild("##side_panel", ImVec2(kStressWidth, avail.y), ImGuiChildFlags_AlwaysUseWindowPadding);
            if (stress_.panel_open) RenderStressPanel(env);
            else RenderInspectPanel(env, renderer);
            ImGui::EndChild();
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor();
        } else {
            RenderScene(env, renderer);
            RenderTimeline(env);
            RenderPlots(env);
            RenderEpisodeStrip(env);
        }
    }
    RenderRecordDialog(env);

    ImGui::End();
}

// =============================================================================
// Run controls (the control row and the Simulation menu)
// =============================================================================

MjViewportPanel::RunState MjViewportPanel::GetRunState() const {
    RunState s;
    s.graph = graph_sim_active_;
    s.playback = playback_;
    s.can_drive = sim_executor_ && !s.graph && !s.playback;
    s.active = s.can_drive && sim_executor_->GetMode() != SimMode::Stopped;
    const SimState state = sim_executor_ ? sim_executor_->GetState() : SimState::Idle;
    s.running = s.active && state == SimState::Running;
    s.paused = s.active && state == SimState::Paused;
    return s;
}

void MjViewportPanel::RunOrResume() {
    const RunState s = GetRunState();
    if (!s.can_drive || s.running) return;
    if (Scrubbing() && current_env_) GoOnFromShownStep(*current_env_);
    if (s.paused) sim_executor_->Resume();
    else sim_executor_->StartManualControl();
}

void MjViewportPanel::PauseRun() {
    if (GetRunState().running) sim_executor_->Pause();
}

void MjViewportPanel::StepOnce() {
    const RunState s = GetRunState();
    if (!s.can_drive) return;
    if (Scrubbing() && current_env_) GoOnFromShownStep(*current_env_);
    if (s.running) sim_executor_->Pause();
    sim_executor_->SingleStep();
}

void MjViewportPanel::StopRun() {
    if (GetRunState().active) sim_executor_->Stop();
}

// =============================================================================
// No model loaded
// =============================================================================

void MjViewportPanel::RenderEmpty() {
    PluginUi* ui = ctx_->GetUi();
    ui->EmptyState("No model loaded", "Pick one in the Environment Library, or open your own MJCF file.");
    const char* library = "Open Environment Library";
    const char* file = "Open MJCF file...";
    const float width = ui->ButtonWidth(library, UiButtonSize::Regular) + ImGui::GetStyle().ItemSpacing.x +
                        ui->ButtonWidth(file, UiButtonSize::Regular);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, (ImGui::GetContentRegionAvail().x - width) * 0.5f));
    if (ui->PrimaryButton(library)) ctx_->ShowPanel(kLibraryPanelId);
    SameLineIfFits(ui->ButtonWidth(file, UiButtonSize::Regular));
    if (ui->SecondaryButton(file)) {
        const std::string path = ui->OpenFileDialog("Open MJCF model", "MJCF model", "xml");
        if (!path.empty()) pending_open_path_ = path;
    }
}

// =============================================================================
// Control row
// =============================================================================

void MjViewportPanel::RenderControls(MjEnvManager& env, MjRenderer& renderer) {
    PluginUi* ui = ctx_->GetUi();
    const UiColors colors = ui->Colors();
    MjSimulationExecutor* exec = sim_executor_;
    const RunState rs = GetRunState();
    const bool graph = rs.graph;
    const bool active = rs.active;
    const bool running = rs.running;
    const bool paused = rs.paused;
    const bool can_drive = rs.can_drive;
    const bool playback = rs.playback;
    const char* busy = graph ? kGraphReason : playback ? kPlaybackReason : nullptr;
    // The scene moves: keep the Engine at full frame rate without input
    if (running || graph || playback) ctx_->RequestFullFrameRate();

    // The model's name and source, then the controls centred on its two lines
    const float row_y = ImGui::GetCursorPosY();
    ImGui::BeginGroup();
    ImGui::TextColored(ToImVec4(colors.text_bright), "%s", model_title_.empty() ? "Model" : model_title_.c_str());
    if (!model_subtitle_.empty()) ImGui::TextColored(ToImVec4(colors.text_dim), "%s", model_subtitle_.c_str());
    ImGui::EndGroup();
    const float button_h = ui->ButtonHeight(UiButtonSize::Regular);
    const float row_h = std::max(ImGui::GetItemRectSize().y, button_h);
    float line_y = row_y;
    // The next item on this line, or at the start of a new line below when it
    // does not fit (the items are placed by hand, so the wrap is too: ImGui's
    // own NewLine goes by the last item's height, not the row's)
    auto next = [&](float width, float height) {
        ImGui::SameLine();
        if (ImGui::GetContentRegionAvail().x < width) {
            line_y += row_h + ImGui::GetStyle().ItemSpacing.y;
            ImGui::SetCursorPosX(ImGui::GetCursorStartPos().x);
        }
        ImGui::SetCursorPosY(line_y + (row_h - height) * 0.5f);
    };
    const float spacing = ImGui::GetStyle().ItemSpacing.x;

    if (running) {
        next(ui->ButtonWidth("Pause", UiButtonSize::Regular) + spacing * 2, button_h);
        if (ui->PrimaryButton("Pause")) PauseRun();
    } else {
        const char* run = paused ? "Resume" : "Run";
        next(ui->ButtonWidth(run, UiButtonSize::Regular) + spacing * 2, button_h);
        if (ui->PrimaryButton(run, can_drive, busy)) RunOrResume();
    }
    next(ui->ButtonWidth("Step", UiButtonSize::Regular), button_h);
    if (ui->SecondaryButton("Step", can_drive, busy)) StepOnce();
    next(ui->ButtonWidth("Reset", UiButtonSize::Regular), button_h);
    if (ui->SecondaryButton("Reset", can_drive, busy)) {
        std::lock_guard phys_lock(env.GetPhysicsMutex());
        env.Reset();
    }
    if (active) {
        next(ui->ButtonWidth("Stop", UiButtonSize::Regular), button_h);
        if (ui->SecondaryButton("Stop")) StopRun();
    }

    // Speed: a typed real-time factor, or as fast as the model steps
    const float field_h = ImGui::GetFrameHeight();
    // The label is drawn at the row's centre: text placed after SameLine sits
    // on the line's text baseline, below the centre of the buttons
    const ImVec2 label_size = ImGui::CalcTextSize("Speed");
    next(label_size.x + spacing * 2, label_size.y);
    const ImVec2 label_pos(ImGui::GetCursorScreenPos().x + spacing * 2, ImGui::GetCursorScreenPos().y);
    ImGui::GetWindowDrawList()->AddText(label_pos, ToU32(colors.text_dim), "Speed");
    ImGui::Dummy(ImVec2(label_size.x + spacing * 2, label_size.y));
    next(56.0f + field_h * 2, field_h);
    ImGui::BeginDisabled(unlimited_ || !can_drive);
    ui->NumberField("##speed", &speed_, 0.25, 8.0, 0.25, "%.2fx", 56.0f, "Times real time");
    ImGui::EndDisabled();
    next(ImGui::CalcTextSize("As fast as possible").x + field_h + spacing, field_h);
    ImGui::BeginDisabled(!can_drive);
    ImGui::Checkbox("As fast as possible", &unlimited_);
    ImGui::EndDisabled();
    if (exec) {
        const float want = unlimited_ ? 0.0f : static_cast<float>(speed_);
        if (exec->GetRealtimeFactor() != want) exec->SetRealtimeFactor(want);
    }

    // The camera and Record menus, the status and the View menu, at the right edge
    const std::string camera = CameraLabel(env.GetModel(), renderer);
    const bool recording = recorder_.IsRecording();
    const bool finishing = recorder_.IsFinishing();
    const char* record = recording ? "Stop recording" : finishing ? "Saving..." : "Record";
    char recording_status[96] = "";
    if (recording) {
        const int seconds = static_cast<int>(recorder_.Seconds());
        std::snprintf(recording_status, sizeof(recording_status), "Recording %d:%02d \xC2\xB7 %d frames",
                      seconds / 60, seconds % 60, recorder_.Frames());
    }
    const char* status = recording  ? recording_status
                         : graph    ? "Driven by the graph run"
                         : playback ? PlaybackStatus()
                         : running  ? "Running"
                         : paused   ? "Paused"
                                    : "Stopped";
    const UiColor& tone = recording                     ? colors.error
                          : graph || playback || paused ? colors.accent_text
                          : running                     ? colors.success
                                                        : colors.pending;
    const float camera_w = ui->ButtonWidth(camera.c_str(), UiButtonSize::Regular);
    const float record_w = ui->ButtonWidth(record, UiButtonSize::Regular);
    const float pill_w = ui->StatusPillWidth(status);
    const float view_w = ui->ButtonWidth("View", UiButtonSize::Regular);
    const char* stress = stress_.panel_open ? "Hide Stress" : "Stress";
    const float stress_w = ui->ButtonWidth(stress, UiButtonSize::Regular);
    const char* inspect = inspect_.panel_open ? "Hide Inspect" : "Inspect";
    const float inspect_w = ui->ButtonWidth(inspect, UiButtonSize::Regular);
    const float right_w = camera_w + record_w + pill_w + view_w + stress_w + inspect_w + spacing * 5;
    next(right_w, button_h);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - right_w));
    if (ui->SecondaryButton(camera.c_str())) ImGui::OpenPopup("mujoco_camera_menu");
    RenderCameraMenu(env, renderer);
    next(record_w, button_h);
    if (recording) {
        if (ui->DangerButton(record)) recorder_.Stop();
    } else if (ui->SecondaryButton(record, !finishing, "The last recording is still being written.")) {
        ImGui::OpenPopup("mujoco_record_menu");
    }
    RenderRecordMenu(env, renderer);
    next(pill_w, field_h);
    ui->StatusPill("##status", status, tone);
    if (busy && !recording && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", busy);
    next(view_w, button_h);
    if (ui->SecondaryButton("View")) ImGui::OpenPopup("mujoco_view_menu");
    RenderViewMenu(env, renderer);
    next(stress_w, button_h);
    if (ui->SecondaryButton(stress)) {
        stress_.panel_open = !stress_.panel_open;
        if (stress_.panel_open) inspect_.panel_open = false;  // one side panel at a time
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Kick the robot, throw objects at it, change the ground: does the policy cope?");
    next(inspect_w, button_h);
    if (ui->SecondaryButton(inspect)) {
        inspect_.panel_open = !inspect_.panel_open;
        if (inspect_.panel_open) stress_.panel_open = false;
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("The selected body, its joints, the actuators, keyframes, Reload the model file");

    ImGui::SetCursorPosY(line_y + row_h + ImGui::GetStyle().ItemSpacing.y);
}

// =============================================================================
// Scene (3D rendered image) with camera input and overlays
// =============================================================================

void MjViewportPanel::SampleSpeed(bool stepping, double sim) {
    using clock = std::chrono::steady_clock;
    if (!stepping) {
        speed_sample_wall_ = clock::time_point{};  // the next run starts a new sample
        achieved_speed_ = 0.0;
        return;
    }
    const auto now = clock::now();
    const double wall = std::chrono::duration<double>(now - speed_sample_wall_).count();
    if (speed_sample_wall_ == clock::time_point{} || sim < speed_sample_sim_) {
        speed_sample_wall_ = now;  // first sample, or a reset sent time back
        speed_sample_sim_ = sim;
        achieved_speed_ = 0.0;
        return;
    }
    if (wall < 0.5) return;
    achieved_speed_ = (sim - speed_sample_sim_) / wall;
    speed_sample_wall_ = now;
    speed_sample_sim_ = sim;
}

void MjViewportPanel::RenderScene(MjEnvManager& env, MjRenderer& renderer) {
    PluginUi* ui = ctx_->GetUi();
    const UiColors colors = ui->Colors();

    // Room for the timeline, the plots and the episode strip below the scene
    ImVec2 avail = ImGui::GetContentRegionAvail();
    avail.y -= ImGui::GetTextLineHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y + StripsHeight();
    if (avail.x < 10.0f || avail.y < 10.0f) return;

    // Debounced resize — avoid expensive MuJoCo context recreation every frame
    // during drag. Only apply after size is stable for N frames.
    const int new_w = static_cast<int>(avail.x);
    const int new_h = static_cast<int>(avail.y);
    if (new_w != pending_w_ || new_h != pending_h_) {
        pending_w_ = new_w;
        pending_h_ = new_h;
        stable_frames_ = 0;
    } else {
        stable_frames_++;
    }
    if (stable_frames_ == kResizeDebounceFrames &&
        (pending_w_ != renderer.GetWidth() || pending_h_ != renderer.GetHeight())) {
        renderer.SetResolution(pending_w_, pending_h_);
    }

    const unsigned int tex_id = renderer.RenderFrame(env.GetModel(), env.GetData());
    if (tex_id != 0) RecordFrame(env, renderer);

    // The scene takes the mouse: dragging orbits or pans the camera instead of
    // moving the window
    ImGui::SetNextItemAllowOverlap();  // the throw dial on top of it takes its own clicks
    ImGui::InvisibleButton("##scene", avail, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    const ImVec2 p0 = ImGui::GetItemRectMin();
    const ImVec2 p1 = ImGui::GetItemRectMax();
    ImGuiIO& io = ImGui::GetIO();
    // Ctrl+drag a body: a spring pull (the Stress panel's push) instead of the camera
    HandlePush(env, renderer, ImGui::IsItemActive(), (io.MousePos.x - p0.x) / std::max(1.0f, p1.x - p0.x),
               (io.MousePos.y - p0.y) / std::max(1.0f, p1.y - p0.y), std::max(1.0f, p1.y - p0.y));
    if (ImGui::IsItemActive() && !pushing_) {
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) renderer.RotateCamera(io.MouseDelta.x, io.MouseDelta.y);
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Right)) renderer.PanCamera(io.MouseDelta.x, io.MouseDelta.y);
    }
    if (ImGui::IsItemHovered() && io.MouseWheel != 0.0f) renderer.ZoomCamera(io.MouseWheel);
    // Double-click a body: select it (Inspect shows it); with Inspect closed the
    // camera also follows it, as before Inspect
    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        const float x = (io.MousePos.x - p0.x) / std::max(1.0f, p1.x - p0.x);
        const float y = (io.MousePos.y - p0.y) / std::max(1.0f, p1.y - p0.y);
        const int body = renderer.BodyAt(env.GetModel(), env.GetData(), x, y);
        if (body > 0) {
            selected_body_ = body;
            if (!inspect_.panel_open) renderer.FollowBody(env.GetModel(), env.GetData(), body);
        }
    }
    {
        const UiColor marker = ctx_->GetUi()->Colors().success;
        const float rgba[4] = {marker.r, marker.g, marker.b, marker.a};
        renderer.Select(selected_body_, rgba);
    }

    ImDrawList* dl = ImGui::GetWindowDrawList();
    if (tex_id != 0) {
        // MuJoCo renders bottom-up: flip V
        dl->AddImage(static_cast<ImTextureID>(static_cast<uintptr_t>(tex_id)), p0, p1, ImVec2(0.0f, 1.0f),
                     ImVec2(1.0f, 0.0f));
    } else {
        dl->AddRectFilled(p0, p1, ToU32(colors.bg_panel), 4.0f);
        const char* text = renderer.IsEnabled() ? "The renderer did not draw this frame. The log has the reason."
                                                : "Drawing is off: View > Draw the scene.";
        const ImVec2 size = ImGui::CalcTextSize(text);
        dl->AddText(ImVec2(p0.x + (avail.x - size.x) * 0.5f, p0.y + (avail.y - size.y) * 0.5f),
                    ToU32(colors.text_dim), text);
    }

    // Overlays: time, steps and achieved speed at the bottom left, the camera
    // hint at the bottom right
    const SceneSnapshot snap = Snapshot(env, history_, scrub_index_);
    const RunState rs = GetRunState();
    const bool graph = rs.graph;
    const bool active = rs.active;
    const bool running = rs.running;
    // Simulated time that only moves forward: an episode's reset sends the
    // model's clock back, so a session counts its steps instead
    const double stepped = active ? sim_executor_->GetMetrics().step_count * snap.model_step : snap.time;
    SampleSpeed(running || graph, stepped);

    struct Part {
        std::string text;
        ImU32 colour;
    };
    std::vector<Part> parts;
    char buf[128];
    std::snprintf(buf, sizeof(buf), "Time %.2f s", snap.time);
    parts.push_back({buf, ToU32(colors.text)});
    if (Scrubbing()) {  // a step back: its own step, not the run's count
        std::snprintf(buf, sizeof(buf), "Episode %d step %d", snap.episode, snap.episode_step);
        parts.push_back({buf, ToU32(colors.text)});
    } else if (active) {
        std::snprintf(buf, sizeof(buf), "Step %lld",
                      static_cast<long long>(sim_executor_->GetMetrics().step_count));
        parts.push_back({buf, ToU32(colors.text)});
    }
    if (Scrubbing()) {
        parts.push_back({"A step back", ToU32(colors.accent_text)});
    } else if (running || graph) {
        const double asked = unlimited_ || graph ? 0.0 : speed_;
        if (achieved_speed_ <= 0.0) {
            std::snprintf(buf, sizeof(buf), "Starting...");
            parts.push_back({buf, ToU32(colors.text_dim)});
        } else if (asked > 0.0 && achieved_speed_ < asked * 0.9) {
            std::snprintf(buf, sizeof(buf), "%.1fx real time (asked %.1fx: this model steps slower)", achieved_speed_,
                          asked);
            parts.push_back({buf, ToU32(colors.caution)});
        } else {
            std::snprintf(buf, sizeof(buf), "%.1fx real time", achieved_speed_);
            parts.push_back({buf, ToU32(colors.success)});
        }
    } else if (playback_) {  // the Engine's playback moves the scene, at its own pace
        parts.push_back({PlaybackStatus(), ToU32(colors.accent_text)});
    } else {
        parts.push_back({active ? "Paused" : "Stopped", ToU32(colors.text_dim)});
    }

    const float pad = 8.0f;
    const float margin = 12.0f;
    const float gap = 16.0f;
    const float line_h = ImGui::GetTextLineHeight();
    float width = -gap;
    for (const auto& part : parts) width += ImGui::CalcTextSize(part.text.c_str()).x + gap;
    const ImVec2 box0(p0.x + margin, p1.y - margin - line_h - pad * 2);
    dl->AddRectFilled(box0, ImVec2(box0.x + width + pad * 2, p1.y - margin), ToU32(colors.bg_window, 0.85f), 6.0f);
    float x = box0.x + pad;
    for (const auto& part : parts) {
        dl->AddText(ImVec2(x, box0.y + pad), part.colour, part.text.c_str());
        x += ImGui::CalcTextSize(part.text.c_str()).x + gap;
    }

    // The camera hint at the bottom right, or a notice for a few seconds
    const bool noticing = !notice_.empty() && std::chrono::steady_clock::now() < notice_until_;
    std::string hint;
    if (noticing) {
        hint = notice_;
    } else {
        switch (renderer.GetCameraMode()) {
            case MjRenderer::CameraMode::Free:
                hint = inspect_.panel_open
                           ? "Drag to orbit  |  Right-drag to pan  |  Scroll to zoom  |  Double-click a body to select it"
                           : "Drag to orbit  |  Right-drag to pan  |  Scroll to zoom  |  Double-click a body to follow it";
                break;
            case MjRenderer::CameraMode::FollowBody:
                hint = "Following a body: drag to orbit  |  Scroll to zoom  |  Camera > Free to pan";
                break;
            case MjRenderer::CameraMode::ModelCamera:
                hint = "The model's own camera: Camera > Free or Follow to move the view";
                break;
        }
    }
    if (pushing_ && !noticing) hint = "Pushing: release the mouse to let go";
    const ImVec2 hint_size = ImGui::CalcTextSize(hint.c_str());
    const float hint_x = p1.x - margin - hint_size.x - pad * 2;
    if (hint_x > box0.x + width + pad * 2 + gap) {  // only when it does not cover the speed
        dl->AddRectFilled(ImVec2(hint_x, box0.y), ImVec2(p1.x - margin, p1.y - margin),
                          ToU32(colors.bg_window, 0.85f), 6.0f);
        const UiColor& tone = noticing ? (notice_failed_ ? colors.caution : colors.success) : colors.text_dim;
        dl->AddText(ImVec2(hint_x + pad, box0.y + pad), ToU32(tone), hint.c_str());
    }
    if (playback_ && live_numbers_) RenderLiveNumbers(p0.x, p0.y, env.Scene().ground);
    if (stress_.panel_open) RenderThrowDial(p1.x, p0.y);
}

// The played episode's numbers at the top left of the scene
void MjViewportPanel::RenderLiveNumbers(float scene_left, float scene_top, const MjGround& ground) {
    const UiColors colors = ctx_->GetUi()->Colors();
    struct Row {
        std::string label;
        std::string value;
    };
    std::vector<Row> rows;
    char buf[64];
    for (const auto& [label, value] : numbers_.rows) rows.push_back({label, value});  // the Engine's first
    rows.push_back({"Episode", std::to_string(numbers_.episode)});
    if (numbers_.max_steps > 0) std::snprintf(buf, sizeof(buf), "%d / %d", numbers_.step, numbers_.max_steps);
    else std::snprintf(buf, sizeof(buf), "%d", numbers_.step);
    rows.push_back({"Step", buf});
    std::snprintf(buf, sizeof(buf), "%.1f", numbers_.episode_return);
    rows.push_back({"Return", buf});
    if (numbers_.has_last_return) {
        std::snprintf(buf, sizeof(buf), "%.1f", numbers_.last_return);
        rows.push_back({"Last episode", buf});
    }
    if (!numbers_.action.empty()) {
        constexpr size_t kShown = 8;
        std::string action;
        for (size_t i = 0; i < std::min(kShown, numbers_.action.size()); ++i) {
            std::snprintf(buf, sizeof(buf), "%s%.2f", i ? "  " : "", numbers_.action[i]);
            action += buf;
        }
        if (numbers_.action.size() > kShown)
            action += "  ... (" + std::to_string(numbers_.action.size()) + " values)";
        rows.push_back({"Action", action});
    }
    if (ground.kind != MjGroundKind::Flat) {
        std::snprintf(buf, sizeof(buf), "%s \xC2\xB7 difficulty %.2f", GroundName(ground.kind), ground.difficulty);
        rows.push_back({"Ground", buf});
    }

    const float pad = 8.0f;
    const float margin = 12.0f;
    const float gap = 14.0f;
    const float line_h = ImGui::GetTextLineHeightWithSpacing();
    float label_w = 0.0f, value_w = 0.0f;
    for (const auto& row : rows) {
        label_w = std::max(label_w, ImGui::CalcTextSize(row.label.c_str()).x);
        value_w = std::max(value_w, ImGui::CalcTextSize(row.value.c_str()).x);
    }
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 box0(scene_left + margin, scene_top + margin);
    const ImVec2 box1(box0.x + pad * 2 + label_w + gap + value_w,
                      box0.y + pad * 2 + line_h * static_cast<float>(rows.size()) - ImGui::GetStyle().ItemSpacing.y);
    dl->AddRectFilled(box0, box1, ToU32(colors.bg_window, 0.85f), 6.0f);
    float y = box0.y + pad;
    for (const auto& row : rows) {
        dl->AddText(ImVec2(box0.x + pad, y), ToU32(colors.text_dim), row.label.c_str());
        dl->AddText(ImVec2(box0.x + pad + label_w + gap, y), ToU32(colors.text), row.value.c_str());
        y += line_h;
    }
}

// =============================================================================
// Episode strip
// =============================================================================

void MjViewportPanel::RenderEpisodeStrip(MjEnvManager& env) {
    const UiColors colors = ctx_->GetUi()->Colors();
    if (playback_) {  // the Engine's Policy Playback panel has the episode's numbers
        ImGui::TextColored(ToImVec4(colors.text_dim), "%s", PlaybackStatus());
        ImGui::SameLine();
        ImGui::TextColored(ToImVec4(colors.text), "%s", playback_label_.c_str());
        return;
    }
    const SceneSnapshot snap = Snapshot(env, history_, scrub_index_);
    char value[64];
    bool first = true;
    auto item = [&](const char* label, const char* text) {
        const float width = ImGui::CalcTextSize(label).x + ImGui::CalcTextSize(text).x + 24.0f;
        if (!first) SameLineIfFits(width);
        if (!first) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 16.0f);
        ImGui::TextColored(ToImVec4(colors.text_dim), "%s", label);
        ImGui::SameLine();
        ImGui::TextColored(ToImVec4(colors.text), "%s", text);
        first = false;
    };
    std::snprintf(value, sizeof(value), "%d", snap.episode);
    item("Episode", value);
    std::snprintf(value, sizeof(value), "%d / %d", snap.episode_step, snap.max_episode_steps);
    item("Episode step", value);
    std::snprintf(value, sizeof(value), "%.2f", snap.episode_reward);
    item("Episode reward", value);
    std::snprintf(value, sizeof(value), "%.4g s", snap.model_step);
    item("Model step", value);
    const MjGround& ground = env.Scene().ground;
    if (ground.kind != MjGroundKind::Flat) item("Ground", DescribeGround(ground).c_str());
}

} // namespace cyxwiz::plugin::mujoco
