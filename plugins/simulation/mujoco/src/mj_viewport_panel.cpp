#include "mj_viewport_panel.h"
#include "mj_simulation_executor.h"

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
constexpr const char* kGraphReason =
    "The graph run is stepping this model. Stop the graph run to control it here.";

ImVec4 ToImVec4(const UiColor& c) { return ImVec4(c.r, c.g, c.b, c.a); }
ImU32 ToU32(const UiColor& c, float alpha = 1.0f) {
    return ImGui::ColorConvertFloat4ToU32(ImVec4(c.r, c.g, c.b, c.a * alpha));
}

// Moves to the same line when `width` still fits, else starts a new line, so
// nothing hides off the right edge of a narrow panel.
void SameLineIfFits(float width) {
    ImGui::SameLine();
    if (ImGui::GetContentRegionAvail().x < width) ImGui::NewLine();
}

// What the panel shows of the model, read in one physics transaction.
struct SceneSnapshot {
    double time = 0.0;
    double model_step = 0.0;  // timestep x frame skip
    int episode = 1;
    int episode_step = 0;
    int max_episode_steps = 0;
    float episode_reward = 0.0f;
};

SceneSnapshot Snapshot(MjEnvManager& env) {
    SceneSnapshot s;
    std::lock_guard phys_lock(env.GetPhysicsMutex());
    if (const mjData* d = env.GetData()) s.time = d->time;
    if (const mjModel* m = env.GetModel()) s.model_step = m->opt.timestep * env.GetFrameSkip();
    s.episode = env.GetEpisodeCount() + 1;
    s.episode_step = env.GetCurrentStep();
    s.max_episode_steps = env.GetMaxEpisodeSteps();
    s.episode_reward = env.GetEpisodeReward();
    return s;
}

}  // namespace

void MjViewportPanel::Render(MjEnvManager& env, MjRenderer& renderer, bool* visible) {
    if (!visible || !*visible) return;

    // A model file the user picked: loaded before this frame's ImGui calls,
    // since loading switches GL contexts.
    if (!pending_open_path_.empty()) {
        const std::string path = std::move(pending_open_path_);
        pending_open_path_.clear();
        if (open_model_) open_model_(path);
    }

    // Auto-clear graph_sim_active if EvaluateNode hasn't been called recently
    if (graph_sim_active_ && ++graph_sim_frames_ > 30) {  // ~0.5s at 60fps without EvaluateNode call
        graph_sim_active_ = false;
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

    if (!env.IsLoaded()) {
        RenderEmpty();
    } else if (!renderer.IsInitialized()) {
        ctx_->GetUi()->EmptyState("The 3D view could not start",
                                  "The model is loaded, but its renderer did not start. The log has the reason.");
    } else {
        RenderControls(env, renderer);
        RenderScene(env, renderer);
        RenderEpisodeStrip(env);
    }

    ImGui::End();
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

    const bool graph = graph_sim_active_;
    const bool active = exec && !graph && exec->GetMode() != SimMode::Stopped;
    const SimState state = exec ? exec->GetState() : SimState::Idle;
    const bool running = active && state == SimState::Running;
    const bool paused = active && state == SimState::Paused;
    const bool can_drive = exec && !graph;
    const char* busy = graph ? kGraphReason : nullptr;

    // The model's name and source, then the controls centred on its two lines
    const float row_y = ImGui::GetCursorPosY();
    ImGui::BeginGroup();
    ImGui::TextColored(ToImVec4(colors.text_bright), "%s", model_title_.empty() ? "Model" : model_title_.c_str());
    if (!model_subtitle_.empty()) ImGui::TextColored(ToImVec4(colors.text_dim), "%s", model_subtitle_.c_str());
    ImGui::EndGroup();
    const float button_h = ui->ButtonHeight(UiButtonSize::Regular);
    const float row_h = std::max(ImGui::GetItemRectSize().y, button_h);
    float line_y = row_y;
    auto next = [&](float width, float height) {
        const float before = ImGui::GetCursorPosY();
        SameLineIfFits(width);
        if (ImGui::GetCursorPosY() > before) line_y = ImGui::GetCursorPosY();  // wrapped
        ImGui::SetCursorPosY(line_y + (row_h - height) * 0.5f);
    };
    const float spacing = ImGui::GetStyle().ItemSpacing.x;

    if (running) {
        next(ui->ButtonWidth("Pause", UiButtonSize::Regular) + spacing * 2, button_h);
        if (ui->PrimaryButton("Pause")) exec->Pause();
    } else {
        const char* run = paused ? "Resume" : "Run";
        next(ui->ButtonWidth(run, UiButtonSize::Regular) + spacing * 2, button_h);
        if (ui->PrimaryButton(run, can_drive, busy)) {
            if (paused) exec->Resume();
            else exec->StartManualControl();
        }
    }
    next(ui->ButtonWidth("Step", UiButtonSize::Regular), button_h);
    if (ui->SecondaryButton("Step", can_drive, busy)) {
        if (running) exec->Pause();
        exec->SingleStep();
    }
    next(ui->ButtonWidth("Reset", UiButtonSize::Regular), button_h);
    if (ui->SecondaryButton("Reset", can_drive, busy)) {
        std::lock_guard phys_lock(env.GetPhysicsMutex());
        env.Reset();
    }
    if (active) {
        next(ui->ButtonWidth("Stop", UiButtonSize::Regular), button_h);
        if (ui->SecondaryButton("Stop")) exec->Stop();
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

    // Status and the View menu, at the right edge
    const char* status = graph ? "Driven by the graph run" : running ? "Running" : paused ? "Paused" : "Stopped";
    const UiColor& tone = graph || paused ? colors.accent_text : running ? colors.success : colors.pending;
    const float pill_w = ui->StatusPillWidth(status);
    const float view_w = ui->ButtonWidth("View", UiButtonSize::Regular);
    const float right_w = pill_w + spacing + view_w;
    next(right_w, field_h);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - right_w));
    ui->StatusPill("##status", status, tone);
    if (graph && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", kGraphReason);
    next(view_w, button_h);
    if (ui->SecondaryButton("View")) ImGui::OpenPopup("mujoco_view_menu");
    RenderViewMenu(env, renderer);

    ImGui::SetCursorPosY(line_y + row_h + ImGui::GetStyle().ItemSpacing.y);
}

void MjViewportPanel::RenderViewMenu(MjEnvManager& env, MjRenderer& renderer) {
    if (!ImGui::BeginPopup("mujoco_view_menu")) return;
    ImGui::TextDisabled("Show in the scene");
    struct VisItem {
        const char* label;
        int flag;
    };
    static constexpr VisItem kItems[] = {
        {"Contact points", mjVIS_CONTACTPOINT},
        {"Contact forces", mjVIS_CONTACTFORCE},
        {"Joints", mjVIS_JOINT},
        {"Actuators", mjVIS_ACTUATOR},
    };
    // Ticking an option keeps the menu open, to set several
    ImGui::PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);
    for (const auto& item : kItems) {
        const bool on = renderer.VisFlag(item.flag);
        if (ImGui::MenuItem(item.label, nullptr, on)) renderer.SetVisFlag(item.flag, !on);
    }
    const bool wireframe = renderer.Wireframe();
    if (ImGui::MenuItem("Wireframe", nullptr, wireframe)) renderer.SetWireframe(!wireframe);
    ImGui::Separator();
    const bool drawing = renderer.IsEnabled();
    if (ImGui::MenuItem("Draw the scene", nullptr, drawing)) renderer.SetEnabled(!drawing);
    ImGui::PopItemFlag();
    if (ImGui::MenuItem("Reset camera")) renderer.ResetCamera(env.GetModel());
    ImGui::Separator();
    ImGui::TextDisabled("Render size %d x %d", renderer.GetWidth(), renderer.GetHeight());
    ImGui::EndPopup();
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

    // Room for the episode strip below the scene
    ImVec2 avail = ImGui::GetContentRegionAvail();
    avail.y -= ImGui::GetTextLineHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y;
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

    // The scene takes the mouse: dragging orbits or pans the camera instead of
    // moving the window
    ImGui::InvisibleButton("##scene", avail, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    const ImVec2 p0 = ImGui::GetItemRectMin();
    const ImVec2 p1 = ImGui::GetItemRectMax();
    ImGuiIO& io = ImGui::GetIO();
    if (ImGui::IsItemActive()) {
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) renderer.RotateCamera(io.MouseDelta.x, io.MouseDelta.y);
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Right)) renderer.PanCamera(io.MouseDelta.x, io.MouseDelta.y);
    }
    if (ImGui::IsItemHovered() && io.MouseWheel != 0.0f) renderer.ZoomCamera(io.MouseWheel);

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
    const SceneSnapshot snap = Snapshot(env);
    const bool graph = graph_sim_active_;
    const bool active = sim_executor_ && !graph && sim_executor_->GetMode() != SimMode::Stopped;
    const bool running = active && sim_executor_->GetState() == SimState::Running;
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
    if (active) {
        std::snprintf(buf, sizeof(buf), "Step %lld",
                      static_cast<long long>(sim_executor_->GetMetrics().step_count));
        parts.push_back({buf, ToU32(colors.text)});
    }
    if (running || graph) {
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

    const char* hint = "Drag to orbit  |  Right-drag to pan  |  Scroll to zoom";
    const ImVec2 hint_size = ImGui::CalcTextSize(hint);
    const float hint_x = p1.x - margin - hint_size.x - pad * 2;
    if (hint_x > box0.x + width + pad * 2 + gap) {  // only when it does not cover the speed
        dl->AddRectFilled(ImVec2(hint_x, box0.y), ImVec2(p1.x - margin, p1.y - margin),
                          ToU32(colors.bg_window, 0.85f), 6.0f);
        dl->AddText(ImVec2(hint_x + pad, box0.y + pad), ToU32(colors.text_dim), hint);
    }
}

// =============================================================================
// Episode strip
// =============================================================================

void MjViewportPanel::RenderEpisodeStrip(MjEnvManager& env) {
    const UiColors colors = ctx_->GetUi()->Colors();
    const SceneSnapshot snap = Snapshot(env);
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
}

} // namespace cyxwiz::plugin::mujoco
