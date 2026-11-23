// The MuJoCo Viewport's Inspect panel (TOFIX212 P2, approved board 3): the
// body double-clicked in the scene (mass, centre of mass, speed, its joints),
// the actuators' controls typed in, keyframes saved with the project, and
// Reload the model file. It changes only the Viewport's own run; a played
// policy's copy or a graph run's model is shown, not changed.

#include "mj_viewport_panel.h"
#include "mj_simulation_executor.h"
#include "mj_ui_helpers.h"
#include "env/mj_inspect.h"
#include "env/mj_physics.h"

#include "plugin/plugin_context.h"
#include "plugin/plugin_ui.h"

#include <imgui.h>

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <string>

namespace cyxwiz::plugin::mujoco {
namespace {

const char* JointKind(int type) {
    switch (type) {
        case mjJNT_FREE: return "free joint: moves and turns freely";
        case mjJNT_BALL: return "ball joint: turns about every axis";
        case mjJNT_SLIDE: return "slide";
        default: return "hinge";
    }
}

std::string ClockNow() {  // "12:04"
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#if defined(_WIN32)
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char text[16];
    std::strftime(text, sizeof(text), "%H:%M", &local);
    return text;
}

}  // namespace

std::string MjViewportPanel::InspectBlocked(MjEnvManager& env) const {
    if (graph_sim_active_) return kGraphReason;
    if (playback_) return "A policy plays this copy: Inspect shows it and changes nothing.";
    if (!env.IsLoaded()) return "Open a model first.";
    return {};
}

void MjViewportPanel::ApplyHeldControls(MjEnvManager& env) {
    if (!sim_executor_) return;
    std::vector<float> controls;
    if (inspect_.hold_controls)
        for (const double c : typed_controls_) controls.push_back(static_cast<float>(c));
    std::lock_guard phys_lock(env.GetPhysicsMutex());
    sim_executor_->SetHeldControls(std::move(controls));
}

void MjViewportPanel::RenderInspectPanel(MjEnvManager& env, MjRenderer& renderer) {
    PluginUi* ui = ctx_->GetUi();
    const UiColors colors = ui->Colors();
    const std::string blocked = InspectBlocked(env);
    const bool can = blocked.empty();
    const bool running = sim_executor_ && sim_executor_->GetState() == SimState::Running;
    const char* joint_why = !can ? blocked.c_str() : "Pause the run to set a joint.";
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    constexpr float kName = 70.0f;

    auto section = [&](const char* title) {
        ImGui::Spacing();
        ImGui::TextColored(ToImVec4(colors.text_dim), "%s", title);
    };
    auto hint = [&](const std::string& text) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(ToImVec4(colors.text_dim), "%s", text.c_str());
        ImGui::PopTextWrapPos();
    };

    ImGui::TextColored(ToImVec4(colors.text_bright), "Inspect");
    hint(can ? "This Viewport's run." : blocked);

    // What the panel shows, read under the physics lock
    MjBodyInfo body;
    std::vector<MjActuatorInfo> actuators;
    std::vector<std::string> model_keys;
    {
        std::lock_guard phys_lock(env.GetPhysicsMutex());
        const mjModel* m = env.GetModel();
        const mjData* d = env.GetData();
        if (m && d) {
            if (selected_body_ >= m->nbody) selected_body_ = -1;
            if (selected_body_ > 0) body = InspectBody(m, d, selected_body_, env.Robot());
            actuators = InspectActuators(m, d);
            model_keys = ModelKeyframes(m);
        }
    }

    // The selected body
    section("Selected body");
    if (body.id < 0) {
        hint("Double-click a body in the scene to select it. Ctrl+drag still pushes.");
    } else {
        ImGui::TextColored(ToImVec4(colors.text_bright), "%s", body.name.c_str());
        ImGui::SameLine();
        ImGui::TextColored(ToImVec4(colors.text_dim), "body %d \xC2\xB7 parent %s", body.id, body.parent_name.c_str());
        if (ImGui::BeginTable("##inspect_body", 2, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("##l", ImGuiTableColumnFlags_WidthFixed, 110.0f);
            ImGui::TableSetupColumn("##v", ImGuiTableColumnFlags_WidthStretch);
            auto row = [&](const char* label, const std::string& value) {
                ImGui::TableNextColumn();
                ImGui::TextColored(ToImVec4(colors.text_dim), "%s", label);
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(value.c_str());
            };
            char text[96];
            std::snprintf(text, sizeof(text), "%.2f kg (robot %.1f kg)", body.mass, body.robot_mass);
            row("Mass", text);
            std::snprintf(text, sizeof(text), "%.2f, %.2f, %.2f m", body.centre_of_mass[0], body.centre_of_mass[1],
                          body.centre_of_mass[2]);
            row("Centre of mass", text);
            std::snprintf(text, sizeof(text), "%.2f m/s \xC2\xB7 %.1f rad/s", body.speed, body.angular_speed);
            row("Speed", text);
            ImGui::EndTable();
        }
        if (ui->LinkButton("Follow it with the camera")) renderer.FollowBody(env.GetModel(), env.GetData(), body.id);
        if (body.parent > 0) {
            ImGui::SameLine();
            if (ui->LinkButton("Select parent")) selected_body_ = body.parent;
        }
        ImGui::SameLine();
        if (ui->LinkButton("Clear")) selected_body_ = -1;

        // Its joints
        if (!body.joints.empty()) section(running ? "Its joints" : "Its joints \xC2\xB7 typed while paused");
        for (const auto& joint : body.joints) {
            ImGui::PushID(joint.id);
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(joint.name.c_str());
            if (joint.scalar) {
                ImGui::SameLine(kName);
                const bool hinge = joint.type == mjJNT_HINGE;
                double value = joint.value;
                const double low = joint.limited ? joint.low : -1.0e6;
                const double high = joint.limited ? joint.high : 1.0e6;
                ImGui::BeginDisabled(!can || running);
                const bool changed =
                    ui->NumberField("##joint", &value, low, high, hinge ? 0.05 : 0.01, hinge ? "%.2f rad" : "%.3f m",
                                    80.0f, "The joint's position (the state is computed again)");
                ImGui::EndDisabled();
                if ((!can || running) && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                    ImGui::SetTooltip("%s", joint_why);
                if (changed && can && !running) {
                    std::lock_guard phys_lock(env.GetPhysicsMutex());
                    SetJointValue(env.GetModel(), env.GetMutableData(), joint.id, value);
                }
                if (joint.limited) {
                    ImGui::SameLine();
                    ImGui::TextColored(ToImVec4(colors.text_dim), "%.2f \xE2\x80\xA6 %.2f", joint.low, joint.high);
                }
                char line[128];
                std::snprintf(line, sizeof(line), "%s \xC2\xB7 velocity %.2f %s \xC2\xB7 stiffness %g, damping %g",
                              JointKind(joint.type), joint.velocity, hinge ? "rad/s" : "m/s", joint.stiffness,
                              joint.damping);
                hint(line);
            } else {
                ImGui::SameLine(kName);
                ImGui::TextColored(ToImVec4(colors.text_dim), "%s", JointKind(joint.type));
            }
            ImGui::PopID();
        }
    }

    // The actuators: each control typed in, its force
    section("Actuators \xC2\xB7 control typed in \xC2\xB7 force");
    if (actuators.empty()) {
        hint("This model has no actuators.");
    } else {
        // Held: the fields keep what was typed (the run steps with it); else they
        // show the model's own controls (a Reset or a keyframe changes them)
        if (typed_controls_.size() != actuators.size() || !inspect_.hold_controls) {
            typed_controls_.clear();
            for (const auto& a : actuators) typed_controls_.push_back(a.control);
        }
        bool controls_changed = false;
        for (size_t i = 0; i < actuators.size(); ++i) {
            const auto& a = actuators[i];
            ImGui::PushID(a.id);
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(a.name.c_str());
            ImGui::SameLine(kName);
            const double low = a.limited ? a.low : -1.0e6;
            const double high = a.limited ? a.high : 1.0e6;
            ImGui::BeginDisabled(!can);
            if (ui->NumberField("##control", &typed_controls_[i], low, high, 0.1, "%.2f", 64.0f,
                                "The actuator's control (its range when the model limits it)")) {
                controls_changed = true;
                if (!running) {  // paused: the force shows now; a run steps with zeros unless held
                    std::lock_guard phys_lock(env.GetPhysicsMutex());
                    if (mjData* d = env.GetMutableData(); d && a.id < env.GetModel()->nu) {
                        d->ctrl[a.id] = typed_controls_[i];
                        mj_forward(env.GetModel(), d);
                    }
                }
            }
            ImGui::EndDisabled();
            if (!can && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("%s", blocked.c_str());
            ImGui::SameLine();
            ImGui::TextColored(ToImVec4(colors.text_dim), "%.1f", a.force);
            ImGui::PopID();
        }
        ImGui::BeginDisabled(!can);
        if (ImGui::Checkbox("Hold these controls while it runs", &inspect_.hold_controls)) controls_changed = true;
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ui->LinkButton("Set all to 0", can)) {
            std::fill(typed_controls_.begin(), typed_controls_.end(), 0.0);
            controls_changed = true;
        }
        if (controls_changed) ApplyHeldControls(env);
    }

    // Keyframes: the model file's own, and those saved with the project
    section("Keyframes");
    const std::filesystem::path project = ctx_->GetProjectDir();
    const std::string model_path = env.ModelPath();
    if (place_path_ != model_path) {
        place_path_ = model_path;
        place_ = {};
        place_.ref.path = model_path;
        if (model_place_) place_ = model_place_(model_path);
    }
    const MjModelRef& model_ref = place_.ref;
    if (keyframes_model_ != model_path) {
        keyframes_model_ = model_path;
        keyframes_ = project.empty() ? std::vector<MjKeyframe>{} : LoadKeyframes(KeyframesFile(project, model_ref));
    }
    auto save = [&]() {
        if (project.empty()) return;  // kept for this session only
        std::string error;
        if (!SaveKeyframes(KeyframesFile(project, model_ref), keyframes_, model_path, error)) Notify(error, true);
    };
    for (size_t k = 0; k < model_keys.size(); ++k) {
        ImGui::PushID(static_cast<int>(k) + 10000);
        ImGui::TextUnformatted(model_keys[k].c_str());
        ImGui::SameLine();
        ImGui::TextColored(ToImVec4(colors.text_dim), "\xC2\xB7 in the model file");
        ImGui::SameLine();
        if (ui->LinkButton("Load", can)) {
            std::lock_guard phys_lock(env.GetPhysicsMutex());
            mj_resetDataKeyframe(env.GetModel(), env.GetMutableData(), static_cast<int>(k));
            mj_forward(env.GetModel(), env.GetMutableData());
        }
        ImGui::PopID();
    }
    int remove = -1;
    for (size_t k = 0; k < keyframes_.size(); ++k) {
        ImGui::PushID(static_cast<int>(k));
        ImGui::TextUnformatted(keyframes_[k].name.c_str());
        if (!keyframes_[k].saved.empty()) {
            ImGui::SameLine();
            ImGui::TextColored(ToImVec4(colors.text_dim), "\xC2\xB7 saved %s", keyframes_[k].saved.c_str());
        }
        ImGui::SameLine();
        if (ui->LinkButton("Load", can)) {
            std::string error;
            bool loaded = false;
            {
                std::lock_guard phys_lock(env.GetPhysicsMutex());
                loaded = ApplyKeyframe(env.GetModel(), env.GetMutableData(), keyframes_[k], error);
            }
            if (!loaded) Notify(error, true);
        }
        ImGui::SameLine();
        if (ui->LinkButton("Delete", can)) remove = static_cast<int>(k);
        ImGui::PopID();
    }
    if (remove >= 0) {
        keyframes_.erase(keyframes_.begin() + remove);
        save();
    }
    if (model_keys.empty() && keyframes_.empty()) hint("None yet: save the state to come back to it.");
    ImGui::SetNextItemWidth(std::max(80.0f, ImGui::GetContentRegionAvail().x -
                                                ui->ButtonWidth("Save the state", UiButtonSize::Small) - spacing));
    ImGui::InputTextWithHint("##keyframe_name", "Name, e.g. stand", keyframe_name_, sizeof(keyframe_name_));
    ImGui::SameLine();
    const bool named = keyframe_name_[0] != '\0';
    if (ui->SecondaryButton("Save the state", can && named, !can ? blocked.c_str() : "Name it first.",
                            UiButtonSize::Small)) {
        MjKeyframe key;
        {
            std::lock_guard phys_lock(env.GetPhysicsMutex());
            key = CaptureKeyframe(env.GetModel(), env.GetData(), keyframe_name_, ClockNow());
        }
        const auto same = std::find_if(keyframes_.begin(), keyframes_.end(),
                                       [&key](const MjKeyframe& k) { return k.name == key.name; });
        if (same != keyframes_.end()) *same = std::move(key);  // the same name: replaced
        else keyframes_.push_back(std::move(key));
        keyframe_name_[0] = '\0';
        save();
    }
    hint(project.empty() ? "No project is open: keyframes are kept until the Engine closes."
                         : "Saved with the project (mujoco/keyframes).");

    // Physics: the Viewport's own run; a played copy's are its environment's
    section("Physics");
    mjOption option{};
    mjOption file{};
    bool has_option = false;
    int frame_skip = 1;
    {
        std::lock_guard phys_lock(env.GetPhysicsMutex());
        if (const mjOption* now = env.Option(); now && env.FileOption()) {
            option = *now;
            file = *env.FileOption();
            has_option = true;
        }
        frame_skip = env.GetFrameSkip();
    }
    if (has_option) {
        bool changed = false;
        constexpr float kLabel = 84.0f;
        auto label = [&](const char* text) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(ToImVec4(colors.text_dim), "%s", text);
            ImGui::SameLine(kLabel);
        };
        ImGui::BeginDisabled(!can);
        label("Timestep");
        changed |= ui->NumberField("##timestep", &option.timestep, 0.0001, 0.1, 0.001, "%.4f s", 80.0f,
                                   "The physics step (an environment step is this times its frame skip)");
        ImGui::SameLine();
        ImGui::TextColored(ToImVec4(colors.text_dim), "frame skip %d", frame_skip);
        label("Integrator");
        int integrator = std::clamp(option.integrator, 0, 3);
        if (ui->SegmentedControl("##integrator", kIntegratorNames, 4, &integrator)) {
            option.integrator = integrator;
            changed = true;
        }
        label("Solver");
        int solver = std::clamp(option.solver, 0, 2);
        if (ui->SegmentedControl("##solver", kSolverNames, 3, &solver)) {
            option.solver = solver;
            changed = true;
        }
        ImGui::SameLine();
        double iterations = option.iterations;
        if (ui->NumberField("##iterations", &iterations, 1.0, 1000.0, 10.0, "%.0f", 48.0f, "Solver iterations")) {
            option.iterations = static_cast<int>(iterations);
            changed = true;
        }
        label("Gravity");
        static const char* kAxes[3] = {"##gx", "##gy", "##gz"};
        for (int i = 0; i < 3; ++i) {
            if (i) ImGui::SameLine();
            changed |= ui->NumberField(kAxes[i], &option.gravity[i], -100.0, 100.0, 0.5, "%.2f", 52.0f,
                                       "m/s\xC2\xB2 along x, y, z");
        }
        float row = 0.0f;
        const float avail = ImGui::GetContentRegionAvail().x;
        for (const auto& s : PhysicsSwitches()) {
            const float w = ui->ToggleChipWidth(s.label);
            if (row > 0.0f && row + spacing + w <= avail) ImGui::SameLine();
            else row = 0.0f;
            bool on = SwitchOn(option, s);
            if (ui->ToggleChip(s.label, s.label, &on)) {
                SetSwitch(option, s, on);
                changed = true;
            }
            row += (row > 0.0f ? spacing : 0.0f) + w;
        }
        ImGui::EndDisabled();
        if (changed && can) {
            std::lock_guard phys_lock(env.GetPhysicsMutex());
            env.SetOption(option);
        }
        const std::string differs = PhysicsChanges(option, file);
        if (ui->LinkButton("Back to the model file's values", can && !differs.empty())) {
            std::lock_guard phys_lock(env.GetPhysicsMutex());
            env.ClearOption();
        }
        ImGui::SameLine();
        ImGui::TextColored(ToImVec4(colors.text_dim), "%s",
                           differs.empty() ? "as in the file" : ("changed: " + differs).c_str());
    }

    // A second policy beside the played one (Policy Playback starts it: a
    // plugin cannot open the Engine's panels)
    section("Compare");
    if (playback_ && ghost_numbers_.active) {
        ImGui::TextColored(ToImVec4(colors.text_dim), "Ghost");
        ImGui::SameLine();
        ImGui::TextColored(ToImVec4(colors.accent_text), "%s", ghost_numbers_.name.c_str());
        hint(ghost_numbers_.not_drawn.empty()
                 ? "Plays beside the policy on the same ground and seed, drawn see-through."
                 : "Not drawn: " + ghost_numbers_.not_drawn);
        hint("Choose another or remove it in Policy Playback (Compare with).");
    } else {
        hint("A second saved policy can play beside a played one as a see-through ghost: Policy Playback, "
             "Compare with, then Show as a ghost.");
    }

    // The model: where it is (the same name on every PC), read again, copied into the project
    section("Model");
    const bool shared = place_.shared;
    std::string where;
    switch (model_ref.kind) {
        case MjRefKind::Builtin: where = "Built in \xC2\xB7 " + model_ref.file; break;
        case MjRefKind::Menagerie:
            where = "Menagerie " + model_ref.robot + " \xC2\xB7 " +
                    (model_ref.commit.empty() ? std::string("version unknown") : "version " + ShortCommit(model_ref.commit)) +
                    (shared ? " \xC2\xB7 shared download" : " \xC2\xB7 copy in this project");
            break;
        case MjRefKind::Project: where = "In this project \xC2\xB7 " + model_ref.file; break;
        case MjRefKind::Path: where = model_path + " \xC2\xB7 not in the project"; break;
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(where.c_str());
    ImGui::PopTextWrapPos();
    if (ui->SecondaryButton("Reload the model file", can && reload_model_, blocked.c_str(), UiButtonSize::Small)) {
        std::string error;
        if (reload_model_(error)) Notify("Reloaded " + std::filesystem::path(model_path).filename().string());
        else Notify("Reload: " + error, true);
    }
    // A copy promises other PCs the same version: only a download whose version is known.
    // A file outside the project: its folder is copied (it then travels with the project).
    const bool pinned = !model_ref.commit.empty();
    const bool outside = model_ref.kind == MjRefKind::Path;
    if ((shared || outside) && copy_model_ && !project.empty()) {
        ImGui::SameLine();
        if (ui->SecondaryButton("Copy into the project", outside || pinned,
                                "Its version is unknown (downloaded before versions were kept): Download again (pinned) "
                                "in the Environment Library first.",
                                UiButtonSize::Small))
            copy_model_(model_ref, model_path);
    }
    hint(shared    ? "This is a shared download used by every project: copy it into the project before editing its "
                     "file. Reload keeps the camera and the scene; the state starts over."
         : outside ? "Other PCs will not find this file: copy its folder into the project. Reload keeps the camera "
                     "and the scene; the state starts over."
                   : "Reload keeps the camera and the scene; the state starts over.");
}

}  // namespace cyxwiz::plugin::mujoco
