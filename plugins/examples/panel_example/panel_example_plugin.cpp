// Example plugin with a panel, built from the installed SDK alone (SDK-10).
//
// It shows the three things a plugin with a screen needs:
//   1. SetImGuiContext: draw in the Engine's ImGui context;
//   2. IPanelProvider: the panel in the Engine's menus, drawn every frame it is open;
//   3. PluginContext::GetUi(): the Engine's own buttons, typed number fields and theme colours,
//      so the panel looks like the Engine's.
// Link CyxWiz::PluginSDK and CyxWiz::PluginImGui (CMakeLists.txt): the ImGui the Engine was built
// with. A plugin built with any other ImGui is refused when it loads (the build fingerprint).
#include "plugin/plugin_context.h"
#include "plugin/plugin_types.h"
#include "plugin/plugin_ui.h"
#include "plugin/interfaces/i_panel_provider.h"

#include <imgui.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace cyxwiz::plugin::examples {

namespace {
constexpr const char* kPanelId = "com.cyxwiz.examples.panel.oscillator";
}

// A damped spring, x'' = -(k/m) x - (c/m) x', shown as a small live panel: a typed field for each
// setting with its unit, a primary action, and the position plotted with ImGui's own line plot.
class PanelExamplePlugin : public IPlugin, public IPanelProvider {
public:
    bool OnLoad(PluginContext& ctx) override {
        ctx_ = &ctx;
        state_ = PluginState::Loaded;
        return true;
    }
    bool OnInitialize(PluginContext&) override {
        state_ = PluginState::Active;
        return true;
    }
    void OnShutdown(PluginContext&) override { state_ = PluginState::Loaded; }
    void OnUnload(PluginContext&) override {
        ctx_ = nullptr;
        state_ = PluginState::Unloaded;
    }

    const PluginManifest& GetManifest() const override { return manifest_; }
    PluginPermissionFlags GetRequiredPermissions() const override {
        return static_cast<uint32_t>(PluginPermission::UIModify);
    }
    PluginState GetState() const override { return state_; }
    void SetImGuiContext(void* imgui_context) override {
        ImGui::SetCurrentContext(static_cast<ImGuiContext*>(imgui_context));
    }
    void* QueryInterface(const char* name) override {
        if (std::string(name) == "IPanelProvider") return static_cast<IPanelProvider*>(this);
        return nullptr;
    }

    std::vector<PluginPanelInfo> GetPanels() override {
        PluginPanelInfo panel;
        panel.panel_id = kPanelId;
        panel.title = "Spring Example";
        panel.category = "Plugins";
        return {panel};
    }

    void RenderPanel(const std::string& panel_id, bool* visible) override {
        if (panel_id != kPanelId) return;
        if (!ImGui::Begin("Spring Example", visible)) {
            ImGui::End();
            return;
        }
        PluginUi* ui = ctx_ ? ctx_->GetUi() : nullptr;
        if (!ui) {  // a host without windows
            ImGui::TextUnformatted("This host has no Engine widgets.");
            ImGui::End();
            return;
        }
        ImGui::PushStyleColor(ImGuiCol_Text, ToImVec4(ui->Colors().text_dim));
        ImGui::TextWrapped("A damped spring, drawn by a plugin built from the installed SDK.");
        ImGui::PopStyleColor();

        // Labels in one column sized to the longest, typed fields in the other: nothing depends
        // on the window's width.
        if (ImGui::BeginTable("##settings", 2, ImGuiTableFlags_SizingFixedFit)) {
            Field(ui, "Mass (kg)", "##mass", &mass_kg_, 0.01, 100.0, 0.1);
            Field(ui, "Stiffness (N/m)", "##stiffness", &stiffness_n_per_m_, 0.0, 10000.0, 1.0);
            Field(ui, "Damping (N s/m)", "##damping", &damping_n_s_per_m_, 0.0, 1000.0, 0.1);
            ImGui::EndTable();
        }

        if (ui->PrimaryButton(running_ ? "Stop" : "Release")) {
            if (!running_) Reset();
            running_ = !running_;
        }
        if (running_) {
            Step(ImGui::GetIO().DeltaTime);
            ctx_->RequestFullFrameRate();  // keep the Engine drawing while it moves
        }
        ImGui::PlotLines("##position", trace_.data(), static_cast<int>(trace_.size()), 0, "position (m)", -1.0f,
                         1.0f, ImVec2(-1.0f, 120.0f));
        ImGui::End();
    }

private:
    static ImVec4 ToImVec4(const UiColor& c) { return ImVec4(c.r, c.g, c.b, c.a); }

    static void Field(PluginUi* ui, const char* label, const char* id, double* value, double lo, double hi,
                      double step) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(label);
        ImGui::TableNextColumn();
        ui->NumberField(id, value, lo, hi, step, "%.2f", 120.0f);
    }

    void Reset() {
        position_m_ = 1.0;
        velocity_m_per_s_ = 0.0;
        carry_s_ = 0.0;
        trace_.assign(240, 0.0f);
    }

    // Semi-implicit Euler. The step is 1 ms, or smaller when the settings need it to stay stable
    // (a tenth of m/c and of 1/omega), and the time a frame leaves over carries to the next frame,
    // so the spring keeps real time at any frame rate.
    void Step(float frame_seconds) {
        double step = 0.001;
        if (damping_n_s_per_m_ > 0.0) step = std::fmin(step, 0.1 * mass_kg_ / damping_n_s_per_m_);
        if (stiffness_n_per_m_ > 0.0) step = std::fmin(step, 0.1 * std::sqrt(mass_kg_ / stiffness_n_per_m_));
        carry_s_ += std::fmin(static_cast<double>(frame_seconds), 0.1);  // at most 0.1 s after a stall
        while (carry_s_ >= step) {
            const double accel = (-stiffness_n_per_m_ * position_m_ - damping_n_s_per_m_ * velocity_m_per_s_) / mass_kg_;
            velocity_m_per_s_ += accel * step;
            position_m_ += velocity_m_per_s_ * step;
            carry_s_ -= step;
        }
        trace_.erase(trace_.begin());
        trace_.push_back(static_cast<float>(position_m_));
    }

    PluginManifest manifest_;
    PluginState state_ = PluginState::Unloaded;
    PluginContext* ctx_ = nullptr;
    double mass_kg_ = 1.0;
    double stiffness_n_per_m_ = 40.0;
    double damping_n_s_per_m_ = 0.8;
    double position_m_ = 1.0;
    double velocity_m_per_s_ = 0.0;
    double carry_s_ = 0.0;  // simulated time owed from earlier frames
    bool running_ = false;
    std::vector<float> trace_ = std::vector<float>(240, 0.0f);
};

}  // namespace cyxwiz::plugin::examples

CYXWIZ_PLUGIN_ENTRY(cyxwiz::plugin::examples::PanelExamplePlugin)
