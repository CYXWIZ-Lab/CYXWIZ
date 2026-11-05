#pragma once

// MuJoCo Simulation Plugin for CyxWiz Engine
//
// - IPlugin lifecycle: initialize/shutdown MuJoCo
// - IPanelProvider: MuJoCo 3D viewport and Environment Library panels
// - INodeProvider: simulation nodes for the node editor

#include <imgui.h>
#include "plugin/plugin_types.h"
#include "mj_simulation_executor.h"
#include "mj_menagerie_downloader.h"
#include "plugin/plugin_context.h"
#include "plugin/interfaces/i_panel_provider.h"
#include "plugin/interfaces/i_node_provider.h"

#include "mj_env_manager.h"
#include "mj_env_library.h"
#include "mj_renderer.h"
#include "mj_viewport_panel.h"
#include "mj_env_browser_panel.h"
#include "rl/observation_filter.h"

#include <string>

struct GLFWwindow;  // Forward declare

namespace cyxwiz::plugin::mujoco {

class MuJoCoPlugin : public IPlugin,
                     public IPanelProvider,
                     public INodeProvider {
public:
    MuJoCoPlugin() = default;
    ~MuJoCoPlugin() override = default;

    // ===== IPlugin Lifecycle =====
    bool OnLoad(PluginContext& ctx) override;
    bool OnInitialize(PluginContext& ctx) override;
    void OnShutdown(PluginContext& ctx) override;
    void OnUnload(PluginContext& ctx) override;

    const PluginManifest& GetManifest() const override { return manifest_; }
    PluginPermissionFlags GetRequiredPermissions() const override {
        return static_cast<uint32_t>(PluginPermission::UIModify);
    }
    PluginState GetState() const override { return state_; }
    void SetImGuiContext(void* ctx) override { ImGui::SetCurrentContext(static_cast<ImGuiContext*>(ctx)); }
    void* QueryInterface(const char* name) override {
        if (std::string(name) == "IPanelProvider") return static_cast<IPanelProvider*>(this);
        if (std::string(name) == "INodeProvider") return static_cast<INodeProvider*>(this);
        return nullptr;
    }

    // ===== IPanelProvider =====
    std::vector<PluginPanelInfo> GetPanels() override;
    void RenderPanel(const std::string& panel_id, bool* visible) override;

    // ===== INodeProvider =====
    std::vector<PluginNodeTypeInfo> GetNodeTypes() override;
    std::string GenerateCode(
        const std::string& node_type_name,
        const std::map<std::string, std::string>& parameters,
        const std::string& framework
    ) override;
    DynamicPinResult ResolveDynamicPins(
        const std::string& node_type_name,
        const std::map<std::string, std::string>& parameters
    ) override;
    PluginNodeEvalResult EvaluateNode(const PluginNodeEvalContext& ctx) override;

    // ===== Public API =====
    MjEnvManager& GetEnvManager() { return env_manager_; }
    const MjEnvManager& GetEnvManager() const { return env_manager_; }
    MjRenderer& GetRenderer() { return renderer_; }

    bool LoadEnvironment(const std::string& mjcf_path);
    bool IsEnvironmentLoaded() const { return env_manager_.IsLoaded(); }
    MjSimulationExecutor& GetSimExecutor() { return sim_executor_; }

    void SetMainWindow(GLFWwindow* window) { main_window_ = window; }
    void SetGraphSimActive(bool active) { viewport_panel_.SetGraphSimActive(active); }

private:
    PluginManifest manifest_;
    PluginState state_ = PluginState::Unloaded;

    MjEnvManager env_manager_;
    MjEnvLibrary env_library_;
    MjRenderer renderer_;
    MjSimulationExecutor sim_executor_;
    MjMenagerieDownloader menagerie_downloader_;
    MjViewportPanel viewport_panel_;
    MjEnvBrowserPanel browser_panel_;

    GLFWwindow* main_window_ = nullptr;

    std::string loaded_mjcf_path_;  // Path of the currently loaded MJCF model

    // Observation Filter state across the ticks of one graph simulation
    // (running normalisation); reset when a new simulation starts.
    rl::ObservationFilter obs_filter_;
    double obs_filter_last_time_ = -1.0;
    bool viewport_visible_ = true;
    bool renderer_needs_init_ = false;
};

} // namespace cyxwiz::plugin::mujoco
