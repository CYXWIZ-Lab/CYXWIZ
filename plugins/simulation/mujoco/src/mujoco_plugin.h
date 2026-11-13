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
#include "plugin/interfaces/i_command_provider.h"
#include "plugin/interfaces/i_simulation_provider.h"
#include "plugin/interfaces/i_environment_provider.h"
#include "plugin/interfaces/i_preferences_provider.h"
#include "plugin/plugin_ui.h"

#include "mj_env_manager.h"
#include "mj_graph_clock.h"
#include "mj_env_library.h"
#include "mj_renderer.h"
#include "mj_viewport_panel.h"
#include "mj_env_browser_panel.h"
#include "env/mj_view_slot.h"
#include "rl/observation_filter.h"

#include <optional>

#include <cstdint>
#include <map>
#include <mutex>
#include <memory>
#include <string>

struct GLFWwindow;  // Forward declare

namespace cyxwiz::plugin::mujoco {

class MuJoCoPlugin : public IPlugin,
                     public IPanelProvider,
                     public INodeProvider,
                     public ICommandProvider,
                     public ISimulationProvider,
                     public IEnvironmentProvider,
                     public IPreferencesProvider {
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
        return PluginPermission::UIModify | PluginPermission::Training;
    }
    PluginState GetState() const override { return state_; }
    void SetImGuiContext(void* ctx) override { ImGui::SetCurrentContext(static_cast<ImGuiContext*>(ctx)); }
    void* QueryInterface(const char* name) override {
        if (std::string(name) == "IPanelProvider") return static_cast<IPanelProvider*>(this);
        if (std::string(name) == "INodeProvider") return static_cast<INodeProvider*>(this);
        if (std::string(name) == "ICommandProvider") return static_cast<ICommandProvider*>(this);
        if (std::string(name) == "ISimulationProvider") return static_cast<ISimulationProvider*>(this);
        if (std::string(name) == "IEnvironmentProvider") return static_cast<IEnvironmentProvider*>(this);
        if (std::string(name) == "IPreferencesProvider") return static_cast<IPreferencesProvider*>(this);
        return nullptr;
    }

    // ===== IPreferencesProvider: the Viewport's saved choices =====
    std::vector<PluginPreferencesSection> GetPreferenceSections() override;
    void RenderPreferences(const std::string& section_id) override;

    // ===== ISimulationProvider: graph runs =====
    // A graph run with a MuJoCo Plant holds the physics: the viewport's own
    // run cannot start meanwhile, and the graph run is refused while it goes.
    std::vector<PluginSimulationNodeInfo> GetSimulationNodes() override;
    PluginSimulationBeginResult BeginGraphRun(const std::vector<PluginSimulationGraphNode>& nodes) override;
    void EndGraphRun() override;

    // ===== IEnvironmentProvider: tasks for reinforcement learning =====
    // Gymnasium's MuJoCo tasks (env/mj_task.h); each batch has its own
    // model and copies, apart from the viewport's model.
    std::vector<PluginEnvironmentInfo> GetEnvironments() override;
    // As the settings make it: Look ahead adds to the observation (from the
    // listed description, kept: the Engine asks while it draws Properties)
    bool DescribeEnvironment(const std::string& id, const std::map<std::string, std::string>& settings,
                             PluginEnvironmentInfo& info, std::string& error) override;
    PluginEnvironmentCreateResult CreateEnvironments(const std::string& id,
                                                     const std::map<std::string, std::string>& settings,
                                                     int count) override;

    // ===== IPanelProvider =====
    std::vector<PluginPanelInfo> GetPanels() override;
    void RenderPanel(const std::string& panel_id, bool* visible) override;

    // ===== ICommandProvider: the Simulation menu =====
    std::vector<PluginCommandInfo> GetCommands() override;
    PluginCommandState GetCommandState(const std::string& command_id) override;
    void RunCommand(const std::string& command_id) override;

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
    PluginContext* ctx_ = nullptr;  // from OnLoad until OnUnload

    std::string loaded_mjcf_path_;  // Path of the currently loaded MJCF model
    std::string assets_dir_;        // the plugin's assets folder (task models)

    // Observation Filter running normalisation, one per settings (see EvaluateNode)
    std::map<std::string, rl::ObservationFilter> obs_filters_;
    // The graph run's clock: the Plant keeps the physics on it
    MjGraphClock graph_clock_;
    bool viewport_visible_ = true;
    bool renderer_needs_init_ = false;

    // Policy playback (TOFIX195 S5.2): a training copy the Engine shows here.
    // Batches publish into the slot; each frame the viewport follows it.
    void SyncPlayback();
    std::shared_ptr<MjViewSlot> view_slot_ = std::make_shared<MjViewSlot>();
    uint64_t shown_version_ = 0;
    bool playback_ = false;
    // What the viewport showed before playback, given back when it ends
    std::string restore_path_;
    std::string restore_title_;
    std::string restore_subtitle_;
    MjScene restore_scene_;  // the scene of the model open before a playback
    std::string shown_as_ = "policy playback";  // what the last shown copy was (its status, lower case)

    // The Viewport's Stress panel (TOFIX212 P4): its requests go to the played
    // copy (the view slot), or act on the Viewport's own run here
    void ForwardStressRequests();
    // The played copy's scene in the Viewport's model (the same build as the batch's)
    void MirrorPlaybackScene(const MjScene& scene);
    bool SceneFailed(const MjScene& scene) const;  // the played scene did not build here
    std::optional<MjScene> failed_scene_;
    MjStressRun local_stress_;
    // The tasks' listed descriptions (GetEnvironments), for DescribeEnvironment
    std::mutex described_mutex_;
    std::map<std::string, PluginEnvironmentInfo> described_;
    int local_episodes_seen_ = 0;
};

} // namespace cyxwiz::plugin::mujoco
