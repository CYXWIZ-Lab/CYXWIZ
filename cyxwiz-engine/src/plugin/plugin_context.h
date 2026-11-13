// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

#include "plugin_types.h"
#include <filesystem>
#include <string>

namespace cyxwiz::plugin {

// Forward declarations for interfaces
class INodeProvider;
class IPanelProvider;
class IDataProvider;
class ITrainingHook;
class IAnalyticsProvider;
class PluginUi;

struct AssistantContextSnapshot {
    std::string engine_version;
    std::string build_id;
    std::string workspace_root;
    std::string active_graph_path;
    std::string selected_run_id;
    std::string selected_node_id;
    std::string selected_trace_id;
    std::string selected_panel;
    std::string debugger_context_json;
    std::string training_context_json;
};

// ============================================================================
// PluginContext - The sole API surface for plugins to interact with the engine.
//
// An interface the Engine implements (EnginePluginContext): every call runs
// the Engine's code, so a plugin needs no library of its own to call it, and
// registrations reach the Engine's registries, not a copy inside the plugin
// (TOFIX125 P5 step 5.2).
// ============================================================================

class PluginContext {
public:
    virtual ~PluginContext() = default;

    // === Identity ===
    virtual const std::string& GetPluginId() const = 0;
    virtual const std::filesystem::path& GetPluginDir() const = 0;

    // === Assistant context ===
    virtual AssistantContextSnapshot GetAssistantContextSnapshot() const = 0;

    // === Logging (the Engine's log, prefixed with the plugin id) ===
    virtual void LogInfo(const std::string& message) = 0;
    virtual void LogWarn(const std::string& message) = 0;
    virtual void LogError(const std::string& message) = 0;
    virtual void LogDebug(const std::string& message) = 0;

    // === Registration ===
    // Each checks the plugin's declared and approved permissions first.

    // Register custom node types (requires UIModify permission)
    virtual bool RegisterNodeProvider(INodeProvider* provider) = 0;

    // Register custom ImGui panels (requires UIModify permission)
    virtual bool RegisterPanelProvider(IPanelProvider* provider) = 0;

    // Register custom dataset loaders (requires DataRegistry permission)
    virtual bool RegisterDataProvider(IDataProvider* provider) = 0;

    // Register training lifecycle hooks (requires Training permission)
    virtual bool RegisterTrainingHook(ITrainingHook* hook) = 0;

    // Register custom analytics computations (requires DataRegistry permission)
    virtual bool RegisterAnalyticsProvider(IAnalyticsProvider* provider) = 0;

    // === UI (plugin API 1.4) ===
    // The Engine's buttons, fields and theme colours for plugin panels
    // (plugin_ui.h). Null where the host draws no windows (command-line
    // tools); the Engine always provides it. Declare api_version 1.4.0 or
    // newer in plugin.json to call it.
    virtual PluginUi* GetUi() = 0;

    // Opens one of this plugin's own panels and brings it to the front (a
    // button in one panel that leads to another). False when the plugin has
    // no panel with that id.
    virtual bool ShowPanel(const std::string& panel_id) = 0;

    // Keeps the Engine drawing at full frame rate for the next second. Call
    // it every frame while something in your panel moves (a running
    // simulation, an animation): without mouse or key input the Engine draws
    // 10 frames a second.
    virtual void RequestFullFrameRate() = 0;

    // === Project and settings (plugin API 1.4) ===
    // The open project's folder (it holds exports/, models/ ...); empty when
    // no project is open.
    virtual std::filesystem::path GetProjectDir() const = 0;

    // The user's settings for this plugin, kept by the Engine between
    // sessions (with its own settings). GetSetting returns `fallback` for a
    // key never set; SetSetting saves at once. Values are text: write
    // numbers and switches as text ("1.5", "on") and read them back.
    virtual std::string GetSetting(const std::string& key, const std::string& fallback = {}) const = 0;
    virtual void SetSetting(const std::string& key, const std::string& value) = 0;

protected:
    PluginContext() = default;
    PluginContext(const PluginContext&) = delete;
    PluginContext& operator=(const PluginContext&) = delete;
};

} // namespace cyxwiz::plugin
