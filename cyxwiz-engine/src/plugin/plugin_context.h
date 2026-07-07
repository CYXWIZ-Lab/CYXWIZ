#pragma once

#include "plugin_types.h"
#include <string>
#include <spdlog/spdlog.h>

namespace cyxwiz::plugin {

// Forward declarations for interfaces
class INodeProvider;
class IPanelProvider;
class IDataProvider;
class ITrainingHook;
class IAnalyticsProvider;

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
// Phase 1: Logging + identity.
// Phase 2: Registration methods for 5 subsystems.
// ============================================================================

class PluginContext {
public:
    PluginContext(const std::string& plugin_id, IPlugin* plugin,
                  std::filesystem::path plugin_dir = {})
        : plugin_id_(plugin_id), plugin_(plugin), plugin_dir_(std::move(plugin_dir)) {}

    ~PluginContext() = default;

    // Move-only: prevent accidental copies
    PluginContext(const PluginContext&) = delete;
    PluginContext& operator=(const PluginContext&) = delete;
    PluginContext(PluginContext&& other) noexcept
        : plugin_id_(std::move(other.plugin_id_)),
          plugin_(other.plugin_),
          assistant_context_(std::move(other.assistant_context_)) {
        other.plugin_ = nullptr;
    }
    PluginContext& operator=(PluginContext&& other) noexcept {
        if (this != &other) {
            plugin_id_ = std::move(other.plugin_id_);
            plugin_ = other.plugin_;
            assistant_context_ = std::move(other.assistant_context_);
            other.plugin_ = nullptr;
        }
        return *this;
    }

    // === Identity ===
    const std::string& GetPluginId() const { return plugin_id_; }
    const std::filesystem::path& GetPluginDir() const { return plugin_dir_; }

    // === Assistant context ===
    void SetAssistantContextSnapshot(const AssistantContextSnapshot& snapshot) {
        assistant_context_ = snapshot;
    }
    const AssistantContextSnapshot& GetAssistantContextSnapshot() const {
        return assistant_context_;
    }

    // === Logging ===
    void LogInfo(const std::string& message) {
        spdlog::info("[Plugin:{}] {}", plugin_id_, message);
    }
    void LogWarn(const std::string& message) {
        spdlog::warn("[Plugin:{}] {}", plugin_id_, message);
    }
    void LogError(const std::string& message) {
        spdlog::error("[Plugin:{}] {}", plugin_id_, message);
    }
    void LogDebug(const std::string& message) {
        spdlog::debug("[Plugin:{}] {}", plugin_id_, message);
    }

    // === Phase 2: Registration APIs ===
    // Each checks permissions before delegating to the appropriate registry.

    // Register custom node types (requires UIModify permission)
    bool RegisterNodeProvider(INodeProvider* provider);

    // Register custom ImGui panels (requires UIModify permission)
    bool RegisterPanelProvider(IPanelProvider* provider);

    // Register custom dataset loaders (requires DataRegistry permission)
    bool RegisterDataProvider(IDataProvider* provider);

    // Register training lifecycle hooks (requires Training permission)
    bool RegisterTrainingHook(ITrainingHook* hook);

    // Register custom analytics computations (requires DataRegistry permission)
    bool RegisterAnalyticsProvider(IAnalyticsProvider* provider);

private:
    std::string plugin_id_;
    IPlugin* plugin_;  // Non-owning
    std::filesystem::path plugin_dir_;
    AssistantContextSnapshot assistant_context_;

    // Permission check helper
    bool CheckPermission(PluginPermission required, const char* action) const;
};

} // namespace cyxwiz::plugin
