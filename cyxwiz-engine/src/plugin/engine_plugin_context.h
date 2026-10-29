#pragma once

// The Engine's PluginContext (TOFIX125 P5 step 5.2): one per loaded plugin,
// owned by the PluginManager. Plugins see only the PluginContext interface.

#include "plugin_context.h"

#include <mutex>

namespace cyxwiz::plugin {

class EnginePluginContext final : public PluginContext {
public:
    // plugin_version: from the plugin.json the Engine read, which keys the
    // user's permission approvals (not the plugin's own GetManifest()).
    // plugin_api: the plugin API plugin.json declares.
    EnginePluginContext(const std::string& plugin_id, std::string plugin_version, PluginVersion plugin_api,
                        IPlugin* plugin, std::filesystem::path plugin_dir = {})
        : plugin_id_(plugin_id), plugin_version_(std::move(plugin_version)), plugin_api_(plugin_api),
          plugin_(plugin), plugin_dir_(std::move(plugin_dir)) {}

    const std::string& GetPluginId() const override { return plugin_id_; }
    const std::filesystem::path& GetPluginDir() const override { return plugin_dir_; }

    AssistantContextSnapshot GetAssistantContextSnapshot() const override;
    // Engine side: the Engine updates what the assistant plugin reads.
    void SetAssistantContextSnapshot(const AssistantContextSnapshot& snapshot);

    void LogInfo(const std::string& message) override;
    void LogWarn(const std::string& message) override;
    void LogError(const std::string& message) override;
    void LogDebug(const std::string& message) override;

    bool RegisterNodeProvider(INodeProvider* provider) override;
    bool RegisterPanelProvider(IPanelProvider* provider) override;
    bool RegisterDataProvider(IDataProvider* provider) override;
    bool RegisterTrainingHook(ITrainingHook* hook) override;
    bool RegisterAnalyticsProvider(IAnalyticsProvider* provider) override;

    // Whether the plugin may do `action` (declared in plugin.json, and
    // approved when the permission is dangerous); logs the refusal.
    bool CheckPermission(PluginPermission required, const char* action) const;
    static const char* PermissionLabel(PluginPermission permission);

    // Whether the plugin is built for the current IDataProvider /
    // IAnalyticsProvider (kDataInterfacesApi); logs the refusal.
    bool CheckDataInterfaces(const char* interface_name) const;

private:

    std::string plugin_id_;
    std::string plugin_version_;
    PluginVersion plugin_api_;
    IPlugin* plugin_;  // Non-owning
    std::filesystem::path plugin_dir_;
    AssistantContextSnapshot assistant_context_;
    mutable std::mutex assistant_context_mutex_;
};

} // namespace cyxwiz::plugin
