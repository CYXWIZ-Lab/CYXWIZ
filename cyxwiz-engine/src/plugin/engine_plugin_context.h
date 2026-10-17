#pragma once

// The Engine's PluginContext (TOFIX125 P5 step 5.2): one per loaded plugin,
// owned by the PluginManager. Plugins see only the PluginContext interface.

#include "plugin_context.h"

#include <mutex>

namespace cyxwiz::plugin {

class EnginePluginContext final : public PluginContext {
public:
    EnginePluginContext(const std::string& plugin_id, IPlugin* plugin, std::filesystem::path plugin_dir = {})
        : plugin_id_(plugin_id), plugin_(plugin), plugin_dir_(std::move(plugin_dir)) {}

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

private:
    bool CheckPermission(PluginPermission required, const char* action) const;

    std::string plugin_id_;
    IPlugin* plugin_;  // Non-owning
    std::filesystem::path plugin_dir_;
    AssistantContextSnapshot assistant_context_;
    mutable std::mutex assistant_context_mutex_;
};

} // namespace cyxwiz::plugin
