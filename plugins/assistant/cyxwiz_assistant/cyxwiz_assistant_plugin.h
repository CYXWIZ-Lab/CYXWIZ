#pragma once

#include "assistant_backend_contract.h"

#include "plugin/plugin_context.h"
#include "plugin/plugin_types.h"
#include "plugin/interfaces/i_assistant_provider.h"

#include <memory>
#include <string>

namespace cyxwiz::plugin::assistant {

class CyxWizAssistantPlugin final : public IPlugin,
                                    public IAssistantProvider {
public:
    CyxWizAssistantPlugin();
    ~CyxWizAssistantPlugin() override = default;

    bool OnLoad(PluginContext& ctx) override;
    bool OnInitialize(PluginContext& ctx) override;
    void OnShutdown(PluginContext& ctx) override;
    void OnUnload(PluginContext& ctx) override;

    const PluginManifest& GetManifest() const override { return manifest_; }
    PluginPermissionFlags GetRequiredPermissions() const override {
        return PluginPermission::FileSystem | PluginPermission::Network;
    }
    PluginState GetState() const override { return state_; }
    void* QueryInterface(const char* name) override;

    AssistantCommandResponse RunAssistantCommand(
        const AssistantCommandRequest& request) override;

private:
    static std::filesystem::path ResolveDefaultPackPath(const PluginContext* context);
    AssistantRequest BuildCommandRequest(const AssistantCommandRequest& command) const;
    bool HasSelectedNodeContext() const;
    bool HasSelectedTraceContext() const;
    bool HasTrainingTerminalContext() const;

    PluginManifest manifest_;
    PluginState state_ = PluginState::Unloaded;
    PluginContext* context_ = nullptr;
    std::unique_ptr<IAssistantBackend> backend_;
};

} // namespace cyxwiz::plugin::assistant
