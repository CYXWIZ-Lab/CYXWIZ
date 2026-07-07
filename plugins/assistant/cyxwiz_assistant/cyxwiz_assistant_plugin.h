#pragma once

#include "assistant_backend_contract.h"

#include "plugin/plugin_context.h"
#include "plugin/plugin_types.h"
#include "plugin/interfaces/i_assistant_provider.h"
#include "plugin/interfaces/i_panel_provider.h"

#include <imgui.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz::plugin::assistant {

class CyxWizAssistantPlugin final : public IPlugin,
                                    public IPanelProvider,
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
        return static_cast<uint32_t>(PluginPermission::UIModify);
    }
    PluginState GetState() const override { return state_; }
    void SetImGuiContext(void* ctx) override { ImGui::SetCurrentContext(static_cast<ImGuiContext*>(ctx)); }
    void* QueryInterface(const char* name) override;

    std::vector<PluginPanelInfo> GetPanels() override;
    void RenderPanel(const std::string& panel_id, bool* visible) override;
    AssistantCommandResponse RunAssistantCommand(
        const AssistantCommandRequest& request) override;

private:
    enum class ContextMode {
        General = 0,
        Trace = 1,
        Training = 2,
    };

    static std::string ContextModeToString(ContextMode mode);
    static std::filesystem::path ResolveDefaultPackPath(const PluginContext* context);
    AssistantRequest BuildRequest() const;
    AssistantRequest BuildCommandRequest(const AssistantCommandRequest& command) const;
    static std::string FormatCommandResponse(const AssistantResponse& response);
    const AssistantContextSnapshot* CurrentSnapshot() const;
    bool HasSelectedNodeContext() const;
    bool HasSelectedTraceContext() const;
    bool HasTrainingTerminalContext() const;
    void SetQuestionText(const std::string& question);
    void SetMissingContextResponse(const std::string& message);
    void ReloadBackend();
    void RunAsk();
    void RunExplainSelectedTrace();
    void RunExplainTrainingStopReason();
    void RunFindSelectedNodeSource();
    void RenderStatus() const;
    void RenderControls();
    void RenderResponse() const;

    PluginManifest manifest_;
    PluginState state_ = PluginState::Unloaded;
    PluginContext* context_ = nullptr;
    std::unique_ptr<IAssistantBackend> backend_;
    AssistantResponse last_response_;

    std::array<char, 1024> question_buffer_{};
    ContextMode context_mode_ = ContextMode::General;
    bool retrieval_only_ = false;
    bool show_citations_ = true;
    int top_k_ = 3;
    int timeout_seconds_ = 120;
    std::array<char, 512> knowledge_pack_path_buffer_{};
    std::array<char, 256> runtime_endpoint_buffer_{};
};

} // namespace cyxwiz::plugin::assistant
