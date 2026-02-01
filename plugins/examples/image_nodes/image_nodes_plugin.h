#pragma once

// Image Processing Nodes Plugin - Example CyxWiz Plugin
//
// Demonstrates:
//   - INodeProvider for custom node types in the visual editor
//   - IPanelProvider for a custom settings panel
//   - Code generation for PyTorch framework
//   - Multiple interface implementation in one plugin

#include <imgui.h>
#include "../../cyxwiz-engine/src/plugin/plugin_types.h"
#include "../../cyxwiz-engine/src/plugin/plugin_context.h"
#include "../../cyxwiz-engine/src/plugin/interfaces/i_node_provider.h"
#include "../../cyxwiz-engine/src/plugin/interfaces/i_panel_provider.h"

#include <string>
#include <vector>

namespace cyxwiz::plugin::examples {

class ImageNodesPlugin : public IPlugin,
                         public INodeProvider,
                         public IPanelProvider {
public:
    ImageNodesPlugin() = default;
    ~ImageNodesPlugin() override = default;

    // ===== IPlugin Lifecycle =====
    bool OnLoad(PluginContext& ctx) override;
    bool OnInitialize(PluginContext& ctx) override;
    void OnShutdown(PluginContext& ctx) override;
    void OnUnload(PluginContext& ctx) override;

    const PluginManifest& GetManifest() const override { return manifest_; }
    PluginPermissionFlags GetRequiredPermissions() const override {
        return static_cast<uint32_t>(PluginPermission::UIModify) |
               static_cast<uint32_t>(PluginPermission::DataRegistry);
    }
    PluginState GetState() const override { return state_; }
    void SetImGuiContext(void* ctx) override { ImGui::SetCurrentContext(static_cast<ImGuiContext*>(ctx)); }
    void* QueryInterface(const char* name) override {
        if (std::string(name) == "INodeProvider") return static_cast<INodeProvider*>(this);
        if (std::string(name) == "IPanelProvider") return static_cast<IPanelProvider*>(this);
        return nullptr;
    }

    // ===== INodeProvider =====
    std::vector<PluginNodeTypeInfo> GetNodeTypes() override;
    std::string GenerateCode(const std::string& node_type_name,
                             const std::map<std::string, std::string>& parameters,
                             const std::string& framework) override;

    // ===== IPanelProvider =====
    std::vector<PluginPanelInfo> GetPanels() override;
    void RenderPanel(const std::string& panel_id, bool* visible) override;

private:
    PluginManifest manifest_;
    PluginState state_ = PluginState::Unloaded;

    // Settings panel state
    int default_kernel_size_ = 5;
    bool use_gpu_ = true;
};

} // namespace cyxwiz::plugin::examples
