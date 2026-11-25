// A plugin whose OnInitialize fails, for test_plugin_unload_registrations:
// after a successful OnLoad the Engine must still give it OnUnload, with the
// context OnLoad got, while that context lives.
#include "../src/plugin/plugin_context.h"
#include "../src/plugin/plugin_types.h"

namespace {

class LifecycleTestPlugin : public cyxwiz::plugin::IPlugin {
public:
    bool OnLoad(cyxwiz::plugin::PluginContext& ctx) override {
        context_ = &ctx;
        ctx.LogInfo("Lifecycle test: OnLoad");
        return true;
    }
    bool OnInitialize(cyxwiz::plugin::PluginContext&) override { return false; }
    void OnShutdown(cyxwiz::plugin::PluginContext& ctx) override { ctx.LogInfo("Lifecycle test: OnShutdown"); }
    void OnUnload(cyxwiz::plugin::PluginContext& ctx) override {
        ctx.LogInfo(&ctx == context_ ? "Lifecycle test: OnUnload" : "Lifecycle test: OnUnload with another context");
        context_ = nullptr;
    }
    const cyxwiz::plugin::PluginManifest& GetManifest() const override { return manifest_; }
    cyxwiz::plugin::PluginPermissionFlags GetRequiredPermissions() const override { return 0; }
    cyxwiz::plugin::PluginState GetState() const override { return cyxwiz::plugin::PluginState::Loaded; }

private:
    cyxwiz::plugin::PluginManifest manifest_;
    cyxwiz::plugin::PluginContext* context_ = nullptr;
};

}  // namespace

CYXWIZ_PLUGIN_ENTRY(LifecycleTestPlugin)
