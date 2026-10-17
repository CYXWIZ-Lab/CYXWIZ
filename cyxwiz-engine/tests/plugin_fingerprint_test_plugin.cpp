// A plugin that does nothing, for test_plugin_fingerprint (TOFIX125 P5 step
// 5.1). Built three ways:
//   fingerprint_ok_plugin        as any plugin: CYXWIZ_PLUGIN_ENTRY
//   fingerprint_other_plugin     with FINGERPRINT_TEST_OTHER: reports a Debug build
//   fingerprint_missing_plugin   with FINGERPRINT_TEST_NO_FINGERPRINT: exports only
//                                CreatePlugin / DestroyPlugin, as before API 1.2
#ifdef FINGERPRINT_TEST_OTHER
// "Built in Debug" (with no ArrayFire); defined here: CMake would split ';'.
#define CYXWIZ_TEST_BUILD_FINGERPRINT     "sdk=0.2;api=1.2;compiler=msvc-19;stdlib=msvc-stl;build=debug;ptr=64;arrayfire=none"
#endif
#include "../src/plugin/plugin_types.h"

namespace {

class FingerprintTestPlugin : public cyxwiz::plugin::IPlugin {
public:
    bool OnLoad(cyxwiz::plugin::PluginContext&) override { return true; }
    bool OnInitialize(cyxwiz::plugin::PluginContext&) override { return true; }
    void OnShutdown(cyxwiz::plugin::PluginContext&) override {}
    void OnUnload(cyxwiz::plugin::PluginContext&) override {}
    const cyxwiz::plugin::PluginManifest& GetManifest() const override { return manifest_; }
    cyxwiz::plugin::PluginPermissionFlags GetRequiredPermissions() const override { return 0; }
    cyxwiz::plugin::PluginState GetState() const override { return cyxwiz::plugin::PluginState::Loaded; }

private:
    cyxwiz::plugin::PluginManifest manifest_;
};

}  // namespace

#ifdef FINGERPRINT_TEST_NO_FINGERPRINT
CYXWIZ_PLUGIN_EXPORT cyxwiz::plugin::IPlugin* CreatePlugin() { return new FingerprintTestPlugin(); }
CYXWIZ_PLUGIN_EXPORT void DestroyPlugin(cyxwiz::plugin::IPlugin* plugin) { delete plugin; }
#else
CYXWIZ_PLUGIN_ENTRY(FingerprintTestPlugin)
#endif
