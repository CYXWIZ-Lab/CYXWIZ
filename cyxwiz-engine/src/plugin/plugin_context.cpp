#include "plugin_context.h"
#include "security/permission_store.h"
#include "../core/extension_node_registry.h"
#include "registries/plugin_panel_registry.h"
#include "registries/plugin_data_loader_registry.h"
#include "registries/plugin_training_hook_manager.h"
#include "registries/plugin_analytics_registry.h"
#include <spdlog/spdlog.h>

namespace cyxwiz::plugin {

static const char* PermissionName(PluginPermission p) {
    switch (p) {
        case PluginPermission::FileSystem:     return "FileSystem";
        case PluginPermission::Network:        return "Network";
        case PluginPermission::SystemCommands: return "SystemCommands";
        case PluginPermission::Python:         return "Python";
        case PluginPermission::GPU:            return "GPU";
        case PluginPermission::DataRegistry:   return "DataRegistry";
        case PluginPermission::Training:       return "Training";
        case PluginPermission::UIModify:       return "UIModify";
        default: return "Unknown";
    }
}

bool PluginContext::CheckPermission(PluginPermission required, const char* action) const {
    if (!plugin_) {
        spdlog::error("[Plugin:{}] No plugin instance for permission check", plugin_id_);
        return false;
    }

    // Check manifest declares the permission
    PluginPermissionFlags requested = plugin_->GetRequiredPermissions();
    if (!HasPermission(requested, required)) {
        spdlog::warn("[Plugin:{}] Permission denied for {}: not declared in manifest ({})",
                     plugin_id_, action, PermissionName(required));
        return false;
    }

    // For dangerous permissions, also check PermissionStore approval
    if (security::PermissionStore::IsDangerousPermission(required)) {
        auto& manifest = plugin_->GetManifest();
        auto granted = security::PermissionStore::Instance().GetGrantedPermissions(
            plugin_id_, manifest.version.ToString(), requested);
        if (!HasPermission(granted, required)) {
            spdlog::warn("[Plugin:{}] Permission denied for {}: {} not approved by user",
                         plugin_id_, action, PermissionName(required));
            return false;
        }
    }

    return true;
}

bool PluginContext::RegisterNodeProvider(INodeProvider* provider) {
    if (!CheckPermission(PluginPermission::UIModify, "RegisterNodeProvider")) return false;
    if (!provider) return false;

    // NOTE: GetNodeTypes() returns a vector allocated by the plugin library.
    // The engine registers through EnumerateNodeTypes in plugin_manager.cpp,
    // which copies strings before the library's vector is destroyed.
    bool all_registered = true;
    for (const auto& info : provider->GetNodeTypes()) {
        ExtensionNodeRegistry::Registration registration;
        registration.descriptor = DescribeSignalNode(plugin_id_, info);
        registration.descriptor.source_path = plugin_dir_.string();
        registration.signal_provider = provider;
        std::string error;
        if (!ExtensionNodeRegistry::Instance().Register(std::move(registration), error)) {
            spdlog::warn("[Plugin:{}] node not registered: {}", plugin_id_, error);
            all_registered = false;
        }
    }
    return all_registered;
}

bool PluginContext::RegisterPanelProvider(IPanelProvider* provider) {
    if (!CheckPermission(PluginPermission::UIModify, "RegisterPanelProvider")) return false;
    PluginPanelRegistry::Instance().Register(plugin_id_, provider);
    return true;
}

bool PluginContext::RegisterDataProvider(IDataProvider* provider) {
    if (!CheckPermission(PluginPermission::DataRegistry, "RegisterDataProvider")) return false;
    PluginDataLoaderRegistry::Instance().Register(plugin_id_, provider);
    return true;
}

bool PluginContext::RegisterTrainingHook(ITrainingHook* hook) {
    if (!CheckPermission(PluginPermission::Training, "RegisterTrainingHook")) return false;
    PluginTrainingHookManager::Instance().RegisterHook(plugin_id_, hook);
    return true;
}

bool PluginContext::RegisterAnalyticsProvider(IAnalyticsProvider* provider) {
    if (!CheckPermission(PluginPermission::DataRegistry, "RegisterAnalyticsProvider")) return false;
    PluginAnalyticsRegistry::Instance().Register(plugin_id_, provider);
    return true;
}

} // namespace cyxwiz::plugin
