#include "plugin_context.h"
#include "registries/plugin_node_registry.h"
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
    PluginPermissionFlags flags = plugin_->GetRequiredPermissions();
    if (!HasPermission(flags, required)) {
        spdlog::warn("[Plugin:{}] Permission denied for {}: missing {}",
                     plugin_id_, action, PermissionName(required));
        return false;
    }
    return true;
}

bool PluginContext::RegisterNodeProvider(INodeProvider* provider) {
    if (!CheckPermission(PluginPermission::UIModify, "RegisterNodeProvider")) return false;
    PluginNodeRegistry::Instance().Register(plugin_id_, provider);
    return true;
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
