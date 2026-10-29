#include "engine_plugin_context.h"
#include "security/permission_store.h"
#include "../core/extension_node_registry.h"
#include "registries/plugin_panel_registry.h"
#include "registries/plugin_data_loader_registry.h"
#include "registries/plugin_training_hook_manager.h"
#include "registries/plugin_analytics_registry.h"
#include <spdlog/spdlog.h>

namespace cyxwiz::plugin {

AssistantContextSnapshot EnginePluginContext::GetAssistantContextSnapshot() const {
    std::lock_guard lock(assistant_context_mutex_);
    return assistant_context_;
}

void EnginePluginContext::SetAssistantContextSnapshot(const AssistantContextSnapshot& snapshot) {
    std::lock_guard lock(assistant_context_mutex_);
    assistant_context_ = snapshot;
}

void EnginePluginContext::LogInfo(const std::string& message) { spdlog::info("[Plugin:{}] {}", plugin_id_, message); }
void EnginePluginContext::LogWarn(const std::string& message) { spdlog::warn("[Plugin:{}] {}", plugin_id_, message); }
void EnginePluginContext::LogError(const std::string& message) { spdlog::error("[Plugin:{}] {}", plugin_id_, message); }
void EnginePluginContext::LogDebug(const std::string& message) { spdlog::debug("[Plugin:{}] {}", plugin_id_, message); }

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

const char* EnginePluginContext::PermissionLabel(PluginPermission permission) {
    return PermissionName(permission);
}

bool EnginePluginContext::CheckPermission(PluginPermission required, const char* action) const {
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
        auto granted = security::PermissionStore::Instance().GetGrantedPermissions(
            plugin_id_, plugin_version_, requested);
        if (!HasPermission(granted, required)) {
            spdlog::warn("[Plugin:{}] Permission denied for {}: {} not approved by user",
                         plugin_id_, action, PermissionName(required));
            return false;
        }
    }

    return true;
}

bool EnginePluginContext::RegisterNodeProvider(INodeProvider* provider) {
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

bool EnginePluginContext::RegisterPanelProvider(IPanelProvider* provider) {
    if (!CheckPermission(PluginPermission::UIModify, "RegisterPanelProvider")) return false;
    PluginPanelRegistry::Instance().Register(plugin_id_, provider);
    return true;
}

bool EnginePluginContext::CheckDataInterfaces(const char* interface_name) const {
    if (plugin_api_ >= kDataInterfacesApi) return true;
    spdlog::warn("[Plugin:{}] {} not used: it changed in plugin API {} and the plugin is built for {}; "
                 "rebuild it against the current SDK",
                 plugin_id_, interface_name, kDataInterfacesApi.ToString(), plugin_api_.ToString());
    return false;
}

bool EnginePluginContext::RegisterDataProvider(IDataProvider* provider) {
    if (!CheckPermission(PluginPermission::DataRegistry, "RegisterDataProvider")) return false;
    if (!CheckDataInterfaces("IDataProvider")) return false;
    PluginDataLoaderRegistry::Instance().Register(plugin_id_, plugin_version_, provider);
    return true;
}

bool EnginePluginContext::RegisterTrainingHook(ITrainingHook* hook) {
    if (!CheckPermission(PluginPermission::Training, "RegisterTrainingHook")) return false;
    PluginTrainingHookManager::Instance().RegisterHook(plugin_id_, hook);
    return true;
}

bool EnginePluginContext::RegisterAnalyticsProvider(IAnalyticsProvider* provider) {
    if (!CheckPermission(PluginPermission::DataRegistry, "RegisterAnalyticsProvider")) return false;
    if (!CheckDataInterfaces("IAnalyticsProvider")) return false;
    PluginAnalyticsRegistry::Instance().Register(plugin_id_, provider);
    return true;
}

} // namespace cyxwiz::plugin
