#include "plugin_manager.h"
#include "plugin_context.h"
#include <imgui.h>
#include "security/safe_execute.h"
#include "security/permission_store.h"
#include "registries/plugin_node_registry.h"
#include "registries/plugin_panel_registry.h"
#include "registries/plugin_data_loader_registry.h"
#include "registries/plugin_training_hook_manager.h"
#include "registries/plugin_analytics_registry.h"
#include "interfaces/i_panel_provider.h"
#include "interfaces/i_node_provider.h"
#include "interfaces/i_training_hook.h"
#include "interfaces/i_data_provider.h"
#include "interfaces/i_analytics_provider.h"
#include "interfaces/i_assistant_provider.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <queue>
#include <set>

namespace cyxwiz::plugin {

// ============================================================================
// Singleton
// ============================================================================

PluginManager& PluginManager::Instance() {
    static PluginManager instance;
    return instance;
}

PluginManager::~PluginManager() {
    ShutdownAll();
    UnloadAll();
}

// ============================================================================
// Discovery
// ============================================================================

void PluginManager::SetSearchPaths(const std::vector<std::filesystem::path>& paths) {
    std::lock_guard lock(mutex_);
    search_paths_ = paths;
    spdlog::info("PluginManager: {} search paths configured", search_paths_.size());
    for (const auto& p : search_paths_) {
        spdlog::debug("  Plugin search path: {}", p.string());
    }
}

std::vector<std::filesystem::path> PluginManager::DiscoverPlugins() const {
    std::vector<std::filesystem::path> plugin_dirs;

    for (const auto& search_path : search_paths_) {
        if (!std::filesystem::exists(search_path)) {
            continue;
        }

        try {
            // Use recursive iterator to find nested plugins (e.g., simulation/mujoco/)
            for (const auto& entry : std::filesystem::recursive_directory_iterator(search_path)) {
                if (!entry.is_directory()) continue;

                // Check if directory contains plugin.json
                auto manifest_path = entry.path() / "plugin.json";
                if (std::filesystem::exists(manifest_path)) {
                    plugin_dirs.push_back(entry.path());
                }
            }
        } catch (const std::filesystem::filesystem_error& e) {
            spdlog::warn("PluginManager: Error scanning {}: {}", search_path.string(), e.what());
        }
    }

    spdlog::info("PluginManager: Discovered {} plugin directories", plugin_dirs.size());
    return plugin_dirs;
}

// ============================================================================
// Loading
// ============================================================================

bool PluginManager::LoadPlugin(const std::filesystem::path& plugin_dir) {
    // Pre-check: parse manifest to get ID before loading DLL.
    // Avoids loading (and then unloading) a DLL that's already in use,
    // which would FreeLibrary the active plugin's code and crash.
    {
        PluginManifest manifest;
        std::string manifest_error;
        if (PluginLoader::ParseManifest(plugin_dir, manifest, manifest_error)) {
            std::lock_guard lock(mutex_);
            if (plugins_.count(manifest.id)) {
                spdlog::debug("PluginManager: Plugin {} is already loaded", manifest.id);
                return true;
            }
        }
    }

    std::string error;
    auto loaded = PluginLoader::LoadFromDirectory(plugin_dir, error);

    if (!loaded) {
        spdlog::error("PluginManager: Failed to load plugin from {}: {}", plugin_dir.string(), error);
        return false;
    }

    const std::string& id = loaded->manifest.id;

    std::lock_guard lock(mutex_);

    // Double-check for duplicate (race condition guard)
    if (plugins_.count(id)) {
        spdlog::warn("PluginManager: Plugin {} already loaded, skipping", id);
        return false;
    }

    spdlog::info("PluginManager: Loaded plugin {} ({})", loaded->manifest.name, id);
    plugins_[id] = std::move(loaded);
    return true;
}

void PluginManager::LoadAllFromSearchPaths() {
    auto plugin_dirs = DiscoverPlugins();

    for (const auto& dir : plugin_dirs) {
        LoadPlugin(dir);
    }
}

// ============================================================================
// Lifecycle
// ============================================================================

bool PluginManager::InitializePlugin(const std::string& plugin_id) {
    std::lock_guard lock(mutex_);

    auto it = plugins_.find(plugin_id);
    if (it == plugins_.end()) {
        spdlog::error("PluginManager: Cannot initialize unknown plugin {}", plugin_id);
        return false;
    }

    auto& plugin = it->second;

    if (plugin->state == PluginState::Initialized ||
        plugin->state == PluginState::Active) {
        return true;
    }

    if (plugin->state != PluginState::Loaded) {
        spdlog::warn("PluginManager: Plugin {} is in state {}, expected Loaded",
                     plugin_id, PluginStateToString(plugin->state));
        return false;
    }

    if (!plugin->instance) {
        spdlog::error("PluginManager: Plugin {} has no instance", plugin_id);
        plugin->state = PluginState::Failed;
        return false;
    }

    // Check for undecided dangerous permissions
    {
        auto& perm_store = security::PermissionStore::Instance();
        auto undecided = perm_store.GetUndecidedDangerous(
            plugin_id, plugin->manifest.version.ToString(), plugin->manifest.permissions);

        if (!undecided.empty()) {
            // Queue permission dialog — initialization will be deferred
            if (permission_dialog_.RequestApproval(plugin->manifest, undecided,
                    [this, plugin_id](const security::PermissionDialog::PendingApproval& approval) {
                        auto& store = security::PermissionStore::Instance();
                        for (size_t i = 0; i < approval.permissions.size(); ++i) {
                            store.SetDecision(
                                approval.plugin_id, approval.plugin_version,
                                approval.permissions[i],
                                approval.allowed[i] ? security::PermissionDecision::Allowed
                                                    : security::PermissionDecision::Denied);
                        }
                        store.Save();
                        spdlog::info("PluginManager: Permission decisions saved for '{}', retrying init",
                                     approval.plugin_name);
                        // Retry initialization now that decisions are stored
                        InitializePlugin(plugin_id);
                    })) {
                spdlog::info("PluginManager: Plugin {} awaiting permission approval", plugin_id);
                return false;
            }
        }
    }

    // Create PluginContext for this plugin
    auto ctx = std::make_unique<PluginContext>(plugin_id, plugin->instance, plugin->plugin_dir);

    // Share engine's ImGui context with the DLL so plugin UI rendering works
    plugin->instance->SetImGuiContext(ImGui::GetCurrentContext());

    // Call OnLoad (with crash isolation)
    {
        bool load_ok = false;
        auto result = security::SafeExecuteBool(plugin_id, "OnLoad",
            [&]() { return plugin->instance->OnLoad(*ctx); }, load_ok);
        if (result.crashed || !result.success) {
            plugin->state = PluginState::Failed;
            plugin->error_message = result.crashed ? result.error_message : "OnLoad() failed";
            return false;
        }
        if (!load_ok) {
            spdlog::error("PluginManager: Plugin {} OnLoad() returned false", plugin_id);
            plugin->state = PluginState::Failed;
            plugin->error_message = "OnLoad() returned false";
            return false;
        }
    }

    // Call OnInitialize (with crash isolation)
    {
        bool init_ok = false;
        auto result = security::SafeExecuteBool(plugin_id, "OnInitialize",
            [&]() { return plugin->instance->OnInitialize(*ctx); }, init_ok);
        if (result.crashed || !result.success) {
            plugin->state = PluginState::Failed;
            plugin->error_message = result.crashed ? result.error_message : "OnInitialize() failed";
            return false;
        }
        if (!init_ok) {
            spdlog::error("PluginManager: Plugin {} OnInitialize() returned false", plugin_id);
            plugin->state = PluginState::Failed;
            plugin->error_message = "OnInitialize() returned false";
            return false;
        }
    }

    // Engine-side registration via QueryInterface (bypasses DLL singleton duplication).
    // Virtual dispatch on IPlugin* resolves to DLL code, returning correct interface pointers.
    // Each registration is wrapped in SafeExecute to catch crashes from buggy plugins.
    {
        auto* instance = plugin->instance;

        // Register panels (with crash isolation)
        if (auto* p = static_cast<IPanelProvider*>(instance->QueryInterface("IPanelProvider"))) {
            auto reg_result = security::SafeExecute(plugin_id, "RegisterPanels", [&]() {
                auto dll_panels = p->GetPanels();
                for (const auto& pi : dll_panels) {
                    PluginPanelRegistry::Instance().RegisterDirect(
                        plugin_id,
                        std::string(pi.panel_id.c_str()),
                        std::string(pi.title.c_str()),
                        std::string(pi.category.c_str()),
                        pi.show_by_default, p);
                }
                spdlog::info("PluginManager: Registered {} panels for '{}'", dll_panels.size(), plugin_id);
            });
            if (reg_result.crashed) {
                plugin->state = PluginState::Failed;
                plugin->error_message = reg_result.error_message;
                return false;
            }
        }

        // Register nodes (with crash isolation)
        if (auto* p = static_cast<INodeProvider*>(instance->QueryInterface("INodeProvider"))) {
            struct EnumState {
                std::string plugin_id;
                INodeProvider* provider;
                PluginNodeTypeInfo current;
                size_t count = 0;
            };
            EnumState state;
            state.plugin_id = plugin_id;
            state.provider = p;

            NodeTypeCallback cb{};
            cb.user_data = &state;
            cb.on_node = [](void* ud, const char* type_name, const char* display_name,
                           const char* category, const char* description,
                           uint32_t color, const char* icon,
                           bool supports_dynamic_pins, const char* dynamic_pin_trigger) {
                auto* s = static_cast<EnumState*>(ud);
                s->current = PluginNodeTypeInfo{};
                s->current.type_name = type_name ? type_name : "";
                s->current.display_name = display_name ? display_name : "";
                s->current.category = category ? category : "";
                s->current.description = description ? description : "";
                s->current.color = color;
                s->current.icon = icon ? icon : "";
                s->current.supports_dynamic_pins = supports_dynamic_pins;
                s->current.dynamic_pin_trigger = dynamic_pin_trigger ? dynamic_pin_trigger : "";
            };
            cb.on_pin = [](void* ud, const char* name, const char* type, bool is_input) {
                auto* s = static_cast<EnumState*>(ud);
                s->current.pins.push_back({name ? name : "", type ? type : "", is_input});
            };
            cb.on_param = [](void* ud, const char* key, const char* value) {
                auto* s = static_cast<EnumState*>(ud);
                if (key && value) {
                    s->current.default_parameters[key] = value;
                }
            };
            cb.on_node_done = [](void* ud) {
                auto* s = static_cast<EnumState*>(ud);
                PluginNodeRegistry::Instance().RegisterDirect(
                    s->plugin_id, std::move(s->current), s->provider);
                s->count++;
            };

            auto reg_result = security::SafeExecute(plugin_id, "RegisterNodes", [&]() {
                p->EnumerateNodeTypes(cb);
            });
            if (reg_result.crashed) {
                plugin->state = PluginState::Failed;
                plugin->error_message = reg_result.error_message;
                return false;
            }
            spdlog::info("PluginManager: Registered {} nodes for '{}'", state.count, plugin_id);
        }

        // Register training hook (with crash isolation)
        if (auto* p = static_cast<ITrainingHook*>(instance->QueryInterface("ITrainingHook"))) {
            auto reg_result = security::SafeExecute(plugin_id, "RegisterTrainingHook", [&]() {
                PluginTrainingHookManager::Instance().RegisterHook(plugin_id, p);
            });
            if (!reg_result.crashed) {
                spdlog::info("PluginManager: Registered ITrainingHook for '{}'", plugin_id);
            }
        }

        // Register data provider
        if (auto* p = static_cast<IDataProvider*>(instance->QueryInterface("IDataProvider"))) {
            PluginDataLoaderRegistry::Instance().Register(plugin_id, p);
        }

        // Register analytics provider
        if (auto* p = static_cast<IAnalyticsProvider*>(instance->QueryInterface("IAnalyticsProvider"))) {
            PluginAnalyticsRegistry::Instance().Register(plugin_id, p);
        }
    }

    plugin->state = PluginState::Initialized;
    contexts_[plugin_id] = std::move(ctx);

    spdlog::info("PluginManager: Plugin {} initialized successfully", plugin_id);
    return true;
}

void PluginManager::InitializeAll() {
    // Get load order from dependency resolution
    auto order = ResolveLoadOrder();

    size_t ready_count = 0;
    size_t initialized_count = 0;
    for (const auto& id : order) {
        const auto state = GetPluginState(id);
        if (state == PluginState::Initialized || state == PluginState::Active) {
            ready_count++;
            continue;
        }
        if (state != PluginState::Loaded) {
            continue;
        }
        if (InitializePlugin(id)) {
            ready_count++;
            initialized_count++;
        }
    }

    spdlog::info("PluginManager: {}/{} plugins ready ({} initialized)",
                 ready_count, order.size(), initialized_count);
}

void PluginManager::ShutdownPlugin(const std::string& plugin_id) {
    std::lock_guard assistant_lock(assistant_command_mutex_);
    std::lock_guard lock(mutex_);

    auto it = plugins_.find(plugin_id);
    if (it == plugins_.end()) return;

    auto& plugin = it->second;
    if (plugin->state != PluginState::Initialized && plugin->state != PluginState::Active) {
        return;
    }

    auto ctx_it = contexts_.find(plugin_id);
    if (ctx_it == contexts_.end()) return;

    security::SafeExecute(plugin_id, "OnShutdown",
        [&]() { plugin->instance->OnShutdown(*ctx_it->second); });

    // Cleanup all registry registrations for this plugin
    PluginNodeRegistry::Instance().RemoveByPlugin(plugin_id);
    PluginPanelRegistry::Instance().RemoveByPlugin(plugin_id);
    PluginDataLoaderRegistry::Instance().RemoveByPlugin(plugin_id);
    PluginTrainingHookManager::Instance().RemoveByPlugin(plugin_id);
    PluginAnalyticsRegistry::Instance().RemoveByPlugin(plugin_id);

    plugin->state = PluginState::Loaded;
    contexts_.erase(ctx_it);

    spdlog::info("PluginManager: Plugin {} shut down", plugin_id);
}

void PluginManager::ShutdownAll() {
    // Get order, then shutdown outside lock
    auto order = ResolveLoadOrder();
    std::reverse(order.begin(), order.end());

    for (const auto& id : order) {
        ShutdownPlugin(id);
    }
}

void PluginManager::UnloadPlugin(const std::string& plugin_id) {
    std::lock_guard assistant_lock(assistant_command_mutex_);
    std::lock_guard lock(mutex_);

    // Clear any pending permission requests for this plugin
    permission_dialog_.ClearPendingForPlugin(plugin_id);

    auto it = plugins_.find(plugin_id);
    if (it == plugins_.end()) return;

    auto& plugin = it->second;
    auto ctx_it = contexts_.find(plugin_id);

    // Shutdown if initialized/active
    if (plugin->state == PluginState::Initialized || plugin->state == PluginState::Active) {
        if (plugin->instance && ctx_it != contexts_.end()) {
            security::SafeExecute(plugin_id, "OnShutdown",
                [&]() { plugin->instance->OnShutdown(*ctx_it->second); });
        }
        plugin->state = PluginState::Loaded;
    }

    // Call OnUnload while context still exists (with crash isolation)
    if (plugin->instance && ctx_it != contexts_.end()) {
        security::SafeExecute(plugin_id, "OnUnload",
            [&]() { plugin->instance->OnUnload(*ctx_it->second); });
    }

    spdlog::info("PluginManager: Unloading plugin {}", plugin_id);
    contexts_.erase(plugin_id);
    plugins_.erase(it);
}

void PluginManager::UnloadAll() {
    auto order = ResolveLoadOrder();
    std::reverse(order.begin(), order.end());

    for (const auto& id : order) {
        UnloadPlugin(id);
    }
}

bool PluginManager::EnablePlugin(const std::string& plugin_id) {
    std::lock_guard lock(mutex_);

    auto it = plugins_.find(plugin_id);
    if (it == plugins_.end()) return false;

    auto& plugin = it->second;
    if (plugin->state == PluginState::Disabled) {
        plugin->state = PluginState::Loaded;
        // Re-initialize will be done by caller
        return true;
    }
    return false;
}

void PluginManager::DisablePlugin(const std::string& plugin_id) {
    ShutdownPlugin(plugin_id);

    std::lock_guard lock(mutex_);
    auto it = plugins_.find(plugin_id);
    if (it != plugins_.end()) {
        it->second->state = PluginState::Disabled;
        spdlog::info("PluginManager: Plugin {} disabled", plugin_id);
    }
}

// ============================================================================
// Querying
// ============================================================================

IPlugin* PluginManager::GetPlugin(const std::string& plugin_id) const {
    std::lock_guard lock(mutex_);
    auto it = plugins_.find(plugin_id);
    if (it == plugins_.end()) return nullptr;
    return it->second->instance;
}

LoadedPlugin* PluginManager::GetLoadedPlugin(const std::string& plugin_id) const {
    std::lock_guard lock(mutex_);
    auto it = plugins_.find(plugin_id);
    if (it == plugins_.end()) return nullptr;
    return it->second.get();
}

std::vector<const LoadedPlugin*> PluginManager::GetAllPlugins() const {
    std::lock_guard lock(mutex_);
    std::vector<const LoadedPlugin*> result;
    result.reserve(plugins_.size());
    for (const auto& [id, plugin] : plugins_) {
        result.push_back(plugin.get());
    }
    return result;
}

std::vector<std::string> PluginManager::GetPluginIds() const {
    std::lock_guard lock(mutex_);
    std::vector<std::string> ids;
    ids.reserve(plugins_.size());
    for (const auto& [id, _] : plugins_) {
        ids.push_back(id);
    }
    return ids;
}

size_t PluginManager::GetPluginCount() const {
    std::lock_guard lock(mutex_);
    return plugins_.size();
}

PluginState PluginManager::GetPluginState(const std::string& plugin_id) const {
    std::lock_guard lock(mutex_);
    auto it = plugins_.find(plugin_id);
    if (it == plugins_.end()) return PluginState::Unloaded;
    return it->second->state;
}

void PluginManager::SetPluginState(const std::string& plugin_id, PluginState state) {
    std::lock_guard lock(mutex_);
    auto it = plugins_.find(plugin_id);
    if (it != plugins_.end()) {
        it->second->state = state;
    }
}

PluginContext* PluginManager::GetContext(const std::string& plugin_id) const {
    std::lock_guard lock(mutex_);
    auto it = contexts_.find(plugin_id);
    if (it == contexts_.end()) return nullptr;
    return it->second.get();
}

void PluginManager::SetAssistantContextSnapshotForAll(
    const AssistantContextSnapshot& snapshot) {
    std::lock_guard lock(mutex_);
    for (auto& [id, context] : contexts_) {
        if (context) {
            context->SetAssistantContextSnapshot(snapshot);
        }
    }
}

AssistantCommandResponse PluginManager::RunAssistantCommand(
    const AssistantCommandRequest& request) {
    std::lock_guard assistant_lock(assistant_command_mutex_);

    IAssistantProvider* provider = nullptr;
    {
        std::lock_guard lock(mutex_);
        for (const auto& [plugin_id, plugin] : plugins_) {
            if (!plugin || !plugin->instance) {
                continue;
            }
            if (plugin->state != PluginState::Initialized &&
                plugin->state != PluginState::Active) {
                continue;
            }
            provider = static_cast<IAssistantProvider*>(
                plugin->instance->QueryInterface("IAssistantProvider"));
            if (provider) {
                break;
            }
        }
    }

    if (!provider) {
        AssistantCommandResponse response;
        response.handled = false;
        response.success = false;
        response.error = "No assistant provider plugin is loaded.";
        return response;
    }

    auto response = provider->RunAssistantCommand(request);
    response.handled = true;
    return response;
}

// ============================================================================
// Dependency Resolution (Kahn's topological sort)
// ============================================================================

std::vector<std::string> PluginManager::ResolveLoadOrder() const {
    std::lock_guard lock(mutex_);

    // Build dependency graph
    std::unordered_map<std::string, std::vector<std::string>> graph;
    // Snapshot of loaded plugin IDs for filtering
    std::set<std::string> loaded_ids;

    for (const auto& [id, plugin] : plugins_) {
        loaded_ids.insert(id);
        if (graph.find(id) == graph.end()) {
            graph[id] = {};
        }

        for (const auto& dep : plugin->manifest.dependencies) {
            // Only add dependency edges for plugins that are actually loaded
            if (plugins_.count(dep.plugin_id)) {
                graph[id].push_back(dep.plugin_id);
            } else {
                spdlog::warn("PluginManager: Plugin {} depends on {} which is not loaded",
                             id, dep.plugin_id);
            }
            if (graph.find(dep.plugin_id) == graph.end()) {
                graph[dep.plugin_id] = {};
            }
        }
    }

    return TopologicalSortInternal(graph, loaded_ids);
}

std::vector<std::string> PluginManager::TopologicalSortInternal(
    const std::unordered_map<std::string, std::vector<std::string>>& graph,
    const std::set<std::string>& loaded_ids
) const {
    // Build reverse graph: edge = "is depended upon by"
    // In our graph, edges point from plugin -> its dependencies.
    // For topological order, we want dependencies loaded FIRST.
    std::unordered_map<std::string, std::vector<std::string>> reverse_graph;
    std::unordered_map<std::string, int> reverse_in_degree;

    for (const auto& [node, _] : graph) {
        reverse_graph[node] = {};
        reverse_in_degree[node] = 0;
    }

    for (const auto& [node, deps] : graph) {
        reverse_in_degree[node] = static_cast<int>(deps.size());
        for (const auto& dep : deps) {
            reverse_graph[dep].push_back(node);
        }
    }

    // Kahn's algorithm
    std::queue<std::string> ready;
    for (const auto& [node, deg] : reverse_in_degree) {
        if (deg == 0) {
            ready.push(node);
        }
    }

    std::vector<std::string> order;
    while (!ready.empty()) {
        auto current = ready.front();
        ready.pop();

        // Only include if actually loaded
        if (loaded_ids.count(current)) {
            order.push_back(current);
        }

        for (const auto& dependent : reverse_graph[current]) {
            reverse_in_degree[dependent]--;
            if (reverse_in_degree[dependent] == 0) {
                ready.push(dependent);
            }
        }
    }

    // Check for circular dependencies
    if (order.size() < loaded_ids.size()) {
        spdlog::error("PluginManager: Circular dependency detected! Ordered {} of {} plugins",
                     order.size(), loaded_ids.size());
    }

    return order;
}

// ============================================================================
// Security — Permission Dialog Rendering
// ============================================================================

void PluginManager::RenderPermissionDialogs() {
    permission_dialog_.Render();
}

} // namespace cyxwiz::plugin
