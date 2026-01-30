#include "plugin_manager.h"
#include "plugin_context.h"
#include "registries/plugin_node_registry.h"
#include "registries/plugin_panel_registry.h"
#include "registries/plugin_data_loader_registry.h"
#include "registries/plugin_training_hook_manager.h"
#include "registries/plugin_analytics_registry.h"
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
            for (const auto& entry : std::filesystem::directory_iterator(search_path)) {
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
    std::string error;
    auto loaded = PluginLoader::LoadFromDirectory(plugin_dir, error);

    if (!loaded) {
        spdlog::error("PluginManager: Failed to load plugin from {}: {}", plugin_dir.string(), error);
        return false;
    }

    const std::string& id = loaded->manifest.id;

    std::lock_guard lock(mutex_);

    // Check for duplicate
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

    // Create PluginContext for this plugin
    auto ctx = std::make_unique<PluginContext>(plugin_id, plugin->instance);

    // Call OnLoad
    try {
        if (!plugin->instance->OnLoad(*ctx)) {
            spdlog::error("PluginManager: Plugin {} OnLoad() returned false", plugin_id);
            plugin->state = PluginState::Failed;
            plugin->error_message = "OnLoad() returned false";
            return false;
        }
    } catch (const std::exception& e) {
        spdlog::error("PluginManager: Plugin {} OnLoad() threw: {}", plugin_id, e.what());
        plugin->state = PluginState::Failed;
        plugin->error_message = "OnLoad() threw: " + std::string(e.what());
        return false;
    } catch (...) {
        spdlog::error("PluginManager: Plugin {} OnLoad() threw unknown exception", plugin_id);
        plugin->state = PluginState::Failed;
        plugin->error_message = "OnLoad() threw unknown exception";
        return false;
    }

    // Call OnInitialize
    try {
        if (!plugin->instance->OnInitialize(*ctx)) {
            spdlog::error("PluginManager: Plugin {} OnInitialize() returned false", plugin_id);
            plugin->state = PluginState::Failed;
            plugin->error_message = "OnInitialize() returned false";
            return false;
        }
    } catch (const std::exception& e) {
        spdlog::error("PluginManager: Plugin {} OnInitialize() threw: {}", plugin_id, e.what());
        plugin->state = PluginState::Failed;
        plugin->error_message = "OnInitialize() threw: " + std::string(e.what());
        return false;
    } catch (...) {
        spdlog::error("PluginManager: Plugin {} OnInitialize() threw unknown exception", plugin_id);
        plugin->state = PluginState::Failed;
        plugin->error_message = "OnInitialize() threw unknown exception";
        return false;
    }

    plugin->state = PluginState::Initialized;
    contexts_[plugin_id] = std::move(ctx);

    spdlog::info("PluginManager: Plugin {} initialized successfully", plugin_id);
    return true;
}

void PluginManager::InitializeAll() {
    // Get load order from dependency resolution
    auto order = ResolveLoadOrder();

    size_t success_count = 0;
    for (const auto& id : order) {
        if (InitializePlugin(id)) {
            success_count++;
        }
    }

    spdlog::info("PluginManager: Initialized {}/{} plugins", success_count, order.size());
}

void PluginManager::ShutdownPlugin(const std::string& plugin_id) {
    std::lock_guard lock(mutex_);

    auto it = plugins_.find(plugin_id);
    if (it == plugins_.end()) return;

    auto& plugin = it->second;
    if (plugin->state != PluginState::Initialized && plugin->state != PluginState::Active) {
        return;
    }

    auto ctx_it = contexts_.find(plugin_id);
    if (ctx_it == contexts_.end()) return;

    try {
        plugin->instance->OnShutdown(*ctx_it->second);
    } catch (const std::exception& e) {
        spdlog::error("PluginManager: Plugin {} OnShutdown() threw: {}", plugin_id, e.what());
    } catch (...) {
        spdlog::error("PluginManager: Plugin {} OnShutdown() threw unknown exception", plugin_id);
    }

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
    std::lock_guard lock(mutex_);

    auto it = plugins_.find(plugin_id);
    if (it == plugins_.end()) return;

    auto& plugin = it->second;
    auto ctx_it = contexts_.find(plugin_id);

    // Shutdown if initialized/active
    if (plugin->state == PluginState::Initialized || plugin->state == PluginState::Active) {
        if (plugin->instance && ctx_it != contexts_.end()) {
            try {
                plugin->instance->OnShutdown(*ctx_it->second);
            } catch (const std::exception& e) {
                spdlog::error("PluginManager: Plugin {} OnShutdown() threw: {}", plugin_id, e.what());
            } catch (...) {
                spdlog::error("PluginManager: Plugin {} OnShutdown() threw unknown exception", plugin_id);
            }
        }
        plugin->state = PluginState::Loaded;
    }

    // Call OnUnload while context still exists
    if (plugin->instance && ctx_it != contexts_.end()) {
        try {
            plugin->instance->OnUnload(*ctx_it->second);
        } catch (...) {
            spdlog::error("PluginManager: Plugin {} OnUnload() threw", plugin_id);
        }
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

} // namespace cyxwiz::plugin
