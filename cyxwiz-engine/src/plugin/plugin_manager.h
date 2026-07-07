#pragma once

#include "plugin_types.h"
#include "plugin_loader.h"
#include "security/permission_dialog.h"
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <set>
#include <filesystem>
#include <string>

namespace cyxwiz::plugin {

// Forward declarations
class PluginContext;
struct AssistantContextSnapshot;
struct AssistantCommandRequest;
struct AssistantCommandResponse;

// ============================================================================
// PluginManager - Singleton managing plugin discovery, loading, and lifecycle
// ============================================================================

class PluginManager {
public:
    static PluginManager& Instance();

    // Non-copyable
    PluginManager(const PluginManager&) = delete;
    PluginManager& operator=(const PluginManager&) = delete;

    // ===== Discovery =====

    // Set directories to search for plugins (in priority order)
    void SetSearchPaths(const std::vector<std::filesystem::path>& paths);
    const std::vector<std::filesystem::path>& GetSearchPaths() const { return search_paths_; }

    // Discover all plugin directories in search paths
    std::vector<std::filesystem::path> DiscoverPlugins() const;

    // ===== Loading =====

    // Load a single plugin from a directory containing plugin.json
    bool LoadPlugin(const std::filesystem::path& plugin_dir);

    // Load all plugins from search paths
    void LoadAllFromSearchPaths();

    // ===== Lifecycle =====

    // Initialize a loaded plugin (calls OnLoad + OnInitialize)
    bool InitializePlugin(const std::string& plugin_id);

    // Initialize all loaded plugins (respects dependency order)
    void InitializeAll();

    // Shutdown a running plugin (calls OnShutdown)
    void ShutdownPlugin(const std::string& plugin_id);

    // Shutdown all running plugins (reverse dependency order)
    void ShutdownAll();

    // Unload a plugin (calls OnUnload, frees DLL)
    void UnloadPlugin(const std::string& plugin_id);

    // Unload all plugins
    void UnloadAll();

    // Enable a disabled plugin
    bool EnablePlugin(const std::string& plugin_id);

    // Disable a running plugin
    void DisablePlugin(const std::string& plugin_id);

    // ===== Querying =====

    // Get plugin by ID (returns nullptr if not found)
    IPlugin* GetPlugin(const std::string& plugin_id) const;

    // Get loaded plugin info by ID
    LoadedPlugin* GetLoadedPlugin(const std::string& plugin_id) const;

    // Get all loaded plugins
    std::vector<const LoadedPlugin*> GetAllPlugins() const;

    // Get plugin IDs
    std::vector<std::string> GetPluginIds() const;

    // Get plugin count
    size_t GetPluginCount() const;

    // Get plugin state
    PluginState GetPluginState(const std::string& plugin_id) const;

    // Set plugin state (used by security/error handling)
    void SetPluginState(const std::string& plugin_id, PluginState state);

    // Get context for plugin
    PluginContext* GetContext(const std::string& plugin_id) const;

    // Publish assistant context to all initialized plugin contexts.
    void SetAssistantContextSnapshotForAll(const AssistantContextSnapshot& snapshot);

    // Run a slash-command request through the first loaded assistant provider.
    AssistantCommandResponse RunAssistantCommand(const AssistantCommandRequest& request);

    // ===== Dependency Resolution =====

    // Get load order respecting dependencies (topological sort)
    std::vector<std::string> ResolveLoadOrder() const;

    // ===== Security =====

    // Render permission approval dialogs (call from main render loop)
    void RenderPermissionDialogs();

private:
    PluginManager() = default;
    ~PluginManager();

    // Plugin storage: id -> LoadedPlugin
    std::unordered_map<std::string, std::unique_ptr<LoadedPlugin>> plugins_;

    // Plugin contexts: id -> PluginContext
    std::unordered_map<std::string, std::unique_ptr<PluginContext>> contexts_;

    // Search paths for plugin directories
    std::vector<std::filesystem::path> search_paths_;

    // Thread safety
    mutable std::recursive_mutex mutex_;
    mutable std::mutex assistant_command_mutex_;

    // Permission approval dialog
    security::PermissionDialog permission_dialog_;

    // Topological sort for dependency resolution (Kahn's algorithm)
    // Must be called with mutex_ already held.
    std::vector<std::string> TopologicalSortInternal(
        const std::unordered_map<std::string, std::vector<std::string>>& graph,
        const std::set<std::string>& loaded_ids
    ) const;
};

} // namespace cyxwiz::plugin
