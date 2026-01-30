#pragma once

#include "plugin_types.h"
#include <memory>
#include <filesystem>
#include <string>

namespace cyxwiz::plugin {

// ============================================================================
// LoadedPlugin - Represents a plugin loaded from a DLL/SO
// ============================================================================

struct LoadedPlugin {
    void* dll_handle = nullptr;             // HMODULE on Windows, void* on Unix
    std::filesystem::path dll_path;         // Full path to DLL
    std::filesystem::path plugin_dir;       // Plugin directory (contains plugin.json)
    IPlugin* instance = nullptr;            // Plugin instance (owned via destroy_func)
    CreatePluginFunc create_func = nullptr;
    DestroyPluginFunc destroy_func = nullptr;
    PluginManifest manifest;
    PluginState state = PluginState::Unloaded;
    std::string error_message;

    ~LoadedPlugin();
};

// ============================================================================
// PluginLoader - Cross-platform DLL loading
// ============================================================================

class PluginLoader {
public:
    // Load plugin from a directory containing plugin.json
    // Returns nullptr on failure, sets error_out
    static std::unique_ptr<LoadedPlugin> LoadFromDirectory(
        const std::filesystem::path& plugin_dir,
        std::string& error_out
    );

    // Load plugin from a specific DLL path (no manifest)
    static std::unique_ptr<LoadedPlugin> LoadFromDLL(
        const std::filesystem::path& dll_path,
        std::string& error_out
    );

    // Unload a plugin (destroy instance, free DLL)
    static void Unload(LoadedPlugin& plugin);

private:
    // Platform abstraction
    static void* LoadDLL(const std::filesystem::path& path, std::string& error_out);
    static void  UnloadDLL(void* handle);
    static void* GetSymbol(void* handle, const char* name);
    static std::string GetLastDLLError();

    // Parse plugin.json from directory
    static bool ParseManifest(
        const std::filesystem::path& plugin_dir,
        PluginManifest& manifest_out,
        std::string& error_out
    );

    // Validate API version compatibility
    static bool ValidateApiVersion(
        const PluginVersion& plugin_api,
        std::string& error_out
    );
};

} // namespace cyxwiz::plugin
