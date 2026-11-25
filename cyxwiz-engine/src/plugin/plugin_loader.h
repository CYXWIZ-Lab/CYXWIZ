#pragma once

#include "plugin_pack.h"
#include "plugin_types.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <filesystem>
#include <string>
#include <vector>

namespace cyxwiz::plugin {

// ============================================================================
// LoadedPlugin - Represents a plugin loaded from a DLL/SO
// ============================================================================

struct LoadedPlugin {
    void* dll_handle = nullptr;             // HMODULE on Windows, void* on Unix
    std::filesystem::path dll_path;         // Full path to DLL
    std::filesystem::path loaded_from;      // The private copy of dll_path that is loaded
    std::filesystem::path plugin_dir;       // Plugin directory (contains plugin.json)
    IPlugin* instance = nullptr;            // Plugin instance (owned via destroy_func)
    CreatePluginFunc create_func = nullptr;
    DestroyPluginFunc destroy_func = nullptr;
    PluginManifest manifest;
    PluginState state = PluginState::Unloaded;
    std::string error_message;
    // Why nodes the plugin offered were not added (wrong pins, bad ids);
    // filled when the plugin is initialised.
    std::vector<std::string> skipped_nodes;

    // Where it came from: a developer folder (as built), or an installed
    // add-on checked against its signed inventory before it loaded.
    enum class Origin { Developer, SignedPack };
    Origin origin = Origin::Developer;
    std::string pack_id;                 // a SignedPack's pack and version
    std::string package_version;
    std::size_t pack_files_checked = 0;  // what the check read, and how long it took
    std::uint64_t pack_bytes_checked = 0;
    double pack_check_ms = 0.0;
    std::chrono::system_clock::time_point pack_checked_at{};

    ~LoadedPlugin();
};

// ============================================================================
// PluginLoader - Cross-platform DLL loading
// ============================================================================

class PluginLoader {
public:
    // Load plugin from a directory containing plugin.json
    // Returns nullptr on failure, sets error_out.
    // `pack`: an installed add-on. Its folder is checked against the signed
    // inventory before plugin.json is read, its libraries are copied and the
    // copies hashed (held open against writes) until they are loaded; a
    // difference refuses the load and is described in `pack_result`.
    static std::unique_ptr<LoadedPlugin> LoadFromDirectory(
        const std::filesystem::path& plugin_dir,
        std::string& error_out,
        const PackLoadCheck* pack = nullptr,
        PackLoadResult* pack_result = nullptr
    );

    // Load plugin from a specific DLL path (no manifest)
    static std::unique_ptr<LoadedPlugin> LoadFromDLL(
        const std::filesystem::path& dll_path,
        std::string& error_out
    );

    // Unload a plugin (destroy instance, free DLL)
    static void Unload(LoadedPlugin& plugin);

    // Removes the per-user copies of plugin libraries that no Engine has
    // loaded (call once at start-up, before plugins load).
    static void PruneLibraryCache();

    // Parse plugin.json from directory (public for pre-load duplicate checks)
    static bool ParseManifest(
        const std::filesystem::path& plugin_dir,
        PluginManifest& manifest_out,
        std::string& error_out
    );

private:
    // Platform abstraction
    static void* LoadDLL(const std::filesystem::path& path, std::string& error_out);
    static void  UnloadDLL(void* handle);
    static void* GetSymbol(void* handle, const char* name);
    static std::string GetLastDLLError();

    // The plugin's build fingerprint against this Engine's (TOFIX125 P5).
    // False with the reason when it is missing or differs.
    static bool CheckBuildFingerprint(void* handle, std::string& error_out);

    // Validate API version compatibility
    static bool ValidateApiVersion(
        const PluginVersion& plugin_api,
        std::string& error_out
    );
};

} // namespace cyxwiz::plugin
