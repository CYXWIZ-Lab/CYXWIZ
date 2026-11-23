#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace cyxwiz::runtime {

struct ActivePack {
    std::string backend;
    std::string pack_id;
    std::filesystem::path directory;
};

struct ActivePackState {
    std::string backend;
    std::string pack_id;
};

struct ActiveRuntimeState {
    std::string runtime_set_id;
    std::uint64_t generation = 0;
    std::string base_pack_id;
    std::vector<ActivePackState> packs;
};

// runtime/active-plugins.json: the active plugin packs, kept apart from
// active-runtime.json so plugin activation never changes the ArrayFire runtime
// identity, its generation, loader paths or route evidence (TOFIX195).
inline constexpr const char* kActivePluginsFileName = "active-plugins.json";

struct ActivePluginPackState {
    std::string plugin_id;
    std::string pack_id;
};

struct ActivePluginsState {
    // The base the plugins were activated with; plugins of another base are inactive.
    std::string base_pack_id;
    std::vector<ActivePluginPackState> plugins;
};

struct ActivePluginPack {
    std::string plugin_id;
    std::string pack_id;
    std::filesystem::path directory;
};

struct ActiveRuntime {
    std::string runtime_set_id;
    std::uint64_t generation = 0;
    std::string base_pack_id;
    std::filesystem::path runtime_root;
    std::filesystem::path base_directory;
    std::filesystem::path engine_executable;
    std::vector<ActivePack> packs;
    std::vector<std::filesystem::path> dll_directories;
};

bool ResolveActiveRuntime(
    const std::filesystem::path& runtime_root,
    ActiveRuntime& output,
    std::string& error);
bool LoadActiveRuntimeState(
    const std::filesystem::path& path,
    ActiveRuntimeState& output,
    std::string& error);
bool SaveActiveRuntimeStateAtomic(
    const std::filesystem::path& path,
    const ActiveRuntimeState& state,
    std::string& error);
bool ResolveRuntimeState(
    const std::filesystem::path& runtime_root,
    const ActiveRuntimeState& state,
    ActiveRuntime& output,
    std::string& error);

// A missing file is an empty state (no plugin pack was ever activated).
bool LoadActivePluginsState(
    const std::filesystem::path& path,
    ActivePluginsState& output,
    std::string& error);
bool SaveActivePluginsStateAtomic(
    const std::filesystem::path& path,
    const ActivePluginsState& state,
    std::string& error);
// runtime_root/plugins/<plugin-id>/<pack-id>
std::filesystem::path PluginPackDirectory(
    const std::filesystem::path& runtime_root,
    const std::string& plugin_id,
    const std::string& pack_id);
// The active plugin packs of the active base. Entries recorded for another base
// are inactive and skipped; an entry whose folder is missing, escapes the runtime
// root or has no plugin.json is skipped and described in `skipped`. Fails only
// when active-plugins.json itself is unreadable or invalid.
bool ResolveActivePluginPacks(
    const std::filesystem::path& runtime_root,
    const std::string& active_base_pack_id,
    std::vector<ActivePluginPack>& output,
    std::vector<std::string>& skipped,
    std::string& error);

void AppendBootstrapDiagnostic(
    const std::filesystem::path& runtime_root,
    const std::string& message);

}  // namespace cyxwiz::runtime
