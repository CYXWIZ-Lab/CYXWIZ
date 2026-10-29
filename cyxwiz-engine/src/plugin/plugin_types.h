// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

#include <cstdio>  // sscanf in PluginVersion::Parse
#include <string>
#include <vector>
#include <cstdint>
#include <functional>
#include <filesystem>
#include "plugin_fingerprint.h"

namespace cyxwiz::plugin {

// ============================================================================
// Plugin API Version
// ============================================================================

constexpr uint16_t PLUGIN_API_VERSION_MAJOR = 1;
constexpr uint16_t PLUGIN_API_VERSION_MINOR = 2;  // 1.1: ITensorNodeProvider; 1.2: build fingerprint (TOFIX125)
constexpr uint16_t PLUGIN_API_VERSION_PATCH = 0;

// ============================================================================
// PluginVersion - Semantic versioning
// ============================================================================

struct PluginVersion {
    uint16_t major = 0;
    uint16_t minor = 0;
    uint16_t patch = 0;

    static PluginVersion Parse(const std::string& str) {
        PluginVersion v;
        if (sscanf(str.c_str(), "%hu.%hu.%hu", &v.major, &v.minor, &v.patch) < 1) {
            v = {0, 0, 0};
        }
        return v;
    }

    std::string ToString() const {
        return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
    }

    bool IsCompatibleWith(const PluginVersion& engine_api) const {
        // Major must match exactly, plugin minor must be <= engine minor
        if (major != engine_api.major) return false;
        if (minor > engine_api.minor) return false;
        return true;
    }

    bool operator==(const PluginVersion& o) const {
        return major == o.major && minor == o.minor && patch == o.patch;
    }
    bool operator<(const PluginVersion& o) const {
        if (major != o.major) return major < o.major;
        if (minor != o.minor) return minor < o.minor;
        return patch < o.patch;
    }
    bool operator<=(const PluginVersion& o) const { return *this == o || *this < o; }
    bool operator>(const PluginVersion& o) const { return !(*this <= o); }
    bool operator>=(const PluginVersion& o) const { return !(*this < o); }
    bool operator!=(const PluginVersion& o) const { return !(*this == o); }
};

// ============================================================================
// Plugin Permissions (bitmask)
// ============================================================================

enum class PluginPermission : uint32_t {
    None           = 0,
    FileSystem     = 1 << 0,
    Network        = 1 << 1,
    SystemCommands = 1 << 2,
    Python         = 1 << 3,
    GPU            = 1 << 4,
    DataRegistry   = 1 << 5,
    Training       = 1 << 6,
    UIModify       = 1 << 7,
};

using PluginPermissionFlags = uint32_t;

inline PluginPermissionFlags operator|(PluginPermission a, PluginPermission b) {
    return static_cast<uint32_t>(a) | static_cast<uint32_t>(b);
}
inline PluginPermissionFlags operator|(PluginPermissionFlags a, PluginPermission b) {
    return a | static_cast<uint32_t>(b);
}
inline bool HasPermission(PluginPermissionFlags flags, PluginPermission p) {
    return (flags & static_cast<uint32_t>(p)) != 0;
}

inline PluginPermissionFlags ParsePermissions(const std::vector<std::string>& perms) {
    PluginPermissionFlags flags = 0;
    for (const auto& p : perms) {
        if (p == "FileSystem")          flags |= static_cast<uint32_t>(PluginPermission::FileSystem);
        else if (p == "Network")        flags |= static_cast<uint32_t>(PluginPermission::Network);
        else if (p == "SystemCommands") flags |= static_cast<uint32_t>(PluginPermission::SystemCommands);
        else if (p == "Python")         flags |= static_cast<uint32_t>(PluginPermission::Python);
        else if (p == "GPU")            flags |= static_cast<uint32_t>(PluginPermission::GPU);
        else if (p == "DataRegistry")   flags |= static_cast<uint32_t>(PluginPermission::DataRegistry);
        else if (p == "Training")       flags |= static_cast<uint32_t>(PluginPermission::Training);
        else if (p == "UIModify")       flags |= static_cast<uint32_t>(PluginPermission::UIModify);
    }
    return flags;
}

inline std::vector<std::string> PermissionsToStrings(PluginPermissionFlags flags) {
    std::vector<std::string> result;
    if (HasPermission(flags, PluginPermission::FileSystem))     result.push_back("FileSystem");
    if (HasPermission(flags, PluginPermission::Network))        result.push_back("Network");
    if (HasPermission(flags, PluginPermission::SystemCommands)) result.push_back("SystemCommands");
    if (HasPermission(flags, PluginPermission::Python))         result.push_back("Python");
    if (HasPermission(flags, PluginPermission::GPU))            result.push_back("GPU");
    if (HasPermission(flags, PluginPermission::DataRegistry))   result.push_back("DataRegistry");
    if (HasPermission(flags, PluginPermission::Training))       result.push_back("Training");
    if (HasPermission(flags, PluginPermission::UIModify))       result.push_back("UIModify");
    return result;
}

// ============================================================================
// Plugin Capability Flags (what the plugin provides)
// ============================================================================

enum class PluginCapability : uint32_t {
    None            = 0,
    ProvidesNodes   = 1 << 0,
    ProvidesPanels  = 1 << 1,
    ProvidesData    = 1 << 2,
    ProvidesTraining = 1 << 3,
    ProvidesAnalytics = 1 << 4,
    RequiresPython  = 1 << 5,
    RequiresGPU     = 1 << 6,
    RequiresNetwork = 1 << 7,
    ProvidesTensorNodes = 1 << 8,  // trainable nodes, ITensorNodeProvider (API 1.1)
};

using PluginCapabilityFlags = uint32_t;

inline PluginCapabilityFlags operator|(PluginCapability a, PluginCapability b) {
    return static_cast<uint32_t>(a) | static_cast<uint32_t>(b);
}

// ============================================================================
// Plugin State Machine
// ============================================================================

enum class PluginState {
    Unloaded,       // DLL not loaded
    Loaded,         // DLL loaded, CreatePlugin called, manifest read
    Initialized,    // OnInitialize called, capabilities registered
    Active,         // Running (alias for Initialized in v1)
    Failed,         // Error occurred, plugin disabled
    Disabled,       // User explicitly disabled
};

inline const char* PluginStateToString(PluginState state) {
    switch (state) {
        case PluginState::Unloaded:    return "Unloaded";
        case PluginState::Loaded:      return "Loaded";
        case PluginState::Initialized: return "Initialized";
        case PluginState::Active:      return "Active";
        case PluginState::Failed:      return "Failed";
        case PluginState::Disabled:    return "Disabled";
        default: return "Unknown";
    }
}

// ============================================================================
// Dependency declaration
// ============================================================================

struct PluginDependency {
    std::string plugin_id;
    std::string version_constraint;  // e.g. ">=1.0.0", "^1.0.0"
};

// ============================================================================
// Plugin Manifest (loaded from plugin.json by the Engine's loader)
// ============================================================================

struct PluginManifest {
    // Identity
    std::string id;                     // "com.vendor.myplugin"
    std::string name;                   // "My Plugin"
    std::string description;
    PluginVersion version;
    PluginVersion api_version;          // Required engine API version

    // Metadata
    std::string author;
    std::string license;
    std::string homepage;
    std::string repository;

    // Capabilities and requirements
    PluginCapabilityFlags capabilities = 0;
    PluginPermissionFlags permissions = 0;
    std::vector<PluginDependency> dependencies;

    // Platform-specific library path (relative to plugin directory)
    std::string library;                // e.g. "bin/myplugin.dll"

    // Ed25519 signature of the DLL (hex-encoded)
    std::string signature;

    // Plugin directory (set at load time, not from JSON)
    std::filesystem::path plugin_dir;

    bool IsValid() const {
        return !id.empty() && !name.empty() && (version.major > 0 || version.minor > 0 || version.patch > 0);
    }
};

// ============================================================================
// IPlugin Interface - Base class all plugins must implement
// ============================================================================

class PluginContext;  // Forward declaration

class IPlugin {
public:
    virtual ~IPlugin() = default;

    // Lifecycle (called in order: OnLoad -> OnInitialize -> [running] -> OnShutdown -> OnUnload)
    virtual bool OnLoad(PluginContext& ctx) = 0;
    virtual bool OnInitialize(PluginContext& ctx) = 0;
    virtual void OnShutdown(PluginContext& ctx) = 0;
    virtual void OnUnload(PluginContext& ctx) = 0;

    // Metadata
    virtual const PluginManifest& GetManifest() const = 0;
    virtual PluginPermissionFlags GetRequiredPermissions() const = 0;
    virtual PluginState GetState() const = 0;

    // Interface query (COM-style) — returns typed pointer for cross-DLL safety.
    // Plugins override to return their interface pointers (e.g. IPanelProvider*, INodeProvider*).
    // This avoids static singleton duplication across DLL boundaries.
    virtual void* QueryInterface(const char* /*interface_name*/) { return nullptr; }

    // Set the engine's ImGui context so DLL plugins share the same UI state.
    // Called by the engine before OnLoad. Plugins must call ImGui::SetCurrentContext(ctx)
    // in their implementation to ensure ImGui calls work across DLL boundaries.
    virtual void SetImGuiContext(void* /*ctx*/) {}
};

// ============================================================================
// Plugin Export Macros
// ============================================================================

#ifdef _WIN32
    #define CYXWIZ_PLUGIN_EXPORT extern "C" __declspec(dllexport)
#else
    #define CYXWIZ_PLUGIN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// Factory function signatures
using CreatePluginFunc  = IPlugin* (*)();
using DestroyPluginFunc = void (*)(IPlugin*);
using BuildFingerprintFunc = const char* (*)();

// Convenience macro for plugin authors. It also exports the plugin's build
// fingerprint (plugin_fingerprint.h): the Engine refuses a plugin built
// differently before creating it (API 1.2).
#define CYXWIZ_PLUGIN_ENTRY(PluginClass) \
    CYXWIZ_PLUGIN_EXPORT const char* CyxwizPluginBuildFingerprint() { \
        static const std::string fingerprint = ::cyxwiz::plugin::CurrentBuildFingerprint(); \
        return fingerprint.c_str(); \
    } \
    CYXWIZ_PLUGIN_EXPORT cyxwiz::plugin::IPlugin* CreatePlugin() { \
        return new PluginClass(); \
    } \
    CYXWIZ_PLUGIN_EXPORT void DestroyPlugin(cyxwiz::plugin::IPlugin* plugin) { \
        delete plugin; \
    }

} // namespace cyxwiz::plugin
